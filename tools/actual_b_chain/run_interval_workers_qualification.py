"""Run the unchanged seventeen interval workers; emit engineering evidence only."""
from pathlib import Path, PurePosixPath, PureWindowsPath
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib
import uuid

JOBS = ('smoke', 'scanner', 'independent2272', 'public-baseline', 'public-candidate',
        'cache514', 'original', 'stage4', 'stream-baseline', 'stream-candidate',
        'training362', 'warm44', 'tenk-baseline', 'tenk-candidate', 'grammar-producer', 'large79', 'disk-fit')
RUNTIMES = ('1.12.6', '1.13.1')
DRIVER = 'test/test_interval_mean_forecaster.jl'
WORKER = 'test/interval_mean_forecaster/worker.jl'
BINDINGS = 'test/interval_mean_forecaster/bindings.jl'
OWNER = 'research/dbdt_completed_source/candidate_native_owner.py'
STAGE = 'tools/actual_b_chain/run_matched342_native.py'
MEMORY = 'tools/run_canonical_harness.py'
PINS = {
    DRIVER: '72dd65b36d313b39703c93a60a4a92a91146c786aa4e4069fac4e4aa99d1fbcd',
    WORKER: 'e3cd1ca92c85c544b812b8debf56920c47ba7fba49cd156dd9387e4b9df56842',
    BINDINGS: '5cbef72d434cae309d382c110f832566aefe068f1661b31c7fd9464f11c7c939',
    OWNER: 'c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00',
    'research/dbdt_completed_source/native_stage.py': '30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
    STAGE: 'ea2f94ec709b5e58086b69fbd142f902499d9a2d2f5dabc8e0c871c452b4533b',
    MEMORY: '219669fea1b718ab997215762105f74dfb5dd8c0f630bc2ed906f00c4eae2236',
}
IMF_PINS = {
    'DstIntervalObserver.jl': '48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816',
    'DstIntervalReplay.jl': 'f2962c092bc6bfd6be82a5b8ad95e2c494dddbc4a2addda4f57140e31d0c6fa5',
    'DstIntervalAdapter.jl': 'b432a8fc65d3e425a67ff2b225a6c0e5323ae7fdb8822dd228677b7998a5fe5a',
    'DstIntervalStream.jl': '20a63654c429aa8be219f857b1e83a2529f334499e38198d8f332a07f22fcd44',
    'DstIntervalTraining.jl': '2e13631e04cd48b36b033721935cee7dc47da76bd692f84ef0d3daae5c142b89',
    'DstIntervalArtifacts.jl': 'fbeb40ecfe1c9ff32a17ba6c97b9eadccdd4c88e924aeddeb7ce00616fc3c29e',
    'DstIntervalExperiment.jl': '50e9bd1876af69a927f1bf6211925285f532e2f6a040cff5460b12c308d0fedd',
    'sandu_fullchannel_features.jl': '2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c',
}
GIB = 1024**3
META, LOG, SOURCE, ARTIFACT = 4*1024**2, 64*1024**2, 64*1024**2, 2*GIB
WHOLE_SECONDS, SOURCE_SECONDS, NATIVE_SECONDS = 5500, 900, 3600
POST_RESERVE, FINAL_RESERVE = 900 + 20, 80
WARNING = 'Warning: detected a stack overflow; program state may be corrupted, so further execution might be unreliable.'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def clock(deadline):
    need(time.monotonic_ns() < deadline, 'original deadline exhausted')


def utc():
    return datetime.now(timezone.utc).isoformat()


def ordinary(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve(strict=True), 'canonical ordinary path required')
    for part in (path, *path.parents):
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
             'link/reparse path rejected')
    need(stat.S_ISDIR(path.stat().st_mode) if directory else stat.S_ISREG(path.stat().st_mode), 'ordinary kind differs')
    return path


def raw(path, limit, deadline):
    path = ordinary(path)
    before = path.stat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns)
    need(0 <= before.st_size <= limit, 'file byte bound')
    clock(deadline)
    with path.open('rb') as stream:
        need(identity(os.fstat(stream.fileno())) == identity(before), 'file changed before read')
        value = stream.read(limit + 1)
        need(identity(os.fstat(stream.fileno())) == identity(before), 'file changed while reading')
    need(len(value) == before.st_size and identity(path.stat()) == identity(before), 'file changed after read')
    clock(deadline)
    return value


def digest(path, deadline, limit=SOURCE):
    path = ordinary(path)
    before = path.stat()
    need(0 <= before.st_size <= limit, 'hash byte bound')
    value, total = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            clock(deadline)
            part = stream.read(65536)
            if not part:
                break
            total += len(part)
            need(total <= limit, 'file grew while hashing')
            value.update(part)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, total, after.st_mtime_ns), 'file changed while hashing')
    return value.hexdigest()


def strict(body):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(body, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))


def put(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')
    need(len(body) <= META, 'metadata exceeds4MiB')
    with Path(path).open('xb') as stream:
        need(stream.write(body) == len(body), 'short metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def import_file(path, expected, deadline, name):
    need(digest(path, deadline) == expected, 'pinned helper changed before import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(digest(path, deadline) == expected, 'pinned helper changed during import')
    return module


def parse_jobs(driver_bytes):
    text = driver_bytes.decode('utf8', errors='strict')
    begin, end = 'const JOBS = (', 'const EVIDENCE ='
    need(text.count(begin) == text.count(end) == 1, 'closed JOBS declaration required')
    literal = text.split(begin)[1].split(end)[0].strip()
    need(literal.endswith(')'), 'JOBS tuple terminator differs')
    values, rest = [], literal[:-1].strip()
    while rest:
        need(rest.startswith('"'), 'JOBS tuple must contain literal strings')
        stop = rest.find('"', 1)
        need(stop > 1, 'JOBS string incomplete')
        value = rest[1:stop]
        need(re.fullmatch('[a-z0-9-]+', value), 'JOBS string grammar differs')
        values.append(value)
        rest = rest[stop+1:].strip()
        if rest:
            need(rest.startswith(','), 'JOBS separator differs')
            rest = rest[1:].strip()
    need(tuple(values) == JOBS, 'exact ordered seventeen jobs required')
    return tuple(values)


def check_admission(available, disk):
    need(type(available) is int and type(disk) is int and available >= 8*GIB and disk >= 16*GIB,
         '8GiB available physical memory and16GiB disk required')
    return dict(available_physical_bytes=available, free_disk_bytes=disk)


def build_driver_argv(science_checkout, julia):
    return [str(julia), '--startup-file=no', '--pkgimages=no', '--threads=1',
            '--project=' + str(science_checkout), str(Path(science_checkout) / DRIVER)]


def require_native_budget(deadline_ns, *, now_ns):
    need(type(deadline_ns) is int and type(now_ns) is int and
         deadline_ns - now_ns > (NATIVE_SECONDS + POST_RESERVE + FINAL_RESERVE)*10**9,
         'complete native and postcheck reserves do not fit original clock')


def validate_native(native, launch, *, native_started_ns, native_deadline_ns):
    need(type(native['pid']) is int and native['pid'] > 0 and native['pid'] == launch['pid'] and
         type(native['native_exit']) is int and native['native_exit'] == 0 and native['timed_out'] is False and
         native['leader_reaped'] is True and native['ownership_boundary_closed'] is True and
         native['all_descendants_os_reaped'] is False, 'genuine untimed/reaped/closed native0 required')
    for wall in (native['wall_seconds'], native['wrapper_observed_wall_seconds']):
        need(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall < NATIVE_SECONDS,
             'finite original native wall required')
    need((native['ownership'] == 'start_new_session=True' and
          native['descendant_cleanup'] == 'owned POSIX session has no live members after stage') or
         (native['ownership'] == 'CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread' and
          native['descendant_cleanup'] == 'private Windows Job Object closed with KILL_ON_JOB_CLOSE'),
         'production direct ownership evidence differs')
    need(
         native_deadline_ns - native_started_ns == NATIVE_SECONDS*10**9 and
         native_started_ns <= launch['monotonic_ns'] < native_deadline_ns,
         'native launch/original clock differs')
    return True


def _origin_path(value, directory=False):
    need(isinstance(value, (str, Path)) and str(value) and not any(ord(c) < 32 for c in str(value)),
         'origin path text differs')
    text = str(value)
    windows = bool(re.match(r'^[A-Za-z]:[\\/]', text)) or text.startswith('\\\\')
    path = PureWindowsPath(text) if windows else PurePosixPath(text)
    normalized = text.replace('\\', '/') if windows else text
    tail = normalized[3:] if windows and path.drive and not path.drive.startswith('\\\\') else normalized[2:] if windows else normalized[1:]
    if directory and tail.endswith('/'):
        tail = tail[:-1]
    need(path.is_absolute() and (windows or '\\' not in text) and
         all(part not in ('', '.', '..') for part in tail.split('/')), 'nonalias absolute origin path required')
    return path


def _origin_equal(left, right, directory=False):
    return _origin_path(left, directory) == _origin_path(right, directory)


def _child_argv(argv, science_checkout, job):
    root = _origin_path(science_checkout, True)
    worker = root.joinpath(*WORKER.split('/'))
    need(type(argv) is list and len(argv) == 14 and all(type(x) is str for x in argv), 'closed Base.julia_cmd argv required')
    _origin_path(argv[0])
    need(argv[1] == '-C' and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.:+,;=\-]{0,2047}', argv[2]) and
         argv[3].startswith('-J') and len(argv[3]) > 2 and argv[4] == '--pkgimages=no' and
         re.fullmatch(r'-g[012]', argv[5]) and argv[6] == '--startup-file=no', 'known Base.julia_cmd prefix differs')
    _origin_path(argv[3][2:])
    need(argv[7:10] == ['--startup-file=no', '--pkgimages=no', '--threads=1'] and
         argv[10].startswith('--project=') and _origin_equal(argv[10][10:], str(root), True) and
         _origin_equal(argv[11], str(worker)) and argv[12] == job and _origin_path(argv[13]).name == job,
         'fixed worker/project/role/output tail differs')


def validate_workers(driver_terminal, worker_receipts, logs, *, runtime, worker_sha256, bindings_sha256, science_checkout):
    need(runtime in RUNTIMES and driver_terminal['complete'] is True and
         driver_terminal['required_jobs'] == list(JOBS) and len(driver_terminal['runs']) == 17 and
         driver_terminal['worker_sha256'] == worker_sha256 and
         set(worker_receipts) == set(logs) == set(JOBS), 'complete ordered17 evidence required')
    result, kernels, architectures = [], set(), set()
    root = _origin_path(science_checkout, True)
    for job, run in zip(JOBS, driver_terminal['runs']):
        receipt, log = worker_receipts[job], logs[job]
        need(run['job'] == job and type(run['pid']) is int and run['pid'] > 0 and
             type(run['native_exit']) is int and run['native_exit'] == 0 and run['timed_out'] is False and
             run['receipt_valid'] is True and type(run['started_utc']) is str and type(run['completed_utc']) is str and
             datetime.fromisoformat(run['started_utc']) <= datetime.fromisoformat(run['completed_utc']),
             'worker native/time evidence differs')
        argv = run['argv']
        _child_argv(argv, science_checkout, job)
        need(receipt['complete'] is True and receipt['job'] == job and receipt['julia_version'] == runtime and
             receipt['worker_sha256'] == worker_sha256 and receipt['bindings_sha256'] == bindings_sha256 and
             _origin_equal(receipt['package_source'], str(root/'src'/'SolarSINDy.jl')) and
             receipt['source_sha256'] == IMF_PINS and type(receipt['blas_threads']) is int and receipt['blas_threads'] == 1 and
             type(receipt['julia_threads']) is int and receipt['julia_threads'] == 1, 'worker source/runtime/threads differ')
        need(run['log_sha256'] == log['sha256'] and log['complete_marker'] == job and
             log['unsafe_runtime_warning'] is False, 'worker original raw join differs')
        kernels.add(receipt['kernel'])
        architectures.add(receipt['architecture'])
        result.append(dict(job=job, pid=run['pid']))
    need(len(kernels) == len(architectures) == 1, 'worker runtime platform differs')
    return result


def run_direct(owner, argv, *, cwd, environment, logpath, timeout, on_launch, stage_helper):
    return stage_helper._run_owned_stage(owner, argv, cwd=cwd, environment=environment,
                                         logpath=logpath, timeout=timeout, on_launch=on_launch)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0', GCM_INTERACTIVE='never',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    return env


def git(root, args, deadline, *, body=None, limit=META):
    clock(deadline)
    allowance = min(120, (deadline-time.monotonic_ns())/1e9)
    need(allowance > 0, 'no Git allowance')
    done = subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args], input=body,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=allowance, env=environment())
    need(done.returncode == 0 and len(done.stdout) <= limit and len(done.stderr) <= META, 'bounded read-only Git failed')
    clock(deadline)
    return done.stdout


def simple_source(root, head, deadline):
    config = git(root, ['config', '--null', '--list'], deadline)
    for row in config.split(b'\0'):
        key, _, value = row.partition(b'\n')
        if key.lower() == b'core.fsmonitor':
            need(value.lower() in (b'false', b'0', b'no', b'off'), 'external fsmonitor rejected')
    need(git(root, ['rev-parse', 'HEAD'], deadline).decode().strip() == head and
         not git(root, ['status', '--porcelain=v1', '-z', '--untracked-files=all'], deadline), 'clean exact source HEAD required')
    for name, pin in PINS.items():
        need(digest(root/name, deadline) == pin, 'fixed numerical driver/helper changed: ' + name)
    parse_jobs(raw(root/DRIVER, META, deadline))
    return git(root, ['rev-parse', 'HEAD^{tree}'], deadline).decode().strip()


def source_proof(root, head, deadline):
    tree = simple_source(root, head, deadline)
    index, committed = {}, {}
    for row in git(root, ['ls-files', '--stage', '-z'], deadline).split(b'\0')[:-1]:
        header, name = row.split(b'\t', 1)
        mode, oid, stage = header.decode().split()
        name = name.decode('utf8')
        need(mode in ('100644', '100755') and stage == '0' and name not in index, 'regular stage0 source required')
        index[name] = (mode, oid)
    for row in git(root, ['ls-tree', '-r', '-z', '--full-tree', 'HEAD'], deadline).split(b'\0')[:-1]:
        header, name = row.split(b'\t', 1)
        mode, kind, oid = header.decode().split()
        name = name.decode('utf8')
        need(kind == 'blob' and name not in committed, 'committed regular source required')
        committed[name] = (mode, oid)
    need(index == committed and 1 <= len(index) <= 8192, 'whole source index/tree differs')
    names = sorted(index)
    attrs = git(root, ['check-attr', '-z', '--stdin', 'filter'], deadline,
                body=b''.join(x.encode('utf8')+b'\0' for x in names)).split(b'\0')
    need(attrs[-1] == b'' and len(attrs) == 3*len(names)+1 and
         all(attrs[i+1] == b'filter' and attrs[i+2] in (b'unspecified', b'unset') for i in range(0, len(attrs)-1, 3)),
         'external clean/smudge filter rejected')
    hashes, objects, physical = {}, {}, set()
    for name in names:
        need(PurePosixPath(name).as_posix() == name and not PurePosixPath(name).is_absolute() and
             '\\' not in name and ':' not in name and all(x not in ('', '.', '..') for x in name.split('/')), 'unsafe source name')
        path = ordinary(root/name)
        key = os.path.normcase(str(path))
        need(key not in physical and path.is_relative_to(root), 'source alias/escape')
        physical.add(key)
        body = raw(path, SOURCE, deadline)
        oid = index[name][1]
        need(re.fullmatch('[0-9a-f]{40}', oid), 'source OID grammar differs')
        hashes[name] = hashlib.sha256(body).hexdigest()
        clean = git(root, ['hash-object', '--path='+name, '--stdin'], deadline, body=body, limit=128).decode().strip()
        blob = git(root, ['cat-file', 'blob', oid], deadline, limit=SOURCE)
        need(clean == oid and hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest() == oid, 'Git clean/raw object differs')
        filtered = git(root, ['cat-file', '--filters', '--path='+name, oid], deadline, limit=SOURCE)
        need(filtered == body and digest(path, deadline) == hashes[name], 'Git checkout filter/source differs')
        objects[name] = oid
    need(simple_source(root, head, deadline) == tree and
         git(root, ['ls-files', '--stage', '-z'], deadline) == b''.join(
             (index[n][0]+' '+index[n][1]+' 0\t'+n).encode()+b'\0' for n in names), 'source changed during proof')
    return dict(complete=True, head=head, tree=tree, source_files=len(names), source_sha256=hashes,
                commit_object_ids=objects, all_clean_and_checkout_filters_match=True)


def controls(root, deadline):
    result = {str(Path(__file__).resolve()): digest(Path(__file__).resolve(), deadline)}
    result.update({str(root/n): digest(root/n, deadline) for n in PINS})
    return result


def runtime_files(julia, deadline):
    prefix = julia.parent.parent
    if os.name == 'nt':
        lib, image = julia.parent/'libjulia.dll', prefix/'lib/julia/sys.dll'
    else:
        pattern = 'libjulia*.dylib' if sys.platform == 'darwin' else 'libjulia.so*'
        candidates = {p.resolve(strict=True) for p in (prefix/'lib').glob(pattern)}
        need(len(candidates) == 1, 'one canonical libjulia identity required')
        lib = candidates.pop()
        image = prefix/('lib/julia/sys.dylib' if sys.platform == 'darwin' else 'lib/julia/sys.so')
    result = {}
    for role, path in (('julia', julia), ('libjulia', lib), ('sysimage', image)):
        path = ordinary(path.resolve(strict=True))
        result[role] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path, deadline, GIB))
    return result


def log_metadata(path, deadline, job=None):
    body = raw(path, LOG, deadline)
    markers = re.findall(rb'(?m)^PACKAGE_BOUND_WORKER_COMPLETE=([^\r\n]+)\r?$', body)
    bad = any(WARNING.encode(encoding) in body for encoding in ('utf8', 'utf-16-le', 'utf-16-be'))
    return dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                complete_marker=markers[0].decode('ascii') if len(markers) == 1 else None,
                unsafe_runtime_warning=bad)


def _terminal_identity(path):
    value=Path(path).lstat()
    need(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode) and
         not getattr(value,'st_file_attributes',0)&0x400 and value.st_ino!=0,
         'ordinary terminal native identity unavailable')
    return value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode)


def _quarantine_terminal(path, identity):
    path=Path(path)
    try:
        if not os.path.lexists(path):return dict(state='absent',owned_object_moved=False)
        if _terminal_identity(path)!=identity:
            return dict(state='unknown_replaced',owned_object_moved=False,path=str(path))
        ordinary(path.parent,True)
        target=path.with_name(path.name+'.unaccepted-'+uuid.uuid4().hex)
        need(not os.path.lexists(target),'fresh terminal quarantine required')
        try:
            path.rename(target)
        except BaseException as error:
            if os.path.lexists(target) and _terminal_identity(target)==identity and not os.path.lexists(path):
                return dict(state='quarantined',owned_object_moved=True,path=str(target),rename_error=repr(error))
            raise
        need(_terminal_identity(target)==identity and not os.path.lexists(path),'terminal quarantine outcome unknown')
        return dict(state='quarantined',owned_object_moved=True,path=str(target))
    except BaseException as error:
        return dict(state='unknown_error',owned_object_moved=False,path=str(path),error=repr(error))


def finalize_interval_terminal(path, record, deadline, *, emit):
    """Output/file protocol only; main supplies genuine verified native facts."""
    path=Path(path);ordinary(path.parent,True)
    need(not os.path.lexists(path),'fresh accepted terminal path required')
    payload=json.loads(json.dumps(record,allow_nan=False))
    pending=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    identity=None;attempted=False
    try:
        clock(deadline)
        with pending.open('xb') as stream:
            initial=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
            need(len(initial)<=META and stream.write(initial)==len(initial),'bounded staged terminal write required')
            stream.flush();os.fsync(stream.fileno())
            value=os.fstat(stream.fileno());identity=(value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode))
            need(value.st_ino!=0,'created terminal native identity unavailable')
            emit()  # No accepted complete file exists while output may block/fail.
            clock(deadline)
            if 'finished_ns' in payload:payload['finished_ns']=time.monotonic_ns()
            if 'finished_utc' in payload:payload['finished_utc']=utc()
            final_body=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
            need(len(final_body)<=META,'terminal metadata exceeds4MiB')
            stream.seek(0);stream.truncate()
            need(stream.write(final_body)==len(final_body),'short final staged terminal write')
            stream.flush();os.fsync(stream.fileno())
        need(_terminal_identity(pending)==identity and not os.path.lexists(path),'owned staged terminal changed')
        clock(deadline)
        attempted=True;pending.rename(path)
        need(_terminal_identity(path)==identity,'accepted terminal object changed')
        clock(deadline)
        return payload.get('complete') is True
    except BaseException as error:
        status=_quarantine_terminal(path,identity) if attempted else dict(state='not_attempted',owned_object_moved=False)
        diagnostic=dict(schema='interval_terminal_failure_v1',complete=False,error=repr(error),
                        rollback_status=status,original_record=payload,source_qualified=False,
                        software_qualified=False,scientific_qualified=False)
        try:
            put(path.with_name(path.name+'.failure-'+uuid.uuid4().hex+'.json'),diagnostic)
        except BaseException as diagnostic_error:
            status['diagnostic_error']=repr(diagnostic_error)
        failure=RuntimeError('interval terminal not accepted; rollback status='+status['state'])
        failure.rollback_status=status
        raise failure from error


def main(argv=None):
    entered, entered_utc = time.monotonic_ns(), utc()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science-checkout', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--runtime', choices=RUNTIMES, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--julia', type=Path)
    parser.add_argument('--preflight-record', type=Path)
    parser.add_argument('--preflight-sha256')
    args = parser.parse_args(argv)
    need(not sys.flags.optimize and sys.version_info >= (3, 11) and
         (sys.platform != 'darwin' or sys.version_info >= (3, 13)), 'supported unoptimized Python required')
    need(re.fullmatch('[0-9a-f]{40}', args.expected_head), 'explicit40hex science head required')
    root = ordinary(args.science_checkout, True)
    out = args.evidence_dir
    ordinary(out.parent, True)
    need(out.is_absolute() and not os.path.lexists(out) and not out.is_relative_to(root) and not root.is_relative_to(out), 'fresh disjoint output required')
    deadline = entered + (180 if args.preflight_only else WHOLE_SECONDS)*10**9
    for name in (MEMORY,):
        need(digest(root/name, deadline) == PINS[name], 'memory helper changed')
    memory = import_file(root/MEMORY, PINS[MEMORY], deadline, 'interval_worker_memory')
    available, method = memory.available_memory()
    admission = check_admission(available, shutil.disk_usage(out.parent).free)
    admission['memory_method'] = method
    admission['observed_ns'] = time.monotonic_ns()
    tree = simple_source(root, args.expected_head, deadline)
    initial_controls = controls(root, deadline)
    if args.preflight_only:
        need(args.julia is None and args.preflight_record is None and args.preflight_sha256 is None, 'preflight launches no Julia')
        out.mkdir()
        record = dict(schema='interval_workers_preflight_v1', complete=True, mode='preflight_only', head=args.expected_head,
            tree=tree, science_checkout=str(root), runtime=args.runtime, started_ns=entered, finished_ns=time.monotonic_ns(),
            admission=admission, project_sha256=digest(root/'Project.toml', deadline),
            manifest_sha256=digest(root/'Manifest.toml', deadline), controls=initial_controls,
            source_qualified=False, software_qualified=False, scientific_qualified=False)
        try:
            clock(deadline)
            put(out/'PREFLIGHT.json', record)
            clock(deadline)
            print(json.dumps(dict(complete=True, preflight=str(out/'PREFLIGHT.json'), sha256=digest(out/'PREFLIGHT.json', deadline))), flush=True)
            clock(deadline)
            return 0
        except BaseException:
            if (out/'PREFLIGHT.json').exists():
                (out/'PREFLIGHT.json').rename(out/'PREFLIGHT.before-failure.json')
            record.update(complete=False, finished_ns=time.monotonic_ns())
            put(out/'PREFLIGHT.json', record)
            raise
    need(args.julia is not None and args.preflight_record is not None and
         re.fullmatch('[0-9a-f]{64}', args.preflight_sha256 or ''), 'original preflight ref and Julia file required')
    need(digest(args.preflight_record, deadline) == args.preflight_sha256, 'original preflight changed')
    pre = strict(raw(args.preflight_record, META, deadline))
    need(pre['schema'] == 'interval_workers_preflight_v1' and pre['complete'] is True and pre['mode'] == 'preflight_only' and
         pre['head'] == args.expected_head and pre['tree'] == tree and pre['science_checkout'] == str(root) and
         pre['runtime'] == args.runtime and pre['started_ns'] <= pre['finished_ns'] < entered and
         pre['project_sha256'] == digest(root/'Project.toml', deadline) and
         pre['manifest_sha256'] == digest(root/'Manifest.toml', deadline) and pre['controls'] == initial_controls,
         'science checkout/Project/Manifest changed through dependency setup')
    julia = ordinary(args.julia.resolve(strict=True))
    out.mkdir()
    artifact = out/'artifact'
    target = artifact/'workers17'/args.runtime
    target.mkdir(parents=True)
    work = out/'work'
    work.mkdir()
    selected_bytes, selected_count = 0, 0
    def reference(path):
        return dict(path=path.relative_to(artifact).as_posix(), sha256=digest(path, deadline, max(LOG, META)))
    def copy_selected(source, destination, limit):
        nonlocal selected_bytes, selected_count
        source = ordinary(source)
        need(source.stat().st_nlink == 1 and source.stat().st_size <= limit, 'selected evidence must be bounded single-link ordinary file')
        selected_bytes += source.stat().st_size
        selected_count += 1
        need(selected_bytes <= ARTIFACT and selected_count <= 128, 'selected artifact aggregate bound')
        value = raw(source, limit, deadline)
        with destination.open('xb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        need(digest(destination, deadline, limit) == hashlib.sha256(value).hexdigest(), 'retained original copy differs')
        return reference(destination)
    record = dict(schema='interval_workers_qualification_terminal_v1', complete=False, head=args.expected_head, tree=tree,
        julia_version=args.runtime, science_checkout=str(root), started_ns=entered, finished_ns=None, deadline_ns=deadline,
        native_deadline_ns=None, started_utc=entered_utc, finished_utc=None, preflight=None, source_proof=None,
        source_after=None, runtime=None, argv=build_driver_argv(root, julia), owner=None, launch=None, raw=None,
        driver_evidence_root=None, driver_terminal=None, workers=[], admission=admission, controls=initial_controls,
        error=None, source_qualified=False, software_qualified=False, scientific_qualified=False)
    native = None
    original_evidence = None
    try:
        record['preflight'] = copy_selected(args.preflight_record, target/'PREFLIGHT.json', META)
        before = source_proof(root, args.expected_head, min(deadline, time.monotonic_ns()+SOURCE_SECONDS*10**9))
        put(target/'SOURCE_PROOF.json', before)
        record['source_proof'] = reference(target/'SOURCE_PROOF.json')
        runtime_before = runtime_files(julia, deadline)
        owner = import_file(root/OWNER, PINS[OWNER], deadline, 'interval_workers_direct_owner')
        stage = import_file(root/STAGE, PINS[STAGE], deadline, 'interval_workers_stage_composition')
        env = environment()
        env.update(TMP=str(work), TEMP=str(work), TMPDIR=str(work), JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', JULIA_PKG_PRECOMPILE_AUTO='0', SOLARSINDY_NO_OPEN='1')
        available, method = memory.available_memory()
        record['admission'] = check_admission(available, shutil.disk_usage(out).free)
        record['admission'].update(memory_method=method, observed_ns=time.monotonic_ns())
        native_started = time.monotonic_ns()
        require_native_budget(deadline, now_ns=native_started)
        record['native_deadline_ns'] = native_started+NATIVE_SECONDS*10**9
        def launched(pid):
            need(type(pid) is int and pid > 0, 'actual direct Julia PID required')
            put(target/'LAUNCH.json', dict(pid=pid, monotonic_ns=time.monotonic_ns()))
        native = run_direct(owner, record['argv'], cwd=str(root), environment=env, logpath=target/'driver.log',
                            timeout=NATIVE_SECONDS, on_launch=launched, stage_helper=stage)
        put(target/'NATIVE.json', native)
        record['owner'], record['launch'] = reference(target/'NATIVE.json'), reference(target/'LAUNCH.json')
        validate_native(native, strict(raw(target/'LAUNCH.json', META, deadline)),
                        native_started_ns=native_started, native_deadline_ns=record['native_deadline_ns'])
        need(time.monotonic_ns() < record['native_deadline_ns'], 'native result returned late')
        body = raw(target/'driver.log', LOG, deadline)
        need(not any(WARNING.encode(x) in body for x in ('utf8', 'utf-16-le', 'utf-16-be')), 'unsafe Julia runtime warning')
        markers = re.findall(rb'(?m)^INTERVAL_FORECASTER_RETAINED_EVIDENCE=([^\r\n]+)\r?$', body)
        need(len(markers) == 1, 'one original interval evidence marker required')
        original_evidence = ordinary(Path(markers[0].decode('utf8')), True)
        need(original_evidence.parent == work and original_evidence.name.startswith('solarsindy-interval-functional-'), 'driver evidence escaped owned work root')
        record['driver_evidence_root'] = str(original_evidence)
        driver = tomllib.loads(raw(original_evidence/'terminal.toml', META, deadline).decode('utf8'))
        receipts = {job: tomllib.loads(raw(original_evidence/job/'worker.toml', 2*1024**2, deadline).decode('utf8')) for job in JOBS}
        logs = {job: log_metadata(original_evidence/(job+'.log'), deadline, job) for job in JOBS}
        validate_workers(driver, receipts, logs, runtime=args.runtime, worker_sha256=PINS[WORKER],
                         bindings_sha256=PINS[BINDINGS], science_checkout=root)
        need(all(run['argv'][0] == str(julia) for run in driver['runs']), 'worker binary differs from direct Julia binary')
        need(all(Path(run['argv'][-1]) == original_evidence/run['job'] for run in driver['runs']),
             'worker argv output differs from original owned evidence tree')
        need(all(len([x for x in run['argv'] if x.startswith('-J')]) == 1 and
                 Path(next(x[2:] for x in run['argv'] if x.startswith('-J'))).resolve(strict=True) ==
                 Path(runtime_before['sysimage']['path']) for run in driver['runs']),
             'worker sysimage differs from recorded runtime')
        need(all(datetime.fromisoformat(run['started_utc']).replace(tzinfo=timezone.utc) >= datetime.fromisoformat(entered_utc)
                 for run in driver['runs']), 'worker UTC evidence predates this invocation')
        record['driver_terminal'] = copy_selected(original_evidence/'terminal.toml', target/'driver-terminal.toml', META)
        for job in JOBS:
            receipt = copy_selected(original_evidence/job/'worker.toml', target/(job+'.worker.toml'), 2*1024**2)
            log = copy_selected(original_evidence/(job+'.log'), target/(job+'.log'), LOG)
            record['workers'].append(dict(job=job, worker=receipt, raw=log))
        first = receipts[JOBS[0]]
        record['runtime'] = dict(julia_version=args.runtime, julia_path=str(julia), julia_binary_sha256=runtime_before['julia']['sha256'],
            blas_threads=first['blas_threads'], julia_threads=first['julia_threads'], omp_threads=1,
            kernel=first['kernel'], architecture=first['architecture'], runtime_files=runtime_before)
        need(runtime_files(julia, deadline) == runtime_before, 'Julia runtime bytes changed')
        after = source_proof(root, args.expected_head, min(deadline, time.monotonic_ns()+SOURCE_SECONDS*10**9))
        need(after == before and controls(root, deadline) == initial_controls, 'source/control bytes changed')
        put(target/'SOURCE_AFTER.json', after)
        record['source_after'] = reference(target/'SOURCE_AFTER.json')
        record['complete'] = True
    except BaseException as error:
        record.update(complete=False, error=type(error).__name__+': '+str(error))
    finally:
        # Preserve only the formal ordinary evidence, including partial failed runs.
        try:
            if (target/'driver.log').exists():
                record['raw'] = reference(target/'driver.log')
            if (target/'LAUNCH.json').exists() and record['launch'] is None:
                record['launch'] = reference(target/'LAUNCH.json')
            if original_evidence is None and work.exists():
                candidates = [p for p in work.iterdir() if p.name.startswith('solarsindy-interval-functional-')]
                if len(candidates) == 1:
                    original_evidence = ordinary(candidates[0], True)
            if not record['complete'] and original_evidence is not None:
                record['driver_evidence_root'] = str(original_evidence)
                if (original_evidence/'terminal.toml').is_file() and not (target/'driver-terminal.toml').exists():
                    record['driver_terminal'] = copy_selected(original_evidence/'terminal.toml', target/'driver-terminal.toml', META)
                existing = {row['job'] for row in record['workers']}
                for job in JOBS:
                    if job in existing:
                        continue
                    row = dict(job=job, worker=None, raw=None)
                    for key, source, destination, bound in (
                        ('worker', original_evidence/job/'worker.toml', target/(job+'.worker.toml'), 2*1024**2),
                        ('raw', original_evidence/(job+'.log'), target/(job+'.log'), LOG)):
                        if destination.exists():
                            row[key] = reference(destination)
                        elif source.is_file():
                            row[key] = copy_selected(source, destination, bound)
                    if row['worker'] is not None or row['raw'] is not None:
                        record['workers'].append(row)
        except BaseException as error:
            record.update(complete=False, error=str(record['error'])+'; retention: '+repr(error))
        record.update(finished_ns=time.monotonic_ns(), finished_utc=utc())
        if record['finished_ns'] >= deadline:
            record.update(complete=False, error='original whole deadline exhausted')
        def emit_pending():
            print(json.dumps(dict(complete=False,state='verified_pending_terminal' if record['complete'] else 'failed_pending_terminal',
                head=args.expected_head,runtime=args.runtime,terminal=str(target/'TERMINAL.json'),
                error=record['error'],source_qualified=False,software_qualified=False,scientific_qualified=False)),flush=True)
        accepted=finalize_interval_terminal(target/'TERMINAL.json',record,deadline,emit=emit_pending)
    return 0 if accepted else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
