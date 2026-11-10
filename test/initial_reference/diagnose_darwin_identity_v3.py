"""Bounded native Darwin identity observations; no ownership qualification.

One pinned harmless file child is observed while live and after a terminal
WNOWAIT event, before its actual Popen wait. No PID or group signal is sent.
"""
from pathlib import Path
import argparse
import ctypes as C
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'research/dbdt_completed_source/native_stage.py'
CANDIDATE = LEGACY.with_name('candidate_native_owner.py')
MEMORY = ROOT / 'tools/run_canonical_harness.py'
PINS = {
    LEGACY: '8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836',
    CANDIDATE: '3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1',
    MEMORY: '209a6b35ebd820d5928a5a91d70062a2fbf0e7caeb3112a18db34bd0e8cfb4ee',
}
WHOLE_SECONDS = 90
CHILD_SECONDS = 15
META_LIMIT = 64 * 1024
RAW_LIMIT = 4 * 1024**2
AVAILABLE_REQUIRED = 512 * 1024**2
DISK_REQUIRED = 8 * 1024**3


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path, maximum=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'ordinary source/raw file required')
    before = path.stat()
    if maximum is not None:
        need(before.st_size <= maximum, 'file exceeds diagnostic byte bound')
    result = hashlib.sha256()
    count = 0
    with path.open('rb') as source:
        while True:
            data = source.read(65536)
            if not data:
                break
            count += len(data)
            need(count <= before.st_size, 'diagnostic file grew during read')
            result.update(data)
    after = path.stat()
    need(count == before.st_size == after.st_size and
         (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
         'diagnostic file identity/size changed')
    return result.hexdigest()


def ordinary_directory(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.normpath(str(path)),
         'canonical absolute directory required')
    for item in (path, *path.parents):
        info = item.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             'ordinary directory components required')
    return path


def write_new(path, value):
    body = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')
    need(len(body) <= META_LIMIT, 'diagnostic metadata byte cap')
    with Path(path).open('xb') as output:
        need(output.write(body) == len(body), 'short diagnostic metadata write')
        output.flush()
        os.fsync(output.fileno())


def load(path, name, expected):
    need(digest(path) == expected, 'diagnostic dependency source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(digest(path) == expected, 'diagnostic dependency changed during import')
    return module


def failure(error):
    return dict(type=type(error).__name__, errno=getattr(error, 'errno', None),
                message=str(error)[:1024])


def observe(operation):
    try:
        return dict(returned=True, value=operation())
    except BaseException as error:
        return dict(returned=False, error=failure(error))


class BSDInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ('flags', 'status', 'xstatus', 'pid', 'ppid', 'uid', 'gid',
                 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]
    _fields_ += [('comm', C.c_char * 16), ('name', C.c_char * 32)]
    _fields_ += [(name, C.c_uint32) for name in
                 ('nfiles', 'pgid', 'pjobc', 'ttydev', 'ttypgid')]
    _fields_ += [('nice', C.c_int32), ('start_sec', C.c_uint64),
                ('start_usec', C.c_uint64)]


class ShortBSDInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('pid', 'ppid', 'pgid', 'status')]
    _fields_ += [('comm', C.c_char * 16)]
    _fields_ += [(name, C.c_uint32) for name in
                 ('flags', 'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]


class NativeRecords:
    def __init__(self):
        self.abi = dict(full_bytes=C.sizeof(BSDInfo), short_bytes=C.sizeof(ShortBSDInfo),
                        full_pid_offset=BSDInfo.pid.offset, full_pgid_offset=BSDInfo.pgid.offset,
                        short_uid_offset=ShortBSDInfo.uid.offset)
        need(self.abi == dict(full_bytes=136, short_bytes=64, full_pid_offset=12,
                              full_pgid_offset=100, short_uid_offset=36),
             'native BSD record ABI differs')
        self.library = C.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.library.proc_pidinfo.argtypes = (C.c_int, C.c_int, C.c_uint64,
                                             C.c_void_p, C.c_int)
        self.library.proc_pidinfo.restype = C.c_int

    def record(self, pid, flavor, kind):
        value = kind()
        C.set_errno(0)
        count = self.library.proc_pidinfo(pid, flavor, 1, C.byref(value), C.sizeof(value))
        error = C.get_errno()
        if count != C.sizeof(value) or value.pid != pid:
            raise OSError(error or errno.EPROTO,
                          f'BSD record unavailable: flavor={flavor} bytes={count} pid={pid}')
        names = ('pid', 'ppid', 'pgid', 'uid', 'ruid', 'status', 'flags')
        result = {name: int(getattr(value, name)) for name in names}
        if kind is BSDInfo:
            result['birth'] = [int(value.start_sec), int(value.start_usec)]
        result['session_leader_flag_0x20'] = bool(result['flags'] & 0x20)
        return result


def snapshot(reader, production, pid):
    # Native fields and both current reader outcomes are observations, not an
    # identity admission. In particular, the terminal leader() need not pass.
    return dict(pid=pid, monotonic_ns=time.monotonic_ns(),
        short=observe(lambda: reader.record(pid, 13, ShortBSDInfo)),
        full=observe(lambda: reader.record(pid, 3, BSDInfo)),
        getsid=observe(lambda: os.getsid(pid)),
        current_info=observe(lambda: production._info(pid)),
        current_leader=observe(lambda: production.leader(pid)))


def raw_pair(snapshot_value):
    need(snapshot_value['short']['returned'] and snapshot_value['full']['returned'],
         'complete native short/full observations required')
    return snapshot_value['short']['value'], snapshot_value['full']['value']


def child(root, expected):
    need(sys.platform == 'darwin' and os.name == 'posix' and not sys.flags.optimize,
         'fixed child requires unoptimized Darwin Python')
    need(digest(__file__) == expected, 'fixed diagnostic child source changed')
    root = ordinary_directory(root)
    ready = dict(pid=os.getpid(), pgid=os.getpgrp(), sid=os.getsid(0), source_sha256=expected)
    write_new(root / 'READY.tmp', ready)
    os.replace(root / 'READY.tmp', root / 'READY.json')
    sys.stdout.buffer.write(b'DARWIN_IDENTITY_CHILD_READY\n')
    sys.stdout.buffer.flush()
    deadline = time.monotonic() + CHILD_SECONDS
    while not (root / 'release').is_file():
        if time.monotonic() >= deadline:
            return 17
        time.sleep(.005)
    need((root / 'release').read_bytes() == b'1\n', 'diagnostic release differs')
    return 7


def main():
    started = time.monotonic_ns()
    deadline = started + WHOLE_SECONDS * 10**9
    need(sys.platform == 'darwin' and os.name == 'posix' and
         sys.version_info >= (3, 13) and not sys.flags.optimize,
         'diagnostic requires unoptimized Darwin Python>=3.13')
    required = ('waitid', 'P_PID', 'WEXITED', 'WNOHANG', 'WNOWAIT',
                'CLD_EXITED', 'CLD_KILLED', 'CLD_DUMPED')
    need(all(hasattr(os, name) for name in required), 'terminal WNOWAIT API required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    parent = ordinary_directory(output.parent)
    need(output.is_absolute() and str(output) == os.path.normpath(str(output)) and
         not os.path.lexists(output) and output != ROOT and
         not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output),
         'fresh disjoint ordinary diagnostic output required')
    protected = {str(path): pin for path, pin in PINS.items()}
    protected[str(Path(__file__).resolve())] = digest(__file__)
    need(all(digest(path) == pin for path, pin in protected.items()), 'fixed source pin differs')
    memory = load(MEMORY, 'darwin_identity_diagnostic_memory', PINS[MEMORY])
    available, method = memory.available_memory()
    free = shutil.disk_usage(parent).free
    need(available >= AVAILABLE_REQUIRED and free >= DISK_REQUIRED,
         'diagnostic512MiB/8GiB admission failed')
    legacy = load(LEGACY, 'darwin_identity_diagnostic_current_reader', PINS[LEGACY])
    production = legacy._DarwinProcesses()
    reader = NativeRecords()
    python_binary = Path(sys.executable).resolve(strict=True)
    output.mkdir()
    raw = output / 'child.raw'
    record = dict(schema='darwin_identity_v3_diagnostic_v1', complete=False,
        observation_only=True, source_qualified=False, software_qualified=False,
        scientific_qualified=False, native_owner_qualified=False,
        ownership_boundary_closed_observed=False, individual_descendant_OS_reaping_observed=False,
        Julia_launched=False, actual_archive_exposed=False, Q_issued=False,
        signals_sent=0, descendants_launched=0, source_pins=protected,
        started_ns=started, deadline_ns=deadline, observer_pid=os.getpid(),
        runtime=dict(uname=list(os.uname()), python_version=sys.version,
                     python_invocation_path=sys.executable, python_binary_path=str(python_binary),
                     python_binary_sha256=digest(python_binary)), native_abi=reader.abi,
        admission=dict(available_physical_bytes=available, memory_method=method,
                       free_disk_bytes=free, required_available_bytes=AVAILABLE_REQUIRED,
                       required_disk_bytes=DISK_REQUIRED,
                       native_admission_utility=dict(path='/usr/bin/vm_stat', timeout_seconds=30,
                           separate_from_fixed_identity_child=True, native_pid_observed=False)))
    write_new(output / 'PREEXECUTION.json', record)
    process = None
    waited = False
    try:
        need(time.monotonic_ns() < deadline, 'diagnostic budget exhausted before spawn')
        argv = [str(python_binary), '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                '--child', str(output), protected[str(Path(__file__).resolve())]]
        with raw.open('xb') as log:
            process = subprocess.Popen(argv, cwd=str(ROOT), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            record['actual_popen_pid'] = process.pid
            record['child_argv'] = argv
            ready_deadline = min(deadline, time.monotonic_ns() + 10 * 10**9)
            while not (output / 'READY.json').is_file():
                need(time.monotonic_ns() < ready_deadline, 'diagnostic child READY deadline')
                time.sleep(.005)
            ready_path = output / 'READY.json'
            need(ready_path.stat().st_size <= 1024 and not ready_path.is_symlink(), 'READY metadata bound')
            ready = json.loads(ready_path.read_bytes())
            need(ready == dict(pid=process.pid, pgid=process.pid, sid=process.pid,
                               source_sha256=protected[str(Path(__file__).resolve())]),
                 'fixed child READY identity differs')
            record['ready'] = ready
            record['live'] = snapshot(reader, production, process.pid)
            short, full = raw_pair(record['live'])
            need(all(row['pid'] == row['pgid'] == process.pid and
                     row['ppid'] == os.getpid() and row['uid'] == os.geteuid() and
                     row['ruid'] == os.getuid() and row['status'] != 5 for row in (short, full)) and
                 record['live']['getsid'] == dict(returned=True, value=process.pid) and
                 full['birth'][0] > 0, 'independent live child calibration failed')
            need(process.returncode is None, 'child must remain held before terminal observation')
            with (output / 'release').open('xb') as release:
                release.write(b'1\n')
                release.flush()
                os.fsync(release.fileno())
            terminal_deadline = min(deadline, time.monotonic_ns() + 20 * 10**9)
            attempts = 0
            while True:
                need(time.monotonic_ns() < terminal_deadline, 'terminal WNOWAIT observation deadline')
                attempts += 1
                event = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if event is not None:
                    need(event.si_pid == process.pid and
                         event.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
                         'unexpected native waitid PID/nonterminal event')
                    record['terminal_waitid'] = {name: int(getattr(event, name)) for name in
                        ('si_pid', 'si_uid', 'si_signo', 'si_status', 'si_code')}
                    break
                time.sleep(.005)
            record['waitid_attempts'] = attempts
            need(process.returncode is None, 'Popen must not reap before terminal snapshot')
            record['terminal'] = snapshot(reader, production, process.pid)
            raw_pair(record['terminal'])
            record['direct_child_native_exit'] = process.wait(
                timeout=max(0., min(20., (deadline - time.monotonic_ns()) / 1e9)))
            waited = True
            record['actual_popen_wait_returned'] = True
            record['terminal_observed_before_first_wait'] = True
            expected_exit = event.si_status if event.si_code == os.CLD_EXITED else -event.si_status
            need(record['direct_child_native_exit'] == expected_exit,
                 'actual Popen exit differs from terminal WNOWAIT event')
        need(time.monotonic_ns() < deadline, 'bounded observation completed late')
        record['complete'] = True
    except BaseException as error:
        record['failure'] = failure(error)
        record['failure_traceback'] = traceback.format_exc(limit=8)[-8000:]
    finally:
        if process is not None and not waited:
            # Failure-only intervention cannot create a successful observation.
            # The pinned child self-expires; wait only this retained child object.
            record['failure_only_wait_attempted'] = True
            try:
                record['failure_only_native_exit'] = process.wait(
                    timeout=max(0., min(20., (deadline - time.monotonic_ns()) / 1e9)))
                record['failure_only_actual_wait_returned'] = True
            except BaseException as error:
                record['failure_only_wait_error'] = failure(error)
        record['direct_child_reaping_observed'] = waited or record.get('failure_only_actual_wait_returned', False)
        try:
            record['source_pins_unchanged'] = all(digest(path) == pin for path, pin in protected.items())
        except BaseException as error:
            record['source_pins_unchanged'] = False
            record['source_guard_error'] = failure(error)
        try:
            if raw.is_file():
                record['raw'] = dict(path=str(raw), bytes=raw.stat().st_size,
                                     sha256=digest(raw, RAW_LIMIT))
        except BaseException as error:
            record['complete'] = False
            record['raw_guard_error'] = failure(error)
        record['finished_ns'] = time.monotonic_ns()
        record['complete'] = bool(record['complete'] and record['source_pins_unchanged'] and
                                   record['finished_ns'] < deadline and waited)
        write_new(output / 'RESULT.json', record)
        metadata_bytes = sum(path.stat().st_size for path in output.iterdir()
                             if path.is_file() and path.suffix == '.json')
        if metadata_bytes > META_LIMIT or time.monotonic_ns() >= deadline:
            (output / 'RESULT.json').rename(output / 'RESULT_UNQUALIFIED.json')
            write_new(output / 'LATE_OR_OVERSIZE.json', dict(complete=False,
                observed_monotonic_ns=time.monotonic_ns(), deadline_ns=deadline,
                metadata_bytes=metadata_bytes, metadata_limit=META_LIMIT))
            record['complete'] = False
    print(json.dumps(dict(complete=record['complete'], observation_only=True,
                          evidence=str(output), source_qualified=False,
                          software_qualified=False, scientific_qualified=False), sort_keys=True), flush=True)
    if time.monotonic_ns() >= deadline:
        path = output / 'RESULT.json'
        if path.exists():
            path.rename(output / 'RESULT_LATE_UNQUALIFIED.json')
        return 1
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--child':
        raise SystemExit(child(Path(sys.argv[2]), sys.argv[3]))
    raise SystemExit(main())
