"""File-based Darwin observation diagnostic V2; no ownership repair or qualification.

ROOT must provide an outer finite native owner and >=512MiB available RAM /
8GiB free disk before running this file on Darwin. Only the pinned existing
helper and generated Python children are exercised; no Julia/data/fits/Git.
"""
import argparse
import ctypes as C
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

OWNER_SHA = '15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8'
ROLES = ('tools/initial_reference_audit/owned_process.py',
         'research/dbdt_completed_source/native_stage.py')
CASES = ('exit_without_descendant', 'exit_with_live_descendant',
         'exit_with_other_group_descendant', 'stopped_child')
MAX_PIDS = 4097
MAX_EVENTS = 512
CASE_SECONDS = 8.
WHOLE_SECONDS = 150.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 2 * 1024**2:
        raise RuntimeError('diagnostic metadata byte cap')
    with Path(path).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


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


class DarwinProcesses:
    def __init__(self):
        if (C.sizeof(BSDInfo) != 136 or BSDInfo.pid.offset != 12 or BSDInfo.pgid.offset != 100 or
                C.sizeof(ShortBSDInfo) != 64 or ShortBSDInfo.uid.offset != 36):
            raise RuntimeError('pinned proc_bsdinfo ABI layout differs')
        self.library = C.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.library.proc_listpids.argtypes = (C.c_uint32, C.c_uint32, C.c_void_p, C.c_int)
        self.library.proc_listpids.restype = C.c_int
        self.library.proc_pidinfo.argtypes = (C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int)
        self.library.proc_pidinfo.restype = C.c_int
        own = self.info(os.getpid())
        if own['pid'] != os.getpid() or own['pgid'] != os.getpgrp() or own['uid'] != os.getuid():
            raise RuntimeError('native libproc self ABI calibration differs')
        if os.getpid() not in self.pids(os.getpgrp()):
            raise RuntimeError('native libproc self group calibration differs')

    def pids(self, pgid, *, all_processes=False):
        if not (pgid == 0 if all_processes else 0 < pgid < 2**31):
            raise RuntimeError('diagnostic owned PGID outside bound')
        buffer = (C.c_int * MAX_PIDS)()
        C.set_errno(0)
        count = self.library.proc_listpids(1 if all_processes else 2, pgid, buffer, C.sizeof(buffer))
        error = C.get_errno()
        if count < 0 or count % C.sizeof(C.c_int) or count >= C.sizeof(buffer) or (count == 0 and error):
            raise RuntimeError(f'incomplete/failed bounded group enumeration: bytes={count} errno={error}')
        result = tuple(sorted(buffer[:count // C.sizeof(C.c_int)]))
        if len(result) != len(set(result)) or any(pid < 0 or (pid == 0 and not all_processes) for pid in result):
            raise RuntimeError('invalid/duplicate group PID result')
        return result

    def info(self, pid):
        value = BSDInfo()
        C.set_errno(0)
        # arg=1 is necessary for the kernel's explicit zombie lookup path.
        count = self.library.proc_pidinfo(pid, 3, 1, C.byref(value), C.sizeof(value))
        error = C.get_errno()
        if count != C.sizeof(value) or value.pid != pid:
            raise OSError(error or errno.EPROTO, f'BSD process information unavailable: pid={pid} bytes={count}')
        return {key: int(getattr(value, key)) for key in
                ('pid', 'ppid', 'pgid', 'status', 'uid', 'ruid', 'start_sec', 'start_usec')}

    def short_info(self, pid):
        value = ShortBSDInfo()
        C.set_errno(0)
        count = self.library.proc_pidinfo(pid, 13, 1, C.byref(value), C.sizeof(value))
        error = C.get_errno()
        if count != C.sizeof(value) or value.pid != pid:
            raise OSError(error or errno.EPROTO, f'short BSD information unavailable: pid={pid} bytes={count}')
        return {key: int(getattr(value, key)) for key in ('pid', 'ppid', 'pgid', 'status', 'uid')}

    def snapshot(self, pgid):
        before = self.pids(pgid)
        rows = [self.info(pid) for pid in before]
        after = self.pids(pgid)
        if before != after or any(row['pgid'] != pgid for row in rows):
            raise RuntimeError('group changed during diagnostic snapshot')
        return dict(pgid=pgid, members=rows, repeated_pid_lists_equal=True,
                    owned_leader_present=pgid in before,
                    live_members_observed=sum(row['status'] != 5 for row in rows),
                    owned_group_zombie_only_observed=(pgid in before and
                                                     all(row['status'] == 5 for row in rows)))

    def session_snapshot(self, sid):
        before = self.pids(0, all_processes=True)
        live = []
        for pid in before:
            if pid == 0:
                continue  # Kernel PID0 is excluded by killpg itself, never a user child.
            row = self.short_info(pid)
            if row['status'] == 5:
                continue  # Zombies cannot fork or join a different process group.
            current_sid = os.getsid(pid)
            if current_sid == sid:
                full = self.info(pid)
                if full['status'] == 5 or full['pgid'] != row['pgid']:
                    raise RuntimeError('private-session member changed during census')
                live.append(dict(full, sid=current_sid))
        after = self.pids(0, all_processes=True)
        if before != after:
            raise RuntimeError('full process list changed during session census')
        return dict(sid=sid, live_members=live, all_pid_count=len(before),
                    repeated_all_pid_lists_equal=True, kernel_pid0_excluded=True,
                    zero_live_session_members_observed=not live)


def record_snapshot(api, pid):
    result = {}
    for name, operation in (('group', api.snapshot), ('private_session', api.session_snapshot)):
        try:
            result[name] = dict(observed=True, data=operation(pid))
        except BaseException as error:
            result[name] = dict(observed=False, type=type(error).__name__,
                                errno=getattr(error, 'errno', None), message=str(error))
    return result


DESCENDANT = r'''
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);mode=sys.argv[2]
if mode=='exit_with_other_group_descendant':os.setpgid(0,0)
record=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0))
raw=json.dumps(record,sort_keys=True).encode('utf-8')
with (root/'descendant.tmp').open('xb') as stream:
    stream.write(raw);stream.flush();os.fsync(stream.fileno())
os.replace(root/'descendant.tmp',root/'descendant.json')
time.sleep(12)
'''

CHILD = r'''
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);mode=sys.argv[2]
descendant=Path(sys.argv[3]);expected_sha=sys.argv[4]
if hashlib.sha256(descendant.read_bytes()).hexdigest()!=expected_sha:
    raise RuntimeError('generated descendant source pin differs before launch')
raw=json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0)),sort_keys=True).encode('utf-8')
with (root/'leader-start.tmp').open('xb') as stream:
    stream.write(raw);stream.flush();os.fsync(stream.fileno())
os.replace(root/'leader-start.tmp',root/'leader-start.json')
deadline=time.monotonic()+2
while not (root/'release').is_file():
    if time.monotonic()>=deadline:raise RuntimeError('generated leader release deadline')
    time.sleep(.005)
if mode in ('exit_with_live_descendant','exit_with_other_group_descendant'):
    subprocess.Popen([sys.executable,'-B','-X','utf8',str(descendant),str(root),mode])
    deadline=time.monotonic()+2
    while not (root/'descendant.json').is_file():
        if time.monotonic()>=deadline:raise RuntimeError('generated descendant readiness deadline')
        time.sleep(.005)
elif mode=='stopped_child':
    print('GENERATED_CHILD_SELF_STOP',flush=True)
    os.kill(os.getpid(),signal.SIGSTOP)
if hashlib.sha256(descendant.read_bytes()).hexdigest()!=expected_sha:
    raise RuntimeError('generated descendant source changed during launch')
sys.exit(7)
'''


def write_program(path, body):
    raw = body.encode('utf-8')
    if not 0 < len(raw) < 8192:
        raise RuntimeError('fixed generated program size bound')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()

def exercise(owner, api, mode, root):
    child_path, descendant_path = root/'leader.py', root/'descendant.py'
    child_sha = write_program(child_path, CHILD)
    descendant_sha = write_program(descendant_path, DESCENDANT)
    events = []
    spawned = []
    original_os, original_subprocess = owner.os, owner.subprocess
    leader_start = None

    def event(row):
        if len(events) >= MAX_EVENTS:
            raise RuntimeError('diagnostic event count cap')
        events.append(row)

    class ObservedOS:
        def __getattr__(self, key):
            return getattr(os, key)

        def waitid(self, kind, pid, flags):
            value = os.waitid(kind, pid, flags)
            if spawned and pid == spawned[0].pid:
                fields = None if value is None else {key: int(getattr(value, key)) for key in
                                                     ('si_pid', 'si_uid', 'si_signo', 'si_status', 'si_code')}
                event(dict(operation='waitid', flags=int(flags), result=fields))
            return value

        def killpg(self, pid, number):
            if len(spawned) != 1 or pid != spawned[0].pid or number != signal.SIGKILL:
                raise RuntimeError('diagnostic refuses an unowned group signal')
            row = dict(operation='first_group_signal', pid=pid, signal=int(number),
                       leader_returncode_before=spawned[0].returncode,
                       before=record_snapshot(api, pid))
            event(row)
            try:
                # Preserve exact legacy behavior: EPERM is recorded AND raised.
                os.killpg(pid, number)
            except BaseException as error:
                row.update(signal_returned=False, type=type(error).__name__,
                           errno=getattr(error, 'errno', None), message=str(error),
                           after=record_snapshot(api, pid))
                raise
            else:
                row.update(signal_returned=True, after=record_snapshot(api, pid))

    class ObservedSubprocess:
        STDOUT = subprocess.STDOUT

        @staticmethod
        def Popen(argv, **options):
            if options.get('start_new_session') is not True or spawned:
                raise RuntimeError('diagnostic requires one fresh owned session')
            process = subprocess.Popen(argv, **options)
            spawned.append(process)
            event(dict(operation='spawn', pid=process.pid, start_new_session=True))
            return process

    owner.os, owner.subprocess = ObservedOS(), ObservedSubprocess
    started = time.monotonic()
    outcome = error_record = None
    def on_launch(pid):
        nonlocal leader_start
        deadline = time.monotonic()+2
        path = root/'leader-start.json'
        while not path.is_file():
            if time.monotonic()>=deadline:
                raise RuntimeError('diagnostic leader handshake deadline')
            time.sleep(.005)
        if path.stat().st_size>1024:
            raise RuntimeError('diagnostic handshake byte cap')
        declared = json.loads(path.read_text())
        if declared != dict(pid=pid,pgid=pid,sid=pid) or os.getpgid(pid)!=pid or os.getsid(pid)!=pid:
            raise RuntimeError('generated fresh session ownership calibration differs')
        leader_start = api.info(pid)
        if leader_start['uid']!=os.getuid() or leader_start['pgid']!=pid:
            raise RuntimeError('generated leader native credentials/group differ')
        event(dict(operation='launch_ownership_calibration',leader=leader_start,sid=pid))
        with (root/'release').open('xb') as gate:
            gate.write(b'1\n')
    try:
        outcome = owner.run_owned([sys.executable, '-B', '-X', 'utf8', str(child_path), str(root), mode,
                                   str(descendant_path), descendant_sha],
                                  cwd=str(root), environment=dict(os.environ),
                                  logpath=root/'native.log', timeout=CASE_SECONDS,
                                  on_launch=on_launch)
    except BaseException as error:
        error_record = dict(type=type(error).__name__, errno=getattr(error, 'errno', None), message=str(error))
    finally:
        owner.os, owner.subprocess = original_os, original_subprocess
    elapsed = time.monotonic() - started
    # Reading returncode does not poll or reap. The pinned owner performed its
    # own direct-child wait; no signal is sent here after that first reap.
    result = dict(mode=mode, outcome=outcome, error=error_record, events=events,
                  leader_native_start=leader_start,
                  wall_seconds=elapsed, direct_child_returncode=None if not spawned else spawned[0].returncode,
                  post_owner_group=None if not spawned else record_snapshot(api, spawned[0].pid),
                  owner_qualified=False, cleanup_semantics_changed=False)
    if spawned and spawned[0].returncode is None:
        result['diagnostic_incomplete'] = 'owner did not return with its direct child reaped'
    # The deliberate other-PGID control uses a short self-expiry. Never signal
    # any group after the owned leader has been reaped, nor signal an orphan by
    # a potentially reused PID. Retain explicit observation of this limitation.
    path = root/'descendant.json'
    if path.is_file():
        if path.stat().st_size > 1024:
            raise RuntimeError('diagnostic descendant record cap')
        descendant = json.loads(path.read_text())
        result['generated_descendant'] = descendant
        deadline = time.monotonic()+15
        while True:
            try:
                row = api.info(descendant['pid'])
            except OSError as error:
                result['descendant_last_observation'] = dict(type=type(error).__name__,errno=error.errno)
                break
            result['descendant_last_observation'] = row
            if row['status']==5:
                break
            if time.monotonic()>=deadline:
                result['diagnostic_incomplete'] = 'generated descendant self-expiry was not observed'
                break
            time.sleep(.02)
    if digest(child_path) != child_sha or digest(descendant_path) != descendant_sha:
        raise RuntimeError('fixed generated Python program changed')
    result['generated_programs'] = dict(leader=dict(path=str(child_path),sha256=child_sha),
                                        descendant=dict(path=str(descendant_path),sha256=descendant_sha))
    write_new(root/'case.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', required=True)
    arguments = parser.parse_args()
    if platform.system() != 'Darwin' or os.name != 'posix':
        raise RuntimeError('prepared Darwin diagnostic must not run on another OS')
    if not __debug__:
        raise RuntimeError('optimized diagnostic execution refused')
    checkout = Path(arguments.checkout).resolve(strict=True)
    paths = [checkout / role for role in ROLES]
    if any(digest(path) != OWNER_SHA for path in paths):
        raise RuntimeError('existing owner source pin differs; diagnostic refuses repair code')
    api = DarwinProcesses()
    parent = Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir())).resolve(strict=True)
    if parent == checkout or checkout in parent.parents or parent in checkout.parents:
        raise RuntimeError('diagnostic output parent overlaps source checkout')
    root = Path(tempfile.mkdtemp(prefix='solarsindy-darwin-owner-diag-', dir=parent)).resolve(strict=True)
    started = time.monotonic()
    records = []
    for index, (role, path) in enumerate(zip(ROLES, paths)):
        spec = importlib.util.spec_from_file_location(f'diagnostic_pinned_owner_{index}', path)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        for mode in CASES:
            if time.monotonic() - started >= WHOLE_SECONDS - 60:
                raise RuntimeError('whole diagnostic monotonic deadline')
            case_root = root/f'{index}-{mode}'
            case_root.mkdir()
            records.append(dict(role=role, result=exercise(owner, api, mode, case_root)))
            if digest(path) != OWNER_SHA:
                raise RuntimeError('owner source changed during diagnostic')
    uname = platform.uname()
    summary = dict(schema='darwin_owned_group_observation_diagnostic_v2',
                   observation_records_complete=True, owner_sha256=OWNER_SHA,
                   system=uname.system, release=uname.release, version=uname.version, machine=uname.machine,
                   python_version=sys.version, python_binary_sha256=digest(sys.executable),
                   diagnostic_source_sha256=digest(__file__), proc_bsdinfo_size=C.sizeof(BSDInfo),
                   waitid_codes={key: int(getattr(os, key)) for key in
                                 ('CLD_EXITED', 'CLD_KILLED', 'CLD_DUMPED',
                                  'CLD_STOPPED', 'CLD_CONTINUED', 'CLD_TRAPPED')},
                   records=records, wall_seconds=time.monotonic()-started,
                   source_or_cleanup_modified=False, production_owner_qualified=False,
                   original_CI_root_cause_proven=False, scientific_qualified=False)
    write_new(root/'OBSERVATIONS.json', summary)
    print(json.dumps(dict(output=str(root), summary_sha256=digest(root/'OBSERVATIONS.json'),
                          production_owner_qualified=False), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
