"""Bounded native execution inside a private Windows Job or POSIX session.

POSIX success requires observed zero live members of the private session while
the direct leader is still held. Individual descendant reaping is unobserved.
"""
import errno
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

_MAX_PIDS = 4096
_CLEANUP_SECONDS = 20.
_POSIX_CLEANUP = 'owned POSIX session has no live members after stage'


def _need(value, message):
    if not value:
        raise RuntimeError(message)


def _before(deadline):
    _need(time.monotonic() < deadline, 'owned process census/cleanup deadline exhausted')


def _identity(row):
    return {key: row[key] for key in ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid')}


class _NativeCensus:
    def leader(self, pid):
        row = self._info(pid)
        _need(row['pid'] == row['pgid'] == row['sid'] == pid and row['birth'] is not None and
              row['uid'] == row['ruid'] == os.getuid() and row['ppid'] == os.getpid(),
              'direct child native PID/birth/private-session identity differs')
        return _identity(row)

    def _pass(self, deadline):
        _before(deadline)
        before = self._ids()
        rows = []
        for pid in before:
            _before(deadline)
            if pid:
                rows.append(self._info(pid))
        _before(deadline)
        _need(before == self._ids(), 'native process inventory changed during census')
        return rows

    @staticmethod
    def _keys(rows):
        # State transitions between running/sleeping are not identity changes.
        # Zombie transitions and all PID/birth/session/group/UID changes are.
        return tuple((r['pid'], r['birth'], r['pgid'], r['sid'], r['uid'], r['ruid'], r['zombie']) for r in rows)

    def _stable(self, deadline):
        first = self._pass(deadline)
        second = self._pass(deadline)
        _need(self._keys(first) == self._keys(second), 'native process identities changed during census')
        return second

    def preflight(self, deadline):
        last = None
        deadline = min(deadline, time.monotonic() + _CLEANUP_SECONDS)
        while time.monotonic() < deadline:
            try:
                rows = self._stable(deadline)
                own = next(r for r in rows if r['pid'] == os.getpid())
                _need(own['pgid'] == os.getpgrp() and own['sid'] == os.getsid(0) and
                      own['uid'] == os.getuid() and own['birth'] is not None,
                      'native process reader self calibration differs')
                return
            except (OSError, RuntimeError, StopIteration) as error:
                last = error
            time.sleep(min(.01, max(0., deadline - time.monotonic())))
        raise RuntimeError('native session census unavailable before launch') from last

    def snapshot(self, identity, deadline):
        rows = self._stable(deadline)
        leader = next((r for r in rows if r['pid'] == identity['pid']), None)
        _need(leader is not None and _identity(leader) == identity,
              'held leader birth/private-session identity changed')
        group = tuple(r for r in rows if r['pgid'] == identity['pgid'])
        live = tuple(r for r in rows if not r['zombie'] and r['sid'] == identity['sid'])
        _need(all(r['uid'] == r['ruid'] == identity['uid'] for r in live),
              'private session contains a foreign-UID live member')
        return dict(complete=True, leader_held=True,
                    group_live=sum(not r['zombie'] for r in group),
                    group_zombie_only=bool(group) and all(r['zombie'] for r in group),
                    session_live=len(live), group_members=group, session_live_members=live)


class _LinuxProcesses(_NativeCensus):
    def __init__(self):
        _need(Path('/proc/self/stat').is_file(), 'Linux native process census unavailable')

    def _ids(self):
        with os.scandir('/proc') as entries:
            result = []
            for entry in entries:
                if entry.name.isascii() and entry.name.isdigit():
                    result.append(int(entry.name))
                    _need(len(result) <= _MAX_PIDS, 'native PID inventory exceeds bound')
        _need(len(result) == len(set(result)) and all(pid > 0 for pid in result),
              'native PID inventory is duplicated or invalid')
        return tuple(sorted(result))

    def _info(self, pid):
        _need(type(pid) is int and 0 < pid < 2**31, 'native PID outside bound')
        root = Path('/proc') / str(pid)
        before = root.stat()
        with (root / 'stat').open('rb') as stream:
            raw = stream.read(8193)
        with (root / 'status').open('rb') as stream:
            status = stream.read(8193)
        after = root.stat()
        _need(0 < len(raw) <= 8192, 'Linux process record exceeds bound')
        _need(0 < len(status) <= 8192 and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
              'Linux status record exceeds bound or process directory changed')
        closing = raw.rfind(b')')
        _need(closing > 0 and raw[:raw.find(b' ')].decode('ascii') == str(pid), 'Linux process PID framing differs')
        fields = raw[closing + 1:].split()
        _need(len(fields) >= 20 and fields[0] in (b'R', b'S', b'D', b'T', b't', b'Z', b'X', b'I'),
              'Linux native process state differs')
        ids = [line.split()[1:] for line in status.splitlines() if line.startswith(b'Uid:')]
        pids = [line.split()[1:] for line in status.splitlines() if line.startswith(b'Pid:')]
        _need(len(ids) == 1 and len(ids[0]) == 4 and
              all(0 < len(v) <= 10 and v.isdigit() and int(v) < 2**32 for v in ids[0]) and
              pids == [[str(pid).encode('ascii')]], 'Linux native real/effective UID or PID record differs')
        ruid, uid = int(ids[0][0]), int(ids[0][1])
        return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                    birth=(int(fields[19]),), uid=uid, ruid=ruid, zombie=fields[0] in (b'Z', b'X'))


class _DarwinProcesses(_NativeCensus):
    def __init__(self):
        import ctypes as C
        self.C = C

        class BSDInfo(C.Structure):
            _fields_ = [(n, C.c_uint32) for n in
                        ('flags', 'status', 'xstatus', 'pid', 'ppid', 'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]
            _fields_ += [('comm', C.c_char * 16), ('name', C.c_char * 32)]
            _fields_ += [(n, C.c_uint32) for n in ('nfiles', 'pgid', 'pjobc', 'ttydev', 'ttypgid')]
            _fields_ += [('nice', C.c_int32), ('start_sec', C.c_uint64), ('start_usec', C.c_uint64)]

        class ShortBSDInfo(C.Structure):
            _fields_ = [(n, C.c_uint32) for n in ('pid', 'ppid', 'pgid', 'status')]
            _fields_ += [('comm', C.c_char * 16)]
            _fields_ += [(n, C.c_uint32) for n in ('flags', 'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]

        _need(C.sizeof(BSDInfo) == 136 and BSDInfo.pid.offset == 12 and BSDInfo.pgid.offset == 100 and
              C.sizeof(ShortBSDInfo) == 64 and ShortBSDInfo.uid.offset == 36, 'native libproc ABI layout differs')
        self.BSDInfo, self.ShortBSDInfo = BSDInfo, ShortBSDInfo
        self.library = C.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.library.proc_listpids.argtypes = (C.c_uint32, C.c_uint32, C.c_void_p, C.c_int)
        self.library.proc_listpids.restype = C.c_int
        self.library.proc_pidinfo.argtypes = (C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int)
        self.library.proc_pidinfo.restype = C.c_int
        own = self._info(os.getpid())
        _need(own['uid'] == os.getuid() and own['pgid'] == os.getpgrp() and own['sid'] == os.getsid(0),
              'native libproc self calibration differs')

    def _ids(self):
        C = self.C
        values = (C.c_int * (_MAX_PIDS + 1))()
        C.set_errno(0)
        count = self.library.proc_listpids(1, 0, values, C.sizeof(values))
        error = C.get_errno()
        _need(0 <= count < C.sizeof(values) and count % C.sizeof(C.c_int) == 0 and not (count == 0 and error),
              'native libproc PID enumeration failed/truncated')
        result = tuple(sorted(values[:count // C.sizeof(C.c_int)]))
        _need(len(result) == len(set(result)) and all(pid >= 0 for pid in result), 'native PID inventory differs')
        return result

    def _record(self, pid, kind, structure):
        C = self.C
        value = structure()
        C.set_errno(0)
        count = self.library.proc_pidinfo(pid, kind, 1, C.byref(value), C.sizeof(value))
        error = C.get_errno()
        if count != C.sizeof(value) or value.pid != pid:
            raise OSError(error or errno.EPROTO, 'native process information unavailable')
        return value

    def _info(self, pid):
        _need(type(pid) is int and 0 < pid < 2**31, 'native PID outside bound')
        short = self._record(pid, 13, self.ShortBSDInfo)
        zombie = short.status == 5
        # Full BSD metadata requires the same user. Short BSD metadata is the
        # unprivileged global enumeration primitive; foreign users are not opened
        # with the privileged/full flavor merely to inspect another session.
        full = None
        if short.uid == short.ruid == os.getuid():
            full = self._record(pid, 3, self.BSDInfo)
            _need(full.pgid == short.pgid and full.uid == short.uid and full.ruid == short.ruid and
                  (full.status == 5) == zombie, 'native process identity changed between records')
        flags = int(full.flags if full is not None else short.flags)
        sid = (pid if flags & 0x20 else None) if zombie else os.getsid(pid)
        if full is not None:
            _need(bool(flags & 0x20) == (sid == pid), 'native session-leader flag calibration differs')
        return dict(pid=pid, ppid=int(short.ppid), pgid=int(short.pgid), sid=sid,
                    birth=None if full is None else (int(full.start_sec), int(full.start_usec)),
                    uid=int(short.uid), ruid=int(short.ruid), zombie=zombie)


def _processes():
    if sys.platform == 'darwin':
        return _DarwinProcesses()
    if sys.platform.startswith('linux'):
        return _LinuxProcesses()
    raise RuntimeError('native private-session census unsupported on this POSIX platform')


def _posix_features(deadline=None):
    required = ('waitid', 'P_PID', 'WEXITED', 'WNOHANG', 'WNOWAIT', 'CLD_EXITED', 'CLD_KILLED', 'CLD_DUMPED', 'CLD_STOPPED')
    _need(os.name == 'posix' and all(hasattr(os, key) for key in required) and
          (sys.platform != 'darwin' or hasattr(os, 'WSTOPPED')), 'nonreaping POSIX terminal-event support is required')
    _need(signal.getsignal(signal.SIGCHLD) != signal.SIG_IGN,
          'SIGCHLD auto-reaping would release the owned PID before cleanup')
    try:
        os.waitid(os.P_PID, os.getpid(), os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        pass
    except OSError as error:
        raise RuntimeError('nonreaping waitid unavailable before launch') from error
    adapter = _processes()
    adapter.preflight(time.monotonic() + _CLEANUP_SECONDS if deadline is None else deadline)
    return adapter


def _checked_snapshot(adapter, identity, deadline, summary):
    last = None
    while time.monotonic() < deadline:
        summary['census_attempts'] += 1
        try:
            value = adapter.snapshot(identity, deadline)
            _need(type(value) is dict and value.get('complete') is True and value.get('leader_held') is True and
                  type(value.get('group_live')) is int and type(value.get('session_live')) is int and
                  0 <= value['group_live'] <= value['session_live'] <= _MAX_PIDS and
                  type(value.get('group_zombie_only')) is bool, 'native session census is incomplete')
            _before(deadline)
            return value
        except (OSError, RuntimeError) as error:
            last = error
        time.sleep(min(.01, max(0., deadline - time.monotonic())))
    raise RuntimeError('native session closure remains unknown at cleanup deadline') from last


def _acquire_leader(adapter, pid, deadline):
    last = None
    while time.monotonic() < deadline:
        try:
            identity = adapter.leader(pid)
            _before(deadline)
            return identity
        except (OSError, RuntimeError) as error:
            # A child can change from live to zombie between libproc's short
            # and full records. Retry that unknown state without reaping it.
            last = error
        time.sleep(min(.01, max(0., deadline - time.monotonic())))
    raise RuntimeError('direct-child native identity unavailable within owned deadline') from last


def _cleanup_posix(process, adapter, identity, summary):
    # This is one shared allowance, including every census and the first wait.
    deadline = time.monotonic() + _CLEANUP_SECONDS
    error = None
    native = None
    group_attempted = False
    try:
        _before(deadline)
        if identity is None:
            identity = _acquire_leader(adapter, process.pid, deadline)
            summary['leader_identity'] = identity.copy()
        _need(_acquire_leader(adapter, process.pid, deadline) == identity,
              'owned leader identity no longer held before group signal')
        try:
            group_attempted = True
            summary['group_signal'] = 'attempted'
            os.killpg(process.pid, signal.SIGKILL)
            summary['group_signal'] = 'returned'
        except ProcessLookupError:
            summary['group_signal'] = 'missing'
        except PermissionError as denied:
            summary['group_signal'] = 'denied'
            evidence = _checked_snapshot(adapter, identity, deadline, summary)
            if not evidence['group_zombie_only'] or evidence['group_live']:
                raise denied
            summary['group_signal'] = 'verified_nonsignalable'
            summary['group_zombie_only_verified'] = True
        while True:
            evidence = _checked_snapshot(adapter, identity, deadline, summary)
            if evidence['session_live'] == 0:
                summary['session_zero_live_verified'] = True
                break
            # Other process groups are observed, never signaled by borrowed PID.
            time.sleep(min(.01, max(0., deadline - time.monotonic())))
        _before(deadline)
    except BaseException as caught:
        error = caught
        # An error never becomes a boundary-closure claim. A still-held direct
        # child may be killed on this failure path; arbitrary descendants may not.
        try:
            if time.monotonic() < deadline and group_attempted and adapter.leader(process.pid) == identity:
                # Popen.kill() polls first and can silently reap a zombie. Use
                # the verified, still-owned direct PID without a hidden wait.
                os.kill(process.pid, signal.SIGKILL)
        except BaseException:
            pass
    try:
        native = process.wait(timeout=max(0., deadline - time.monotonic()))
    except BaseException as caught:
        if error is None:
            error = caught
    if time.monotonic() >= deadline and error is None:
        error = RuntimeError('complete POSIX cleanup exceeded its shared20-second deadline')
    if error is not None:
        try:
            error.ownership_boundary_closed = False
            error.leader_reaped = native is not None
            error.native_exit = native
            error.closure_summary = summary.copy()
        except BaseException:
            pass
        raise error
    return native


def _run_posix(argv, cwd, environment, logpath, timeout, on_launch, *, started=None):
    started = time.monotonic() if started is None else started
    _need(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0 and math.isfinite(started + timeout),
          'positive finite owned stage timeout required')
    deadline = started + timeout
    adapter = _posix_features(deadline)
    _before(deadline)
    expired = False
    summary = dict(leader_identity=None, terminal_observation=None, group_signal='not_attempted',
                   group_zombie_only_verified=False, session_zero_live_verified=False,
                   census_attempts=0, stop_notifications_consumed=0)
    with Path(logpath).open('xb', buffering=0) as log:
        _before(deadline)
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        identity = None
        anchor_held = True
        try:
            identity = _acquire_leader(adapter, process.pid, deadline)
            summary['leader_identity'] = identity.copy()
            if on_launch:
                on_launch(process.pid)
            terminal_codes = (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)
            while True:
                observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                expired = time.monotonic() >= deadline
                if observed is not None:
                    _need(observed.si_pid == process.pid, 'waitid returned another process identity')
                    if observed.si_code in terminal_codes:
                        summary['terminal_observation'] = {key: int(getattr(observed, key)) for key in
                                                           ('si_pid', 'si_code', 'si_status', 'si_signo', 'si_uid')}
                        break
                    if observed.si_code == os.CLD_STOPPED and sys.platform == 'darwin':
                        # CPython3.14.10 asyncio uses this stop-only consumption;
                        # the terminal child remains unreaped under WNOWAIT.
                        stopped = os.waitid(os.P_PID, process.pid, os.WSTOPPED | os.WNOHANG)
                        if stopped is not None and (stopped.si_pid != process.pid or stopped.si_code != os.CLD_STOPPED):
                            anchor_held = False
                            raise RuntimeError('stop-only wait returned an unexpected process/event; anchor not trusted')
                        summary['stop_notifications_consumed'] += 1
                    elif observed.si_code not in (os.CLD_STOPPED, getattr(os, 'CLD_CONTINUED', None)):
                        raise RuntimeError('unsupported nonterminal waitid event')
                if expired:
                    break
                time.sleep(min(.02, max(0., deadline - time.monotonic())))
        finally:
            if not anchor_held:
                # No numeric signal is safe after loss of the identity anchor.
                # This is an explicit failed ownership path, never a result.
                try:
                    process.wait(timeout=0)
                except BaseException:
                    pass
            else:
                native = _cleanup_posix(process, adapter, identity, summary)
    elapsed = time.monotonic() - started
    _need(summary['session_zero_live_verified'], 'native private-session closure not observed')
    return (dict(pid=process.pid, native_exit=native, timed_out=expired or elapsed >= timeout,
                 wall_seconds=elapsed, descendant_cleanup=_POSIX_CLEANUP, ownership='start_new_session=True'), summary)


def run_owned(argv, *, cwd, environment, logpath, timeout, on_launch=None):
    if os.name == 'nt':
        started = time.monotonic()
        result = _run_windows(argv, cwd, environment, logpath, timeout, on_launch)
        elapsed = time.monotonic() - started
        result['timed_out'] = (result['timed_out'] or elapsed >= timeout or
                              result.get('wall_seconds', elapsed) >= timeout)
        return result
    started = time.monotonic()
    return _run_posix(argv, cwd, environment, logpath, timeout, on_launch, started=started)[0]


def _run_windows(argv, cwd, environment, logpath, timeout, on_launch):
    import ctypes as C
    from ctypes import wintypes as W
    import msvcrt

    kernel = C.WinDLL('kernel32', use_last_error=True)
    SIZE_T = C.c_size_t

    class STARTUPINFO(C.Structure):
        _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR),
                    ('lpDesktop', W.LPWSTR), ('lpTitle', W.LPWSTR),
                    ('dwX', W.DWORD), ('dwY', W.DWORD), ('dwXSize', W.DWORD),
                    ('dwYSize', W.DWORD), ('dwXCountChars', W.DWORD),
                    ('dwYCountChars', W.DWORD), ('dwFillAttribute', W.DWORD),
                    ('dwFlags', W.DWORD), ('wShowWindow', W.WORD),
                    ('cbReserved2', W.WORD), ('lpReserved2', C.POINTER(W.BYTE)),
                    ('hStdInput', W.HANDLE), ('hStdOutput', W.HANDLE),
                    ('hStdError', W.HANDLE)]

    class PROCESS_INFORMATION(C.Structure):
        _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                    ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]

    class BASIC_LIMIT(C.Structure):
        _fields_ = [('PerProcessUserTimeLimit', C.c_longlong),
                    ('PerJobUserTimeLimit', C.c_longlong), ('LimitFlags', W.DWORD),
                    ('MinimumWorkingSetSize', SIZE_T), ('MaximumWorkingSetSize', SIZE_T),
                    ('ActiveProcessLimit', W.DWORD), ('Affinity', SIZE_T),
                    ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]

    class IO_COUNTERS(C.Structure):
        _fields_ = [(name, C.c_ulonglong) for name in
                    ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                     'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class EXTENDED_LIMIT(C.Structure):
        _fields_ = [('BasicLimitInformation', BASIC_LIMIT), ('IoInfo', IO_COUNTERS),
                    ('ProcessMemoryLimit', SIZE_T), ('JobMemoryLimit', SIZE_T),
                    ('PeakProcessMemoryUsed', SIZE_T), ('PeakJobMemoryUsed', SIZE_T)]

    kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    kernel.CreateJobObjectW.restype = W.HANDLE
    kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    kernel.SetInformationJobObject.restype = W.BOOL
    kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    kernel.AssignProcessToJobObject.restype = W.BOOL
    kernel.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
    kernel.TerminateJobObject.restype = W.BOOL
    kernel.CreateProcessW.argtypes = [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p,
                                     W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR,
                                     C.POINTER(STARTUPINFO), C.POINTER(PROCESS_INFORMATION)]
    kernel.CreateProcessW.restype = W.BOOL
    kernel.ResumeThread.argtypes = [W.HANDLE]
    kernel.ResumeThread.restype = W.DWORD
    kernel.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    kernel.WaitForSingleObject.restype = W.DWORD
    kernel.GetExitCodeProcess.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
    kernel.GetExitCodeProcess.restype = W.BOOL
    kernel.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
    kernel.TerminateProcess.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.CloseHandle.restype = W.BOOL

    def checked(value):
        if not value:
            raise C.WinError(C.get_last_error())
        return value

    job = checked(kernel.CreateJobObjectW(None, None))
    process = PROCESS_INFORMATION()
    created = assigned = False
    expired = False
    started = time.monotonic()
    try:
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        checked(kernel.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)))
        with Path(logpath).open('xb', buffering=0) as log, open(os.devnull, 'rb') as null:
            os.set_inheritable(log.fileno(), True)
            os.set_inheritable(null.fileno(), True)
            startup = STARTUPINFO()
            startup.cb = C.sizeof(startup)
            startup.dwFlags = 0x100  # STARTF_USESTDHANDLES
            startup.hStdInput = msvcrt.get_osfhandle(null.fileno())
            startup.hStdOutput = startup.hStdError = msvcrt.get_osfhandle(log.fileno())
            command = C.create_unicode_buffer(subprocess.list2cmdline([str(a) for a in argv]))
            block = '\0'.join(k+'='+v for k,v in sorted(environment.items(), key=lambda pair: pair[0].upper()))+'\0\0'
            environment_block = C.create_unicode_buffer(block)
            flags = 0x4 | 0x400 | 0x08000000  # SUSPENDED | UNICODE_ENVIRONMENT | NO_WINDOW
            checked(kernel.CreateProcessW(str(argv[0]), command, None, None, True,
                    flags, environment_block, str(cwd), C.byref(startup), C.byref(process)))
            created = True
            checked(kernel.AssignProcessToJobObject(job, process.hProcess))
            assigned = True
            if on_launch:
                on_launch(int(process.dwProcessId))
            if kernel.ResumeThread(process.hThread) == 0xFFFFFFFF:
                raise C.WinError(C.get_last_error())
            checked(kernel.CloseHandle(process.hThread))
            process.hThread = None
            while True:
                state = kernel.WaitForSingleObject(process.hProcess, 100)
                if state == 0:  # process handle signaled: actual native exit
                    break
                if state != 258:
                    raise C.WinError(C.get_last_error())
                if time.monotonic()-started >= timeout:
                    expired = True
                    checked(kernel.TerminateJobObject(job, 0xDEAD))
                    if kernel.WaitForSingleObject(process.hProcess, 20000) != 0:
                        raise RuntimeError('owned process did not terminate within cleanup bound')
                    break
            code = W.DWORD()
            checked(kernel.GetExitCodeProcess(process.hProcess, C.byref(code)))
            result = dict(pid=int(process.dwProcessId), native_exit=int(code.value),
                timed_out=expired, descendant_cleanup='private Windows Job Object closed with KILL_ON_JOB_CLOSE',
                ownership='CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread',
                wall_seconds=time.monotonic()-started)
        return result
    finally:
        # Assignment failure leaves a suspended process with no descendants.
        if created and not assigned:
            kernel.TerminateProcess(process.hProcess, 0xDEAD)
            kernel.WaitForSingleObject(process.hProcess, 20000)
        if job:
            kernel.CloseHandle(job)  # also cancels owned descendants after normal root exit
        if process.hThread:
            kernel.CloseHandle(process.hThread)
        if process.hProcess:
            kernel.CloseHandle(process.hProcess)
