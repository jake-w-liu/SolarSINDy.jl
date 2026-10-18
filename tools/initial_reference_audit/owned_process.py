"""Bounded native process execution with owned descendant cleanup.

On Windows the process is created suspended and assigned to a private kill-on-
close Job Object before its primary thread resumes. POSIX uses a fresh session.
The utility never attaches an existing process to its ownership boundary.
"""
import os
from pathlib import Path
import signal
import subprocess
import time


def run_owned(argv, *, cwd, environment, logpath, timeout, on_launch=None):
    if os.name == 'nt':
        return _run_windows(argv, cwd, environment, logpath, timeout, on_launch)
    with Path(logpath).open('xb', buffering=0) as log:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        expired = False
        try:
            if on_launch:
                on_launch(process.pid)
            try:
                native = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                expired = True
                os.killpg(process.pid, signal.SIGKILL)
                native = process.wait(timeout=20)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=20)
        return dict(pid=process.pid, native_exit=native, timed_out=expired,
                    descendant_cleanup='owned POSIX session killed after stage',
                    ownership='start_new_session=True')


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
