"""Fixed, generated-only B1 native fixtures. Never a production CLI option."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tools/actual_b_chain/run_initial_l0_native.py'


def write_all(fd, body):
    body = memoryview(body)
    while body:
        count = os.write(fd, body)
        if type(count) is not int or not 0 < count <= len(body):
            raise OSError('fixture write made no progress')
        body = body[count:]


def emit(fd, count):
    while count:
        take = min(count, 65536)
        write_all(fd, b'B' * take)
        count -= take


def save(path, value):
    body = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def driver():
    spec = importlib.util.spec_from_file_location('b1_generated_diagnostic_fixture', DRIVER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main(mode, directory):
    root = Path(directory)
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError('existing canonical fixture root required')
    if mode == 'exit0':
        write_all(1, b'B1_FIXED_EXIT0\n')
        return 0
    if mode == 'exit7':
        write_all(2, b'B1_FIXED_EXIT7\n')
        return 7
    if mode in ('raw-at64MiB', 'raw-over64MiB'):
        emit(1, 67108864 + (mode == 'raw-over64MiB'))
        return 0
    if mode == 'inherited-writer':
        time.sleep(4)
        return 0
    allowed = {'diagnostic-below', 'diagnostic-at', 'diagnostic-mixed',
               'diagnostic-overflow', 'diagnostic-record-failure',
               'diagnostic-write-failure', 'diagnostic-read-failure',
               'diagnostic-overflow-fsync-failure', 'diagnostic-overflow-close-failure',
               'diagnostic-held-writer'}
    if mode not in allowed:
        raise ValueError('unsupported fixed generated fixture mode')
    value = driver()
    if mode == 'diagnostic-record-failure':
        # Publication fault is fixed to the designated generated root. The
        # production controller would reject this stale name before launch.
        save(root / 'SUPERVISOR_DIAGNOSTIC_WAIT.json', {'fixture_poison': True})
    if mode == 'diagnostic-read-failure':
        def read_fault(fd, count):
            raise OSError('fixed generated reader read failure')
        value.os.read = read_fault
    state = value.start_diagnostics(str(root))
    if mode == 'diagnostic-overflow-fsync-failure':
        original_fsync = os.fsync
        raised = [False]
        def fsync_fault(fd):
            if not raised[0]:
                raised[0] = True
                raise OSError('fixed generated post-overflow fsync failure')
            return original_fsync(fd)
        value.os.fsync = fsync_fault
    if mode == 'diagnostic-overflow-close-failure':
        original_close = os.close
        raised = [False]
        def close_fault(fd):
            original_close(fd)
            if fd not in (1, 2) and not raised[0]:
                raised[0] = True
                raise OSError('fixed generated post-overflow close failure')
        value.os.close = close_fault
    if mode == 'diagnostic-write-failure':
        # Inject only the reader's generated raw write; fd1/fd2 still use the
        # actual OS pipe, and all tests remain inside a genuine owned child.
        original_write = os.write
        raised = [False]
        def fault(fd, body):
            if fd not in (1, 2) and not raised[0]:
                raised[0] = True
                raise OSError('fixed generated reader storage failure')
            return original_write(fd, body)
        value.os.write = fault
    if mode == 'diagnostic-held-writer':
        subprocess.Popen([sys.executable, '-I', '-S', '-u', '-B', '-X', 'utf8',
                          str(Path(__file__)), 'inherited-writer', str(root)],
                         stdin=subprocess.DEVNULL, close_fds=False)
        emit(1, 17)
    elif mode == 'diagnostic-mixed':
        sys.stderr.write('B' * 16)
        sys.stderr.flush()
        emit(1, 4194296)
        emit(2, 4194296)
    else:
        count = {'diagnostic-below': 17, 'diagnostic-at': 8388608,
                 'diagnostic-overflow': 8388609,
                 'diagnostic-overflow-fsync-failure': 8388609,
                 'diagnostic-overflow-close-failure': 8388609}.get(mode, 17)
        emit(2, count)
    # Fixed short deadline, solely for this unqualified capture fixture. It
    # is never passed to the actual request/permit admission path.
    now = time.monotonic_ns()
    context = {'start_ns': now, 'deadline_ns': now + 10_000_000_000}
    result = value.finish_diagnostics(state, context)
    save(root / 'FIXTURE_RESULT.json', result)
    status = 0 if result['reader_joined'] and result['capture_complete'] else 1
    os._exit(status)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('fixed mode and generated root required')
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
