"""Fixed, finite source-file children for independent native owner controls."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def record(root, name, data):
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    if len(raw) > 2048:
        raise RuntimeError('fixture metadata bound')
    path = root / (name + '.tmp')
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise RuntimeError('fixture short write')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(path, root / (name + '.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('mode')
    parser.add_argument('--descendant', action='store_true')
    args = parser.parse_args()
    if args.descendant:
        if args.mode != 'same_group_descendant':
            os.setpgid(0, 0)
        record(args.root, 'descendant', dict(pid=os.getpid(), pgid=os.getpgrp(), sid=os.getsid(0)))
        seconds = dict(same_group_descendant=32, other_group_12s=12,
                       other_group_short=0.5, other_group_persistent=32)[args.mode]
        time.sleep(seconds)
        return 0
    record(args.root, 'leader', dict(pid=os.getpid(), pgid=os.getpgrp(), sid=os.getsid(0)))
    deadline = time.monotonic() + 3
    while not (args.root / 'release').is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('finite leader release wait')
        time.sleep(0.005)
    if args.mode in ('same_group_descendant', 'other_group_12s', 'other_group_short', 'other_group_persistent'):
        subprocess.Popen([sys.executable, '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                          str(args.root), args.mode, '--descendant'])
        deadline = time.monotonic() + 3
        while not (args.root / 'descendant.json').is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError('finite descendant readiness wait')
            time.sleep(0.005)
    if args.mode in ('persistent_stop', 'transient_stop'):
        sys.stdout.buffer.write(b'INDEPENDENT_CHILD_SELF_STOP\n')
        sys.stdout.buffer.flush()
        os.kill(os.getpid(), signal.SIGSTOP)
    return 7


if __name__ == '__main__':
    raise SystemExit(main())
