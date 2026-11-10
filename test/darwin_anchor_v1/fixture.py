"""One fixed harmless Darwin launch child; no descendants or external PID input."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('mode',choices=('exit7','signal9','stop_then_exit7'))
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    with Path(__file__).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=args.source_sha256:
            raise RuntimeError('fixed harmless fixture changed')
    body=(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),sid=os.getsid(0)),sort_keys=True)+'\n').encode()
    temporary=args.directory/'READY.tmp'
    with temporary.open('xb') as stream:
        if stream.write(body)!=len(body):raise RuntimeError('short fixture handshake')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,args.directory/'READY.json')
    deadline=time.monotonic()+3
    while not (args.directory/'release').is_file():
        if time.monotonic()>=deadline:return 17
        time.sleep(.005)
    if args.mode=='stop_then_exit7':
        os.kill(os.getpid(),signal.SIGSTOP)
    if args.mode=='signal9':
        os.kill(os.getpid(),signal.SIGKILL)
        return 18
    return 7


if __name__=='__main__':
    raise SystemExit(main())
