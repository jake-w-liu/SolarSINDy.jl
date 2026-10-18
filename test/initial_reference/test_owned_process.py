"""Independent descendant-lifetime controls for the portable reader runner."""
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
from owned_process import run_owned

CHILD='''import json,socket,sys,time
from pathlib import Path
with socket.socket() as server:
    server.bind(('127.0.0.1',0));server.listen()
    Path(sys.argv[1]).write_text(json.dumps({'port':server.getsockname()[1]}))
    time.sleep(60)
'''
PARENT='''import json,socket,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1])
subprocess.Popen([sys.executable,str(root/'child.py'),str(root/'ready.json')])
deadline=time.monotonic()+10
while not (root/'ready.json').is_file():
    if time.monotonic()>deadline:raise RuntimeError('owned child did not become ready')
    time.sleep(.01)
port=json.loads((root/'ready.json').read_text())['port']
with socket.create_connection(('127.0.0.1',port),timeout=1):pass
(root/'held-control.json').write_text(json.dumps({'listening_verified':True}))
if sys.argv[2]=='timeout':time.sleep(60)
'''


class OwnedProcess(unittest.TestCase):
    def control(self,mode):
        root=Path(tempfile.mkdtemp(prefix='solarsindy-audit-process-')).resolve(strict=True)
        try:
            (root/'child.py').write_text(CHILD,encoding='utf-8')
            (root/'parent.py').write_text(PARENT,encoding='utf-8')
            native=run_owned([sys.executable,str(root/'parent.py'),str(root),mode],cwd=str(root),
                environment=dict(os.environ),logpath=root/'native.log',timeout=5 if mode=='timeout' else 15)
            self.assertTrue(json.loads((root/'held-control.json').read_text())['listening_verified'])
            self.assertEqual(native['timed_out'],mode=='timeout')
            if mode=='exit':self.assertEqual(native['native_exit'],0)
            else:self.assertNotEqual(native['native_exit'],0)
            port=json.loads((root/'ready.json').read_text())['port']
            deadline=time.monotonic()+3
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):pass
                except OSError:break
                if time.monotonic()>=deadline:self.fail('owned descendant listener survived parent cleanup')
                time.sleep(.02)
        except BaseException:
            print('Retained failed owned-process evidence: '+str(root),file=sys.stderr,flush=True)
            raise
        self.assertEqual(root,root.resolve(strict=True))
        self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
        self.assertTrue(root.name.startswith('solarsindy-audit-process-'))
        shutil.rmtree(root)

    def test_normal_root_exit_closes_descendant_listener(self):self.control('exit')
    def test_timeout_closes_descendant_listener(self):self.control('timeout')


if __name__=='__main__':unittest.main()
