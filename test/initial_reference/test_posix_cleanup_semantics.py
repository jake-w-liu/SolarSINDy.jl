"""Cross-platform deterministic POSIX lifetime oracles; no real signals sent."""
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parents[1]/'tools/initial_reference_audit'
sys.path.insert(0,str(TOOLS))
import owned_process

ORACLE=HERE/'frozen/posix_cleanup_oracles_v1.json'
BEFORE=HERE/'frozen/owned_process_before_v1.py'


class PosixCleanupSemantics(unittest.TestCase):
    def setUp(self):
        self.assertEqual(hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
            'cc6a1d5892321862e1826300600c4f0f46d0dfe9a99c0e1b4106131ead16408c')
        self.oracle=json.loads(ORACLE.read_bytes())
        self.assertEqual(hashlib.sha256(BEFORE.read_bytes()).hexdigest(),self.oracle['before_source_sha256'])

    def exercise(self,module,mode):
        events=[];process=SimpleNamespace(pid=424242,returncode=None,waits=0,kills=0)
        def wait(timeout):
            events.append(['wait',timeout]);process.waits+=1
            if process.returncode is not None:return process.returncode
            if mode=='normal_exit':process.returncode=0;return 0
            if process.waits==1:raise subprocess.TimeoutExpired(['fixture'],timeout)
            raise AssertionError('wait before the owned process has terminated')
        process.wait=wait
        def popen(argv,**options):
            self.assertEqual(argv,['fixture'])
            self.assertTrue(options['start_new_session'])
            events.append(['spawn',options['start_new_session']]);return process
        def killpg(pid,number):
            events.append(['killpg',pid,number]);process.kills+=1
            self.assertEqual((pid,number),(424242,9))
            if process.kills>1:raise PermissionError(errno.EPERM,'second signal of terminated group')
            if mode=='permission_denied':raise PermissionError(errno.EPERM,'first signal denied')
            if mode=='disappeared_group':
                process.returncode=0
                raise ProcessLookupError(errno.ESRCH,'group disappeared at deadline')
            if process.returncode is None:process.returncode=-9
        def callback(pid):
            events.append(['callback',pid]);raise RuntimeError('launch callback oracle')
        outcome=None;error=None
        with tempfile.TemporaryDirectory(prefix='solarsindy-posix-control-') as temporary:
            root=Path(temporary).resolve(strict=True)
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
            with patch.object(module,'os',SimpleNamespace(name='posix',killpg=killpg)), \
                    patch.object(module,'signal',SimpleNamespace(SIGKILL=9)), \
                    patch.object(module,'subprocess',SimpleNamespace(Popen=popen,TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT)):
                try:
                    outcome=module.run_owned(['fixture'],cwd=str(root),environment={},logpath=root/'native.log',
                        timeout=5,on_launch=callback if mode=='callback_error' else None)
                except BaseException as caught:error=caught
        return outcome,error,events

    def test_frozen_timeout_reproduces_second_signal_failure(self):
        spec=importlib.util.spec_from_file_location('frozen_owned_process_before',BEFORE)
        before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
        result,error,events=self.exercise(before,'timeout')
        self.assertIsNone(result);self.assertIsInstance(error,PermissionError)
        self.assertEqual(events,self.oracle['timeout']['events']+[['killpg',424242,9]])
        # The Windows native Job Object implementation retains its exact bytes.
        self.assertEqual(BEFORE.read_bytes().split(b'def _run_windows',1)[1],
            Path(owned_process.__file__).read_bytes().split(b'def _run_windows',1)[1])

    def verify(self,mode):
        result,error,events=self.exercise(owned_process,mode)
        expected=self.oracle[mode]
        self.assertEqual(events,expected['events'])
        if mode=='callback_error':self.assertIsInstance(error,RuntimeError);self.assertIsNone(result)
        elif mode=='permission_denied':self.assertIsInstance(error,PermissionError);self.assertIsNone(result)
        else:
            self.assertIsNone(error)
            self.assertEqual(result['native_exit'],expected['native_exit'])
            self.assertEqual(result['timed_out'],expected['timed_out'])
            self.assertEqual(result['ownership'],'start_new_session=True')

    def test_timeout_signals_owned_group_once(self):self.verify('timeout')
    def test_normal_exit_still_signals_owned_descendants(self):self.verify('normal_exit')
    def test_callback_error_still_terminates_and_reaps(self):self.verify('callback_error')
    def test_disappeared_group_remains_a_timed_out_deadline(self):self.verify('disappeared_group')
    def test_first_permission_denial_is_fatal(self):self.verify('permission_denied')


if __name__=='__main__':unittest.main()
