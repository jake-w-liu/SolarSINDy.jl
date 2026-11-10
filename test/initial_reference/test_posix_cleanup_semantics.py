"""Historical V2 POSIX lifetime oracles; no current-owner qualification or signals."""
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
# These frozen V2 spies model the original flags/events, before native session
# evidence and terminal-code admission. Live owner qualification is V3.
V2_OWNER=HERE/'frozen/owned_process_before_v3.py'
if hashlib.sha256(V2_OWNER.read_bytes()).hexdigest()!='15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8':
    raise RuntimeError('historical V2 owner bytes differ')
V2_OWNER_SPEC=importlib.util.spec_from_file_location('historical_owned_process_v2',V2_OWNER)
owned_process=importlib.util.module_from_spec(V2_OWNER_SPEC);V2_OWNER_SPEC.loader.exec_module(owned_process)

ORACLE=HERE/'frozen/posix_cleanup_oracles_v1.json'
BEFORE=HERE/'frozen/owned_process_before_v1.py'
V2_SPEC=importlib.util.spec_from_file_location('reference_posix_oracles_v2',HERE/'posix_lifetime_oracles_v2.py')
V2=importlib.util.module_from_spec(V2_SPEC);V2_SPEC.loader.exec_module(V2)


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
        with tempfile.TemporaryDirectory(prefix='solarsindy-posix-v2-') as temporary:
            return V2.exercise(owned_process,mode,Path(temporary))

    def test_timeout_signals_owned_group_once(self):self.verify('timeout')
    def test_normal_exit_still_signals_owned_descendants(self):self.verify('normal_exit')
    def test_callback_error_still_terminates_and_reaps(self):self.verify('callback_error')
    def test_disappeared_group_remains_a_timed_out_deadline(self):self.verify('disappeared_group')
    def test_first_permission_denial_is_fatal(self):
        for mode in ('permission_denied','direct_kill_error'):
            with self.subTest(mode=mode):self.verify(mode)
    def test_observation_exception_cleans_up_before_propagating(self):self.verify('observation_error')
    def test_observation_and_cleanup_deadlines_preserve_actual_zero(self):
        for mode in ('late_zero','cleanup_late_zero'):
            with self.subTest(mode=mode):self.verify(mode)
    def test_missing_nonreaping_features_fail_before_launch(self):
        for mode in ('missing_feature','unsupported_feature'):
            with self.subTest(mode=mode):self.verify(mode)
    def test_windows_whole_delegate_deadline_preserves_actual_exit(self):V2.verify_windows_deadlines(owned_process)


if __name__=='__main__':unittest.main()
