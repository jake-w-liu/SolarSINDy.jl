"""Independent harmless-child lifecycle controls; no Julia/data/Q authority."""
from pathlib import Path
import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT/'tools/actual_b_chain/run_matched342_native.py'
OWNER = ROOT/'research/dbdt_completed_source/candidate_native_owner.py'
LEGACY = OWNER.with_name('native_stage.py')
OBSERVER = ROOT/'test/posix_owner_v3/observations.py'
DARWIN_OBSERVER = ROOT/'test/initial_reference/diagnose_darwin_groups.py'
MEMORY = ROOT/'tools/run_canonical_harness.py'
PINS = {
    TARGET:'ea2f94ec709b5e58086b69fbd142f902499d9a2d2f5dabc8e0c871c452b4533b',
    OWNER:'c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00',
    LEGACY:'30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
    OBSERVER:'c68ce8f6c8ef38aef394538b6a9b1915b17988d73e5647531a5a63ce6eacbf36',
    DARWIN_OBSERVER:'3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2',
    MEMORY:'defbb2a90690acd83ff0dc8eae05fd8622543871ecfa6c7a595b04069d44b331',
}
RAW = 4*1024**2
WHOLE_SECONDS = 180
REAL_POPEN = subprocess.Popen
REAL_KILLPG = getattr(os,'killpg',None)


def need(ok,message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path,name,expected):
    need(digest(path)==expected,'pinned lifecycle dependency changed')
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    need(digest(path)==expected,'lifecycle dependency changed during import')
    return value


def write_new(path,value):
    body = (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    need(len(body)<=RAW,'lifecycle metadata cap')
    with Path(path).open('xb') as stream:
        need(stream.write(body)==len(body),'short lifecycle evidence write')
        stream.flush()
        os.fsync(stream.fileno())


def child_argv(mode,root):
    return [sys.executable,'-I','-B','-X','utf8',str(Path(__file__).resolve()),'--child',mode,str(root),digest(__file__)]


class OwnerTap:
    def __init__(self,owner):
        self.owner = owner
        self.calls = []
        self.returned = None

    def run_owned(self,argv,**kwargs):
        self.calls.append((argv,kwargs))
        self.returned = self.owner.run_owned(argv,**kwargs)
        return self.returned


class LifecycleTests(unittest.TestCase):
    target = None
    owner = None
    observer_module = None
    output = None
    worker_deadline_ns = None
    simulated_rows = []
    native_rows = []

    def test_windows_forwarding_simulated(self):
        for raises in (False,True):
            with self.subTest(raises=raises):
                calls = []
                argv = ['fixed-file.py']
                kwargs = dict(cwd='literal',environment={},logpath='unused',timeout=8,on_launch=object())
                token = object()
                failure = RuntimeError('literal delegated failure')
                def delegated(actual,**options):
                    calls.append((actual,options))
                    if raises:
                        raise failure
                    return token
                def forbidden(*args):
                    raise AssertionError('Windows helper must not use a signal API')
                with patch.object(self.target,'os',SimpleNamespace(name='nt')), \
                        patch.object(self.target,'signal',SimpleNamespace(getsignal=forbidden,signal=forbidden)):
                    if raises:
                        with self.assertRaises(RuntimeError) as caught:
                            self.target._run_owned_stage(SimpleNamespace(run_owned=delegated),argv,**kwargs)
                        self.assertIs(caught.exception,failure)
                    else:
                        self.assertIs(self.target._run_owned_stage(SimpleNamespace(run_owned=delegated),argv,**kwargs),token)
                self.assertEqual(len(calls),1)
                self.assertIs(calls[0][0],argv)
                self.assertEqual(calls[0][1],kwargs)
                self.simulated_rows.append(dict(case='windows_direct_exception' if raises else 'windows_direct_return',passed=True,evidence_kind='simulated_forwarding'))

    def test_posix_handler_states_simulated(self):
        cases = ('normal_default','normal_ignore','normal_callable','owner_exception','callback_exception',
                 'install_failure','first_term','repeat_term','restore_failure')
        for mode in cases:
            with self.subTest(mode=mode):
                sentinel = lambda number,frame:None
                previous = 0 if mode=='normal_default' else 1 if mode=='normal_ignore' else sentinel
                current = [previous]
                actions = []
                owner_calls = []
                token = object()
                failure = RuntimeError('literal owner/callback failure')
                def getsignal(number):
                    self.assertEqual(number,15)
                    return current[0]
                def sethandler(number,handler):
                    self.assertEqual(number,15)
                    if mode=='install_failure' and not actions:
                        raise ValueError('literal handler installation failure')
                    if mode=='restore_failure' and actions and handler is previous:
                        raise ValueError('literal handler restoration failure')
                    actions.append(handler)
                    old = current[0]
                    current[0] = handler
                    return old
                def delegated(argv,**kwargs):
                    owner_calls.append(argv)
                    if mode in ('owner_exception','callback_exception'):
                        if mode=='callback_exception':
                            kwargs['on_launch'](424242)
                        raise failure
                    if mode in ('first_term','repeat_term'):
                        try:
                            current[0](15,None)
                        finally:
                            # This is simulated; the real repeated signal is a
                            # different native checkpoint case below.
                            self.assertEqual(current[0],1)
                            if mode=='repeat_term':
                                actions.append('simulated ignored repeat')
                    return token
                def callback(pid):
                    self.assertEqual(pid,424242)
                    raise failure
                fake = SimpleNamespace(SIGTERM=15,SIG_IGN=1,getsignal=getsignal,signal=sethandler)
                returned = None
                error = None
                with patch.object(self.target,'os',SimpleNamespace(name='posix')),patch.object(self.target,'signal',fake):
                    try:
                        returned = self.target._run_owned_stage(SimpleNamespace(run_owned=delegated),['fixed'],on_launch=callback)
                    except BaseException as caught:
                        error = caught
                if mode.startswith('normal_'):
                    self.assertIsNone(error)
                    self.assertIs(returned,token)
                else:
                    self.assertIsNone(returned)
                    expected = ValueError if mode in ('install_failure','restore_failure') else InterruptedError if mode in ('first_term','repeat_term') else RuntimeError
                    self.assertIsInstance(error,expected)
                    if mode in ('owner_exception','callback_exception'):
                        self.assertIs(error,failure)
                if mode=='install_failure':
                    self.assertEqual(owner_calls,[])
                else:
                    self.assertEqual(len(owner_calls),1)
                if mode!='restore_failure':
                    self.assertIs(current[0],previous)
                self.simulated_rows.append(dict(case='posix_'+mode,passed=True,evidence_kind='simulated_handler_state',
                    actual_signals_sent=0,actual_processes_started=0,primitive_native_result_fabricated=False))

    def windows_case(self,mode):
        directory = self.output/('windows-'+mode)
        directory.mkdir()
        launched = []
        handle = [None]
        actual_exception = RuntimeError('genuine Windows on_launch rejection')
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        from ctypes import wintypes
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        kernel.CloseHandle.restype=wintypes.BOOL
        def on_launch(pid):
            launched.append(pid)
            if mode=='exception':
                handle[0] = kernel.OpenProcess(0x00100000|0x1000,False,pid)
                need(bool(handle[0]),'retain actual suspended child handle')
                raise actual_exception
        def forbidden(*args):
            raise AssertionError('real Windows helper must not touch signal APIs')
        tap = OwnerTap(self.owner)
        argv = child_argv('ok',directory)
        kwargs = dict(cwd=str(directory),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
            logpath=str(directory/'native.raw'),timeout=8,on_launch=on_launch)
        returned = error = None
        direct_exit = None
        try:
            with patch.object(self.target,'signal',SimpleNamespace(getsignal=forbidden,signal=forbidden)):
                try:
                    returned = self.target._run_owned_stage(tap,argv,**kwargs)
                except BaseException as caught:
                    error = caught
            self.assertEqual(len(tap.calls),1)
            self.assertIs(tap.calls[0][0],argv)
            self.assertEqual(tap.calls[0][1],kwargs)
            self.assertEqual(len(launched),1)
            if mode=='normal':
                self.assertIsNone(error)
                self.assertIs(returned,tap.returned)
                self.assertEqual(returned['pid'],launched[0])
                self.assertEqual(returned['native_exit'],0)
                self.assertIs(returned['timed_out'],False)
                self.assertIs(returned['leader_reaped'],True)
                self.assertIs(returned['ownership_boundary_closed'],True)
                self.assertIs(returned['all_descendants_os_reaped'],False)
                self.assertEqual((directory/'native.raw').read_bytes(),b'LIFECYCLE_CHILD_OK\n')
            else:
                self.assertIs(error,actual_exception)
                self.assertIsNone(returned)
                self.assertIsNone(tap.returned)
                self.assertEqual(kernel.WaitForSingleObject(handle[0],5000),0)
                code = wintypes.DWORD()
                need(kernel.GetExitCodeProcess(handle[0],ctypes.byref(code)),'observe actual handle exit')
                self.assertNotEqual(code.value,259)
                direct_exit = int(code.value)
        finally:
            if handle[0]:
                need(kernel.CloseHandle(handle[0]),'close owned retained observer handle')
        raw = directory/'native.raw'
        need(raw.stat().st_size<=RAW,'native raw bound')
        row = dict(case='windows_'+mode,passed=True,evidence_kind='actual_Windows_Job',actual_on_launch_pid=launched[0],
            native_result_returned=returned is not None,native=returned,direct_exit_via_retained_handle=direct_exit,
            exception_type=None if error is None else type(error).__name__,
            exception_Job_boundary_independently_observed=False,individual_descendant_OS_reaping_observed=False,
            raw=dict(path=str(raw),sha256=digest(raw),bytes=raw.stat().st_size))
        write_new(directory/'OBSERVATIONS.json',row)
        self.native_rows.append(row)

    def posix_case(self,mode):
        directory = self.output/('posix-'+mode)
        directory.mkdir()
        observer = self.observer_module.NativeObservation()
        events = []
        state = dict(process=None,leader=None,signals=0,wait_returned=False,wait_exit=None,
            term_sends=0,reap_started=False,test_intervention=False,intervention_exit=None,
            helper_handler_restored=False)
        tap = OwnerTap(self.owner)
        sentinel_calls = []
        def sentinel(number,frame):
            sentinel_calls.append(number)
        previous = signal.getsignal(signal.SIGTERM)
        def event(operation,**values):
            need(len(events)<256,'bounded lifecycle event trace')
            events.append(dict(operation=operation,monotonic_ns=time.monotonic_ns(),**values))
        class Process:
            def __init__(self,actual):
                self.actual=actual
            def __getattr__(self,name):
                return getattr(self.actual,name)
            def wait(proxy,timeout=None):
                state['reap_started']=True
                context=self.observer_module.observation_context(started,self.worker_deadline_ns,
                    first_wait=True,wait_timeout=timeout)
                snap=self.observer_module.observed_snapshot(observer,state['leader'],**context)
                event('before_actual_terminal_wait',signals=state['signals'],snapshot=snap,timeout=timeout)
                self.assertEqual(state['signals'],1)
                self.assertIs(snap.get('complete'),True)
                self.assertEqual(snap.get('session_live'),0)
                self.assertIs(snap.get('positive_acceptance_eligible'),True)
                if mode=='repeat_term':
                    self.assertIs(signal.getsignal(signal.SIGTERM),signal.SIG_IGN)
                    state['term_sends']+=1
                    event('actual_second_TERM_to_self',pid=os.getpid())
                    os.kill(os.getpid(),signal.SIGTERM)
                    event('actual_second_TERM_returned_ignored')
                value=proxy.actual.wait(timeout=timeout)
                state['wait_returned']=True
                state['wait_exit']=value
                observer.release(proxy.pid)
                event('actual_terminal_wait_returned',native_exit=value)
                return value
            def poll(proxy):
                raise AssertionError('unexpected terminal poll in lifecycle owner')
            def kill(proxy):
                raise AssertionError('unexpected hidden-poll kill in lifecycle owner')
        def popen(argv,**kwargs):
            self.assertIsNone(state['process'])
            self.assertIs(kwargs.get('start_new_session'),True)
            actual=REAL_POPEN(argv,**kwargs)
            state['process']=Process(actual)
            event('actual_spawn',pid=actual.pid)
            return state['process']
        def killpg(pid,number):
            self.assertEqual((pid,number),(state['process'].pid,signal.SIGKILL))
            self.assertFalse(state['reap_started'])
            state['signals']+=1
            self.assertEqual(state['signals'],1)
            event('actual_original_group_SIGKILL',pid=pid)
            return REAL_KILLPG(pid,number)
        def on_launch(pid):
            deadline=time.monotonic()+3
            while not (directory/'READY.json').is_file():
                need(time.monotonic()<deadline,'finite native fixture readiness')
                time.sleep(.005)
            ready=json.loads((directory/'READY.json').read_bytes())
            self.assertEqual(ready,dict(pid=pid,pgid=pid,sid=pid))
            leader=observer.identity(pid)
            self.assertEqual((leader['pid'],leader['pgid'],leader['sid']),(pid,pid,pid))
            observer.hold(leader)
            state['leader']=leader
            event('actual_launch_anchor',leader=leader)
            if mode=='normal':
                (directory/'release').write_bytes(b'1\n')
            else:
                self.assertIsNot(signal.getsignal(signal.SIGTERM),sentinel)
                state['term_sends']+=1
                event('actual_first_TERM_to_self',pid=os.getpid())
                os.kill(os.getpid(),signal.SIGTERM)
                raise AssertionError('first TERM failed to unwind the helper')
        returned=error=None
        case_failure=None
        child_echild=False
        started=time.monotonic_ns()
        signal.signal(signal.SIGTERM,sentinel)
        try:
            with patch.object(subprocess,'Popen',popen),patch.object(os,'killpg',killpg):
                try:
                    returned=self.target._run_owned_stage(tap,child_argv('hold',directory),cwd=str(directory),
                        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=str(directory/'native.raw'),
                        timeout=8,on_launch=on_launch)
                except BaseException as caught:
                    error=caught
            state['helper_handler_restored']=signal.getsignal(signal.SIGTERM) is sentinel
            self.assertTrue(state['helper_handler_restored'])
            self.assertEqual(sentinel_calls,[])
            self.assertEqual(len(tap.calls),1)
            self.assertTrue(state['wait_returned'])
            self.assertEqual(state['signals'],1)
            self.assertIsNotNone(state['leader'])
            expected_exit=0 if mode=='normal' else -signal.SIGKILL
            self.assertEqual(state['wait_exit'],expected_exit)
            if mode=='normal':
                self.assertIsNone(error)
                self.assertIs(returned,tap.returned)
                self.assertEqual(returned['pid'],state['process'].pid)
                self.assertEqual(returned['native_exit'],0)
                self.assertIs(returned['timed_out'],False)
                self.assertIs(returned['leader_reaped'],True)
                self.assertIs(returned['ownership_boundary_closed'],True)
                self.assertIs(returned['all_descendants_os_reaped'],False)
            else:
                self.assertIsInstance(error,InterruptedError)
                self.assertIsNone(returned)
                self.assertIsNone(tap.returned)
                self.assertEqual(state['term_sends'],2 if mode=='repeat_term' else 1)
            try:
                observed=os.waitpid(state['process'].pid,os.WNOHANG)
            except ChildProcessError as no_child:
                self.assertEqual(no_child.errno,errno.ECHILD)
                child_echild=True
            else:
                state['test_intervention']=observed[0]!=0
                self.fail('owner did not genuinely reap the direct child before the test probe')
        except BaseException as caught:
            case_failure=caught
        finally:
            if state['process'] is not None and not state['wait_returned']:
                # An intervention is negative evidence. Wait only on this
                # actual child object; the fixture expires after15s itself.
                # Never signal an observer-supplied or possibly reused PID.
                state['test_intervention']=True
                event('test_only_direct_wait_after_failed_owner')
                try:
                    state['intervention_exit']=state['process'].actual.wait(timeout=17)
                    observer.release(state['process'].pid)
                    event('test_only_direct_wait_returned',native_exit=state['intervention_exit'])
                except BaseException as failed_wait:
                    event('test_only_direct_wait_failed',error_type=type(failed_wait).__name__)
            signal.signal(signal.SIGTERM,previous)
        raw=directory/'native.raw'
        raw_ref=None
        if raw.is_file():
            need(raw.stat().st_size<=RAW,'native lifecycle raw bound')
            raw_ref=dict(path=str(raw),sha256=digest(raw),bytes=raw.stat().st_size)
        zero_observed=any(e['operation']=='before_actual_terminal_wait' and e['snapshot'].get('complete') is True and
            e['snapshot'].get('session_live')==0 and e['snapshot'].get('positive_acceptance_eligible') is True for e in events)
        row=dict(case='posix_'+mode,passed=case_failure is None,evidence_kind='actual_POSIX_lifecycle',native_result_returned=returned is not None,
            returned_native=returned,exception_type=None if error is None else type(error).__name__,events=events,
            failure_type=None if case_failure is None else type(case_failure).__name__,
            actual_child_pid=None if state['process'] is None else state['process'].pid,actual_child_identity=state['leader'],
            actual_owner_terminal_wait_returned=state['wait_returned'],actual_direct_exit_observed=state['wait_exit'],
            waitpid_ECHILD_observed=child_echild,test_intervention_reap=state['test_intervention'],
            test_intervention_exit=state['intervention_exit'],previous_handler_restored=state['helper_handler_restored'],
            actual_TERM_sends_to_self=state['term_sends'],independent_zero_live_SID_before_reap=zero_observed,
            individual_descendant_OS_reaping_observed=False,elapsed_ns=time.monotonic_ns()-started,raw=raw_ref)
        write_new(directory/'OBSERVATIONS.json',row)
        self.native_rows.append(row)
        if case_failure is not None:
            raise case_failure

    def test_native_applicable_platform_routes(self):
        if os.name=='nt':
            for mode in ('normal','exception'):
                with self.subTest(platform='Windows',mode=mode):
                    self.windows_case(mode)
        elif os.name=='posix' and (sys.platform=='darwin' or sys.platform.startswith('linux')):
            for mode in ('normal','first_term','repeat_term'):
                with self.subTest(platform=sys.platform,mode=mode):
                    self.posix_case(mode)
        else:
            self.fail('unsupported native engineering platform')


def child(mode,root,expected_sha):
    need(digest(__file__)==expected_sha,'fixed harmless fixture source changed')
    if mode=='ok':
        sys.stdout.buffer.write(b'LIFECYCLE_CHILD_OK\n')
        sys.stdout.buffer.flush()
        return 0
    need(mode=='hold' and os.name=='posix','fixed POSIX hold fixture required')
    write_new(root/'READY.json',dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0)))
    deadline=time.monotonic()+15
    while not (root/'release').is_file():
        if time.monotonic()>=deadline:
            return 17  # Finite self-expiry; no observer signals orphan PID.
        time.sleep(.005)
    return 0


def main():
    started=time.monotonic_ns()
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_directory
    need(not sys.flags.optimize and output.is_absolute() and not output.exists() and output.parent.is_dir(),
         'unoptimized fixed-file fresh evidence required')
    output=output.parent.resolve(strict=True)/output.name
    need(not output.is_relative_to(ROOT) and output!=ROOT,'evidence must be outside the checkout')
    protected={str(path):expected for path,expected in PINS.items()}
    protected[str(Path(__file__).resolve())]=digest(__file__)
    need(all(digest(path)==expected for path,expected in protected.items()),'lifecycle source pin changed')
    memory=load(MEMORY,'independent_lifecycle_memory',PINS[MEMORY])
    available,method=memory.available_memory()
    free=shutil.disk_usage(output.parent).free
    need(available>=512*1024**2 and free>=8*1024**3,'engineering512MiB/8GiB admission failed')
    LifecycleTests.target=load(TARGET,'independent_lifecycle_subject',PINS[TARGET])
    LifecycleTests.owner=load(OWNER,'independent_lifecycle_real_owner',PINS[OWNER])
    LifecycleTests.observer_module=load(OBSERVER,'independent_lifecycle_observer',PINS[OBSERVER])
    output.mkdir()
    LifecycleTests.output=output
    LifecycleTests.worker_deadline_ns=started+WHOLE_SECONDS*10**9
    write_new(output/'PREEXECUTION.json',dict(source_pins=protected,available_physical_bytes=available,
        memory_method=method,free_disk_bytes=free,started_ns=started,deadline_ns=started+WHOLE_SECONDS*10**9,
        Julia_launched=False,actual_data_exposed=False,Q_issued=False))
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleTests)
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    unchanged=all(digest(path)==expected for path,expected in protected.items())
    posix=os.name=='posix' and (sys.platform=='darwin' or sys.platform.startswith('linux'))
    windows=os.name=='nt'
    native_posix=sum(row['evidence_kind']=='actual_POSIX_lifecycle' for row in LifecycleTests.native_rows)
    native_windows=sum(row['evidence_kind']=='actual_Windows_Job' for row in LifecycleTests.native_rows)
    passed=result.wasSuccessful() and result.testsRun==3 and len(LifecycleTests.simulated_rows)==11 and unchanged and \
        native_posix==(3 if posix else 0) and native_windows==(2 if windows else 0)
    finished=time.monotonic_ns()
    passed=passed and finished<started+WHOLE_SECONDS*10**9
    write_new(output/'RESULT.json',dict(schema='matched342_independent_owner_lifecycle_v1',complete=passed,
        unittest_methods=result.testsRun,simulated_controls=len(LifecycleTests.simulated_rows),simulated_rows=LifecycleTests.simulated_rows,
        actual_POSIX_cases=native_posix,actual_Windows_cases=native_windows,native_rows=LifecycleTests.native_rows,
        native_POSIX_applicable=posix,native_POSIX_qualified=False,source_pins=protected,source_pins_unchanged=unchanged,
        started_ns=started,finished_ns=finished,deadline_ns=started+WHOLE_SECONDS*10**9,
        Julia_launched=False,actual_data_exposed=False,Q_issued=False,source_qualified=False,scientific_qualified=False,
        individual_descendant_OS_reaping_observed=False))
    if time.monotonic_ns()>=started+WHOLE_SECONDS*10**9:
        (output/'RESULT.json').rename(output/'RESULT_LATE_UNQUALIFIED.json')
        write_new(output/'LATE_OUTPUT.json',dict(complete=False,deadline_ns=started+WHOLE_SECONDS*10**9,observed_ns=time.monotonic_ns()))
        return 1
    return 0 if passed else 1


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--child':
        raise SystemExit(child(sys.argv[2],Path(sys.argv[3]),sys.argv[4]))
    raise SystemExit(main())
