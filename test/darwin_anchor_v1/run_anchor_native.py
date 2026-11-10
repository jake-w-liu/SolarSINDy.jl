"""Genuine Darwin-only harmless-child launch-anchor controls; no science/Q."""
from pathlib import Path
import argparse
from contextlib import ExitStack
import ctypes as C
import errno
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
STATE_SHA='99b72148874d2a103bdc003f748be7b9fd3a0f310a3de02d88b7493fd8e455fd'
_path=HERE/'test_anchor_state.py'
_spec=importlib.util.spec_from_file_location('anchor_independent_common',_path)
_common=importlib.util.module_from_spec(_spec)
import hashlib
with _path.open('rb') as _stream:
    if hashlib.file_digest(_stream,'sha256').hexdigest()!=STATE_SHA:raise RuntimeError('pinned literal/common source changed')
_spec.loader.exec_module(_common)
if _common.digest(_path)!=STATE_SHA:raise RuntimeError('literal/common source changed during import')
ROOT=_common.ROOT
need,digest,load,put=_common.need,_common.digest,_common.load,_common.put
REAL_POPEN=subprocess.Popen
REAL_WAITID=os.waitid if hasattr(os,'waitid') else None
REAL_KILLPG=os.killpg if hasattr(os,'killpg') else None
REAL_KILL=os.kill
_CANCELLED=None


def check_cancellation():
    if _CANCELLED is not None:raise _CANCELLED


def read_ready(path):
    need(path.is_file() and 0<path.stat().st_size<=2048 and not path.is_symlink(),'bounded fixture handshake required')
    value=json.loads(path.read_bytes())
    need(set(value)=={'pid','ppid','pgid','sid'},'closed fixture identity required')
    return value


def exercise(subject,observer,abi,mode,directory,*,historical,whole_deadline_ns,failure_recovery=False):
    directory.mkdir()
    started=time.monotonic_ns()
    lock=threading.RLock()
    events=[]
    state=dict(actual=None,proxy=None,leader=None,first_raw=None,proof=None,signals=0,direct_signals=0,
        wait_calls=0,wait_returned=False,wait_exit=None,reap_started=False,on_launch_calls=0,
        intervention=False,unclosed=False,controller_CONT=False,terminal_before_return=None,
        before_return_records=None,live_records=None,returned_to_producer=False,
        identity_fault=False,fault_injected=False,stopped_records=None,intervention_ECHILD=False)
    raw=directory/'native.raw'
    cancelled=None

    def event(operation,**values):
        need(len(events)<4096,'bounded actual native event trace')
        events.append(dict(operation=operation,monotonic_ns=time.monotonic_ns(),**values))

    def records(pid):
        result={}
        for name,kind,structure in (('short',13,abi.ShortBSDInfo),('full',3,abi.BSDInfo)):
            value=structure();C.set_errno(0)
            count=observer.darwin.library.proc_pidinfo(pid,kind,1,C.byref(value),C.sizeof(value))
            error=C.get_errno()
            need(count==C.sizeof(value) and value.pid==pid,'independent bounded native record unavailable')
            keys=('pid','ppid','pgid','status','uid','ruid','flags')+(('start_sec','start_usec') if name=='full' else ())
            result[name]={key:int(getattr(value,key)) for key in keys}
            result[name+'_bytes']=count;result[name+'_errno']=error
        a,b=result['short'],result['full']
        need(all(a[k]==b[k] for k in ('pid','ppid','pgid','uid','ruid')) and (a['status']==5)==(b['status']==5),
             'independent short/full native identity changed')
        try:result['getsid']=dict(value=os.getsid(pid),errno=None)
        except OSError as error:result['getsid']=dict(value=None,errno=error.errno)
        return result

    def matching(row):
        return state['leader'] is not None and all(row[k]==state['leader'][k] for k in ('pid','pgid','sid','birth','uid','ruid'))

    def independent_identity(pid):
        if state['identity_fault'] and state['actual'] is not None and pid==state['actual'].pid:
            raise OSError(errno.EIO,'intentional independent identity-read failure before Popen return')
        return observer.identity(pid)

    def observe(deadline_ns,phase):
        result=_observer.observed_snapshot(observer,state['leader'],deadline_ns=deadline_ns,phase=phase)
        # The protected independent reader can classify an exception as unknown.
        # The caller still preserves its own explicit cancellation unchanged.
        check_cancellation()
        return result

    class Process:
        def __init__(self,actual):self.actual=actual
        def __getattr__(self,name):return getattr(self.actual,name)
        def wait(self,timeout=None):
            with lock:
                need(not state['reap_started'],'only one actual first terminal wait allowed')
                state['reap_started']=True;state['wait_calls']+=1
                context=_observer.observation_context(started,whole_deadline_ns,first_wait=True,
                    wait_timeout=timeout,expected_negative=historical)
                snapshot=observe(context['deadline_ns'],context['phase'])
                event('actual_terminal_wait_entry',received_timeout=timeout,snapshot=snapshot,
                    original_group_signals=state['signals'])
            value=self.actual.wait(timeout=timeout)
            state['wait_returned']=True;state['wait_exit']=value
            observer.release(self.pid)
            event('actual_terminal_wait_return',native_exit=value)
            return value
        def poll(self):
            event('forbidden_hidden_terminal_poll');raise RuntimeError('test refuses hidden terminal poll')
        def kill(self):
            event('forbidden_hidden_poll_kill');raise RuntimeError('test refuses hidden poll in Popen.kill')

    def release_fixture():
        with (directory/'release').open('xb') as stream:
            need(stream.write(b'1\n')==2,'short fixture release');stream.flush();os.fsync(stream.fileno())

    def popen(argv,**kwargs):
        check_cancellation()
        need(state['actual'] is None and kwargs.get('start_new_session') is True,'one actual new private session required')
        need(argv[5:7]==[str(HERE/'fixture.py'),str(directory)],'fixed file fixture argv required')
        actual=REAL_POPEN(argv,**kwargs)
        state['actual']=actual;proxy=Process(actual);state['proxy']=proxy
        event('actual_spawn',pid=actual.pid,argv=argv,start_new_session=True)
        ready_deadline=min(whole_deadline_ns,started+8000000000,time.monotonic_ns()+3000000000)
        while not (directory/'READY.json').is_file():
            check_cancellation()
            need(time.monotonic_ns()<ready_deadline,'finite actual READY handshake')
            time.sleep(.005)
        ready=read_ready(directory/'READY.json')
        need(ready==dict(pid=actual.pid,ppid=os.getpid(),pgid=actual.pid,sid=actual.pid),'native child launch identity differs')
        leader=observer.identity(actual.pid)
        need(not leader['zombie'] and leader['pid']==leader['pgid']==leader['sid']==actual.pid and
             leader['uid']==os.geteuid() and leader['ruid']==os.getuid(),'independent actual live anchor differs')
        observer.hold(leader);state['leader']=leader
        state['live_records']=records(actual.pid)
        need(state['live_records']['full']['ppid']==os.getpid(),'independent actual parent differs')
        event('independent_actual_live_anchor',leader=leader,records=state['live_records'])
        if mode!='live_then_zombie_exit7':
            release_fixture()
            terminal_deadline=min(whole_deadline_ns,started+8000000000)
            while True:
                check_cancellation()
                need(time.monotonic_ns()<terminal_deadline,'finite delayed actual Popen return')
                current=independent_identity(actual.pid)
                need(matching(current) and actual.returncode is None,'unreaped actual child identity changed')
                if failure_recovery and current['stopped']:
                    state['stopped_records']=records(actual.pid)
                    state['identity_fault']=True;state['fault_injected']=True
                    event('intentional_identity_fault_before_Popen_return',pid=actual.pid,
                        independent_stopped_records=state['stopped_records'])
                    independent_identity(actual.pid)
                if mode=='first_seen_zombie_after_transient_stop' and current['stopped'] and not state['controller_CONT']:
                    with lock:
                        need(not state['reap_started'] and state['signals']==0 and matching(observer.identity(actual.pid)),
                             'CONT requires same actual held pre-cleanup child')
                        REAL_KILL(actual.pid,signal.SIGCONT);state['controller_CONT']=True
                        event('actual_controller_CONT',pid=actual.pid,held_identity=current)
                if current['zombie']:
                    state['before_return_records']=records(actual.pid)
                    observed=REAL_WAITID(os.P_PID,actual.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                    state['terminal_before_return']=None if observed is None else {key:int(getattr(observed,key)) for key in
                        ('si_pid','si_code','si_status','si_signo','si_uid')}
                    event('actual_zombie_before_Popen_return',records=state['before_return_records'],
                        WNOWAIT=state['terminal_before_return'])
                    break
                time.sleep(.005)
        state['returned_to_producer']=True
        event('actual_Popen_object_returned_to_producer',pid=actual.pid,returncode=actual.returncode)
        return proxy

    def waitid(kind,pid,flags):
        if state['actual'] is not None and pid==state['actual'].pid and flags & os.WEXITED and not flags & os.WNOWAIT:
            event('forbidden_terminal_consuming_waitid',flags=int(flags),syscall_executed=False)
            raise RuntimeError('test refuses terminal-consuming waitid before syscall')
        try:value=REAL_WAITID(kind,pid,flags)
        except OSError as error:
            if state['actual'] is not None and pid==state['actual'].pid:
                event('actual_owned_waitid_error',pid=pid,flags=int(flags),errno=error.errno,error_type=type(error).__name__)
            check_cancellation()
            raise
        if state['actual'] is not None and pid==state['actual'].pid:
            event('actual_owned_waitid',pid=pid,flags=int(flags),result=None if value is None else
                {key:int(getattr(value,key)) for key in ('si_pid','si_code','si_status','si_signo','si_uid')})
        check_cancellation()
        return value

    def killpg(pid,number):
        with lock:
            need(state['actual'] is not None and (pid,number)==(state['actual'].pid,signal.SIGKILL) and
                 not state['reap_started'] and state['signals']==0,'borrowed/repeated/post-reap original-group signal refused')
            state['signals']+=1
            context=_observer.observation_context(started,whole_deadline_ns,before_signal=True)
            event('actual_original_group_signal',pid=pid,signal=int(number),snapshot=observe(context['deadline_ns'],context['phase']))
        try:return REAL_KILLPG(pid,number)
        except OSError as error:
            context=_observer.observation_context(started,whole_deadline_ns,before_signal=True)
            event('actual_original_group_signal_error',pid=pid,errno=error.errno,
                snapshot=observe(context['deadline_ns'],context['phase']))
            raise

    def kill(pid,number):
        need(state['actual'] is not None and (pid,number)==(state['actual'].pid,signal.SIGKILL) and
             not state['reap_started'],'borrowed direct numeric signal refused')
        state['direct_signals']+=1;event('actual_owned_direct_signal',pid=pid,signal=int(number))
        return REAL_KILL(pid,number)

    def on_launch(pid):
        check_cancellation()
        state['on_launch_calls']+=1
        need(state['actual'] is not None and pid==state['actual'].pid and state['returned_to_producer'],
             'actual owner callback/Popen identity mismatch')
        event('actual_owner_on_launch',pid=pid)
        if mode=='live_then_zombie_exit7':release_fixture()

    returned=error=None
    with ExitStack() as stack:
        def instrument(module):
            original_processes=module._processes
            def processes():
                adapter=original_processes()
                method='_raw_info' if hasattr(adapter,'_raw_info') else '_info'
                original=getattr(adapter,method)
                def raw_read(pid):
                    answer=original(pid)
                    if state['actual'] is not None and pid==state['actual'].pid and state['first_raw'] is None:
                        row,flags=answer if method=='_raw_info' else (answer,None)
                        state['first_raw']=dict(row=row.copy(),flags=flags,method=method)
                        event('producer_first_actual_child_read',data=state['first_raw'])
                    return answer
                setattr(adapter,method,raw_read)
                return adapter
            original_run=module._run_posix
            def run(*args,**kwargs):
                value=original_run(*args,**kwargs);state['proof']=value[1];return value
            stack.enter_context(patch.object(module,'_processes',processes))
            stack.enter_context(patch.object(module,'_run_posix',run))
            return module
        if hasattr(subject,'_posix_module'):
            original_loader=subject._posix_module
            stack.enter_context(patch.object(subject,'_posix_module',lambda:instrument(original_loader())))
        else:instrument(subject)
        stack.enter_context(patch.object(subprocess,'Popen',popen))
        stack.enter_context(patch.object(os,'waitid',waitid))
        stack.enter_context(patch.object(os,'killpg',killpg))
        stack.enter_context(patch.object(os,'kill',kill))
        fixture_mode='signal9' if mode=='first_seen_zombie_signal9' else 'stop_then_exit7' if mode=='first_seen_zombie_after_transient_stop' or failure_recovery else 'exit7'
        argv=[sys.executable,'-I','-B','-X','utf8',str(HERE/'fixture.py'),str(directory),fixture_mode,'--source-sha256',digest(HERE/'fixture.py')]
        try:
            returned=subject.run_owned(argv,cwd=str(directory),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                logpath=raw,timeout=8,on_launch=on_launch)
        except BaseException as caught:
            error=dict(type=type(caught).__name__,errno=getattr(caught,'errno',None),message=str(caught)[:1024],chain=_common.error_chain(caught))
            if isinstance(caught,InterruptedError) and caught.errno is None:cancelled=caught
            if _CANCELLED is not None:cancelled=_CANCELLED
        finally:
            actual=state['actual']
            if actual is not None and not state['wait_returned']:
                state['intervention']=True
                event('test_intervention_after_missing_owner_wait')
                try:
                    observed_identity=independent_identity(actual.pid)
                    event('test_intervention_identity_lookup',matching=matching(observed_identity),identity=observed_identity)
                except BaseException as unavailable:
                    event('test_intervention_identity_unavailable',error_type=type(unavailable).__name__)
                associated=False;terminal=False
                try:
                    need(actual.returncode is None,'retained direct Popen is no longer exclusively unreaped')
                    observed=REAL_WAITID(os.P_PID,actual.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                    if observed is not None:
                        need(observed.si_pid==actual.pid and observed.si_code in
                            (os.CLD_EXITED,os.CLD_KILLED,os.CLD_DUMPED,os.CLD_STOPPED,os.CLD_CONTINUED),
                            'intervention has a contradictory owned-child association')
                        terminal=observed.si_code in (os.CLD_EXITED,os.CLD_KILLED,os.CLD_DUMPED)
                    associated=True
                    event('test_intervention_nonreaping_child_association',pid=actual.pid,
                        result=None if observed is None else {key:int(getattr(observed,key)) for key in
                            ('si_pid','si_code','si_status','si_signo','si_uid')},terminal=terminal)
                except BaseException as lost:
                    state['unclosed']=True
                    event('test_intervention_association_unavailable',error_type=type(lost).__name__,errno=getattr(lost,'errno',None))
                if associated and not terminal:
                    try:
                        # Only this retained, exclusive actual direct child is
                        # signaled. Native identity failure never borrows a PID.
                        REAL_KILL(actual.pid,signal.SIGKILL)
                        event('test_intervention_owned_direct_kill',pid=actual.pid)
                    except BaseException as signal_error:
                        event('test_intervention_direct_signal_error',error_type=type(signal_error).__name__,errno=getattr(signal_error,'errno',None))
                try:
                    # Always try the bounded actual wait, even if independent
                    # metadata or the failure-cleanup signal was unavailable.
                    value=actual.wait(timeout=max(0.,min(4.,(whole_deadline_ns-time.monotonic_ns())/1e9)))
                    observer.release(actual.pid)
                    event('test_intervention_wait_returned',native_exit=value,owned_child_association_verified=associated)
                    try:
                        unexpected=os.waitpid(actual.pid,os.WNOHANG)
                    except ChildProcessError as no_child:
                        state['intervention_ECHILD']=no_child.errno==errno.ECHILD
                        event('test_intervention_postwait_ECHILD',errno=no_child.errno)
                    else:
                        state['unclosed']=True
                        event('test_intervention_unexpected_postwait_result',result=list(unexpected))
                except BaseException as failed:
                    state['unclosed']=True
                    event('test_intervention_unclosed',error_type=type(failed).__name__)
    finished=time.monotonic_ns()
    row=dict(mode=mode,historical=historical,native=returned,error=error,private_proof=state['proof'],events=events,
        actual_child_pid=None if state['actual'] is None else state['actual'].pid,independent_live_anchor=state['leader'],
        independent_live_records=state['live_records'],independent_before_return_records=state['before_return_records'],
        producer_first_read=state['first_raw'],actual_before_return_WNOWAIT=state['terminal_before_return'],
        actual_owner_wait_returned=state['wait_returned'],actual_owner_wait_exit=state['wait_exit'],
        actual_original_group_signals=state['signals'],actual_owned_direct_signals=state['direct_signals'],
        owner_on_launch_calls=state['on_launch_calls'],test_controller_CONT_sent=state['controller_CONT'],
        test_intervention=state['intervention'],ownership_unclosed=state['unclosed'],
        failure_recovery_test_input=failure_recovery,intentional_identity_fault_injected=state['fault_injected'],
        independent_stopped_records=state['stopped_records'],Popen_returned_to_producer=state['returned_to_producer'],
        test_intervention_ECHILD_observed=state['intervention_ECHILD'],
        started_ns=started,finished_ns=finished,elapsed_ns=finished-started,
        raw=None if not raw.is_file() else dict(path=str(raw),sha256=digest(raw),bytes=raw.stat().st_size),
        individual_descendant_OS_reaping_observed=False,evidence_kind='actual_Darwin_harmless_child')
    if row['raw'] is not None:need(row['raw']['bytes']<=4*1024**2,'bounded native raw required')
    return row,cancelled


def check(row,candidate):
    before=row['independent_before_return_records']
    first=row['producer_first_read']
    events=row['events']
    reaps=[e for e in events if e['operation']=='actual_terminal_wait_entry']
    forbidden=any(e['operation'].startswith('forbidden_') for e in events)
    checks=dict(actual_owned_wait=row['actual_owner_wait_returned'] and len(reaps)==1,
        no_intervention=not row['test_intervention'] and not row['ownership_unclosed'],no_hidden_reap=not forbidden,
        direct_signals_zero=row['actual_owned_direct_signals']==0,live_anchor_retained=row['independent_live_anchor'] is not None,
        raw_retained=row['raw'] is not None)
    if row['mode']!='live_then_zombie_exit7':
        checks['actual_first_zombie_trigger']=before is not None and before['full']['status']==5 and before['short']['status']==5 and \
            bool(before['full']['flags']&0x4000) and bool(before['short']['flags']&0x4000) and \
            first is not None and first['row']['zombie'] is True and first['row']['sid'] is None
    if row['historical']:
        checks.update(old_error_retained=row['error'] is not None and row['native'] is None,
            old_exact_exit7=row['actual_owner_wait_exit']==7,old_no_fake_callback=row['owner_on_launch_calls']==0,
            old_no_numeric_group_signal=row['actual_original_group_signals']==0,
            negative_diagnostic_retained=len(reaps)==1 and reaps[0]['snapshot'].get('complete') is True and
                reaps[0]['snapshot'].get('positive_acceptance_eligible') is False)
        return checks
    native=row['native'];expected=-9 if row['mode']=='first_seen_zombie_signal9' else 7
    proof=row['private_proof'] or {}
    keys={'pid','native_exit','timed_out','wall_seconds','ownership','descendant_cleanup'}
    if candidate:keys|={'leader_reaped','ownership_boundary_closed','all_descendants_os_reaped','observation_method','descendant_evidence'}
    checks.update(no_error=row['error'] is None,closed_result=isinstance(native,dict) and set(native)==keys,
        native_PID=isinstance(native,dict) and native.get('pid')==row['actual_child_pid'],
        exact_exit=isinstance(native,dict) and native.get('native_exit')==expected and row['actual_owner_wait_exit']==expected,
        untimed=isinstance(native,dict) and native.get('timed_out') is False,
        inside_stage=row['elapsed_ns']<8000000000,one_original_group=row['actual_original_group_signals']==1,
        one_owner_callback=row['owner_on_launch_calls']==1,
        private_zero_verified=proof.get('session_zero_live_verified') is True,
        independent_zero_before_reap=len(reaps)==1 and reaps[0]['snapshot'].get('complete') is True and
            reaps[0]['snapshot'].get('session_live')==0 and reaps[0]['snapshot'].get('positive_acceptance_eligible') is True,
        correct_origin=proof.get('leader_identity_origin')==('native_live' if row['mode']=='live_then_zombie_exit7' else 'owned_exec_terminal'))
    if row['mode']!='live_then_zombie_exit7':
        terminal=proof.get('launch_terminal_observation')
        checks['actual_terminal_launch_witness']=isinstance(terminal,dict) and terminal.get('si_pid')==row['actual_child_pid'] and terminal.get('si_code') in (os.CLD_EXITED,os.CLD_KILLED,os.CLD_DUMPED)
    if row['mode']=='first_seen_zombie_after_transient_stop':checks['actual_safe_CONT']=row['test_controller_CONT_sent']
    if candidate:checks['candidate_truthful_projection']=native is not None and native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and native.get('all_descendants_os_reaped') is False
    return checks


def check_failure_cleanup(row):
    events=row['events']
    associations=[e for e in events if e['operation']=='test_intervention_nonreaping_child_association']
    kills=[e for e in events if e['operation']=='test_intervention_owned_direct_kill']
    waits=[e for e in events if e['operation']=='test_intervention_wait_returned']
    lookup_failures=[e for e in events if e['operation']=='test_intervention_identity_unavailable']
    stopped=row['independent_stopped_records']
    return dict(explicit_expected_negative_input=row['failure_recovery_test_input'] and row['intentional_identity_fault_injected'],
        genuine_stopped_fixture=stopped is not None and stopped['full']['status']==4 and stopped['short']['status']==4,
        independent_live_anchor_retained=row['independent_live_anchor'] is not None,
        intentional_read_error=row['error'] is not None and row['error']['errno']==errno.EIO,
        no_owner_result=row['native'] is None and row['private_proof'] is None,
        factory_not_returned=not row['Popen_returned_to_producer'] and row['owner_on_launch_calls']==0,
        no_fake_owner_wait=not row['actual_owner_wait_returned'],
        actual_identity_lookup_failure=len(lookup_failures)==1 and lookup_failures[0]['error_type']=='OSError',
        nonreaping_matching_child=len(associations)==1 and associations[0]['pid']==row['actual_child_pid'],
        only_owned_direct_intervention=len(kills)==1 and kills[0]['pid']==row['actual_child_pid'] and
            row['actual_original_group_signals']==0 and row['actual_owned_direct_signals']==0,
        genuine_bounded_test_wait=len(waits)==1 and waits[0]['native_exit']==-9 and waits[0]['owned_child_association_verified'] is True,
        actual_postwait_ECHILD=row['test_intervention_ECHILD_observed'] is True,
        intervention_explicitly_negative=row['test_intervention'] and not row['ownership_unclosed'],
        no_CONT_or_positive_closure=not row['test_controller_CONT_sent'],raw_retained=row['raw'] is not None)


def main():
    entered=time.monotonic_ns();deadline=entered+480*10**9
    parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();output=args.output_directory
    need(not sys.flags.optimize and sys.platform=='darwin' and sys.version_info>=(3,14),
         'genuine Darwin/Python3.14 native controls required; no skip qualification')
    need(output.is_absolute() and not output.exists() and output.parent.is_dir() and not output.is_relative_to(ROOT),
         'fresh external native evidence required')
    value=_common.config();pins=_common.source_pins(value);before=_common.inventory()
    available,method,resource_processes=_common.available_memory();free=shutil.disk_usage(output.parent).free
    need(available>=536870912 and free>=8589934592,'actual engineering512MiB/8GiB admission failed')
    global _observer
    _observer=load(ROOT/'test/posix_owner_v3/observations.py','anchor_independent_observer',value['independent_observer_sha256'])
    abi=load(ROOT/'test/initial_reference/diagnose_darwin_groups.py','anchor_independent_ABI',value['independent_darwin_abi_sha256'])
    observer=_observer.NativeObservation()
    output.mkdir()
    put(output/'PREEXECUTION.json',dict(source_pins=pins,test_sources=before,started_ns=entered,deadline_ns=deadline,
        admission=dict(available_physical_bytes=available,memory_method=method,free_disk_bytes=free,resource_probe_subprocesses=resource_processes)))
    current=[];historical=[];failure_cleanup=[];abort=None
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(number,frame):
        global _CANCELLED
        if _CANCELLED is None:_CANCELLED=InterruptedError('Darwin anchor native controller received SIGTERM')
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        # Defer the same exception to controlled owned-call/loop boundaries.
        # An asynchronous throw must not interrupt failure-cleanup finally.
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for index,role in enumerate(_common.ROLES):
            for mode in value['native_modes']:
                check_cancellation()
                need(time.monotonic_ns()<deadline,'original whole480 native deadline exhausted')
                subject=load(ROOT/role,'anchor_native_current_'+str(index)+mode,
                    value['current_candidate_sha256'] if index==2 else value['current_legacy_sha256'])
                name='current-'+str(index)+'-'+mode
                row,cancelled=exercise(subject,observer,abi,mode,output/name,historical=False,whole_deadline_ns=deadline)
                row.update(role=role,source_sha256=digest(ROOT/role));row['checks']=check(row,index==2);row['passed']=all(row['checks'].values())
                put(output/(name+'.json'),row);current.append(row)
                if cancelled is not None:raise cancelled
                need(not row['ownership_unclosed'],'unclosed intervention cannot launch another native case')
        for index,role in enumerate(_common.ROLES):
            check_cancellation()
            need(time.monotonic_ns()<deadline,'original whole480 historical deadline exhausted')
            path=HERE/'frozen'/('candidate_native_owner.py' if index==2 else 'native_stage.py')
            subject=load(path,'anchor_native_old_'+str(index),value['historical_candidate_sha256'] if index==2 else value['historical_legacy_sha256'])
            name='old-'+str(index)+'-first_seen_zombie_exit7'
            row,cancelled=exercise(subject,observer,abi,'first_seen_zombie_exit7',output/name,historical=True,whole_deadline_ns=deadline)
            row.update(role=role,source_sha256=digest(path));row['checks']=check(row,index==2)
            row['passed']=all(row['checks'].values());row['expected_negative_reproduced']=row['passed']
            put(output/(name+'.json'),row);historical.append(row)
            if cancelled is not None:raise cancelled
            need(not row['ownership_unclosed'],'unclosed historical intervention cannot continue')
        check_cancellation()
        need(time.monotonic_ns()<deadline,'original whole480 fault-recovery deadline exhausted')
        role=_common.ROLES[1]
        subject=load(ROOT/role,'anchor_native_failure_cleanup',value['current_legacy_sha256'])
        name='failure-cleanup-identity-fault'
        row,cancelled=exercise(subject,observer,abi,'failure_cleanup_identity_fault',output/name,
            historical=False,whole_deadline_ns=deadline,failure_recovery=True)
        row.update(role=role,source_sha256=digest(ROOT/role),evidence_kind='actual_Darwin_fault_recovery_expected_negative')
        row['checks']=check_failure_cleanup(row);row['passed']=all(row['checks'].values())
        row['expected_failure_cleanup_negative_reproduced']=row['passed']
        put(output/(name+'.json'),row);failure_cleanup.append(row)
        if cancelled is not None:raise cancelled
        need(not row['ownership_unclosed'],'fault-recovery remains unclosed')
    except BaseException as error:
        abort=error
    finally:
        signal.signal(signal.SIGTERM,previous)
    if abort is None and _CANCELLED is not None:
        abort=_CANCELLED
    unchanged=_common.source_pins(value)==pins and _common.inventory()==before
    finished=time.monotonic_ns()
    passed=abort is None and len(current)==12 and len(historical)==3 and len(failure_cleanup)==1 and all(r['passed'] for r in current+historical+failure_cleanup) and unchanged and finished<deadline
    put(output/'RESULT.json',dict(schema='independent_darwin_anchor_native_result_v1',complete=passed,
        actual_current_cases=len(current),actual_historical_negative_cases=len(historical),actual_failure_cleanup_negative_cases=len(failure_cleanup),
        actual_native_cases=len(current)+len(historical)+len(failure_cleanup),
        current_case_passes=[r['passed'] for r in current],historical_negative_passes=[r['passed'] for r in historical],
        failure_cleanup_negative_passes=[r['passed'] for r in failure_cleanup],
        source_pins=pins,test_sources=before,source_unchanged=unchanged,
        runtime=dict(python_version=sys.version,python_binary_sha256=digest(sys.executable),uname=list(os.uname())),
        admission=dict(available_physical_bytes=available,memory_method=method,free_disk_bytes=free,resource_probe_subprocesses=resource_processes),
        started_ns=entered,finished_ns=finished,deadline_ns=deadline,individual_descendant_OS_reaping_observed=False,
        native_Darwin_qualified=False,Julia_launched=False,actual_data_exposed=False,source_qualified=False,scientific_qualified=False))
    if time.monotonic_ns()>=deadline:
        (output/'RESULT.json').rename(output/'RESULT_LATE_UNQUALIFIED.json');return 1
    if abort is not None:raise abort
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
