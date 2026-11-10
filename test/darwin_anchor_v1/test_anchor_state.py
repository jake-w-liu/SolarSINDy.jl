"""Independent literal Darwin launch controls; no native POSIX test process."""
from pathlib import Path
import argparse
import copy
import ctypes as C
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ORACLE_SHA='51b90716f70fc35fcd5541ab995e49695957a17378b2de44012617c4e6e25e97'
ROLES=('tools/initial_reference_audit/owned_process.py','research/dbdt_completed_source/native_stage.py',
       'research/dbdt_completed_source/candidate_native_owner.py')
PID=424242
PARENT=999999
UID=501
RAW=4*1024**2


def need(value,message):
    if not value:raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path,name,sha):
    need(digest(path)==sha,'pinned independent source changed')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(digest(path)==sha,'independent source changed during import')
    return module


def config():
    need(digest(HERE/'oracles_v1.json')==ORACLE_SHA,'frozen anchor oracle changed')
    value=json.loads((HERE/'oracles_v1.json').read_bytes())
    need(value['schema']=='independent_darwin_launch_anchor_oracles_v1' and value['state_rows']==72 and
         value['state_methods']==3 and len(value['literal_cases'])==24 and len(value['candidate_cases'])==6,
         'literal anchor inventory differs')
    return value


def source_pins(value):
    pins={str(ROOT/role):value['current_candidate_sha256'] if index==2 else value['current_legacy_sha256']
          for index,role in enumerate(ROLES)}
    pins[str(HERE/'frozen/native_stage.py')]=value['historical_legacy_sha256']
    pins[str(HERE/'frozen/candidate_native_owner.py')]=value['historical_candidate_sha256']
    pins[str(HERE/'frozen/before_waitid_identity_v2.py')]=value['before_waitid_identity_v2_sha256']
    pins[str(ROOT/'test/posix_owner_v3/observations.py')]=value['independent_observer_sha256']
    pins[str(ROOT/'test/initial_reference/diagnose_darwin_groups.py')]=value['independent_darwin_abi_sha256']
    need(all(digest(path)==sha for path,sha in pins.items()),'current or historical source pin differs')
    return pins


def inventory():
    paths=sorted(p for p in HERE.rglob('*') if p.is_file())
    need(len(paths)==7 and all(not p.is_symlink() and p.stat().st_size<=256*1024 and '__pycache__' not in p.parts for p in paths),
         'bounded ordinary seven-file anchor test inventory differs')
    return {p.relative_to(HERE).as_posix():digest(p) for p in paths}


def put(path,value):
    body=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    need(len(body)<=RAW,'anchor evidence cap')
    with Path(path).open('xb') as stream:
        need(stream.write(body)==len(body),'short anchor evidence write')
        stream.flush();os.fsync(stream.fileno())


def available_memory():
    if os.name=='nt':
        class Memory(C.Structure):
            _fields_=[('length',C.c_ulong),('load',C.c_ulong)]+[(name,C.c_ulonglong) for name in ('total','available','pt','pa','vt','va','ex')]
        value=Memory();value.length=C.sizeof(value)
        need(C.windll.kernel32.GlobalMemoryStatusEx(C.byref(value)),'native physical memory observation failed')
        return value.available,'Windows GlobalMemoryStatusEx',0
    if sys.platform.startswith('linux'):
        text=Path('/proc/meminfo').read_text()
        rows=re.findall(r'^MemAvailable:\s+([0-9]+) kB$',text,re.M)
        need(len(rows)==1,'Linux physical available memory missing')
        return int(rows[0])*1024,'Linux MemAvailable',0
    if sys.platform=='darwin':
        text=subprocess.check_output(['/usr/bin/vm_stat'],timeout=30).decode()
        page=re.search(r'page size of ([0-9]+) bytes',text)
        rows=dict(re.findall(r'^(Pages free|Pages speculative):\s+([0-9]+)\.$',text,re.M))
        need(page is not None and len(rows)==2,'Darwin physical admission missing')
        return sum(map(int,rows.values()))*int(page[1]),'Darwin free+speculative only',1
    raise RuntimeError('unsupported engineering admission platform')


def error_chain(error):
    pending=[error];seen=set();rows=[]
    while pending:
        item=pending.pop()
        if item is None or id(item) in seen:continue
        seen.add(id(item));need(len(seen)<=16,'bounded error chain')
        rows.append(dict(type=type(item).__name__,message=str(item)[:512]))
        pending.extend((item.__cause__,item.__context__))
    return rows


class LiteralWorld:
    """Literal native-read boundary; the actual concrete producer parses it."""
    def __init__(self,module,abi,mode,directory,variant=None):
        self.module,self.abi,self.mode,self.directory,self.variant=module,abi,mode,directory,variant
        self.now=0.;self.spawned=False;self.reaped=False;self.signals=0;self.direct_signals=0
        self.wait_calls=0;self.waitid_calls=0;self.record_calls=0;self.popen_calls=0
        self.waitid_events=[]
        self.child_lost=False;self.child_reads_after_loss=0;self.on_launch_calls=0;self.callbacks_after_loss=0
        self.borrowed_signal_attempts=0
        self.mutated=False;self.mutation_rejections=[];self.first_child_raw=None;self.proof=None
        self.parent_changed=False;self.process=None
        self.budget_violation=False;self.acquired_identities=[]
        self.live=mode in ('live_anchor_then_zombie_sid_loss','changed_birth_rejected','conflicting_known_sid_rejected',
                           'release_object_and_adapter_isolation','released_anchor_not_reused') or \
            (mode=='wrong_terminal_pid_rejected' and (variant=='main_loop' or str(variant).startswith(('main_','stop_'))))
        self.exit_code=-9 if mode=='first_zombie_exec_signal9' else 7
        adapter=object.__new__(module._DarwinProcesses)
        adapter.C=C;adapter.BSDInfo=abi.BSDInfo;adapter.ShortBSDInfo=abi.ShortBSDInfo
        adapter._launch_parent=None;adapter._launch_process=None;adapter._launch_anchor=None;adapter._launch_used=False
        adapter.library=SimpleNamespace(proc_pidinfo=self.pidinfo,proc_listpids=self.listpids)
        self.adapter=adapter
        actual=adapter._raw_info
        def observed_raw(pid):
            row,flags=actual(pid)
            if pid==PID and self.first_child_raw is None:self.first_child_raw=dict(row=row.copy(),flags=flags)
            return row,flags
        adapter._raw_info=observed_raw
        self.os=SimpleNamespace(name='posix',getpid=lambda:PARENT,getuid=lambda:UID,geteuid=lambda:UID,
            getpgrp=lambda:PARENT-9,getsid=self.getsid,waitid=self.waitid,P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,
            WSTOPPED=16,CLD_EXITED=1,CLD_KILLED=2,CLD_DUMPED=3,CLD_STOPPED=5,killpg=self.killpg,kill=self.kill)
        self.clock=SimpleNamespace(monotonic=lambda:self.now,sleep=self.sleep)

    def sleep(self,seconds):
        self.now+=seconds
        # Fake floating-clock addition only: reserve1us for representation,
        # never another stage/cleanup interval or a real native tolerance.
        if self.now>28.000001:
            self.budget_violation=True
            raise AssertionError('literal original8+20 clock budget exceeded')

    def getsid(self,pid):
        if pid in (0,PARENT):return PARENT-19
        if self.mode=='conflicting_known_sid_rejected' and self.mutated:return PID+9
        if self.live:return PID
        raise ProcessLookupError(errno.ESRCH,'literal zombie getsid')

    def fields(self,pid,kind):
        if pid==PARENT:
            return dict(pid=PARENT,ppid=PARENT-100,pgid=PARENT-9,uid=UID,ruid=UID,status=2,flags=0x4000,
                start_sec=51 if self.parent_changed else 50,start_usec=60)
        value=dict(pid=pid,ppid=PARENT,pgid=PID,uid=UID,ruid=UID,status=2 if self.live else 5,
            flags=0x4020 if self.live else 0x4000,start_sec=101 if self.mutated else 100,start_usec=200)
        if self.child_lost:
            self.child_reads_after_loss+=1
            value.update(start_sec=200,start_usec=300)
        changes={
            'eof_or_preexec_death_is_not_exec':dict(flags=0),
            'zombie_sleader_without_exec_rejected':dict(flags=0x20),
            'inherited_group_rejected':dict(pgid=PARENT-9),
            'wrong_parent_rejected':dict(ppid=PARENT+1),
            'wrong_effective_uid_rejected':dict(uid=UID+1),
            'wrong_real_uid_rejected':dict(ruid=UID+1),
            'invalid_birth_rejected':dict(start_sec=0) if self.variant!='bad_usec' else dict(start_usec=1000000),
            'wrong_raw_pid_rejected':dict(pid=PID+1)}
        value.update(changes.get(self.mode,{}))
        if self.mode=='wrong_terminal_pid_rejected' and self.variant=='surviving_sleader':
            value['flags']=0x4020
        if self.mode=='short_full_mismatch_rejected' and kind==3:
            value.update({self.variant or 'ppid':PARENT+1 if self.variant in (None,'ppid') else UID+1 if self.variant in ('uid','ruid') else PID+1})
        return value

    def pidinfo(self,pid,kind,arg,buffer,size):
        need(kind in (3,13) and arg==1,'actual bounded raw reader flavor differs')
        self.record_calls+=1
        if pid==PID and self.mode=='native_read_error_rejected' and self.variant in ('permission','truncated'):
            C.set_errno(errno.EACCES if self.variant=='permission' else errno.EPROTO)
            return 0 if self.variant=='permission' else size-1
        structure=self.abi.BSDInfo if kind==3 else self.abi.ShortBSDInfo
        value=structure()
        for key,field in self.fields(pid,kind).items():
            if hasattr(value,key):setattr(value,key,field)
        need(C.sizeof(value)==size,'literal ABI buffer size differs')
        C.memmove(buffer,C.byref(value),size)
        return size

    def listpids(self,kind,arg,buffer,size):
        need((kind,arg)==(1,0),'actual census inventory flavor differs')
        ids=(PARENT,PID) if self.spawned else (PARENT,)
        value=(C.c_int*len(ids))(*ids)
        C.memmove(buffer,value,C.sizeof(value))
        return C.sizeof(value)

    def popen(self,argv,**kwargs):
        need(argv==['literal-file-fixture.py'] and kwargs['start_new_session'] is True and not self.spawned,
             'one internally returned literal Popen origin required')
        self.spawned=True;self.popen_calls+=1
        world=self
        class Proxy:
            pid=PID
            returncode=7 if world.mode=='already_reaped_process_rejected' else None
            def wait(self,timeout=None):
                world.wait_calls+=1
                if world.mode in AnchorStateTests.positive:
                    need(world.signals==1,'positive terminal wait preceded original-group signal')
                self.returncode=world.exit_code;world.reaped=True
                return self.returncode
            def poll(self):raise AssertionError('no hidden terminal poll')
            def kill(self):raise AssertionError('no hidden terminal poll in kill')
        self.process=Proxy()
        if self.mode=='changed_launch_parent_rejected':self.parent_changed=True
        return self.process

    def waitid(self,kind,pid,flags):
        need(kind==1,'literal WNOWAIT kind differs')
        if pid==PARENT:raise ChildProcessError(errno.ECHILD,'self is not child')
        need(pid==PID and flags in (14,20),'terminal/STOP-only proof flags differ')
        self.waitid_calls+=1
        if 'echild' in str(self.variant):
            if str(self.variant).startswith('stop_') and flags==14:
                observed=SimpleNamespace(si_pid=PID,si_code=5,si_status=19,si_signo=17,si_uid=UID)
                self.waitid_events.append(dict(flags=flags,**vars(observed)))
                return observed
            self.child_lost=True
            child_error=str(self.variant).startswith('child_') or '_child_' in str(self.variant)
            if child_error:
                exception=ChildProcessError(errno.ECHILD,'literal owned child is no longer waitable')
            else:
                # OSError(errno.ECHILD,...) specializes to ChildProcessError.
                # Exercise the separate generic errno branch with the actual
                # base object, not an event label describing an intended type.
                exception=OSError('literal owned child is no longer waitable')
                exception.errno=errno.ECHILD
            self.waitid_events.append(dict(flags=flags,error_type=type(exception).__name__,errno=exception.errno))
            raise exception
        if self.mode=='missing_terminal_deadline_not_reset':
            self.waitid_events.append(dict(flags=flags,si_pid=None,si_code=None))
            return None
        code=5 if self.mode=='nonterminal_event_not_a_certificate' else 2 if self.exit_code==-9 else 1
        wrong=self.mode=='wrong_terminal_pid_rejected' and not (self.variant=='wrong_then_valid' and self.waitid_calls>1)
        observed=SimpleNamespace(si_pid=PID+1 if wrong else PID,
            si_code=code,si_status=9 if self.exit_code==-9 else 7,si_signo=17,
            si_uid=0 if self.mode in ('actual_zero_terminal_uid_accepted','first_zombie_exec_signal9') else UID)
        self.waitid_events.append(dict(flags=flags,**vars(observed)))
        return observed

    def killpg(self,pid,number):
        if (pid,number)!=(PID,9) or self.reaped or self.signals!=0:self.borrowed_signal_attempts+=1
        need((pid,number)==(PID,9) and not self.reaped and self.signals==0,'borrowed/repeated/post-reap signal forbidden')
        need(bool(self.acquired_identities) and self.acquired_identities[-1]==
            dict(pid=PID,pgid=PID,sid=PID,birth=(100,200),uid=UID,ruid=UID),
            'original-group failure cleanup needs actual native acquisition proof')
        self.signals+=1
        if self.live:self.exit_code=-9;self.live=False

    def kill(self,pid,number):
        if (pid,number)!=(PID,9) or self.reaped:self.borrowed_signal_attempts+=1
        need((pid,number)==(PID,9) and not self.reaped,'borrowed direct signal forbidden')
        self.direct_signals+=1
        if self.live:self.exit_code=-9;self.live=False


class AnchorStateTests(unittest.TestCase):
    output=None;configuration=None;abi=None;rows=[]
    positive={'live_anchor_then_zombie_sid_loss','first_zombie_exec_exit7','first_zombie_exec_signal9',
        'changed_birth_rejected','conflicting_known_sid_rejected','actual_zero_terminal_uid_accepted',
        'release_object_and_adapter_isolation','released_anchor_not_reused'}

    def one(self,module,mode,directory,*,candidate=None,variant=None,historical=False):
        directory.mkdir()
        world=LiteralWorld(module,self.abi,mode,directory,variant)
        def callback(pid):
            world.on_launch_calls+=1
            if world.child_lost:world.callbacks_after_loss+=1
            self.assertEqual(pid,PID)
            self.assertIsNotNone(world.adapter._launch_anchor)
            if mode in ('changed_birth_rejected','conflicting_known_sid_rejected'):
                world.mutated=True
                with self.assertRaises(RuntimeError):world.adapter._info(PID)
                world.mutation_rejections.append(mode);world.mutated=False
            if mode=='release_object_and_adapter_isolation':
                with self.assertRaises(RuntimeError):world.adapter._release_launch(SimpleNamespace(pid=PID,returncode=None))
                other=LiteralWorld(module,self.abi,'first_zombie_exec_exit7',directory).adapter
                with self.assertRaises(RuntimeError):other.leader(PID)
                world.mutation_rejections.extend(('wrong_release_object','unbound_other_adapter'))
            world.live=False
        returned=error=None
        actual=module._run_posix
        acquire=module._acquire_leader
        def acquired(*args,**kwargs):
            identity=acquire(*args,**kwargs)
            world.acquired_identities.append(identity.copy())
            return identity
        def tapped(*args,**kwargs):
            result=actual(*args,**kwargs);world.proof=result[1];return result
        with patch.object(module,'os',world.os),patch.object(module,'sys',SimpleNamespace(platform='darwin')), \
             patch.object(module,'time',world.clock),patch.object(module,'signal',SimpleNamespace(SIGKILL=9,SIGCHLD=17,SIG_IGN=1,getsignal=lambda _:0)), \
             patch.object(module,'subprocess',SimpleNamespace(Popen=world.popen,STDOUT=-2)), \
             patch.object(module,'_processes',return_value=world.adapter),patch.object(module,'_run_posix',tapped), \
             patch.object(module,'_acquire_leader',acquired):
            if mode=='generic_unbound_zombie_has_no_sid':
                self.assertIsNone(world.adapter._info(PID)['sid'])
                with self.assertRaises(RuntimeError):world.adapter.leader(PID)
            else:
                try:
                    if candidate is None:
                        returned=module.run_owned(['literal-file-fixture.py'],cwd=str(directory),environment={},
                            logpath=directory/'literal.raw',timeout=8,on_launch=callback)
                    else:
                        with patch.object(candidate,'os',SimpleNamespace(name='posix')),patch.object(candidate,'time',world.clock), \
                             patch.object(candidate,'_posix_module',return_value=module):
                            returned=candidate.run_owned(['literal-file-fixture.py'],cwd=str(directory),environment={},
                                logpath=directory/'literal.raw',timeout=8,on_launch=callback)
                except BaseException as caught:error=caught
            put(directory/'OBSERVATIONS.json',dict(evidence_kind='simulated_literal_native_read_boundary',
                verdict_pending=True,mode=mode,variant=variant,subject_sha256=digest(module.__file__),
                returned=returned,error_chain=error_chain(error),proof=world.proof,first_raw=world.first_child_raw,
                WNOWAIT_events=world.waitid_events,group_signals=world.signals,direct_signals=world.direct_signals,
                acquire_proofs=world.acquired_identities,fake_clock_seconds=world.now,budget_violation=world.budget_violation,
                child_reads_after_loss=world.child_reads_after_loss,on_launch_calls=world.on_launch_calls,
                callbacks_after_loss=world.callbacks_after_loss,borrowed_signal_attempts=world.borrowed_signal_attempts,
                actual_POSIX_test_processes_started=0,actual_signals_sent=0))
            if historical:
                self.assertEqual(mode,'wrong_terminal_pid_rejected')
                self.assertEqual(digest(module.__file__),self.configuration['before_waitid_identity_v2_sha256'])
                self.assertEqual(world.waitid_events[0]['si_pid'],PID+1)
                if variant=='wrong_then_valid':
                    self.assertIsNone(error)
                    self.assertIsNotNone(returned)
                    self.assertGreaterEqual(world.waitid_calls,2)
                    self.assertEqual(world.waitid_events[1]['si_pid'],PID)
                elif variant=='main_loop':
                    self.assertIsNotNone(error)
                    self.assertIsNone(returned)
                    self.assertEqual(world.waitid_calls,1)
                else:
                    self.fail('only the two frozen semantic historical negatives are admitted')
                self.assertEqual(world.signals,1)
            elif mode in self.positive:
                self.assertIsNone(error)
                self.assertEqual(returned['native_exit'],world.exit_code)
                self.assertIs(returned['timed_out'],False)
                self.assertEqual(world.signals,1)
                self.assertEqual(world.wait_calls,1)
                self.assertTrue(world.reaped)
                self.assertIs(world.proof['session_zero_live_verified'],True)
                origin='native_live' if mode in ('live_anchor_then_zombie_sid_loss','changed_birth_rejected',
                    'conflicting_known_sid_rejected','release_object_and_adapter_isolation','released_anchor_not_reused') else 'owned_exec_terminal'
                self.assertEqual(world.proof['leader_identity_origin'],origin)
                if mode=='actual_zero_terminal_uid_accepted':
                    self.assertEqual(world.proof['launch_terminal_observation']['si_uid'],0)
                base={'pid','native_exit','timed_out','wall_seconds','ownership','descendant_cleanup'}
                self.assertEqual(set(returned),base|({'leader_reaped','ownership_boundary_closed','all_descendants_os_reaped',
                    'observation_method','descendant_evidence'} if candidate else set()))
                if candidate:
                    self.assertIs(returned['ownership_boundary_closed'],True)
                    self.assertIs(returned['all_descendants_os_reaped'],False)
                if mode=='released_anchor_not_reused':
                    self.assertIsNone(world.adapter._info(PID)['sid'])
                    with self.assertRaises(RuntimeError):world.adapter.leader(PID)
                    world.adapter._release_launch(world.process)
                    with self.assertRaises(RuntimeError):world.adapter._prepare_launch(100.)
                    world.mutation_rejections.append('released_anchor_and_adapter_not_reused')
            elif mode!='generic_unbound_zombie_has_no_sid':
                self.assertIsNotNone(error)
                self.assertIsNone(returned)
                if mode=='zombie_sleader_without_exec_rejected':
                    self.assertLessEqual(world.signals,1)
                    self.assertTrue(world.acquired_identities)
                else:
                    self.assertEqual(world.signals,0)
                self.assertEqual(world.direct_signals,0)
                self.assertIsNot(getattr(error,'ownership_boundary_closed',None),True)
                chain=error_chain(error)
                if mode=='wrong_terminal_pid_rejected':
                    if 'echild' in str(variant):
                        self.assertEqual(world.waitid_calls,2 if str(variant).startswith('stop_') else 1)
                        self.assertEqual(world.waitid_events[-1]['errno'],errno.ECHILD)
                        self.assertEqual(world.waitid_events[-1]['error_type'],
                            'ChildProcessError' if str(variant).startswith('child_') or '_child_' in str(variant) else 'OSError')
                        self.assertEqual(world.child_reads_after_loss,0)
                        self.assertEqual(world.on_launch_calls,1 if str(variant).startswith(('main_','stop_')) else 0)
                    else:
                        self.assertEqual(world.waitid_calls,1)
                        self.assertEqual(world.waitid_events[0]['si_pid'],PID+1)
                if mode=='native_read_error_rejected' and 'echild' in str(variant):
                    self.assertEqual(world.waitid_calls,1)
                    self.assertEqual(world.waitid_events[-1]['errno'],errno.ECHILD)
                    self.assertEqual(world.waitid_events[-1]['error_type'],'ChildProcessError' if '_child_' in str(variant) else 'OSError')
                    self.assertEqual(world.child_reads_after_loss,0)
                    self.assertEqual(world.on_launch_calls,0)
                expected={'eof_or_preexec_death_is_not_exec':'native exec witness',
                    'zombie_sleader_without_exec_rejected':'native exec witness','wrong_terminal_pid_rejected':'held unreaped child',
                    'nonterminal_event_not_a_certificate':'held unreaped child','missing_terminal_deadline_not_reset':'held unreaped child',
                    'already_reaped_process_rejected':'fresh private leader','changed_launch_parent_rejected':'launch-parent identity changed',
                    'native_read_error_rejected':'native process information unavailable','short_full_mismatch_rejected':'identity changed between records',
                    'wrong_raw_pid_rejected':'native process information unavailable'}.get(mode,'direct launch native identity differs')
                if mode!='wrong_terminal_pid_rejected' and 'echild' not in str(variant):
                    self.assertTrue(any(expected in item['message'] for item in chain),chain)
        self.assertIsNone(world.adapter._launch_anchor)
        self.assertIsNone(world.adapter._launch_process)
        self.assertIsNone(world.adapter._launch_parent)
        self.assertFalse(world.budget_violation)
        self.assertEqual(world.borrowed_signal_attempts,0)
        self.assertEqual(world.callbacks_after_loss,0)
        self.assertLessEqual(world.now,28.000001)
        if mode!='generic_unbound_zombie_has_no_sid':self.assertEqual(world.popen_calls,1)
        return dict(mode=mode,variant=variant,passed=True,evidence_kind='simulated_historical_waitid_identity_negative' if historical else 'simulated_concrete_Darwin_native_read_boundary',
            expected_historical_negative_reproduced=True if historical else None,
            native_result=returned,proof=world.proof,error_chain=error_chain(error),first_raw=world.first_child_raw,
            fake_clock_seconds=world.now,simulated_group_signals=world.signals,simulated_direct_signals=world.direct_signals,
            simulated_terminal_waits=world.wait_calls,simulated_WNOWAIT_calls=world.waitid_calls,
            simulated_WNOWAIT_events=world.waitid_events,
            native_record_parser_calls=world.record_calls,mutation_rejections=world.mutation_rejections,
            original_native_acquire_proofs=world.acquired_identities,
            child_reads_after_loss=world.child_reads_after_loss,on_launch_calls=world.on_launch_calls,
            callbacks_after_loss=world.callbacks_after_loss,borrowed_signal_attempts=world.borrowed_signal_attempts,
            actual_POSIX_test_processes_started=0,actual_signals_sent=0,native_Darwin_qualified=False)

    def test_legacy_literal_launch_anchor_cases(self):
        for index,role in enumerate(ROLES[:2]):
            for number,mode in enumerate(self.configuration['literal_cases']):
                with self.subTest(role=role,mode=mode):
                    module=load(ROOT/role,'anchor_legacy_'+str(index)+'_'+str(number),self.configuration['current_legacy_sha256'])
                    variants=('bad_usec',None) if mode=='invalid_birth_rejected' else ('ppid','uid','ruid') if mode=='short_full_mismatch_rejected' else ('permission','truncated','wait_child_echild','wait_os_echild') if mode=='native_read_error_rejected' else (None,'surviving_sleader','wrong_then_valid','main_loop','child_echild_then_birthB','os_echild_then_birthB','main_child_echild','main_os_echild','stop_child_echild','stop_os_echild') if mode=='wrong_terminal_pid_rejected' else (None,)
                    details=[self.one(module,mode,self.output/(str(index)+'-'+str(number)+'-'+str(v)),variant=v) for v in variants]
                    historical=[]
                    if mode=='wrong_terminal_pid_rejected':
                        for variant in ('wrong_then_valid','main_loop'):
                            before=load(HERE/'frozen/before_waitid_identity_v2.py','anchor_before_wait_identity_'+str(index)+'_'+variant,
                                self.configuration['before_waitid_identity_v2_sha256'])
                            historical.append(self.one(before,mode,self.output/('before-'+str(index)+'-'+variant),variant=variant,historical=True))
                    self.rows.append(dict(role=role,mode=mode,passed=True,evidence_kind='simulated_literal_anchor_control',variants=details,historical_negatives=historical))

    def test_candidate_delegates_to_real_checked_backend(self):
        for index,mode in enumerate(self.configuration['candidate_cases']):
            with self.subTest(mode=mode):
                candidate=load(ROOT/ROLES[2],'anchor_candidate_'+str(index),self.configuration['current_candidate_sha256'])
                backend=candidate._posix_module()
                self.assertEqual(digest(backend.__file__),self.configuration['current_legacy_sha256'])
                variants=(None,'wrong_then_valid','main_loop','child_echild_then_birthB','os_echild_then_birthB','stop_child_echild','stop_os_echild') if mode=='wrong_terminal_pid_rejected' else (None,)
                details=[self.one(backend,mode,self.output/('candidate-'+str(index)+'-'+str(v)),candidate=candidate,variant=v) for v in variants]
                self.rows.append(dict(role=ROLES[2],mode=mode,passed=True,evidence_kind='simulated_real_candidate_backend',details=details))

    def test_explicit_cancellation_is_distinct_from_native_eintr(self):
        sites=('preflight','checked_snapshot','acquire_leader','bind_launch')
        self.assertEqual(self.configuration['cancellation_retry_sites'],list(sites))
        identity=dict(pid=PID,pgid=PID,sid=PID,birth=(100,200),uid=UID,ruid=UID)
        parent=dict(pid=PARENT,pgid=PARENT-9,sid=PARENT-19,birth=(50,60),uid=UID,ruid=UID,zombie=False)
        snapshot=dict(complete=True,leader_held=True,group_live=0,session_live=0,group_zombie_only=True,
            group_members=(),session_live_members=())

        def retry_case(module,site,explicit):
            state=SimpleNamespace(now=0.,calls=0,spawn_calls=0)
            failure=InterruptedError('literal explicit supervisor cancellation') if explicit else InterruptedError(errno.EINTR,'literal native interrupted syscall')
            def fault(value):
                state.calls+=1
                if state.calls==1:raise failure
                return value
            def forbidden_spawn(*args,**kwargs):
                state.spawn_calls+=1
                raise AssertionError('retry admission must not launch a process')
            clock=SimpleNamespace(monotonic=lambda:state.now,sleep=lambda seconds:setattr(state,'now',state.now+seconds))
            fake_os=SimpleNamespace(getpid=lambda:PARENT,getuid=lambda:UID,getpgrp=lambda:PARENT-9,getsid=lambda _:PARENT-19)
            returned=error=None
            process=None
            world=None
            with patch.object(module,'time',clock),patch.object(module,'os',fake_os), \
                 patch.object(module,'subprocess',SimpleNamespace(Popen=forbidden_spawn)):
                try:
                    if site=='preflight':
                        reader=SimpleNamespace(_stable=lambda deadline:fault([parent.copy()]))
                        returned=module._NativeCensus.preflight(reader,8.)
                    elif site=='checked_snapshot':
                        reader=SimpleNamespace(snapshot=lambda actual,deadline:fault(snapshot.copy()))
                        returned=module._checked_snapshot(reader,identity,8.,dict(census_attempts=0))
                    elif site=='acquire_leader':
                        reader=SimpleNamespace(leader=lambda actual:fault(identity.copy()))
                        returned=module._acquire_leader(reader,PID,8.)
                    else:
                        world=LiteralWorld(module,self.abi,'first_zombie_exec_exit7',self.output)
                        actual_raw=world.adapter._raw_info
                        def interrupted_raw(pid):
                            if pid==PID:fault(None)
                            return actual_raw(pid)
                        world.adapter._raw_info=interrupted_raw
                        with patch.object(module,'os',world.os),patch.object(module,'time',world.clock):
                            world.adapter._prepare_launch(8.)
                            process=world.popen(['literal-file-fixture.py'],start_new_session=True)
                            world.adapter._bind_launch(process,8.)
                            returned=world.adapter._launch_anchor.copy()
                except BaseException as caught:error=caught
                finally:
                    if world is not None:world.adapter._release_launch(process)
            if explicit:
                self.assertIs(error,failure)
                self.assertIsNone(returned)
                self.assertEqual(state.calls,1)
            else:
                self.assertIsNone(error)
                self.assertEqual(state.calls,2)
                if site=='preflight':self.assertIsNone(returned)
                if site=='checked_snapshot':self.assertEqual(returned,snapshot)
                if site=='acquire_leader':self.assertEqual(returned,identity)
                if site=='bind_launch':self.assertEqual(returned['origin'],'owned_exec_terminal')
            self.assertEqual(state.spawn_calls,0)
            self.assertLess(world.now if world is not None else state.now,8.)
            if world is not None:
                self.assertIsNone(world.adapter._launch_anchor)
                self.assertEqual(world.signals,0)
                self.assertEqual(world.direct_signals,0)
            return dict(site=site,explicit_cancel=explicit,passed=True,boundary_calls=state.calls,
                same_exception_preserved=error is failure if explicit else None,native_errno=failure.errno,
                actual_POSIX_test_processes_started=0,actual_signals_sent=0,
                evidence_kind='simulated_actual_retry_boundary',native_Darwin_qualified=False)

        for index,role in enumerate(ROLES[:2]):
            for site in sites:
                for explicit in (True,False):
                    with self.subTest(role=role,site=site,explicit=explicit):
                        module=load(ROOT/role,'anchor_cancel_'+str(index)+site+str(explicit),self.configuration['current_legacy_sha256'])
                        row=retry_case(module,site,explicit)
                        row.update(role=role)
                        self.rows.append(row)
        for explicit in (True,False):
            with self.subTest(role='candidate_preflight',explicit=explicit):
                candidate=load(ROOT/ROLES[2],'anchor_cancel_candidate_'+str(explicit),self.configuration['current_candidate_sha256'])
                module=candidate._posix_module()
                self.assertEqual(digest(module.__file__),self.configuration['current_legacy_sha256'])
                state=SimpleNamespace(now=0.,calls=0)
                failure=InterruptedError('literal candidate supervisor cancellation') if explicit else InterruptedError(errno.EINTR,'literal candidate native EINTR')
                class Reader:
                    def _stable(reader,deadline):
                        state.calls+=1
                        if state.calls==1:raise failure
                        return [parent.copy()]
                    def preflight(reader,deadline):return module._NativeCensus.preflight(reader,deadline)
                reader=Reader()
                def self_probe(*args):raise ChildProcessError(errno.ECHILD,'self is not a child')
                fake_os=SimpleNamespace(name='posix',getpid=lambda:PARENT,getuid=lambda:UID,getpgrp=lambda:PARENT-9,getsid=lambda _:PARENT-19,
                    waitid=self_probe,P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,WSTOPPED=16,
                    CLD_EXITED=1,CLD_KILLED=2,CLD_DUMPED=3,CLD_STOPPED=5)
                returned=error=None
                with patch.object(candidate,'_posix_module',return_value=module),patch.object(module,'os',fake_os), \
                     patch.object(module,'sys',SimpleNamespace(platform='darwin')), \
                     patch.object(module,'signal',SimpleNamespace(SIGCHLD=17,SIG_IGN=1,getsignal=lambda _:0)), \
                     patch.object(module,'time',SimpleNamespace(monotonic=lambda:state.now,sleep=lambda seconds:setattr(state,'now',state.now+seconds))), \
                     patch.object(module,'_processes',return_value=reader),patch.object(module,'subprocess') as no_spawn:
                    try:returned=candidate.posix_features()
                    except BaseException as caught:error=caught
                    no_spawn.Popen.assert_not_called()
                if explicit:
                    self.assertIs(error,failure);self.assertIsNone(returned);self.assertEqual(state.calls,1)
                else:
                    self.assertIsNone(error);self.assertIs(returned,reader);self.assertEqual(state.calls,2)
                self.rows.append(dict(role=ROLES[2],site='candidate_preflight',explicit_cancel=explicit,passed=True,
                    boundary_calls=state.calls,same_exception_preserved=error is failure if explicit else None,
                    native_errno=failure.errno,evidence_kind='simulated_real_candidate_preflight_delegation',
                    actual_POSIX_test_processes_started=0,actual_signals_sent=0,native_Darwin_qualified=False))


def main():
    started=time.monotonic_ns();deadline=started+90*10**9
    parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();output=args.output_directory
    need(not sys.flags.optimize and output.is_absolute() and not output.exists() and output.parent.is_dir() and
         not output.is_relative_to(ROOT),'fresh unoptimized external state evidence required')
    value=config();pins=source_pins(value);before=inventory()
    available,method,resource_processes=available_memory();free=shutil.disk_usage(output.parent).free
    need(available>=536870912 and free>=8589934592,'engineering512MiB/8GiB admission failed')
    abi=load(ROOT/'test/initial_reference/diagnose_darwin_groups.py','anchor_independent_frozen_abi',value['independent_darwin_abi_sha256'])
    need(C.sizeof(abi.BSDInfo)==136 and C.sizeof(abi.ShortBSDInfo)==64,'independent literal ABI differs')
    output.mkdir();AnchorStateTests.output=output;AnchorStateTests.configuration=value;AnchorStateTests.abi=abi
    put(output/'PREEXECUTION.json',dict(source_pins=pins,test_sources=before,started_ns=started,deadline_ns=deadline,
        available_physical_bytes=available,memory_method=method,free_disk_bytes=free,resource_probe_subprocesses=resource_processes))
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnchorStateTests))
    unchanged=source_pins(value)==pins and inventory()==before
    finished=time.monotonic_ns()
    passed=result.wasSuccessful() and result.testsRun==3 and len(AnchorStateTests.rows)==72 and unchanged and finished<deadline
    put(output/'RESULT.json',dict(schema='independent_darwin_anchor_literal_result_v1',complete=passed,
        unittest_methods=result.testsRun,expected_methods=3,literal_rows=len(AnchorStateTests.rows),expected_rows=72,
        rows=AnchorStateTests.rows,source_pins=pins,test_sources=before,source_unchanged=unchanged,
        started_ns=started,finished_ns=finished,deadline_ns=deadline,actual_POSIX_test_processes_started=0,actual_signals_sent=0,
        native_Darwin_qualified=False,Julia_launched=False,actual_data_exposed=False,source_qualified=False,scientific_qualified=False))
    if time.monotonic_ns()>=deadline:
        (output/'RESULT.json').rename(output/'RESULT_LATE_UNQUALIFIED.json');return 1
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
