"""Literal simulated V3 states; no subprocess, OS census or real signal."""
from pathlib import Path
import argparse
import copy
import errno
import io
import hashlib
import importlib.util
import math
from types import SimpleNamespace
import sys
sys.dont_write_bytecode = True
import unittest
from unittest.mock import patch
_observations_path = Path(__file__).resolve().with_name('observations.py')
_observations_sha = 'd9b02137f669e16819c82cdbf3b1cba182ecf9f8c94fc0e52ad6fe7f96bb01bf'
with _observations_path.open('rb') as _stream:
    _observations_body = _stream.read(256*1024+1)
if len(_observations_body)>256*1024 or hashlib.sha256(_observations_body).hexdigest()!=_observations_sha:
    raise RuntimeError('pinned independent observation sibling changed')
_observations_spec = importlib.util.spec_from_file_location('posix_v3_checked_observations',_observations_path)
observations = importlib.util.module_from_spec(_observations_spec)
_observations_spec.loader.exec_module(observations)
if observations.digest(_observations_path)!=_observations_sha:
    raise RuntimeError('independent observation sibling changed during import')
HERE,ROOT,ROLES = observations.HERE,observations.ROOT,observations.ROLES
NativeObservation,load = observations.NativeObservation,observations.load
source_pins,test_inventory,write_new = observations.source_pins,observations.test_inventory,observations.write_new

PID = 424242
PARENT = 999999
POSITIVE = ('normal_exit', 'verified_zombie_eperm', 'esrch_with_verified_zero_sid', 'stop_then_terminal', 'delayed_zero_sid', 'stop_only_none_then_terminal')
NEGATIVE = ('missing_feature', 'unknown_event', 'wrong_event_pid', 'eperm_live_group', 'eperm_unverified_zombie',
            'unknown_census', 'wrong_birth', 'wrong_sid', 'wrong_uid', 'persistent_other_group',
            'callback_error', 'observation_error', 'shared_cleanup_deadline', 'terminal_wait_error',
            'truncated_census', 'boolean_count_census', 'foreign_session_uid', 'esrch_live_session',
            'preflight_budget_exhausted')


def error_chain(error):
    pending = [error]
    seen = set()
    rows = []
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > 16:
            raise AssertionError('bounded exception-chain oracle exceeded')
        rows.append(dict(type=type(item).__name__,message=str(item)))
        pending.extend((item.__cause__,item.__context__))
    return rows


def exercise(module, mode, root):
    # Literal foreign process data. The fixture cannot confer native authority.
    state = SimpleNamespace(now=0., code=None, signals=0, reaped=False, launched=False,
        snapshots=0, stops=0, observations=0, wait_timeout=None, cleanup_start=None,
        borrowed_attempts=0,order_violations=0,direct_signals=0,callback_invocations=0)
    events = []
    baseline = dict(pid=PID, ppid=PARENT, pgid=PID, sid=PID, birth=(100, 200), uid=1000, ruid=1000, zombie=False)

    class Adapter:
        def preflight(self, deadline):
            events.append(('preflight', deadline))
            if mode == 'preflight_budget_exhausted':
                state.now = deadline

        def _info(self, pid):
            assert pid == PID
            row = dict(baseline, zombie=state.code is not None)
            if mode == 'wrong_sid':
                row['sid'] = PID + 1
            if mode == 'wrong_uid':
                row['uid'] = 1001
            if mode == 'wrong_birth' and state.signals:
                row['birth'] = (200, 300)
            return row

        def leader(self, pid):
            return module._NativeCensus.leader(self, pid)

        def _stable(self, deadline):
            state.snapshots += 1
            if mode == 'unknown_census':
                raise OSError(errno.EIO, 'literal unreadable census')
            if mode == 'shared_cleanup_deadline':
                state.now = state.cleanup_start + 19.
            leader = self._info(PID)
            rows = [leader]
            live = mode in ('eperm_live_group', 'persistent_other_group', 'esrch_live_session') or (mode == 'delayed_zero_sid' and state.snapshots <= 2)
            if live:
                rows.append(dict(baseline, pid=PID+1, ppid=PID, pgid=PID if mode == 'eperm_live_group' else PID+1,
                                 birth=(101, 200), zombie=False))
            if mode == 'foreign_session_uid':
                rows.append(dict(baseline, pid=PID+1, ppid=PID, pgid=PID+1,
                                 birth=(101, 200), uid=1001, ruid=1001, zombie=False))
            if mode == 'eperm_unverified_zombie':
                rows[0]['zombie'] = False
            if mode == 'delayed_zero_sid':
                state.now += 1.
            return rows

        def snapshot(self, identity, deadline):
            if mode == 'truncated_census':
                return dict(complete=False, leader_held=True, group_live=0, session_live=0, group_zombie_only=True)
            if mode == 'boolean_count_census':
                return dict(complete=True, leader_held=True, group_live=False, session_live=0, group_zombie_only=True)
            return module._NativeCensus.snapshot(self, identity, deadline)

    adapter = Adapter()

    class Process:
        pid = PID

        def poll(self):
            raise AssertionError('literal first terminal poll forbidden')

        def kill(self):
            raise AssertionError('hidden poll in Process.kill forbidden')

        def wait(self, timeout):
            events.append(('first_wait', timeout, state.now))
            state.wait_timeout = timeout
            assert not state.reaped
            if mode not in ('wrong_sid', 'wrong_uid'):
                assert state.signals == 1, 'terminal wait must follow original group attempt'
            if state.code is None:
                raise RuntimeError('literal child still alive')
            if mode == 'terminal_wait_error':
                raise OSError(errno.EIO, 'literal first terminal wait failure')
            state.reaped = True
            if mode == 'shared_cleanup_deadline':
                state.now += timeout  # Exact cleanup-boundary equality is late.
            return state.code

    def popen(argv, **options):
        assert argv == ['literal-fixture'] and options['start_new_session'] is True
        assert not state.launched
        state.launched = True
        events.append(('spawn', PID))
        return Process()

    def waitid(kind, pid, flags):
        assert kind == 1
        if pid == PARENT:
            events.append(('feature', flags))
            raise ChildProcessError(errno.ECHILD, 'self is not an owned child')
        assert pid == PID and not state.reaped
        if flags == 20:  # WSTOPPED16|WNOHANG4, with no terminal WEXITED/WNOWAIT.
            state.stops += 1
            events.append(('stop_only_consumed', flags))
            if mode == 'stop_only_none_then_terminal':
                return None
            return SimpleNamespace(si_pid=PID, si_code=5, si_status=19, si_signo=17, si_uid=1000)
        assert flags == 14
        state.observations += 1
        state.now += .1
        if mode == 'observation_error':
            raise OSError(errno.EIO, 'literal terminal observation failure')
        if mode == 'unknown_event':
            code = 4
        elif mode in ('stop_then_terminal','stop_only_none_then_terminal') and state.observations == 1:
            code = 5
        elif mode in ('stop_then_terminal','stop_only_none_then_terminal') and state.observations == 2:
            code = 6
        else:
            code = 1
            state.code = 7
        events.append(('observe', code))
        return SimpleNamespace(si_pid=PID+1 if mode == 'wrong_event_pid' else PID,
                               si_code=code, si_status=7, si_signo=17, si_uid=1000)

    def killpg(pid, number):
        if pid != PID or number != 9:
            state.borrowed_attempts += 1
            raise AssertionError('borrowed group signal refused')
        if state.reaped:
            state.order_violations += 1
            raise AssertionError('group signal after terminal reap refused')
        state.signals += 1
        assert state.signals == 1, 'no second original-group signal'
        state.cleanup_start = state.now
        events.append(('killpg', pid, number))
        if mode.startswith('eperm_') or mode == 'verified_zombie_eperm':
            raise PermissionError(errno.EPERM, 'literal group permission outcome')
        if mode in ('esrch_with_verified_zero_sid', 'esrch_live_session'):
            raise ProcessLookupError(errno.ESRCH, 'literal absent group')
        if state.code is None:
            state.code = -9

    def kill(pid, number):
        if pid != PID or number != 9:
            state.borrowed_attempts += 1
            raise AssertionError('borrowed direct signal refused')
        if state.reaped:
            state.order_violations += 1
            raise AssertionError('direct signal after terminal reap refused')
        state.direct_signals += 1
        events.append(('owned_direct_kill', pid, number))
        if state.code is None:
            state.code = -9

    fake_os = SimpleNamespace(name='posix', getpid=lambda:PARENT, getuid=lambda:1000,
        waitid=waitid, P_PID=1, WEXITED=2, WNOHANG=4, WNOWAIT=8, WSTOPPED=16,
        CLD_EXITED=1, CLD_KILLED=2, CLD_DUMPED=3, CLD_STOPPED=5, CLD_CONTINUED=6,
        killpg=killpg, kill=kill)
    if mode == 'missing_feature':
        del fake_os.WNOWAIT
    fake_time = SimpleNamespace(monotonic=lambda:state.now, sleep=lambda seconds:setattr(state, 'now', state.now+seconds))
    fake_signal = SimpleNamespace(SIGKILL=9, SIGCHLD=17, SIG_IGN=1, getsignal=lambda _:0)
    fake_subprocess = SimpleNamespace(Popen=popen, STDOUT=-2)
    result = error = None

    def callback(pid):
        assert pid == PID
        state.callback_invocations += 1
        if mode == 'callback_error':
            raise RuntimeError('literal callback failure')

    with patch.object(module, 'os', fake_os), patch.object(module, 'sys', SimpleNamespace(platform='darwin')), \
            patch.object(module, 'signal', fake_signal), patch.object(module, 'subprocess', fake_subprocess), \
            patch.object(module, 'time', fake_time), patch.object(module, '_processes', return_value=adapter):
        try:
            result = module._run_posix(['literal-fixture'], str(root), {}, root/'literal.raw', 8., callback)
        except BaseException as caught:
            error = caught
    return result, error, events, state


def darwin_retry_literal_scenario(module, mode, *, now=2000000000,
                                  deadline=9000000000, phase='positive_owner_observation'):
    """Primitive-only Darwin fixture; no real PID, signal, library or process."""
    held, vanished, live, partial = 16517, 16518, 16519, 16516
    leader = dict(pid=held,pgid=held,sid=held,birth=[1794290259,821315],uid=501,ruid=501,zombie=True,stopped=False)
    state = SimpleNamespace(now=now,attempt=0,group_reads=0,global_reads=0,vanished=False,sleeps=0,post_changed=False)
    calls=[]
    primitive_path=HERE.parent/'initial_reference'/'diagnose_darwin_groups.py'
    if observations.digest(primitive_path)!='3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2':
        raise AssertionError('literal Darwin primitive source changed')
    primitive=load(primitive_path,'darwin_retry_literal_frozen_primitives')
    def touch(operation,pid=None):
        calls.append((operation,state.attempt,pid));state.now+=1000
    def error(pid,short=True,number=3,count=0,wrong=False):
        prefix='short BSD information unavailable' if short else 'BSD process information unavailable'
        return OSError(number,f'{prefix}: pid={pid+1 if wrong else pid} bytes={count}')
    def native_row(pid):
        row=dict(pid=pid,ppid=1,pgid=held if pid==held else pid,status=5 if pid==held else 1,
                 uid=501,ruid=501,start_sec=1794290259 if pid==held else 1794290260,start_usec=821315)
        if pid==held:
            if mode=='held_info':raise error(pid,short=False)
            if mode=='held_birth' or state.post_changed:row['start_usec']=821316
            if mode=='held_pgid':row['pgid']=held+1
            if mode=='held_uid':row['uid']=502
            if mode=='held_ruid':row['ruid']=502
            if mode in ('held_sid','held_getsid'):row['status']=1
        return row
    def full(pid):
        row=native_row(pid)
        if pid==vanished:
            if mode=='full_vanish' and state.attempt==1:raise error(pid,short=False)
            if mode=='info_permission':raise error(pid,short=False,number=1)
            if mode=='row_conflict':row['pgid']=pid+1
        return row
    class Primitives:
        # The before-negative really calls frozen3395's group/session methods;
        # the candidate independently composes its actual adapter over these primitives.
        def snapshot(self,pgid):return primitive.DarwinProcesses.snapshot(self,pgid)
        def session_snapshot(self,sid):return primitive.DarwinProcesses.session_snapshot(self,sid)
        def pids(self,pgid,*,all_processes=False):
            touch('global_inventory' if all_processes else 'group_inventory',pgid)
            if not all_processes:
                state.group_reads+=1
                if mode=='held_group_absent':return ()
                if mode=='group_change' and state.attempt==1 and state.group_reads%2==0:return (held,vanished)
                return (held,)
            state.global_reads+=1
            if mode=='held_post_change' and state.global_reads==2:state.post_changed=True
            if mode=='getsid_probe_permission' and state.vanished:raise PermissionError(13,'literal complete inventory denied')
            if mode=='session_change' and state.attempt==1 and state.global_reads%2==0:return (held,vanished)
            if mode in ('live_after','partial_then_zero'):
                if state.attempt==1:return (partial,held,vanished) if mode=='partial_then_zero' else (held,vanished)
                return (held,live) if mode=='live_after' else (held,)
            if mode in ('short_vanish','full_vanish','getsid_vanish') and state.attempt>1:return (held,)
            if mode=='getsid_vanish' and state.vanished:return (held,)
            if mode in ('zero','late','group_change','session_change') or mode.startswith('held_'):return (held,)
            return (held,vanished)
        def info(self,pid):touch('info',pid);return full(pid)
        def short_info(self,pid):
            touch('short_info',pid)
            if pid==held and mode=='held_short':raise error(pid)
            if pid==vanished:
                if mode in ('short_vanish','live_after','partial_then_zero') and state.attempt==1 or mode=='unstable':raise error(pid)
                if mode=='short_permission':raise error(pid,number=13)
                if mode=='protocol':raise error(pid,number=71)
                if mode=='partial63':raise error(pid,count=63)
                if mode=='wrong_message_pid':raise error(pid,wrong=True)
                if mode=='generic_ESRCH':raise ProcessLookupError(3,'unclassified literal error')
                if mode=='generic_instability':raise module._CensusInstability('untyped Darwin error',pid)
            value=native_row(pid)
            row={k:value[k] for k in ('pid','ppid','pgid','status','uid')}
            if pid==vanished and mode=='malformed_row':row['uid']=True
            return row
    observer=object.__new__(module.NativeObservation)
    observer._held={held:{k:copy.deepcopy(leader[k]) for k in ('pid','pgid','sid','birth','uid','ruid')}}
    observer.darwin=Primitives()
    def getsid(pid):
        touch('getsid',pid)
        if pid==held and mode=='held_getsid':raise ProcessLookupError(3,'literal held getsid missing')
        if pid==held and mode=='held_sid':return held+1
        if pid==vanished and mode in ('getsid_vanish','getsid_still_listed','getsid_probe_permission'):
            state.vanished=True;raise ProcessLookupError(3,'literal known getsid missing')
        return held
    def snapshot(anchor):
        state.attempt+=1;state.group_reads=0;state.global_reads=0
        calls.append(('fresh_attempt',state.attempt,None))
        value=module.NativeObservation.snapshot(observer,anchor)
        if mode=='late':state.now=2250000000 if deadline is None else min(deadline,2000000000+250000000)
        return value
    observer.snapshot=snapshot
    fake_os=SimpleNamespace(getsid=getsid,getuid=lambda:501,geteuid=lambda:501)
    def sleep(seconds):state.sleeps+=1;state.now+=int(seconds*10**9)
    fake_time=SimpleNamespace(monotonic_ns=lambda:state.now,sleep=sleep)
    with patch.object(module,'os',fake_os),patch.object(module,'time',fake_time),patch.object(primitive,'os',fake_os):
        result=module.observed_snapshot(observer,leader,deadline_ns=deadline,phase=phase)
    return result,calls,state


def darwin_group_literal_scenario(module, mode, *, now=2000000000,
                                  deadline=9000000000, phase='positive_owner_observation',
                                  scope='process_group'):
    """Independent primitive data; no real process, native library or signals."""
    held, member = 26517, 26518
    leader = dict(pid=held,pgid=held,sid=held,birth=[1794299426,355078],
                  uid=501,ruid=501,zombie=True,stopped=False)
    state = SimpleNamespace(now=now,attempt=0,group_reads=0,sleeps=0,
                            post_changed=False,payloads=[])
    calls=[]
    def touch(operation,pid=None):
        calls.append((operation,state.attempt,pid));state.now+=1000
    def native_error(pid,number=3,count=0):
        return OSError(number,f'BSD process information unavailable: pid={pid} bytes={count}')
    class Primitives:
        def pids(self,pgid,*,all_processes=False):
            touch('global_inventory' if all_processes else 'group_inventory',pgid)
            if all_processes:
                raise AssertionError('group scope must not read the global Darwin PID inventory')
            if pgid!=held:
                raise AssertionError('only the held original process group may be listed')
            state.group_reads+=1
            if mode=='held_absent_before':return (member,)
            if mode=='held_absent_after' and state.group_reads%2==0:return (member,)
            if mode=='post_birth' and state.group_reads%2==0:state.post_changed=True
            if mode=='roster_change' and state.attempt==1:
                return (held,member) if state.group_reads%2 else (held,)
            if mode=='unstable':return (held,) if state.group_reads%2 else (held,member)
            if mode=='vanish' and state.attempt==1:return (held,member)
            if mode in ('live_group','member_pgid','malformed_uid','extra_field','wrong_pid',
                        'permission','protocol','partial'):
                return (held,member)
            return (held,)
        def info(self,pid):
            touch('info',pid)
            if pid not in (held,member):
                raise AssertionError('a non-group PID must not be read')
            if mode=='held_lost' and pid==held:raise native_error(pid)
            if mode=='vanish' and pid==member and state.attempt==1:raise native_error(pid)
            if pid==member and mode in ('permission','protocol','partial'):
                raise native_error(pid,number=13 if mode=='permission' else 71 if mode=='protocol' else 3,
                                   count=63 if mode=='partial' else 0)
            row=dict(pid=pid,ppid=1,pgid=held,status=5 if pid==held else 2,uid=501,ruid=501,
                     start_sec=1794299426 if pid==held else 1794299427,start_usec=355078)
            if pid==held:
                if mode=='birth' or state.post_changed:row['start_usec']=355079
                if mode=='uid':row['uid']=502
                if mode=='ruid':row['ruid']=502
                if mode=='held_pgid':row['pgid']=held+1
            else:
                if mode=='member_pgid':row['pgid']=held+1
                if mode=='malformed_uid':row['uid']=True
                if mode=='extra_field':row['future']=0
                if mode=='wrong_pid':row['pid']=pid+1
            return row
        def short_info(self,pid):
            touch('short_info',pid)
            raise AssertionError('group scope requires exact full group rows, not unrelated short rows')
    observer=object.__new__(module.NativeObservation)
    observer.darwin=Primitives()
    observer._held={held:{key:copy.deepcopy(leader[key]) for key in
                         ('pid','pgid','sid','birth','uid','ruid')}}
    def getsid(pid):
        touch('getsid',pid)
        if pid!=held:
            raise AssertionError('group scope must not query a non-held PID session')
        return held
    def group_snapshot(anchor):
        state.attempt+=1;state.group_reads=0
        calls.append(('fresh_group_attempt',state.attempt,None))
        answer=module.NativeObservation.group_snapshot(observer,anchor)
        state.payloads.append(copy.deepcopy(answer))
        if mode=='late':state.now=min(deadline,now+250000000) if deadline is not None else now+250000000
        return answer
    def forbidden_full_session(anchor):
        raise AssertionError('Darwin group scope must not invoke a whole-session snapshot')
    observer.group_snapshot=group_snapshot
    observer.snapshot=forbidden_full_session
    def sleep(seconds):state.sleeps+=1;state.now+=int(seconds*10**9)
    with patch.object(module,'os',SimpleNamespace(getsid=getsid,getuid=lambda:501,geteuid=lambda:501)), \
            patch.object(module,'time',SimpleNamespace(monotonic_ns=lambda:state.now,sleep=sleep)):
        result=module.observed_snapshot(observer,leader,deadline_ns=deadline,phase=phase,scope=scope)
    return result,calls,state


def group_phase_native_predicates(module, group, reap, *, eperm=False, negative=False):
    """Call the actual acceptance predicate over literal observations only."""
    pid=26517
    native=dict(pid=pid,native_exit=7,timed_out=False,wall_seconds=.1,
                ownership='start_new_session=True',
                descendant_cleanup='owned POSIX session has no live members after stage')
    events=[dict(operation='spawn',pid=pid),
            dict(operation='waitid',code=1,flags=10,monotonic_ns=1),
            dict(operation='group_signal_entry',pid=pid,signal=9,reap_started=False,
                 snapshot=group,monotonic_ns=2)]
    if eperm:
        events.append(dict(operation='group_signal_error',errno=1,error_type='PermissionError',
                           snapshot=group,monotonic_ns=3))
    events.append(dict(operation='terminal_wait_entry',first=True,signal_started=True,
                       snapshot=reap,monotonic_ns=4))
    row=dict(events=events,emergency=False,controller_threads_joined=True,actual_direct_exit=7,
             native=None if negative else native,error=dict(type='RuntimeError') if negative else None,
             role=ROLES[0],elapsed_ns=100000000,other_group_live_at_owner_return=True,
             fixture_expiry_observed=True)
    fake_os=SimpleNamespace(CLD_EXITED=1,CLD_KILLED=2,CLD_DUMPED=3,CLD_STOPPED=5,
                            WNOWAIT=8,WEXITED=2,WSTOPPED=16)
    with patch.object(module,'os',fake_os),patch.object(module,'sys',SimpleNamespace(platform='linux')):
        return module.check(row,'other_group_persistent' if negative else 'no_descendant' if eperm else 'same_group_descendant')


class LiteralStateTests(unittest.TestCase):
    output = None
    rows = []

    def test_all_literal_semantic_scenarios(self):
        for index, role in enumerate(ROLES[:2]):
            module = load(ROOT/role, 'independent_v3_state_owner_' + str(index))
            for mode in POSITIVE + NEGATIVE:
                with self.subTest(role=role, mode=mode):
                    root = self.output / (str(index)+'-'+mode)
                    root.mkdir()
                    result, error, events, state = exercise(module, mode, root)
                    if mode in POSITIVE:
                        self.assertIsNone(error)
                        native, summary = result
                        self.assertEqual(native['native_exit'], 7)
                        self.assertTrue(summary['session_zero_live_verified'])
                        self.assertEqual(native['descendant_cleanup'], 'owned POSIX session has no live members after stage')
                        self.assertEqual(state.signals, 1)
                        self.assertTrue(state.reaped)
                        if mode in ('stop_then_terminal','stop_only_none_then_terminal'):
                            self.assertEqual(state.stops, 1)
                            self.assertGreaterEqual(state.observations, 3)
                    else:
                        self.assertIsNotNone(error)
                        self.assertIsNone(result)
                        primary = {
                            'missing_feature':('RuntimeError','nonreaping POSIX terminal-event'),
                            'unknown_event':('RuntimeError','unsupported nonterminal'),
                            'wrong_event_pid':('RuntimeError','another process identity'),
                            'eperm_live_group':('PermissionError','literal group permission'),
                            'eperm_unverified_zombie':('PermissionError','literal group permission'),
                            'wrong_sid':('RuntimeError','direct-child native identity unavailable'),
                            'wrong_uid':('RuntimeError','direct-child native identity unavailable'),
                            'callback_error':('RuntimeError','literal callback failure'),
                            'observation_error':('OSError','literal terminal observation'),
                            'terminal_wait_error':('OSError','literal first terminal wait'),
                            'shared_cleanup_deadline':('RuntimeError','shared20-second deadline')}
                        if mode=='preflight_budget_exhausted':
                            primary[mode] = ('RuntimeError','owned process census/cleanup deadline exhausted')
                        kind, text = primary.get(mode,('RuntimeError','closure remains unknown'))
                        self.assertEqual(type(error).__name__,kind)
                        self.assertIn(text,str(error))
                        cause = {'unknown_census':'literal unreadable census','wrong_birth':'held leader birth',
                                 'truncated_census':'native session census is incomplete',
                                 'boolean_count_census':'native session census is incomplete',
                                 'foreign_session_uid':'foreign-UID live member'}
                        if mode in cause:
                            self.assertIsNotNone(error.__cause__)
                            self.assertIn(cause[mode],str(error.__cause__))
                        if mode in ('wrong_sid','wrong_uid'):
                            chain = error_chain(error)
                            self.assertTrue(any(row['type']=='RuntimeError' and
                                'direct child native PID/birth/private-session identity differs' in row['message'] for row in chain))
                            self.assertEqual(state.signals,0)
                            self.assertEqual(state.direct_signals,0)
                            self.assertEqual(state.callback_invocations,0)
                            self.assertIs(getattr(error,'ownership_boundary_closed',None),False)
                            self.assertIs(error.closure_summary['session_zero_live_verified'],False)
                        if mode in ('missing_feature','preflight_budget_exhausted'):
                            self.assertFalse(state.launched)
                            self.assertFalse((root/'literal.raw').exists())
                        if mode == 'shared_cleanup_deadline':
                            self.assertLessEqual(state.wait_timeout, 1.)
                    self.assertEqual(state.borrowed_attempts,0)
                    self.assertEqual(state.order_violations,0)
                    self.rows.append(dict(role=role, mode=mode, evidence_kind='simulated_literal_state',
                        passed=True, error_type=None if error is None else type(error).__name__, events=events,
                        error_chain=[] if error is None else error_chain(error),
                        borrowed_signal_attempts=state.borrowed_attempts,order_violations=state.order_violations,
                        group_signal_count=state.signals, terminal_reaped_in_simulation=state.reaped))

    def test_linux_status_real_and_effective_uid_are_distinct_evidence(self):
        stat_bytes = b'424242 (independent (name) with spaces) S 999999 424242 424242 '+b'0 '*15+b'123456\n'
        equal = b'Name:\tfixture\nPid:\t424242\nUid:\t1000\t1000\t1000\t1000\n'
        unequal = b'Name:\tfixture\nPid:\t424242\nUid:\t1001\t1000\t1000\t1000\n'
        bad = {
            'missing_uid':b'Name:\tfixture\nPid:\t424242\n',
            'duplicate_uid':equal+b'Uid:\t1000\t1000\t1000\t1000\n',
            'wrong_width':b'Pid:\t424242\nUid:\t1000\t1000\t1000\n',
            'wrong_pid':equal.replace(b'Pid:\t424242',b'Pid:\t424243'),
            'invalid_uid':equal.replace(b'Uid:\t1000',b'Uid:\t+1000'),
            'uid_overflow':equal.replace(b'Uid:\t1000',b'Uid:\t4294967296'),
            'oversize_status':equal+b'x'*(8193-len(equal)),
            'changed_directory':equal,
            'missing_pid':b'Uid:\t1000\t1000\t1000\t1000\n',
            'duplicate_pid':equal+b'Pid:\t424242\n'}
        for index,role in enumerate(ROLES[:2]):
            module = load(ROOT/role,'independent_v3_linux_uid_'+str(index))
            for mode,status in (('equal_uid',equal),('unequal_uid',unequal),*bad.items()):
                with self.subTest(role=role,mode=mode):
                    counters = dict(stats=0)
                    class FileBoundary:
                        def __init__(self,parts):
                            self.parts = parts
                        def __truediv__(self,value):
                            return FileBoundary(self.parts+(str(value),))
                        def stat(self):
                            self_check = self.parts == ('/proc',str(PID))
                            if not self_check:
                                raise AssertionError('unexpected mocked process-directory read')
                            counters['stats'] += 1
                            return SimpleNamespace(st_dev=1,st_ino=77+(1 if mode=='changed_directory' and counters['stats']>1 else 0),st_uid=1000)
                        def open(self,access):
                            if access!='rb' or self.parts[:-1]!=('/proc',str(PID)):
                                raise AssertionError('unexpected mocked Linux file access')
                            if self.parts[-1]=='stat':
                                return io.BytesIO(stat_bytes)
                            if self.parts[-1]=='status':
                                return io.BytesIO(status)
                            raise AssertionError('unexpected mocked Linux file')
                    adapter = object.__new__(module._LinuxProcesses)
                    with patch.object(module,'Path',lambda name:FileBoundary((str(name),))), \
                            patch.object(module,'os',SimpleNamespace(getuid=lambda:1000,getpid=lambda:PARENT)):
                        if mode in bad:
                            with self.assertRaises(RuntimeError):
                                adapter._info(PID)
                        else:
                            row = adapter._info(PID)
                            self.assertEqual((row['pid'],row['uid'],row['ruid']),(PID,1000,1001 if mode=='unequal_uid' else 1000))
                            if mode=='unequal_uid':
                                with self.assertRaisesRegex(RuntimeError,'direct child native PID/birth/private-session identity differs'):
                                    adapter.leader(PID)
                            else:
                                self.assertEqual(adapter.leader(PID)['ruid'],1000)
                    self.rows.append(dict(role=role,mode='linux_status_'+mode,passed=True,
                        evidence_kind='simulated_Linux_file_boundary',actual_proc_files_opened=0))

    def test_darwin_zombie_sid_uses_only_a_matching_unreaped_launch_anchor(self):
        observer = object.__new__(NativeObservation)
        observer._held = {}
        current = dict(pid=PID,pgid=PID,status=2,uid=1000,ruid=1000,start_sec=100,start_usec=200)
        observer.darwin = SimpleNamespace(info=lambda pid:dict(current))
        def getsid(pid):
            self.assertEqual(pid,PID)
            if current['status']==5:
                raise ProcessLookupError(errno.ESRCH,'literal XNU getsid dead-process lookup')
            return PID
        with patch.object(observations.os,'getsid',getsid,create=True), \
                patch.object(observations.os,'getuid',lambda:1000,create=True), \
                patch.object(observations.os,'geteuid',lambda:1000,create=True):
            leader = observer.identity(PID)
            observer.hold(leader)
            self.assertEqual(leader['sid'],PID)
            self.rows.append(dict(mode='darwin_live_launch_anchor',passed=True,evidence_kind='simulated_observer_boundary'))
            current['status'] = 5
            before_path = HERE/'frozen/observations_before_zombie_anchor.py'
            self.assertEqual(observations.digest(before_path),'fda65adc19fd2a150c6911ba233b653106bb4923211d28fad1387eac9644aacc')
            before_module = load(before_path,'independent_observer_before_zombie_anchor')
            before = object.__new__(before_module.NativeObservation)
            before.darwin = observer.darwin
            before._held = dict(observer._held)
            with self.assertRaises(ProcessLookupError) as failure:
                before.identity(PID)
            self.assertEqual(failure.exception.errno,errno.ESRCH)
            self.rows.append(dict(mode='darwin_before_observer_getsid_zombie_failure',passed=True,
                evidence_kind='simulated_historical_observer_negative',source_sha256='fda65adc19fd2a150c6911ba233b653106bb4923211d28fad1387eac9644aacc'))
            self.assertEqual(observer.identity(PID)['sid'],PID)
            self.rows.append(dict(mode='darwin_matching_held_zombie',passed=True,evidence_kind='simulated_observer_boundary'))
            for key,value in (('start_usec',201),('uid',1001),('pgid',PID+1)):
                old = current[key]
                current[key] = value
                with self.assertRaisesRegex(RuntimeError,'held Darwin zombie native identity changed'):
                    observer.identity(PID)
                current[key] = old
                self.rows.append(dict(mode='darwin_changed_'+key,passed=True,evidence_kind='simulated_observer_boundary'))
            observer.release(PID)
            self.assertIsNone(observer.identity(PID)['sid'])
            self.rows.append(dict(mode='darwin_released_anchor_not_reused',passed=True,evidence_kind='simulated_observer_boundary'))

    def test_candidate_uses_the_checked_closure_summary(self):
        candidate = load(ROOT/ROLES[2], 'independent_v3_state_candidate')
        native = dict(pid=PID, native_exit=7, timed_out=False, wall_seconds=.1,
                      descendant_cleanup='owned POSIX session has no live members after stage', ownership='start_new_session=True')
        proof = dict(leader_identity={}, terminal_observation={}, group_signal='verified_nonsignalable',
                     group_zombie_only_verified=True, session_zero_live_verified=True,census_attempts=2,stop_notifications_consumed=1)
        for closed in (True, False):
            backend = SimpleNamespace(_run_posix=lambda *args,**kw:(native.copy(),dict(proof,session_zero_live_verified=closed)))
            ticks = iter((0.,.1))
            with patch.object(candidate,'os',SimpleNamespace(name='posix')), patch.object(candidate,'_posix_module',return_value=backend), \
                    patch.object(candidate,'time',SimpleNamespace(monotonic=lambda:next(ticks))):
                if closed:
                    result = candidate.run_owned(['literal'],cwd='unused',environment={},logpath='unused',timeout=8.)
                    self.assertTrue(result['ownership_boundary_closed'])
                    self.assertFalse(result['all_descendants_os_reaped'])
                    self.assertEqual(result['native_exit'],7)
                else:
                    with self.assertRaises(RuntimeError):
                        candidate.run_owned(['literal'],cwd='unused',environment={},logpath='unused',timeout=8.)
            self.rows.append(dict(role=ROLES[2], mode='candidate_summary_'+str(closed), passed=True,
                                  evidence_kind='simulated_candidate_projection', native_backend_mocked=True))

    def test_sigchld_autoreap_rejected_before_native_probe_or_spawn(self):
        for index, role in enumerate(ROLES[:2]):
            module = load(ROOT/role, 'independent_v3_sigchld_owner_' + str(index))
            fake_os = SimpleNamespace(name='posix', waitid=lambda *args:(_ for _ in ()).throw(AssertionError('probe after auto-reap')),
                P_PID=1, WEXITED=2, WNOHANG=4, WNOWAIT=8, WSTOPPED=16, CLD_EXITED=1, CLD_KILLED=2, CLD_DUMPED=3, CLD_STOPPED=5)
            with patch.object(module,'os',fake_os), patch.object(module,'sys',SimpleNamespace(platform='darwin')), \
                    patch.object(module,'signal',SimpleNamespace(SIGCHLD=17,SIG_IGN=1,getsignal=lambda _:1)), \
                    patch.object(module,'_processes') as census:
                with self.assertRaises(RuntimeError):
                    module._posix_features(8.)
                census.assert_not_called()
            self.rows.append(dict(role=role,mode='SIGCHLD_ignored',passed=True,evidence_kind='simulated_prelaunch_admission'))

    def test_independent_fresh_census_retry_literals(self):
        oracle = observations.read_json(HERE/'census_retry_oracles_v1.json')
        self.assertEqual(observations.digest(HERE/'census_retry_oracles_v1.json'), observations.RETRY_ORACLE_SHA)
        self.assertEqual((oracle['max_attempts'],oracle['observation_subcap_ns'],oracle['retry_sleep_ns']),
                         (8,250000000,5000000))
        self.assertEqual((oracle['expected_simulated_rows'],oracle['expected_test_methods']),(93,6))
        names = (
            'unrelated_enoent_then_fresh_stable_live_census',
            'all_fresh_censuses_unstable_remain_unknown',
            'late_nominal_complete_census_rejected',
            'held_leader_loss_or_changed_identity_not_retried',
            'permission_or_malformed_record_not_retried',
            'stable_complete_zero_census_is_observed',
            'expired_positive_clock_and_ineligible_negative_diagnostic',
            'negative_diagnostic_deadline_cannot_restart')
        self.assertEqual(oracle['cases'],list(names))
        leader = dict(pid=PID,pgid=PID,sid=PID,birth=[100,200],uid=1000,ruid=1000,zombie=True,stopped=False)
        descendant = dict(leader,pid=PID+1,birth=[101,200],zombie=False)
        native_checks = load(HERE/'run_native.py','independent_retry_acceptance_predicates')
        fake_os = SimpleNamespace(CLD_EXITED=1,CLD_KILLED=2,CLD_DUMPED=3,CLD_STOPPED=5,
            WNOWAIT=8,WEXITED=2,WSTOPPED=16)

        def predicate(group, reap, *, negative=False, forged_result=False):
            native = dict(pid=PID,native_exit=7,timed_out=False,wall_seconds=.1,
                ownership='start_new_session=True',descendant_cleanup='owned POSIX session has no live members after stage')
            events = [dict(operation='spawn',pid=PID),
                dict(operation='waitid',code=1,flags=10,monotonic_ns=1),
                dict(operation='group_signal_entry',pid=PID,signal=9,reap_started=False,snapshot=group,monotonic_ns=2),
                dict(operation='terminal_wait_entry',first=True,signal_started=True,snapshot=reap,monotonic_ns=3)]
            row = dict(events=events,emergency=False,controller_threads_joined=True,actual_direct_exit=7,
                native=None if negative else native,error=dict(type='RuntimeError') if negative else None,
                role=ROLES[0],elapsed_ns=100000000,other_group_live_at_owner_return=True,fixture_expiry_observed=True)
            if forged_result:
                row.update(native=native,error=None)
            with patch.object(native_checks,'os',fake_os),patch.object(native_checks,'sys',SimpleNamespace(platform='linux')):
                return native_checks.check(row,'other_group_persistent' if negative else 'same_group_descendant')

        def scenario(mode, *, now=0, deadline=1000000000, phase=observations.POSITIVE_PHASE):
            state = SimpleNamespace(now=now,attempt=0)
            calls = []
            observer = object.__new__(NativeObservation)
            observer._held = {PID:{k:leader[k] for k in ('pid','pgid','sid','birth','uid','ruid')}}
            observer.darwin = None
            def identity(pid):
                calls.append(('identity',state.attempt,pid))
                if mode=='leader_missing' and pid==PID:
                    raise FileNotFoundError(errno.ENOENT,'literal held leader missing',f'/proc/{PID}/stat')
                if mode.startswith('leader_changed_') and pid==PID:
                    key=mode.removeprefix('leader_changed_')
                    return dict(leader,**{key:[200,300] if key=='birth' else PID+9 if key=='sid' else 1001})
                if pid==2333:
                    if mode in ('vanish','unstable'):
                        raise FileNotFoundError(errno.ENOENT,'literal unrelated listed PID vanished','/proc/2333/stat')
                    if mode=='permission':
                        raise PermissionError(errno.EACCES,'literal unrelated PID denied','/proc/2333/stat')
                    if mode=='malformed':
                        raise ValueError('literal malformed independent stat')
                if pid==PID:
                    return copy.deepcopy(leader)
                if pid in (1111,PID+1):
                    return dict(descendant,pid=pid,birth=[pid,200])
                raise AssertionError('unexpected literal independent PID')
            def pids():
                calls.append(('inventory',state.attempt))
                if mode=='vanish' and state.attempt==1:
                    return [1111,2333,PID,PID+1]
                if mode in ('unstable','permission','malformed'):
                    return [2333,PID]
                return [PID,PID+1] if mode in ('vanish','live') else [PID]
            def snapshot(anchor):
                state.attempt+=1
                calls.append(('fresh_attempt',state.attempt))
                answer=NativeObservation.snapshot(observer,anchor)
                if mode=='late':
                    state.now=deadline
                return answer
            observer.identity=identity
            observer._linux_pids=pids
            observer.snapshot=snapshot
            fake_time=SimpleNamespace(monotonic_ns=lambda:state.now,
                sleep=lambda seconds:setattr(state,'now',state.now+int(seconds*10**9)))
            with patch.object(observations,'time',fake_time):
                result=observations.observed_snapshot(observer,leader,deadline_ns=deadline,phase=phase)
            return result,calls,state

        def unknown(result):
            self.assertIs(result['complete'],False)
            self.assertIs(result['unknown'],True)
            self.assertIs(result['positive_acceptance_eligible'],False)
            self.assertNotIn('group_live',result)
            self.assertNotIn('session_live',result)
            checks=predicate(result,result)
            self.assertFalse(checks['live_group_before_signal'])
            self.assertFalse(checks['zero_sid_before_reap'])

        for index,name in enumerate(names):
            with self.subTest(case=name):
                details=[]
                if index==0:
                    result,calls,state=scenario('vanish')
                    self.assertIs(result['complete'],True)
                    self.assertEqual((result['group_live'],result['session_live']),(1,1))
                    self.assertEqual(len(result['observation_attempts']),2)
                    self.assertEqual(result['observation_attempts'][0]['vanished_pid'],2333)
                    self.assertEqual(result['observation_attempts'][0]['error_type'],'FileNotFoundError')
                    self.assertTrue(result['positive_acceptance_eligible'])
                    self.assertIn(('identity',1,1111),calls)
                    self.assertNotIn(('identity',2,1111),calls)
                    self.assertEqual(sum(c[0]=='inventory' and c[1]==2 for c in calls),3)
                    self.assertTrue(predicate(result,result)['live_group_before_signal'])
                    details=[result]
                elif index==1:
                    result,calls,state=scenario('unstable')
                    unknown(result)
                    self.assertEqual((len(result['observation_attempts']),state.attempt),(8,8))
                    self.assertEqual(state.now,35000000)
                    self.assertEqual(sum(c[0]=='fresh_attempt' for c in calls),8)
                    details=[result]
                elif index==2:
                    result,calls,state=scenario('late',deadline=50000000)
                    unknown(result)
                    self.assertEqual(state.attempt,1)
                    self.assertTrue(result['observation_attempts'][0]['late'])
                    self.assertEqual(result['observation_finished_ns'],50000000)
                    details=[result]
                elif index==3:
                    for mode in ('leader_missing','leader_changed_birth','leader_changed_sid','leader_changed_uid'):
                        result,calls,state=scenario(mode)
                        unknown(result)
                        self.assertEqual((state.attempt,len(result['observation_attempts'])),(1,1))
                        self.assertFalse(result['observation_attempts'][0]['retryable'])
                        self.assertFalse(any(c[0]=='inventory' for c in calls))
                        details.append(result)
                elif index==4:
                    for mode in ('permission','malformed'):
                        result,calls,state=scenario(mode)
                        unknown(result)
                        self.assertEqual(state.attempt,1)
                        self.assertFalse(result['observation_attempts'][0]['retryable'])
                        details.append(result)
                elif index==5:
                    result,calls,state=scenario('zero')
                    self.assertTrue(result['complete'])
                    self.assertEqual((result['group_live'],result['session_live']),(0,0))
                    self.assertTrue(result['group_zombie_only'])
                    self.assertTrue(result['positive_acceptance_eligible'])
                    self.assertEqual(state.attempt,1)
                    self.assertEqual(sum(c[0]=='inventory' for c in calls),3)
                    self.assertTrue(predicate(result,result)['zero_sid_before_reap'])
                    details=[result]
                elif index==6:
                    clock=SimpleNamespace(monotonic_ns=lambda:20165543043)
                    with patch.object(observations,'time',clock):
                        positive=observations.observation_context(0,480000000000,first_wait=True,wait_timeout=0)
                        negative=observations.observation_context(0,480000000000,first_wait=True,wait_timeout=0,expected_negative=True)
                    self.assertEqual(positive,dict(phase=observations.POSITIVE_PHASE,deadline_ns=20000000000))
                    self.assertEqual(negative,dict(phase=observations.NEGATIVE_PHASE,deadline_ns=38000000000))
                    result,calls,state=scenario('live',now=20165543043,deadline=positive['deadline_ns'],phase=positive['phase'])
                    unknown(result)
                    self.assertEqual(state.attempt,0)
                    diagnostic,calls,state=scenario('live',now=20165543043,deadline=negative['deadline_ns'],phase=negative['phase'])
                    self.assertIs(diagnostic['complete'],True)
                    self.assertEqual(diagnostic['session_live'],1)
                    self.assertIs(diagnostic['positive_acceptance_eligible'],False)
                    self.assertTrue(all(predicate(diagnostic,diagnostic,negative=True).values()))
                    self.assertFalse(predicate(diagnostic,diagnostic,negative=True,forged_result=True)['real_unclosed_failure'])
                    self.assertFalse(predicate(diagnostic,diagnostic)['positive_reap_observation_budget'])
                    details=[result,diagnostic]
                else:
                    result,calls,state=scenario('live',now=38000000000,deadline=38000000000,phase=observations.NEGATIVE_PHASE)
                    unknown(result)
                    self.assertEqual((state.attempt,len(result['observation_attempts'])),(0,0))
                    self.assertFalse(predicate(result,result,negative=True)['live_sid_before_failed_reap'])
                    late,calls,state=scenario('late',now=37990000000,deadline=38000000000,phase=observations.NEGATIVE_PHASE)
                    unknown(late)
                    self.assertEqual(state.attempt,1)
                    self.assertEqual(late['original_deadline_ns'],38000000000)
                    details=[result,late]
                self.rows.append(dict(mode=name,passed=True,evidence_kind='simulated_independent_census_read_boundary',
                    actual_POSIX_processes_started=0,actual_signals_sent=0,snapshots=details))


    def test_independent_darwin_fresh_complete_census_retry_literals(self):
        oracle=observations.read_json(HERE/'darwin_census_retry_oracles_v1.json')
        self.assertEqual(observations.digest(HERE/'darwin_census_retry_oracles_v1.json'),
                         'a9bed3b64c9e3e35190ec7fa406f30e87a789f53ca9023cc53be4901a6477795')
        self.assertEqual(oracle['schema'],'independent_darwin_census_fresh_retry_oracles_v1')
        self.assertEqual((oracle['max_attempts'],oracle['observation_subcap_ns'],oracle['retry_sleep_ns']),
                         (8,250000000,5000000))
        self.assertEqual((oracle['original_simulated_rows'],oracle['new_simulated_rows'],
                          oracle['expected_simulated_rows'],oracle['expected_test_methods']),(93,8,101,7))
        names=(
            'known_nonheld_zero_byte_ESRCH_requires_fresh_complete_census',
            'getsid_absence_proof_and_inventory_change_restart_whole_census',
            'fresh_complete_census_retains_persistent_other_group_live_member',
            'eight_fresh_instabilities_remain_UNKNOWN_without_partial_acceptance',
            'expired_and_late_complete_censuses_remain_UNKNOWN',
            'held_loss_or_changed_identity_is_first_error_UNKNOWN',
            'permission_protocol_partial_and_unproved_disappearance_are_not_retried',
            'discard_partial_rows_preserve_phase_and_no_deadline_eligibility')
        self.assertEqual([r['name'] for r in oracle['cases']],list(names))
        def run(mode,**kwargs):return darwin_retry_literal_scenario(observations,mode,**kwargs)
        def unknown(result,attempts):
            self.assertIs(result['complete'],False);self.assertIs(result['unknown'],True)
            self.assertIs(result['positive_acceptance_eligible'],False)
            self.assertNotIn('session_live',result);self.assertNotIn('group_live',result)
            self.assertEqual(len(result['observation_attempts']),attempts)
        def zero(result,attempts):
            self.assertIs(result['complete'],True)
            self.assertEqual((result['group_live'],result['session_live']),(0,0))
            self.assertIs(result['group_zombie_only'],True)
            self.assertEqual(result['session_members'],[])
            self.assertIs(result['positive_acceptance_eligible'],True)
            self.assertEqual(len(result['observation_attempts']),attempts)
            self.assertLess(result['observation_finished_ns'],result['effective_deadline_ns'])
        for index,name in enumerate(names):
            with self.subTest(case=name):
                details=[]
                if index==0:
                    for mode in ('short_vanish','full_vanish'):
                        result,calls,state=run(mode);zero(result,2)
                        self.assertEqual(state.attempt,2)
                        self.assertIs(result['observation_attempts'][0]['retryable'],True)
                        self.assertEqual(result['observation_attempts'][0]['errno'],3)
                        self.assertEqual(result['observation_attempts'][0]['error_type'],'ProcessLookupError')
                        self.assertEqual(result['observation_attempts'][0]['vanished_pid'],16518)
                        self.assertEqual(sum(c[0]=='fresh_attempt' for c in calls),2)
                        self.assertTrue(any(c[0]=='group_inventory' and c[1]==2 for c in calls))
                        self.assertTrue(any(c[0]=='global_inventory' and c[1]==2 for c in calls))
                        details.append(result)
                elif index==1:
                    for mode in ('getsid_vanish','group_change','session_change'):
                        result,calls,state=run(mode);zero(result,2)
                        self.assertIs(result['observation_attempts'][0]['retryable'],True)
                        self.assertEqual(state.attempt,2)
                        if mode=='getsid_vanish':
                            self.assertGreaterEqual(sum(c[0]=='global_inventory' and c[1]==1 for c in calls),2)
                        details.append(result)
                elif index==2:
                    result,calls,state=run('live_after')
                    self.assertIs(result['complete'],True);self.assertEqual(len(result['observation_attempts']),2)
                    self.assertEqual((result['group_live'],result['session_live']),(0,1))
                    self.assertEqual([(r['pid'],r['pgid'],r['sid']) for r in result['session_members']],[(16519,16519,16517)])
                    details=[result]
                elif index==3:
                    result,calls,state=run('unstable');unknown(result,8)
                    self.assertEqual((state.attempt,state.sleeps),(8,7))
                    self.assertEqual(sum(c[0]=='fresh_attempt' for c in calls),8)
                    self.assertTrue(all(r['retryable'] for r in result['observation_attempts']))
                    details=[result]
                elif index==4:
                    expired,calls,state=run('zero',now=9000000000,deadline=9000000000)
                    unknown(expired,0);self.assertEqual(state.attempt,0);self.assertEqual(calls,[])
                    late,calls,state=run('late');unknown(late,1)
                    self.assertIs(late['observation_attempts'][0]['late'],True)
                    details=[expired,late]
                elif index==5:
                    for mode in ('held_info','held_short','held_getsid','held_birth','held_post_change',
                                 'held_pgid','held_sid','held_uid','held_ruid','held_group_absent'):
                        result,calls,state=run(mode);unknown(result,1)
                        self.assertEqual(state.attempt,1)
                        self.assertIs(result['observation_attempts'][0]['retryable'],False)
                        details.append(result)
                elif index==6:
                    for mode in ('info_permission','short_permission','protocol','partial63','wrong_message_pid',
                                 'generic_ESRCH','getsid_still_listed','getsid_probe_permission','generic_instability',
                                 'malformed_row','row_conflict'):
                        result,calls,state=run(mode);unknown(result,1)
                        self.assertEqual(state.attempt,1)
                        self.assertIs(result['observation_attempts'][0]['retryable'],False)
                        details.append(result)
                else:
                    fresh,calls,state=run('partial_then_zero');zero(fresh,2)
                    self.assertIn(('info',1,16516),calls);self.assertNotIn(('info',2,16516),calls)
                    diagnostic,calls,state=run('zero',phase=observations.NEGATIVE_PHASE)
                    self.assertIs(diagnostic['complete'],True);self.assertIs(diagnostic['positive_acceptance_eligible'],False)
                    no_clock,calls,state=run('short_vanish',deadline=None);unknown(no_clock,1)
                    details=[fresh,diagnostic,no_clock]
                self.rows.append(dict(mode=name,passed=True,evidence_kind='simulated_independent_Darwin_census_read_boundary',
                    actual_POSIX_processes_started=0,actual_signals_sent=0,snapshots=details))


    def test_darwin_process_group_observation_scope_literals(self):
        path=HERE/'darwin_group_phase_oracles_v1.json'
        self.assertEqual(observations.digest(path),'1b2dc20a907d309914d88eb4c012b7d17011cd0678992c56c6a94565306d8141')
        oracle=observations.read_json(path)
        self.assertEqual(oracle['schema'],'independent_darwin_process_group_phase_oracles_v1')
        self.assertEqual((oracle['old_literal_rows'],oracle['old_unittest_methods'],
                          oracle['new_literal_rows'],oracle['new_unittest_methods'],
                          oracle['expected_literal_rows'],oracle['expected_unittest_methods']),(101,7,12,1,113,8))
        self.assertEqual((oracle['retry_attempts'],oracle['retry_subcap_ns']),(8,250000000))
        self.assertEqual(oracle['scopes'],['whole_session','process_group'])
        names=(
            'zombie_group_complete_without_unrelated_global_reads',
            'live_group_is_complete_but_EPERM_ineligible',
            'held_birth_change_fatal_unknown',
            'held_UID_or_RUID_change_fatal_unknown',
            'held_or_member_group_identity_change_fatal_unknown',
            'held_leader_absent_before_or_after_group_fatal',
            'group_roster_change_then_fresh_complete_retry',
            'eight_group_roster_changes_remain_unknown',
            'nonheld_ESRCH_fresh_retry_held_loss_fatal',
            'malformed_or_access_error_fatal_unknown',
            'preflight_or_postread_deadline_exhaustion_ineligible',
            'group_only_cannot_close_session_or_replace_persistent_negative')
        self.assertEqual([row['name'] for row in oracle['rows']],list(names))
        native_checks=load(HERE/'run_native.py','group_phase_independent_acceptance_predicates')
        def run(mode,**kwargs):return darwin_group_literal_scenario(observations,mode,**kwargs)
        def unknown(result,attempts):
            self.assertIs(result['complete'],False)
            self.assertIs(result['unknown'],True)
            self.assertIs(result['positive_acceptance_eligible'],False)
            self.assertEqual(result['observation_scope'],'process_group')
            self.assertIsNone(result.get('group_live'))
            self.assertIsNone(result.get('session_live'))
            self.assertEqual(len(result['observation_attempts']),attempts)
        def group(result,state,*,live=0,attempts=1):
            self.assertIs(result['complete'],True)
            self.assertEqual(result['observation_scope'],'process_group')
            self.assertIs(result['positive_acceptance_eligible'],True)
            self.assertIs(type(result['group_live']),int)
            self.assertEqual(result['group_live'],live)
            self.assertIs(result['group_zombie_only'],live==0)
            self.assertIs(result['held_leader']['zombie'],True)
            self.assertIsNone(result['session_live'])
            self.assertIsNone(result['session_members'])
            self.assertEqual(len(result['observation_attempts']),attempts)
            self.assertLess(result['observation_finished_ns'],result['effective_deadline_ns'])
            self.assertEqual(set(state.payloads[-1]),{'complete','held_leader','group_live',
                'group_zombie_only','session_live','session_members'})
        full=dict(complete=True,observation_scope='whole_session',session_live=0,session_members=[],
                  group_live=0,group_zombie_only=True,
                  held_leader=dict(pid=26517,pgid=26517,sid=26517,birth=[1794299426,355078],
                                   uid=501,ruid=501,zombie=True,stopped=False),
                  positive_acceptance_eligible=True,observation_phase=observations.POSITIVE_PHASE)
        for index,name in enumerate(names):
            with self.subTest(case=name):
                details=[]
                if index==0:
                    result,calls,state=run('zombie');group(result,state)
                    self.assertEqual(state.attempt,1)
                    self.assertTrue(any(row[0]=='group_inventory' for row in calls))
                    self.assertFalse(any(row[0] in ('global_inventory','short_info') for row in calls))
                    self.assertFalse(any(row[0]=='getsid' and row[2]!=26517 for row in calls))
                    check=group_phase_native_predicates(native_checks,result,full,eperm=True)
                    self.assertTrue(check['eperm_native_zombie_only'])
                    self.assertTrue(check['eperm_observation_budget'])
                    self.assertTrue(check['zero_sid_before_reap'])
                    details=[result]
                elif index==1:
                    result,calls,state=run('live_group');group(result,state,live=1)
                    check=group_phase_native_predicates(native_checks,result,full,eperm=True)
                    self.assertFalse(check['eperm_native_zombie_only'])
                    self.assertTrue(check['eperm_observation_budget'])
                    self.assertTrue(group_phase_native_predicates(native_checks,result,full)['live_group_before_signal'])
                    details=[result]
                elif index in (2,3,4,5):
                    modes=(('birth','post_birth'),('uid','ruid'),('held_pgid','member_pgid'),
                           ('held_absent_before','held_absent_after'))[index-2]
                    for mode in modes:
                        result,calls,state=run(mode);unknown(result,1)
                        self.assertEqual(state.attempt,1)
                        self.assertFalse(result['observation_attempts'][0]['retryable'])
                        details.append(result)
                elif index==6:
                    result,calls,state=run('roster_change');group(result,state,attempts=2)
                    self.assertEqual((state.attempt,state.sleeps),(2,1))
                    self.assertTrue(result['observation_attempts'][0]['retryable'])
                    self.assertIn(('info',1,26518),calls)
                    self.assertNotIn(('info',2,26518),calls)
                    self.assertEqual(sum(row[0]=='fresh_group_attempt' for row in calls),2)
                    self.assertTrue(all(row[0]!='global_inventory' for row in calls))
                    details=[result]
                elif index==7:
                    result,calls,state=run('unstable');unknown(result,8)
                    self.assertEqual((state.attempt,state.sleeps),(8,7))
                    self.assertTrue(all(row['retryable'] for row in result['observation_attempts']))
                    self.assertEqual(sum(row[0]=='fresh_group_attempt' for row in calls),8)
                    self.assertFalse(any(row[0]=='global_inventory' for row in calls))
                    details=[result]
                elif index==8:
                    result,calls,state=run('vanish');group(result,state,attempts=2)
                    first=result['observation_attempts'][0]
                    self.assertEqual((first['error_type'],first['errno'],first['vanished_pid']),
                                     ('ProcessLookupError',3,26518))
                    self.assertTrue(first['retryable'])
                    self.assertIn(('info',1,26518),calls)
                    self.assertNotIn(('info',2,26518),calls)
                    self.assertTrue(any(row[0]=='group_inventory' and row[1]==2 for row in calls))
                    missing,calls,state=run('held_lost');unknown(missing,1)
                    self.assertFalse(missing['observation_attempts'][0]['retryable'])
                    details=[result,missing]
                elif index==9:
                    for mode in ('malformed_uid','extra_field','wrong_pid','permission','protocol','partial'):
                        result,calls,state=run(mode);unknown(result,1)
                        self.assertEqual(state.attempt,1)
                        self.assertFalse(result['observation_attempts'][0]['retryable'])
                        details.append(result)
                elif index==10:
                    expired,calls,state=run('zombie',now=9000000000,deadline=9000000000)
                    unknown(expired,0);self.assertEqual(calls,[])
                    late,calls,state=run('late');unknown(late,1)
                    self.assertTrue(late['observation_attempts'][0]['late'])
                    self.assertEqual(late['original_deadline_ns'],9000000000)
                    diagnostic,calls,state=run('zombie',phase=observations.NEGATIVE_PHASE)
                    self.assertTrue(diagnostic['complete'])
                    self.assertFalse(diagnostic['positive_acceptance_eligible'])
                    self.assertFalse(group_phase_native_predicates(native_checks,diagnostic,full,eperm=True)['eperm_observation_budget'])
                    no_clock,calls,state=run('zombie',deadline=None)
                    self.assertTrue(no_clock['complete'])
                    self.assertFalse(no_clock['positive_acceptance_eligible'])
                    details=[expired,late,diagnostic,no_clock]
                else:
                    result,calls,state=run('zombie');group(result,state)
                    self.assertFalse(group_phase_native_predicates(native_checks,result,result)['zero_sid_before_reap'])
                    forged=dict(result,session_live=0,session_members=[])
                    self.assertFalse(group_phase_native_predicates(native_checks,result,forged)['zero_sid_before_reap'])
                    self.assertFalse(group_phase_native_predicates(native_checks,result,forged)['positive_reap_observation_budget'])
                    live_full=dict(full,session_live=1,session_members=[dict(pid=26519,pgid=26519,sid=26517)],
                                   observation_phase=observations.NEGATIVE_PHASE,positive_acceptance_eligible=False)
                    self.assertTrue(all(group_phase_native_predicates(native_checks,result,live_full,negative=True).values()))
                    self.assertFalse(group_phase_native_predicates(native_checks,result,
                        dict(live_full,observation_scope='process_group'),negative=True)['live_sid_before_failed_reap'])
                    stronger_group=dict(full,group_live=1,group_zombie_only=False)
                    stronger=group_phase_native_predicates(native_checks,stronger_group,full)
                    self.assertTrue(stronger['live_group_before_signal'])
                    self.assertTrue(stronger['positive_group_observation_budget'])
                    self.assertFalse(group_phase_native_predicates(native_checks,stronger_group,full,eperm=True)['eperm_native_zombie_only'])
                    with self.assertRaises(ValueError):run('zombie',scope='undocumented')
                    # Linux must project an already completed unchanged whole
                    # census. It cannot turn an incomplete/failed one into group0.
                    linux=object.__new__(observations.NativeObservation)
                    linux.darwin=None;linux._held={};linux_calls=[]
                    anchor=copy.deepcopy(full['held_leader'])
                    linux.identity=lambda pid:copy.deepcopy(anchor)
                    def whole_snapshot(leader):
                        linux_calls.append(copy.deepcopy(leader))
                        return copy.deepcopy(full)
                    linux.snapshot=whole_snapshot
                    clock=SimpleNamespace(monotonic_ns=lambda:2000000000,sleep=lambda seconds:None)
                    with patch.object(observations,'time',clock):
                        projected=observations.observed_snapshot(linux,anchor,deadline_ns=9000000000,scope='process_group')
                    self.assertTrue(projected['complete']);self.assertEqual(projected['observation_scope'],'process_group')
                    self.assertIsNone(projected['session_live']);self.assertIsNone(projected['session_members'])
                    self.assertEqual(len(linux_calls),1)
                    def incomplete(leader):return dict(full,complete=False)
                    linux.snapshot=incomplete
                    with patch.object(observations,'time',clock):
                        rejected=observations.observed_snapshot(linux,anchor,deadline_ns=9000000000,scope='process_group')
                    unknown(rejected,1)
                    def failed(leader):raise RuntimeError('literal incomplete Linux native census')
                    linux.snapshot=failed
                    with patch.object(observations,'time',clock):
                        failed_result=observations.observed_snapshot(linux,anchor,deadline_ns=9000000000,scope='process_group')
                    unknown(failed_result,1)
                    self.assertFalse(group_phase_native_predicates(native_checks,projected,projected)['zero_sid_before_reap'])
                    details=[result,forged,live_full,projected,rejected,failed_result]
                self.rows.append(dict(mode=name,passed=True,evidence_kind='simulated_independent_process_group_scope',
                    actual_POSIX_processes_started=0,actual_signals_sent=0,snapshots=details))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args()
    if sys.flags.optimize or not args.output_directory.is_absolute() or args.output_directory.exists() or \
            not args.output_directory.parent.is_dir() or args.output_directory.is_relative_to(ROOT):
        raise ValueError('unoptimized fresh absolute simulation evidence required')
    pins = source_pins()
    before = test_inventory()
    args.output_directory.mkdir()
    LiteralStateTests.output = args.output_directory
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LiteralStateTests))
    unchanged = source_pins() == pins and test_inventory() == before
    write_new(args.output_directory/'RESULT.json',dict(schema='owned_posix_v3_simulated_controls_v1',
        complete=result.wasSuccessful() and unchanged and len(LiteralStateTests.rows)==113 and result.testsRun==8, unittest_methods=result.testsRun,
        expected_simulated_rows=113,
        simulated_rows=len(LiteralStateTests.rows), rows=LiteralStateTests.rows, source_pins=pins,
        source_pins_unchanged=unchanged, actual_POSIX_processes_started=0, actual_signals_sent=0,
        native_Linux_or_Darwin_qualified=False, source_qualified=False, scientific_qualified=False))
    return 0 if result.wasSuccessful() and unchanged and len(LiteralStateTests.rows)==113 and result.testsRun==8 else 1


if __name__ == '__main__':
    raise SystemExit(main())
