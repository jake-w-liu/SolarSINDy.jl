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
_observations_sha = 'bc4e0ba607bedd1fd59bdc90f23c048139a072534a64aa68d1fe8d05b4c485e8'
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
        complete=result.wasSuccessful() and unchanged and len(LiteralStateTests.rows)==85, unittest_methods=result.testsRun,
        expected_simulated_rows=85,
        simulated_rows=len(LiteralStateTests.rows), rows=LiteralStateTests.rows, source_pins=pins,
        source_pins_unchanged=unchanged, actual_POSIX_processes_started=0, actual_signals_sent=0,
        native_Linux_or_Darwin_qualified=False, source_qualified=False, scientific_qualified=False))
    return 0 if result.wasSuccessful() and unchanged and len(LiteralStateTests.rows)==85 else 1


if __name__ == '__main__':
    raise SystemExit(main())
