"""Independent native POSIX V3 controls; CI-only, no scientific qualification."""
from pathlib import Path
import argparse
import json
import math
import hashlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch
sys.dont_write_bytecode = True
_observations_path = Path(__file__).resolve().with_name('observations.py')
_observations_sha = 'c68ce8f6c8ef38aef394538b6a9b1915b17988d73e5647531a5a63ce6eacbf36'
with _observations_path.open('rb') as _stream:
    _observations_body = _stream.read(256*1024+1)
if len(_observations_body)>256*1024 or hashlib.sha256(_observations_body).hexdigest()!=_observations_sha:
    raise RuntimeError('pinned independent observation sibling changed')
_observations_spec = importlib.util.spec_from_file_location('posix_v3_checked_observations',_observations_path)
_observations = importlib.util.module_from_spec(_observations_spec)
_observations_spec.loader.exec_module(_observations)
if _observations.digest(_observations_path)!=_observations_sha:
    raise RuntimeError('independent observation sibling changed during import')
HERE,ROOT,ROLES = _observations.HERE,_observations.ROOT,_observations.ROLES
NativeObservation,digest,load = _observations.NativeObservation,_observations.digest,_observations.load
observed_snapshot,read_json = _observations.observed_snapshot,_observations.read_json
observation_context = _observations.observation_context
source_pins,test_inventory,write_new = _observations.source_pins,_observations.test_inventory,_observations.write_new

REAL_POPEN = subprocess.Popen
REAL_WAITID = getattr(os, 'waitid', None)
REAL_KILLPG = getattr(os, 'killpg', None)
REAL_KILL = os.kill
PRINCIPAL = ('no_descendant', 'same_group_descendant', 'other_group_12s', 'persistent_stop', 'transient_stop')
EXTRA = ('other_group_short', 'other_group_persistent')
STAGE = 8.0
CLEANUP = 20.0
WHOLE = 480.0
TOKEN = 'owned POSIX session has no live members after stage'


def exercise(module, observer, mode, directory, *, worker_deadline_ns):
    directory.mkdir()
    lock = threading.RLock()
    done = threading.Event()
    state = dict(process=None, leader=None, signal_started=False, reap_started=False,
                 stop_seen=0, stop_consumed=0, cont_sent=False, emergency=False)
    events = []
    threads = []

    def event(operation, **values):
        with lock:
            if len(events) >= 4096:
                raise RuntimeError('native event cap')
            events.append(dict(operation=operation, monotonic_ns=time.monotonic_ns(), **values))

    def before_reap(operation, wait_timeout=None):
        with lock:
            first = not state['reap_started']
            state['reap_started'] = True
            context = observation_context(started, worker_deadline_ns, first_wait=first and operation=='terminal_wait_entry',
                wait_timeout=wait_timeout, expected_negative=mode=='other_group_persistent')
            snap = observed_snapshot(observer, state['leader'], **context) if first else None
            event(operation, first=first, signal_started=state['signal_started'], snapshot=snap,
                received_wait_timeout=wait_timeout)

    class Process:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def wait(self, timeout=None):
            before_reap('terminal_wait_entry', timeout)
            value = self.actual.wait(timeout=timeout)
            observer.release(self.pid)
            event('terminal_wait_return', native_exit=value)
            return value

        def poll(self):
            before_reap('poll_entry')
            value = self.actual.poll()
            if value is not None:
                observer.release(self.pid)
            return value

        def kill(self):
            # Popen.kill may internally poll/reap. Observe its entry honestly.
            before_reap('direct_kill_entry')
            value = self.actual.kill()
            if self.actual.returncode is not None:
                observer.release(self.pid)
            return value

    def popen(argv, **options):
        if state['process'] is not None or options.get('start_new_session') is not True:
            raise RuntimeError('one genuine fresh private session required')
        actual = REAL_POPEN(argv, **options)
        process = Process(actual)
        state['process'] = process
        event('spawn', pid=actual.pid, start_new_session=True)
        return process

    def waitid(kind, pid, flags):
        value = REAL_WAITID(kind, pid, flags)
        process = state['process']
        if process is not None and pid == process.pid:
            code = None if value is None else int(value.si_code)
            with lock:
                if code == os.CLD_STOPPED:
                    state['stop_seen'] += 1
                if state['stop_seen'] and flags & os.WSTOPPED and not flags & os.WNOWAIT and not flags & os.WEXITED and code in (None, os.CLD_STOPPED):
                    # WNOHANG may return None; the completed stop-only call,
                    # followed by continued terminal observation, is the policy.
                    state['stop_consumed'] += 1
                # Terminal-consuming waitid is a reap attempt, unlike STOP-only.
                if flags & os.WEXITED and not flags & os.WNOWAIT:
                    before_reap('terminal_consuming_waitid')
                event('waitid', pid=pid, flags=int(flags), code=code,
                      result_pid=None if value is None else int(value.si_pid))
        return value

    def killpg(pid, number):
        with lock:
            process = state['process']
            if process is None or pid != process.pid or number != signal.SIGKILL:
                event('borrowed_group_signal', pid=pid, signal=int(number))
                raise RuntimeError('test refuses borrowed process-group signal')
            if state['signal_started']:
                event('repeated_group_signal', pid=pid)
                raise RuntimeError('test refuses repeated group signal')
            state['signal_started'] = True
            context = observation_context(started, worker_deadline_ns, before_signal=True)
            snap = observed_snapshot(observer, state['leader'], **context)
            event('group_signal_entry', pid=pid, signal=int(number), reap_started=state['reap_started'], snapshot=snap)
            try:
                REAL_KILLPG(pid, number)
            except BaseException as error:
                event('group_signal_error', error_type=type(error).__name__, errno=getattr(error, 'errno', None),
                      snapshot=observed_snapshot(observer, state['leader'],
                          **observation_context(started, worker_deadline_ns, before_signal=True)))
                raise
            else:
                event('group_signal_return')

    def kill(pid, number):
        process = state['process']
        if process is None or pid != process.pid or number != signal.SIGKILL:
            event('borrowed_direct_signal', pid=pid, signal=int(number))
            raise RuntimeError('production cannot signal an observer-supplied PID')
        event('owned_direct_signal', pid=pid, signal=int(number))
        return REAL_KILL(pid, number)

    def controller():
        deadline = time.monotonic() + STAGE
        while not done.is_set() and time.monotonic() < deadline:
            with lock:
                if state['signal_started'] or state['reap_started']:
                    event('controller_cont_refused_after_cleanup')
                    return
                try:
                    row = observer.identity(state['leader']['pid'])
                except Exception:
                    row = None
                if row is not None and all(row[k] == state['leader'][k] for k in ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid')):
                    consumed = sys.platform != 'darwin' or state['stop_consumed'] > 0
                    if row['stopped'] and consumed:
                        # The unreaped leader holds this numeric identity; lock
                        # excludes the first group signal/reap entry during CONT.
                        REAL_KILL(row['pid'], signal.SIGCONT)
                        state['cont_sent'] = True
                        event('controller_cont', pid=row['pid'], held_identity=row,
                              signal_started=False, reap_started=False)
                        return
            done.wait(0.005)
        event('controller_cont_not_sent')

    def watchdog():
        if done.wait(STAGE + CLEANUP + 10):
            return
        with lock:
            if state['reap_started'] or state['leader'] is None:
                return
            try:
                row = observer.identity(state['leader']['pid'])
                if any(row[k] != state['leader'][k] for k in ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid')):
                    return
                state['emergency'] = True
                event('emergency_test_direct_kill', pid=row['pid'])
                REAL_KILL(row['pid'], signal.SIGKILL)
            except Exception as error:
                event('emergency_test_unknown', error_type=type(error).__name__)

    def on_launch(pid):
        deadline = time.monotonic() + 3
        while not (directory / 'leader.json').is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError('leader file handshake deadline')
            time.sleep(0.005)
        if read_json(directory / 'leader.json') != dict(pid=pid, pgid=pid, sid=pid):
            raise RuntimeError('file-based leader ownership differs')
        leader = observer.identity(pid)
        if leader['pgid'] != pid or leader['sid'] != pid or leader['uid'] != os.geteuid() or leader['ruid'] != os.getuid():
            raise RuntimeError('independent native leader identity differs')
        state['leader'] = leader
        observer.hold(leader)
        event('on_launch', leader=leader)
        (directory / 'release').write_bytes(b'1\n')
        for function in ((controller, watchdog) if mode == 'transient_stop' else (watchdog,)):
            thread = threading.Thread(target=function, daemon=True)
            threads.append(thread)
            thread.start()

    started = time.monotonic_ns()
    result = None
    error = None
    raw = directory / 'native.raw'
    try:
        with patch.object(os, 'waitid', waitid), patch.object(os, 'killpg', killpg), \
                patch.object(os, 'kill', kill), patch.object(subprocess, 'Popen', popen):
            result = module.run_owned([sys.executable, '-I', '-B', '-X', 'utf8', str(HERE / 'fixture.py'), str(directory), mode],
                cwd=str(directory), environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), logpath=str(raw),
                timeout=STAGE, on_launch=on_launch)
    except BaseException as caught:
        error = dict(type=type(caught).__name__, errno=getattr(caught, 'errno', None), message=str(caught))
    finally:
        ended = time.monotonic_ns()
        done.set()
        for thread in threads:
            thread.join(timeout=2)
    process = state['process']
    if raw.is_file() and raw.stat().st_size > 4 * 1024**2:
        raise RuntimeError('case raw bound')
    row = dict(mode=mode, native=result, error=error, events=events, start_ns=started, end_ns=ended,
        elapsed_ns=ended-started, actual_direct_exit=None if process is None else process.returncode,
        controller_cont_sent=state['cont_sent'], emergency=state['emergency'],
        controller_threads_joined=all(not t.is_alive() for t in threads),
        raw=None if not raw.is_file() else dict(path=str(raw), sha256=digest(raw), bytes=raw.stat().st_size),
        individual_descendant_OS_reaping_observed=False)
    # Deliberate other-group fixtures self-expire. Never send an orphan signal.
    if (directory / 'descendant.json').is_file():
        descendant = read_json(directory / 'descendant.json')
        row['generated_descendant'] = descendant
        if mode == 'other_group_persistent':
            try:
                last = observer.identity(descendant['pid'])
                row['other_group_live_at_owner_return'] = not last['zombie'] and last['sid'] == process.pid and last['pgid'] != process.pid
            except Exception:
                row['other_group_live_at_owner_return'] = None
            # Retain the unclosed failure, then wait only for the fixture's own
            # bounded timer; eventual disappearance does not repair owner output.
            expiry = time.monotonic() + 35
            row['fixture_expiry_observed'] = False
            while time.monotonic() < expiry:
                try:
                    last = observer.identity(descendant['pid'])
                    if last['zombie']:
                        row['fixture_expiry_observed'] = True
                        break
                except ProcessLookupError:
                    row['fixture_expiry_observed'] = True
                    break
                except FileNotFoundError:
                    row['fixture_expiry_observed'] = True
                    break
                except OSError:
                    row['fixture_expiry_unknown'] = True
                    break
                time.sleep(0.02)
            else:
                row['fixture_expiry_timeout'] = True
    return row


def check(row, mode, *, historical=False):
    events = row['events']
    signals = [e for e in events if e['operation'] == 'group_signal_entry']
    reaps = [e for e in events if e.get('first') is True and e['operation'] in ('terminal_wait_entry', 'poll_entry', 'direct_kill_entry', 'terminal_consuming_waitid')]
    terminal = [e for e in events if e['operation'] == 'waitid' and e['code'] in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)
                and e['flags'] & os.WNOWAIT and e['flags'] & os.WEXITED]
    stops = [e for e in events if e['operation'] == 'waitid' and e['code'] in (None, os.CLD_STOPPED)
             and e['flags'] & os.WSTOPPED and not e['flags'] & (os.WNOWAIT | os.WEXITED)]
    bad_signal = any(e['operation'] in ('borrowed_group_signal', 'borrowed_direct_signal', 'repeated_group_signal') for e in events)
    checks = dict(one_original_signal=len(signals) == 1, no_borrowed_signal=not bad_signal,
        signal_before_reap=len(signals) == 1 and not signals[0]['reap_started'],
        no_emergency=not row['emergency'], threads_joined=row['controller_threads_joined'])
    if mode == 'other_group_persistent':
        checks.update(real_unclosed_failure=row['error'] is not None and row['native'] is None,
                      live_other_group=row.get('other_group_live_at_owner_return') is True,
                      fixture_expiry_observed=row.get('fixture_expiry_observed') is True)
        failed = reaps[0].get('snapshot', {}) if len(reaps)==1 else {}
        checks['live_sid_before_failed_reap'] = failed.get('complete') is True and \
            failed.get('observation_phase') == _observations.NEGATIVE_PHASE and \
            failed.get('positive_acceptance_eligible') is False and \
            type(failed.get('session_live')) is int and failed['session_live'] >= 1
        return checks
    native = row['native']
    timeout = mode in ('other_group_12s', 'persistent_stop')
    code = -9 if mode == 'persistent_stop' else 7
    checks.update(no_error=row['error'] is None, actual_exit=row['actual_direct_exit'] == code,
        native_exit=isinstance(native, dict) and native.get('native_exit') == code,
        timeout=isinstance(native, dict) and native.get('timed_out') is timeout,
        cleanup_token=isinstance(native, dict) and native.get('descendant_cleanup') == TOKEN,
        zero_sid_before_reap=len(reaps) == 1 and reaps[0].get('snapshot', {}).get('complete') is True and reaps[0]['snapshot'].get('session_live') == 0)
    checks['positive_reap_observation_budget'] = len(reaps)==1 and \
        reaps[0].get('snapshot',{}).get('positive_acceptance_eligible') is True and \
        reaps[0]['snapshot'].get('observation_phase') == _observations.POSITIVE_PHASE
    base_keys = {'pid','native_exit','timed_out','ownership','descendant_cleanup','wall_seconds'}
    candidate_keys = base_keys | {'leader_reaped','ownership_boundary_closed','all_descendants_os_reaped','observation_method','descendant_evidence'}
    expected_keys = candidate_keys if row.get('role') == ROLES[2] else base_keys
    spawns = [e for e in events if e['operation'] == 'spawn']
    checks['closed_public_result'] = isinstance(native,dict) and set(native) == expected_keys
    checks['actual_pid_binding'] = isinstance(native,dict) and type(native.get('pid')) is int and len(spawns)==1 and native['pid']==spawns[0]['pid']
    checks['native_wall_finite'] = isinstance(native,dict) and type(native.get('wall_seconds')) in (int,float) and math.isfinite(native['wall_seconds']) and native['wall_seconds'] >= 0
    checks['private_session_method'] = isinstance(native,dict) and native.get('ownership') == 'start_new_session=True'
    if timeout:
        checks['full_stage_deadline_observed'] = row['elapsed_ns'] >= int(STAGE * 10**9)
    else:
        checks['inside_original_stage'] = row['elapsed_ns'] < int(STAGE * 10**9)
    for event in events:
        if event['operation'] == 'group_signal_error' and event['errno'] == 1:
            snap = event['snapshot']
            checks['eperm_native_zombie_only'] = snap.get('complete') is True and snap.get('group_zombie_only') is True and snap.get('group_live') == 0 and snap['held_leader']['zombie'] is True
            checks['eperm_observation_budget'] = snap.get('positive_acceptance_eligible') is True
    if mode == 'same_group_descendant':
        checks['live_group_before_signal'] = len(signals) == 1 and signals[0]['snapshot'].get('group_live', 0) >= 1
        checks['positive_group_observation_budget'] = len(signals)==1 and signals[0]['snapshot'].get('positive_acceptance_eligible') is True
    if mode in ('persistent_stop', 'transient_stop') and sys.platform == 'darwin':
        checks['stop_only_consumed'] = len(stops) >= 1
    if mode == 'transient_stop':
        checks['controlled_cont'] = row['controller_cont_sent']
        checks['terminal_observation_before_cleanup'] = bool(terminal) and bool(signals) and terminal[-1]['monotonic_ns'] < signals[0]['monotonic_ns']
    if row.get('role') == ROLES[2]:
        checks['candidate_boundary_closed'] = isinstance(native,dict) and native.get('ownership_boundary_closed') is True
        checks['candidate_leader_reaped'] = isinstance(native,dict) and native.get('leader_reaped') is True
        checks['candidate_descendant_reaping_unobserved'] = isinstance(native,dict) and native.get('all_descendants_os_reaped') is False
        checks['candidate_observation_method_retained'] = isinstance(native,dict) and all(isinstance(native.get(k),str) and bool(native[k]) for k in ('observation_method','descendant_evidence'))
    return checks


def main():
    started = time.monotonic_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or os.name != 'posix' or not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
        raise RuntimeError('unoptimized native Linux/Darwin controls only')
    output = args.output_directory
    if not output.is_absolute() or output.exists() or not output.parent.is_dir() or output.is_relative_to(ROOT):
        raise RuntimeError('fresh external evidence directory required')
    memory = load(ROOT / 'tools/run_canonical_harness.py', 'v3_native_engineering_memory')
    available, method = memory.available_memory()
    free = shutil.disk_usage(output.parent).free
    if available < 512 * 1024**2 or free < 8 * 1024**3:
        raise RuntimeError('native engineering512MiB/8GiB admission failed')
    pins = source_pins()
    tests_before = test_inventory()
    observer = NativeObservation()
    output.mkdir()
    rows = []
    for index, role in enumerate(ROLES):
        module = load(ROOT / role, 'v3_current_native_owner_' + str(index))
        for mode in PRINCIPAL + EXTRA:
            if (time.monotonic_ns()-started)/1e9 >= WHOLE:
                raise RuntimeError('whole native control budget exhausted')
            row = exercise(module, observer, mode, output / (str(index) + '-' + mode), worker_deadline_ns=started+int(WHOLE*10**9))
            row.update(role=role, source_sha256=pins[role], evidence_kind='actual_native_current')
            row['checks'] = check(row, mode)
            row['passed'] = all(row['checks'].values())
            write_new(output / (str(index) + '-' + mode + '.json'), row)
            rows.append(row)
    historical = []
    if sys.platform == 'darwin':
        for index, role in enumerate(ROLES):
            name = 'candidate_native_owner.py' if index == 2 else 'native_stage.py'
            path = HERE / 'frozen' / name
            module = load(path, 'v3_historical_stop_owner_' + str(index))
            row = exercise(module, observer, 'transient_stop', output / ('old-' + str(index) + '-transient_stop'),
                worker_deadline_ns=started+int(WHOLE*10**9))
            row.update(role=role, source_sha256=digest(path), evidence_kind='actual_native_historical_negative')
            row['new_stop_oracle_checks'] = check(row, 'transient_stop', historical=True)
            # Require the semantic stop defect, not merely the historical text.
            row['expected_negative_reproduced'] = not row['new_stop_oracle_checks'].get('stop_only_consumed', True) and any(
                e['operation'] == 'waitid' and e['code'] == os.CLD_STOPPED for e in row['events'])
            write_new(output / ('old-' + str(index) + '-transient_stop.json'), row)
            historical.append(row)
    after = source_pins()
    tests_unchanged = test_inventory() == tests_before
    # Keep the legacy six fields; optional processor is explicitly unknown.
    # Iterating platform.uname() can spawn an unbounded uname -p helper.
    runtime_facts = dict(uname=list(os.uname())+[''], python_version=sys.version,
        python_binary_sha256=digest(sys.executable), oracle_sha256=digest(HERE / 'oracles_v3.json'),
        fixture_sha256=digest(HERE / 'fixture.py'))
    elapsed = time.monotonic_ns()-started
    passed = all(r['passed'] for r in rows) and all(r['expected_negative_reproduced'] for r in historical) and pins == after and tests_unchanged and elapsed < int(WHOLE*10**9)
    write_new(output / 'RESULT.json', dict(schema='owned_posix_native_v3_independent_result', complete=passed,
        actual_current_cases=len(rows), principal_current_cases=15, extra_current_cases=6,
        historical_stop_negative_cases=len(historical), source_pins=pins, source_pins_unchanged=pins == after,
        test_source_sha256=tests_before,test_sources_unchanged=tests_unchanged,
        elapsed_ns=elapsed, **runtime_facts, admission=dict(available_physical_bytes=available,memory_method=method,free_disk_bytes=free),
        case_passes=[r['passed'] for r in rows], historical_negative_passes=[r['expected_negative_reproduced'] for r in historical],
        individual_descendant_OS_reaping_observed=False, source_qualified=False, scientific_qualified=False))
    if time.monotonic_ns()-started >= int(WHOLE*10**9):
        (output/'RESULT.json').rename(output/'RESULT_LATE_UNQUALIFIED.json')
        write_new(output/'LATE_OUTPUT.json',dict(complete=False,deadline_ns=started+int(WHOLE*10**9),observed_ns=time.monotonic_ns()))
        return 1
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
