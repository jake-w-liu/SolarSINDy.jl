"""Independent test observations; never imports the new production census."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_OBSERVER_SHA = '3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2'
ORACLE_SHA = '7d4d444771abb90a0a07f297f9a11b669a7dc78645f7511925ce018e7f9f0e98'
RETRY_ORACLE_SHA = 'cecad892ca941c8b3ac038079d65634c0857fb6c918bca760fbaadf5685df31c'
RETRY_ATTEMPTS = 8
RETRY_SUBCAP_NS = 250000000
RETRY_SLEEP_NS = 5000000
POSITIVE_PHASE = 'positive_owner_observation'
NEGATIVE_PHASE = 'expected_negative_pre_reap_diagnostic'
ROLES = ('tools/initial_reference_audit/owned_process.py',
         'research/dbdt_completed_source/native_stage.py',
         'research/dbdt_completed_source/candidate_native_owner.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 2 * 1024**2:
        raise ValueError('bounded ordinary test metadata required')
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate test key')
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def source_pins():
    if digest(HERE / 'oracles_v3.json') != ORACLE_SHA:
        raise ValueError('independently frozen V3 oracle changed')
    if digest(HERE / 'census_retry_oracles_v1.json') != RETRY_ORACLE_SHA:
        raise ValueError('independently frozen census retry oracle changed')
    data = read_json(HERE / 'current_sources_v3.json')
    if set(data) != {'schema', 'sources'} or data['schema'] != 'owned_posix_current_source_pins_v3':
        raise ValueError('current source binding schema differs')
    pins = data['sources']
    if set(pins) != set(ROLES) or any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in pins.values()):
        raise ValueError('current source bindings are not sealed')
    if any(digest(ROOT / role) != pins[role] for role in ROLES):
        raise ValueError('current source changed')
    if (ROOT / ROLES[0]).read_bytes() != (ROOT / ROLES[1]).read_bytes():
        raise ValueError('paired legacy sources differ')
    before = (HERE / 'frozen/native_stage.py').read_bytes()
    if digest(HERE / 'frozen/native_stage.py') != '15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8':
        raise ValueError('historical legacy changed')
    for role in ROLES[:2]:
        if (ROOT / role).read_bytes().split(b'def _run_windows', 1)[1] != before.split(b'def _run_windows', 1)[1]:
            raise ValueError('preserved Windows implementation changed')
    if digest(HERE / 'frozen/candidate_native_owner.py') != 'ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21':
        raise ValueError('historical candidate changed')
    return pins


def test_inventory():
    paths = sorted(p for p in HERE.rglob('*') if p.is_file())
    if not 0 < len(paths) <= 32 or any(p.is_symlink() or '__pycache__' in p.parts or p.stat().st_size > 256 * 1024 for p in paths):
        raise RuntimeError('bounded ordinary V3 test-source inventory differs')
    return {p.relative_to(HERE).as_posix():digest(p) for p in paths}


class _CensusInstability(RuntimeError):
    def __init__(self, reason, pid=None):
        super().__init__(reason)
        self.reason = reason
        self.pid = pid


class NativeObservation:
    def __init__(self):
        self._held = {}
        if sys.platform == 'darwin':
            path = ROOT / 'test/initial_reference/diagnose_darwin_groups.py'
            if digest(path) != OLD_OBSERVER_SHA:
                raise ValueError('independent old native observer changed')
            self.darwin = load(path, 'v3_frozen_independent_darwin_observer').DarwinProcesses()
        elif sys.platform.startswith('linux'):
            self.darwin = None
        else:
            raise RuntimeError('native Linux/Darwin observation required')
        own = self.identity(os.getpid())
        if own['pgid'] != os.getpgrp() or own['sid'] != os.getsid(0) or own['uid'] != os.getuid():
            raise RuntimeError('independent self calibration differs')

    def identity(self, pid):
        if self.darwin is not None:
            row = self.darwin.info(pid)
            birth = [row['start_sec'], row['start_usec']]
            if row['status'] == 5:
                # XNU getsid uses proc_find, which does not hold dead procs.
                # BSD arg1 can still prove birth/credentials/group for a zombie.
                # Only a matching unreaped live-launch anchor supplies its SID.
                anchor = self._held.get(pid)
                if anchor is not None and (anchor['birth'] != birth or anchor['pgid'] != row['pgid'] or
                        anchor['uid'] != row['uid'] or anchor['ruid'] != row['ruid']):
                    raise RuntimeError('held Darwin zombie native identity changed')
                sid = None if anchor is None else anchor['sid']
            else:
                sid = os.getsid(pid)
            return dict(pid=pid, pgid=row['pgid'], sid=sid,
                        birth=[row['start_sec'], row['start_usec']], uid=row['uid'], ruid=row['ruid'],
                        zombie=row['status'] == 5, stopped=row['status'] in (4,))
        path = Path('/proc') / str(pid)
        with (path / 'stat').open('rb') as stream:
            raw = stream.read(8193)
        if len(raw) > 8192 or raw.find(b'(') < 0 or raw.rfind(b')') < 0:
            raise RuntimeError('bounded independent Linux stat differs')
        head = raw[:raw.find(b'(')].strip()
        fields = raw[raw.rfind(b')') + 1:].split()
        if int(head) != pid or len(fields) < 20:
            raise RuntimeError('independent Linux stat shape differs')
        with (path / 'status').open('rb') as stream:
            status = stream.read(8193)
        if len(status) > 8192:
            raise RuntimeError('bounded independent Linux status differs')
        ids = [line.split()[1:] for line in status.splitlines() if line.startswith(b'Uid:')]
        if len(ids) != 1 or len(ids[0]) != 4:
            raise RuntimeError('independent Linux UID record differs')
        ruid, uid = int(ids[0][0]), int(ids[0][1])
        return dict(pid=pid, pgid=int(fields[2]), sid=int(fields[3]), birth=[int(fields[19])],
                    uid=uid, ruid=ruid,
                    zombie=fields[0] in (b'Z', b'X', b'x'), stopped=fields[0] in (b'T', b't'))

    def hold(self, leader):
        if leader['zombie'] or not leader['pid'] == leader['pgid'] == leader['sid'] or \
                leader['uid'] != os.geteuid() or leader['ruid'] != os.getuid():
            raise RuntimeError('independent live launch anchor differs')
        current = self.identity(leader['pid'])
        if current['zombie'] or any(current[k] != leader[k] for k in ('pid','pgid','sid','birth','uid','ruid')):
            raise RuntimeError('independent live launch anchor changed')
        self._held[leader['pid']] = {k:leader[k] for k in ('pid','pgid','sid','birth','uid','ruid')}

    def release(self, pid):
        self._held.pop(pid,None)

    def _linux_pids(self):
        result = sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal())
        if not 0 < len(result) < 16384:
            raise RuntimeError('independent process count cap')
        return result

    def _linux_rows(self, pids, leader):
        rows = []
        for pid in pids:
            try:
                row = self.identity(pid)
            except FileNotFoundError as error:
                # Only a listed unrelated Linux PID's actual stat/status loss
                # is a retryable census instability. Never drop that PID.
                match = re.fullmatch(r'/proc/([0-9]+)/(?:stat|status)', str(error.filename))
                if error.errno == 2 and pid != leader['pid'] and match is not None and int(match[1]) == pid:
                    raise _CensusInstability('listed unrelated PID stat/status disappeared', pid) from error
                raise
            if pid == leader['pid'] and any(row[k] != leader[k] for k in ('pid','pgid','sid','birth','uid','ruid')):
                raise RuntimeError('independently held leader changed during census')
            rows.append(row)
        return rows

    def snapshot(self, leader):
        # This is the pre-existing diagnostic's independently observed census,
        # not a call to any new production adapter or its positive summary.
        held = self.identity(leader['pid'])
        if any(held[k] != leader[k] for k in ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid')):
            raise RuntimeError('independently held leader changed')
        if self.darwin is not None:
            group = self.darwin.snapshot(leader['pgid'])
            session = self.darwin.session_snapshot(leader['sid'])
            return dict(complete=True, held_leader=held,
                group_live=group['live_members_observed'], group_zombie_only=group['owned_group_zombie_only_observed'],
                session_live=len(session['live_members']), session_members=session['live_members'])
        before = self._linux_pids()
        rows = self._linux_rows(before, leader)
        if before != self._linux_pids():
            raise _CensusInstability('independent census changed, not zero')
        second = self._linux_rows(before, leader)
        keys = ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid', 'zombie')
        if before != self._linux_pids() or any(any(a[k] != b[k] for k in keys) for a,b in zip(rows,second)):
            raise _CensusInstability('independent native identity/state changed, not zero')
        rows = second
        group = [r for r in rows if r['pgid'] == leader['pgid']]
        live = [r for r in rows if r['sid'] == leader['sid'] and not r['zombie']]
        if not any(r['pid'] == leader['pid'] and r['birth'] == leader['birth'] for r in group):
            raise RuntimeError('independent held leader absent')
        return dict(complete=True, held_leader=held, group_live=sum(not r['zombie'] for r in group),
                    group_zombie_only=bool(group) and all(r['zombie'] for r in group),
                    session_live=len(live), session_members=live)


def observation_context(case_started_ns, worker_deadline_ns, *, before_signal=False,
                        first_wait=False, wait_timeout=None, expected_negative=False):
    if type(case_started_ns) is not int or type(worker_deadline_ns) is not int or worker_deadline_ns <= case_started_ns:
        raise ValueError('original absolute observation clocks required')
    now = time.monotonic_ns()
    valid_wait = type(wait_timeout) in (int, float) and math.isfinite(wait_timeout)
    if wait_timeout is not None and not valid_wait:
        raise ValueError('finite received wait-timeout required')
    if expected_negative and first_wait and valid_wait and wait_timeout <= 0:
        return dict(phase=NEGATIVE_PHASE, deadline_ns=min(worker_deadline_ns, case_started_ns+38000000000))
    deadline = min(worker_deadline_ns, case_started_ns+20000000000)
    if before_signal and now < case_started_ns+8000000000:
        deadline = min(deadline, case_started_ns+8000000000)
    if valid_wait:
        deadline = min(deadline, now+max(0, int(wait_timeout*10**9)))
    return dict(phase=POSITIVE_PHASE, deadline_ns=deadline)


def observed_snapshot(observer, leader, *, deadline_ns=None, phase=POSITIVE_PHASE):
    if phase not in (POSITIVE_PHASE, NEGATIVE_PHASE) or (deadline_ns is not None and type(deadline_ns) is not int):
        raise ValueError('closed observation phase/original absolute deadline required')
    started = time.monotonic_ns()
    effective = None if deadline_ns is None else min(deadline_ns, started+RETRY_SUBCAP_NS)
    attempts = []
    last_type, last_message = 'ObservationDeadlineExceeded', 'original observation deadline exhausted'
    limit = 1 if deadline_ns is None else RETRY_ATTEMPTS
    for index in range(limit):
        begin = time.monotonic_ns()
        if effective is not None and begin >= effective:
            break
        answer = None
        error = None
        try:
            # Every attempt rechecks the held leader and begins a whole new
            # inventory/two-pass census. No partial rows or counts are reused.
            answer = observer.snapshot(leader)
            if not isinstance(answer, dict) or answer.get('complete') is not True:
                raise RuntimeError('independent census did not return a complete snapshot')
        except Exception as caught:
            error = caught
        end = time.monotonic_ns()
        late = effective is not None and end >= effective
        retryable = isinstance(error, _CensusInstability) and observer.darwin is None
        source_error = error.__cause__ if error is not None and error.__cause__ is not None else error
        trace = dict(index=index+1, started_ns=begin, finished_ns=end, complete=error is None and not late,
            retryable=retryable, error_type=None if source_error is None else type(source_error).__name__,
            errno=None if source_error is None else getattr(source_error,'errno',None),
            vanished_pid=getattr(error,'pid',None), instability_reason=getattr(error,'reason',None), late=late)
        attempts.append(trace)
        if error is None and not late:
            finished = time.monotonic_ns()
            if effective is None or finished < effective:
                return dict(answer, observation_phase=phase, positive_acceptance_eligible=phase==POSITIVE_PHASE and deadline_ns is not None,
                    original_deadline_ns=deadline_ns, effective_deadline_ns=effective,
                    observation_started_ns=started, observation_finished_ns=finished, observation_attempts=attempts)
            late = True
            trace.update(complete=False, late=True)
        if late:
            last_type, last_message = 'ObservationDeadlineExceeded', 'complete census finished after its original bounded observation clock'
            break
        last_type = type(error).__name__
        last_message = str(error)[:512]
        if not retryable or effective is None or index+1 == limit:
            break
        remaining = effective-time.monotonic_ns()
        if remaining <= 0:
            break
        time.sleep(min(RETRY_SLEEP_NS, remaining)/10**9)
    return dict(complete=False, unknown=True, error_type=last_type, error=last_message,
        observation_phase=phase, positive_acceptance_eligible=False, original_deadline_ns=deadline_ns,
        effective_deadline_ns=effective, observation_started_ns=started,
        observation_finished_ns=time.monotonic_ns(), observation_attempts=attempts)


def write_new(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 4 * 1024**2:
        raise ValueError('test evidence bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise RuntimeError('test short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
