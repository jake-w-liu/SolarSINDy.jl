"""Independent test observations; never imports the new production census."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_OBSERVER_SHA = '3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2'
ORACLE_SHA = '7d4d444771abb90a0a07f297f9a11b669a7dc78645f7511925ce018e7f9f0e98'
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
        def pids():
            result = sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal())
            if not 0 < len(result) < 16384:
                raise RuntimeError('independent process count cap')
            return result
        before = pids()
        rows = [self.identity(pid) for pid in before]
        if before != pids():
            raise RuntimeError('independent census changed, not zero')
        second = [self.identity(pid) for pid in before]
        keys = ('pid', 'pgid', 'sid', 'birth', 'uid', 'ruid', 'zombie')
        if before != pids() or any(any(a[k] != b[k] for k in keys) for a,b in zip(rows,second)):
            raise RuntimeError('independent native identity/state changed, not zero')
        rows = second
        group = [r for r in rows if r['pgid'] == leader['pgid']]
        live = [r for r in rows if r['sid'] == leader['sid'] and not r['zombie']]
        if not any(r['pid'] == leader['pid'] and r['birth'] == leader['birth'] for r in group):
            raise RuntimeError('independent held leader absent')
        return dict(complete=True, held_leader=held, group_live=sum(not r['zombie'] for r in group),
                    group_zombie_only=bool(group) and all(r['zombie'] for r in group),
                    session_live=len(live), session_members=live)


def observed_snapshot(observer, leader):
    try:
        return observer.snapshot(leader)
    except Exception as error:
        return dict(complete=False, unknown=True, error_type=type(error).__name__, error=str(error))


def write_new(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 4 * 1024**2:
        raise ValueError('test evidence bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise RuntimeError('test short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
