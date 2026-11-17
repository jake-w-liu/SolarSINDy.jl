"""Independent simulated metadata tests; no Darwin native evidence is created."""
from pathlib import Path
import argparse
import copy
import errno
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORACLE = HERE / 'ORACLES_V1.json'
ORACLE_SHA = '590493deffeb0e2e7e9ffdc2e7310de0b38f35497dfa2ac478479ed8ed8dd51c'
ABI_SHA = '3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2'
LIMIT = 512 * 1024
ROWS = []


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def ordinary(path, directory=False):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise RuntimeError('Canonical ordinary path required')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('Linked path is not an ordinary test control')
    info = path.lstat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise RuntimeError('Ordinary path kind differs')
    return path


def metadata_error(kind):
    if kind in ('ESRCH', 'ESRCH_PROCESSLOOKUP'):
        return ProcessLookupError(errno.ESRCH, 'independent simulated process lookup')
    if kind == 'ESRCH_BASE_OSERROR':
        # OSError(3, ...) specializes to ProcessLookupError. This deliberately
        # exercises the distinct base OSError/errno branch without that factory.
        error = OSError('independent simulated base errno')
        error.errno = errno.ESRCH
        if type(error) is not OSError:
            raise RuntimeError('Base OSError metadata construction differs')
        return error
    if kind == 'EPERM':
        return PermissionError(errno.EPERM, 'independent simulated permission denial')
    raise RuntimeError('Unknown frozen metadata error: ' + str(kind))


def load_subject(path, expected):
    ordinary(path)
    if path.stat().st_size > LIMIT or digest(path) != expected:
        raise RuntimeError('Bounded subject bytes differ from explicit pin')
    spec = importlib.util.spec_from_file_location('independent_transition_subject', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if digest(path) != expected:
        raise RuntimeError('Subject changed during file-backed import')
    return module


class IdentityTransitionTests(unittest.TestCase):
    subject = None
    oracle = None

    def exercise(self, case):
        oracle = self.oracle
        pid = oracle['base_pid']
        initial = dict(oracle['base_row'], **case.get('initial', {}))
        fresh = dict(oracle['base_row'], **case.get('fresh', {}))
        if 'remove_fresh' in case:
            del fresh[case['remove_fresh']]
        initial_error = metadata_error(case['initial_error']) if 'initial_error' in case else None
        fresh_error = metadata_error(case['fresh_error']) if 'fresh_error' in case else None
        getter_error = metadata_error(case['getter']) if case['getter'] not in ('SUCCESS', 'UNEXPECTED') else None
        calls = dict(info=[], getter=[])

        class LiteralInfo:
            def info(self, asked):
                calls['info'].append(asked)
                if len(calls['info']) == 1:
                    if initial_error is not None:
                        raise initial_error
                    return copy.deepcopy(initial)
                if len(calls['info']) == 2:
                    if fresh_error is not None:
                        raise fresh_error
                    return copy.deepcopy(fresh)
                raise AssertionError('The frozen contract permits only one fresh info read')

        def literal_getsid(asked):
            calls['getter'].append(asked)
            if case['getter'] == 'UNEXPECTED':
                raise AssertionError('Frozen first-zombie branch must not call getsid')
            if case['getter'] == 'SUCCESS':
                return pid
            raise getter_error

        observer = self.subject.NativeObservation.__new__(self.subject.NativeObservation)
        observer.darwin = LiteralInfo()
        observer._held = {pid: copy.deepcopy(oracle['held_anchor'])} if case['held'] else {}
        held_before = copy.deepcopy(observer._held)
        returned = None
        caught = None
        # create=True makes this genuine Python method test portable to Windows,
        # where os.getsid is absent. Both providers are labelled metadata stubs.
        with mock.patch.object(self.subject.os, 'getsid', literal_getsid, create=True):
            try:
                returned = observer.identity(pid)
            except Exception as error:
                caught = error
        checks = dict(
            accepted_matches_literal=(caught is None and returned is not None) is case['accept'],
            info_call_count=len(calls['info']) == case['info_calls'],
            getter_call_count=len(calls['getter']) == case['getter_calls'],
            arguments_remain_bound=all(value == pid for value in calls['info'] + calls['getter']),
            held_cache_unchanged=observer._held == held_before)
        if case['accept']:
            expected = dict(pid=pid, pgid=pid, sid=case['expected_sid'], birth=[111, 222], uid=501, ruid=501,
                            zombie=case['expected_zombie'], stopped=False)
            checks['identity_exact'] = returned == expected
            checks['identity_closed_fields'] = type(returned) is dict and set(returned) == set(expected)
        else:
            checks['exception_retained'] = caught is not None and returned is None
        if case.get('same_error') == 'getter':
            checks['same_original_error'] = caught is getter_error
        elif case.get('same_error') == 'initial':
            checks['same_original_error'] = caught is initial_error
        if case['id'] == 'P02':
            checks['actual_base_OSError_metadata'] = type(getter_error) is OSError and getter_error.errno == errno.ESRCH
        row = dict(id=case['id'], passed=all(value is True for value in checks.values()), checks=checks,
                   returned_identity=returned, exception_type=None if caught is None else type(caught).__name__,
                   exception_errno=None if caught is None else getattr(caught, 'errno', None),
                   literal_getter_error_type=None if getter_error is None else type(getter_error).__name__,
                   info_arguments=calls['info'], getter_arguments=calls['getter'],
                   evidence_kind='simulated_metadata_only', actual_native_observations=0,
                   actual_signals=0, actual_Darwin_processes=0,
                   source_qualified=False, software_qualified=False, scientific_qualified=False)
        ROWS.append(row)
        self.assertTrue(row['passed'], json.dumps(row, sort_keys=True))

    def test_positive_recovery_literals(self):
        for case in self.oracle['recovery_rows']:
            with self.subTest(id=case['id']):
                self.exercise(case)

    def test_negative_boundary_literals(self):
        for case in self.oracle['negative_rows']:
            with self.subTest(id=case['id']):
                self.exercise(case)

    def test_existing_compatibility_literals(self):
        for case in self.oracle['compatibility_rows']:
            with self.subTest(id=case['id']):
                self.exercise(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=Path, required=True)
    parser.add_argument('--expected-subject-sha256', required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__ or len(args.expected_subject_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_subject_sha256):
        raise RuntimeError('Unoptimized execution and explicit64-hex subject pin required')
    subject_path = ordinary(args.subject)
    abi_path = ordinary(subject_path.parents[2] / 'test/initial_reference/diagnose_darwin_groups.py')
    if abi_path.stat().st_size > LIMIT or digest(abi_path) != ABI_SHA:
        raise RuntimeError('Frozen independent ABI source changed')
    output = args.output_directory
    ordinary(output.parent, True)
    if not output.is_absolute() or os.path.lexists(output) or output.parent != output.parent.resolve(strict=True):
        raise RuntimeError('Fresh external ordinary output required')
    if output.is_relative_to(subject_path.parent) or subject_path.parent.is_relative_to(output) or output.is_relative_to(HERE) or HERE.is_relative_to(output):
        raise RuntimeError('Output must be separate from subject/control directories')
    if digest(ORACLE) != ORACLE_SHA:
        raise RuntimeError('Frozen independent oracle changed')
    raw = ORACLE.read_bytes()
    if len(raw) > LIMIT:
        raise RuntimeError('Oracle exceeds bounded metadata')
    oracle = json.loads(raw)
    if [len(oracle[name]) for name in ('recovery_rows', 'negative_rows', 'compatibility_rows')] != [2, 20, 3]:
        raise RuntimeError('Frozen literal inventory differs')
    protected = [Path(__file__).resolve(), ORACLE, subject_path, abi_path]
    before = {str(path): digest(path) for path in protected}
    if before[str(subject_path)] != args.expected_subject_sha256:
        raise RuntimeError('Explicit subject pin differs')
    output.mkdir()
    started = time.monotonic_ns()
    IdentityTransitionTests.oracle = oracle
    IdentityTransitionTests.subject = load_subject(subject_path, args.expected_subject_sha256)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IdentityTransitionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(path): digest(path) for path in protected}
    expected_ids = [case['id'] for group in ('compatibility_rows', 'negative_rows', 'recovery_rows') for case in oracle[group]]
    actual_ids = [row['id'] for row in ROWS]
    complete = result.wasSuccessful() and not result.skipped and result.testsRun == 3 and actual_ids == expected_ids and before == after and all(row['passed'] for row in ROWS)
    record = dict(schema='independent_darwin_observer_transition_literals_v1', complete=complete,
                  unittest_methods=result.testsRun, expected_methods=3, literal_rows=len(ROWS), expected_rows=25,
                  recovery_rows=2, negative_rows=20, compatibility_rows=3, rows=ROWS,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  evidence_kind='simulated_metadata_only', actual_native_observations_requested=0,
                  actual_signals_requested=0, actual_Darwin_processes_requested=0,
                  actual_Darwin_observations=0, actual_native_test_processes_started=0,
                  actual_POSIX_test_processes_started=0, actual_signals_sent=0, Julia_launched=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    body = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')
    if len(body) > 4 * 1024**2:
        raise RuntimeError('Bounded result exceeded4MiB')
    with (output / 'RESULT.json').open('xb') as stream:
        if stream.write(body) != len(body):
            raise RuntimeError('Short literal result write')
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
