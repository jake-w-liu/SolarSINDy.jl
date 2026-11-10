"""Small engineering controls for the matched342 owner; never Julia evidence.

Run file-based under the fixed outer owner on an admitted engineering machine.
This file does not call the runtime main, issue Q, admit a permit, or open data.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'tools/actual_b_chain/run_matched342_native.py'
TARGET_SHA = '32618b32e80b8a88524ddae774bdd80a8a42b0a16c1c3ab98d0869eac0063bb5'
HEX = 'a' * 64
HEAD = 'b' * 40
META = 2 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    # Independent canonical oracle for these integer/string/bool-only fixtures.
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def identified(data, key='request_id'):
    result = copy.deepcopy(data)
    result.pop(key, None)
    result[key] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def request_fixtures(root):
    # Pure grammar data: these nonexistent refs grant no software/actual authority.
    ref = lambda name: dict(path=str(root / name), sha256=HEX)
    runtime = dict(julia_version='1.12.6', julia_binary_sha256=HEX, evotrees_version='0.18.7',
                   evotrees_source_sha256='f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f',
                   threads=1, blas_threads=1, omp_threads=1)
    limits = dict(source_rows=500000, model_member_bytes=4194304, total_model_member_bytes=134217728,
                  required_available_bytes=8589934592, required_disk_bytes=17179869184,
                  stage_wall_seconds=120, fit_wall_seconds=60)
    budget = dict(peak_rss_bytes=8589934592, julia_gc_allocated_bytes=123456789, descriptor_growth=2)
    contracts = dict(protocol_sha256='7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404',
                     feature_contract_sha256='2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683',
                     feature_schema_sha256='109464af1630d16ac766b4f8791ad9f065a528d0aabac201a11fb68a4156249d',
                     loss_contract_sha256='f24c0dd5b9ef4baf58f0595056d5b5a30ab21ff5c278126b5ebefaa9435bc73c',
                     audit_contract_sha256='60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a',
                     evotrees_source_sha256=runtime['evotrees_source_sha256'])
    spec = dict(member='tools/actual_b_chain/fixtures/full_shape342_spec_v1.json', **ref('spec.json'))
    profile = identified(dict(schema='matched342_full_shape_profile_request_v1', evidence_mode='synthetic_only',
        operation='initial32_full_shape_profile', qualified_head=HEAD,
        implementation_sources=[dict(name=f'test/member{i:03d}.jl', sha256=HEX) for i in range(1, 137)],
        source_proof=ref('proof.json'), component_qualification=ref('component.json'), contracts=contracts,
        runtime=runtime, fixture_spec=spec, limits=limits, budgets=budget,
        output_directory=str(root / 'profile'), owner_directory=str(root / 'profile-owner')))
    issuer = identified(dict(schema='matched342_resource_issuance_request_v1',
        operation='issue_initial32_full_shape_resource_qualification', head=HEAD, source_proof=ref('proof.json'),
        root_qualification=ref('root.json'), component_qualification=ref('component.json'), fixture_spec=spec,
        profile_result=ref('result.json'), native=ref('native.json'), raw=ref('raw.log'),
        prelaunch_admission=ref('prelaunch.json'), measurement=ref('measurement.json'),
        limits=limits, budgets=budget, output_directory=str(root / 'issuance')))
    actual = identified(dict(schema='matched342_initial_actual_request_v1', evidence_mode='archival_assumed_completion_research',
        operation='initial_matched342_32fits', execution_authority=ref('permit.json'),
        sources=[dict(delay_hours=k, source_proof=ref(f'source{k}.json'), roots=dict(source=str(root / f'source{k}')))
                 for k in (0, 1)], source_proof=ref('proof.json'), runtime=runtime,
        output_directory=str(root / 'execution'), owner_directory=str(root / 'actual-owner')))
    return dict(profile=profile, issue_resource=issuer, actual_initial32=actual)


class OwnerContractTests(unittest.TestCase):
    target = None
    native_owner = None
    output = None
    native_rows = []

    def setUp(self):
        self.directory = self.output / self._testMethodName
        self.directory.mkdir()

    def mutate(self, request, key, value):
        result = copy.deepcopy(request)
        result[key] = value
        return identified(result)

    def native(self, mode, timeout=10, callback=None):
        raw = self.directory / (mode + '.raw')
        launched = []

        def on_launch(pid):
            launched.append(pid)
            if callback is not None:
                callback(pid)

        started = time.monotonic_ns()
        result = self.native_owner.run_owned(
            [sys.executable, '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--child', mode],
            cwd=str(self.directory), environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            logpath=str(raw), timeout=timeout, on_launch=on_launch)
        ended = time.monotonic_ns()
        self.assertEqual(len(launched), 1)
        self.assertEqual(result['pid'], launched[0])
        self.native_rows.append(dict(mode=mode, native=result, actual_on_launch_pid=launched[0],
            start_ns=started, end_ns=ended, raw=dict(path=str(raw), sha256=sha(raw), bytes=raw.stat().st_size)))
        return result, raw, launched[0]

    def test_canonical_literal_bytes(self):
        value = dict(z=[True, None, 7], a='quarter')
        self.assertEqual(self.target.canonical_bytes(value), b'{"a":"quarter","z":[true,null,7]}\n')
        self.assertEqual(self.target.canonical_bytes(dict(value=1.25)), b'{"value":"3ff4000000000000"}\n')
        self.assertEqual(self.target.canonical_bytes(dict(value=-0.0)), b'{"value":"8000000000000000"}\n')
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                self.target.canonical_bytes(dict(value=value))

    def test_all_closed_request_grammars(self):
        requests = request_fixtures(self.directory)
        self.assertEqual([len(v) for v in requests.values()], [15, 16, 10])
        for operation, data in requests.items():
            self.assertEqual(self.target.validate_request_metadata(operation, data), data)
            for key, value in (('schema', 'future'), ('operation', 'future'), ('future', None),
                               ('output_directory', 'relative')):
                with self.subTest(operation=operation, key=key):
                    with self.assertRaises(ValueError):
                        self.target.validate_request_metadata(operation, self.mutate(data, key, value))
            stale = copy.deepcopy(data)
            stale['operation'] = 'future'
            with self.assertRaises(ValueError):
                self.target.validate_request_metadata(operation, stale)
        for operation in ('profile', 'issue_resource'):
            data = requests[operation]
            for key, value in (('budgets', None), ('limits', dict(data['limits'], source_rows=1024)),
                               ('budgets', dict(data['budgets'], descriptor_growth=True))):
                with self.assertRaises(ValueError):
                    self.target.validate_request_metadata(operation, self.mutate(data, key, value))

    def test_source_runtime_mode_semantics_after_rehash(self):
        requests = request_fixtures(self.directory)
        p = requests['profile']
        for key, value in (('evidence_mode', 'software_qualified'),
                           ('runtime', dict(p['runtime'], threads=True)),
                           ('runtime', dict(p['runtime'], blas_threads=2)),
                           ('implementation_sources', list(reversed(p['implementation_sources']))),
                           ('implementation_sources', p['implementation_sources'][:-1])):
            with self.assertRaises(ValueError):
                self.target.validate_request_metadata('profile', self.mutate(p, key, value))
        a = requests['actual_initial32']
        for sources in (list(reversed(a['sources'])), a['sources'][:1]):
            with self.assertRaises(ValueError):
                self.target.validate_request_metadata('actual_initial32', self.mutate(a, 'sources', sources))
        for label in ('', 'outside/path', 'with space'):
            sources = copy.deepcopy(a['sources'])
            sources[0]['roots'] = {label: str(self.directory / 'source')}
            with self.assertRaises(ValueError):
                self.target.validate_request_metadata('actual_initial32', self.mutate(a, 'sources', sources))

    def test_budget_limits_exact_boundaries(self):
        budget = dict(peak_rss_bytes=8589934592, julia_gc_allocated_bytes=1, descriptor_growth=0)
        self.assertEqual(self.target.validate_budgets(budget), budget)
        for key, value in (('peak_rss_bytes', 8589934593), ('peak_rss_bytes', 0), ('peak_rss_bytes', True),
                           ('julia_gc_allocated_bytes', 0), ('descriptor_growth', -1), ('descriptor_growth', 0.0),
                           ('future', 0)):
            with self.assertRaises(ValueError):
                self.target.validate_budgets(dict(budget, **{key: value}))

    def test_bounded_json_duplicate_nonfinite_and_alias_rejection(self):
        path = self.directory / 'ordinary.json'
        path.write_bytes(b'{"value":3}\n')
        self.assertEqual(self.target.read_json(path), dict(value=3))
        for index, content in enumerate((b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{} trailing')):
            bad = self.directory / f'invalid{index}.json'
            bad.write_bytes(content)
            with self.assertRaises(ValueError):
                self.target.read_json(bad)
        huge = self.directory / 'oversize.json'
        with huge.open('wb') as stream:
            stream.truncate(META + 1)
        with self.assertRaises(ValueError):
            self.target.read_json(huge)
        with self.assertRaises(ValueError):
            self.target.read_json('relative.json')
        alias = self.directory / 'hardlink.json'
        os.link(path, alias)
        with self.assertRaises(ValueError):
            self.target.read_json(alias)

    def test_bridge_invalid_request_is_rejected_before_write(self):
        # No successful/prelaunch8GiB observation is fabricated for a unit test.
        with self.assertRaises(ValueError):
            self.target.write_bridge(self.directory, 'invalid', {})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_attempt_marker_survives_real_failed_child_and_owner_change(self):
        execution = str(self.directory / 'execution')
        first_owner = self.directory / 'owner-one'
        second_owner = self.directory / 'owner-two'
        first_owner.mkdir()
        second_owner.mkdir()
        marker = self.target.attempt_marker_path(execution, HEX)
        self.assertEqual(marker.parent, self.directory)
        self.assertEqual(marker.name, 'execution.matched342-' + HEX + '.attempt.json')
        payload = identified(dict(schema='matched342_permanent_actual_attempt_v1', execution_directory=execution,
            permit_sha256=HEX, request_sha256=HEX, owner_pid=os.getpid(), created_monotonic_ns=time.monotonic_ns()), 'marker_id')
        ref = self.target.create_attempt_marker(marker, payload)
        before = marker.read_bytes()
        result, raw, pid = self.native('fail')
        self.assertEqual(result['native_exit'], 7)
        self.assertTrue(result['leader_reaped'])
        self.assertTrue(result['ownership_boundary_closed'])
        self.assertEqual(raw.read_bytes(), b'PROFILE_OWNER_CHILD_FAIL\n')
        with self.assertRaises(ValueError):
            self.target.check_native_result(result, pid, time.monotonic_ns() + 10**9, raw)
        self.assertEqual(self.target.attempt_marker_path(execution, HEX), marker)
        # Changing the separate owner folder never changes this permit's marker.
        self.assertNotEqual(first_owner, second_owner)
        with self.assertRaises(ValueError):
            self.target.create_attempt_marker(self.target.attempt_marker_path(execution, HEX), payload)
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(sha(marker), ref['sha256'])

    def test_real_native_pid_wait_and_raw(self):
        result, raw, pid = self.native('ok')
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(raw.read_bytes(), b'PROFILE_OWNER_CHILD_OK\n')
        self.assertIs(self.target.check_native_result(result, pid, time.monotonic_ns() + 10**9, raw), result)
        self.assertFalse(result['all_descendants_os_reaped'])

    def test_native_predicates_reject_modified_observations(self):
        result, raw, pid = self.native('ok')
        for key, value in (('pid', pid + 1000000), ('pid', True), ('native_exit', 1), ('native_exit', False),
                           ('timed_out', True), ('leader_reaped', False), ('ownership_boundary_closed', False),
                           ('all_descendants_os_reaped', True), ('wall_seconds', float('nan')),
                           ('wall_seconds', True), ('future', None)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.target.check_native_result(dict(result, **{key: value}), pid, time.monotonic_ns() + 10**9, raw)
        with self.assertRaises(ValueError):
            self.target.check_native_result(result, pid, time.monotonic_ns() - 1, raw)
        with self.assertRaises(ValueError):
            self.target.check_native_result(result, os.getpid(), time.monotonic_ns() + 10**9, raw)
        huge = self.directory / 'oversize.raw'
        with huge.open('wb') as stream:
            stream.truncate(64 * 1024**2 + 1)
        with self.assertRaises(ValueError):
            self.target.check_native_result(result, pid, time.monotonic_ns() + 10**9, huge)
        with self.assertRaises(FileNotFoundError):
            self.target.check_native_result(result, pid, time.monotonic_ns() + 10**9, self.directory / 'missing.raw')

    def test_real_timeout_is_not_positive_wait(self):
        result, raw, pid = self.native('sleep', timeout=0.3)
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['leader_reaped'])
        self.assertTrue(result['ownership_boundary_closed'])
        with self.assertRaises(ValueError):
            self.target.check_native_result(result, pid, time.monotonic_ns() + 10**9, raw)

    def test_real_callback_exception_never_returns_native_success(self):
        launched = []
        raw = self.directory / 'callback-error.raw'

        def reject(pid):
            launched.append(pid)
            raise RuntimeError('independent callback rejection')

        with self.assertRaisesRegex(RuntimeError, 'independent callback rejection'):
            self.native_owner.run_owned(
                [sys.executable, '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--child', 'ok'],
                cwd=str(self.directory), environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                logpath=str(raw), timeout=10, on_launch=reject)
        self.assertEqual(len(launched), 1)
        self.assertIs(type(launched[0]), int)
        self.assertGreater(launched[0], 0)
        self.assertNotEqual(launched[0], os.getpid())
        self.assertTrue(raw.is_file())
        # An exception supplies no primitive result. Do not infer success,
        # closure or reaping merely from the callback having been invoked.
        self.native_rows.append(dict(mode='callback-error', actual_on_launch_pid=launched[0],
            expected_exception='RuntimeError: independent callback rejection', native_result_returned=False,
            native_exit_observed=False, leader_reaping_observed=False, ownership_boundary_closed_observed=False,
            raw=dict(path=str(raw), sha256=sha(raw), bytes=raw.stat().st_size)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or not args.output_directory.is_absolute():
        raise ValueError('unoptimized file-based engineering test and absolute output required')
    output = args.output_directory
    source_name, output_name = os.path.normcase(str(ROOT)), os.path.normcase(str(output))
    try:
        common = os.path.commonpath((source_name, output_name))
    except ValueError:
        common = None  # Distinct Windows volumes have no common ancestor.
    if common in (source_name, output_name):
        raise ValueError('engineering evidence must be outside the source tree')
    if output.exists() or not output.parent.is_dir():
        raise ValueError('fresh engineering evidence root required')
    memory = load(ROOT / 'tools/run_canonical_harness.py', 'profile_owner_engineering_memory')
    available, method = memory.available_memory()
    free = shutil.disk_usage(output.parent).free
    if available < 512 * 1024**2 or free < 8 * 1024**3:
        raise ValueError('engineering512MiB/8GiB admission failed')
    if sha(TARGET) != TARGET_SHA:
        raise ValueError('held runtime owner changed')
    target = load(TARGET, 'profile_owner_contract_target')
    if sha(target.OWNER) != target.OWNER_SHA or sha(target.LEGACY) != target.LEGACY_SHA:
        raise ValueError('fixed native owner bytes changed')
    output.mkdir()
    OwnerContractTests.output = output
    OwnerContractTests.target = target
    OwnerContractTests.native_owner = load(target.OWNER, 'profile_owner_contract_fixed_native')
    started = time.monotonic_ns()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OwnerContractTests)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    protected = sha(TARGET) == TARGET_SHA and sha(target.OWNER) == target.OWNER_SHA and sha(target.LEGACY) == target.LEGACY_SHA
    record = dict(schema='matched342_owner_small_engineering_tests_v1', complete=result.wasSuccessful() and protected,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        start_ns=started, end_ns=time.monotonic_ns(), target_sha256=TARGET_SHA, protected_source_pins_unchanged=protected,
        admission=dict(available_physical_bytes=available, memory_method=method, free_disk_bytes=free),
        native_rows=OwnerContractTests.native_rows, Julia_launched=False, Q_issued=False, actual_permit_admitted=False,
        source_qualified=False, scientific_qualified=False)
    with (output / 'RESULT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        mode = sys.argv[2]
        if mode == 'ok':
            sys.stdout.buffer.write(b'PROFILE_OWNER_CHILD_OK\n')
            sys.stdout.buffer.flush()
            raise SystemExit(0)
        if mode == 'fail':
            sys.stdout.buffer.write(b'PROFILE_OWNER_CHILD_FAIL\n')
            sys.stdout.buffer.flush()
            raise SystemExit(7)
        if mode == 'sleep':
            sys.stdout.buffer.write(b'PROFILE_OWNER_CHILD_SLEEP\n')
            sys.stdout.buffer.flush()
            time.sleep(30)
            raise SystemExit(0)
        raise SystemExit(2)
    raise SystemExit(main())
