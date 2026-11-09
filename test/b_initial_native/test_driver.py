"""Independent B1 driver controls; generated metadata/owned Python only."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRODUCTION = ROOT / 'tools/actual_b_chain/run_initial_l0_native.py'
CHILD = HERE / 'fixed_child.py'
OWNER = ROOT / 'research/dbdt_completed_source/candidate_native_owner.py'
UTILITY = ROOT / 'research/dbdt_completed_source/run_candidate_stage.py'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def exclusive(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


class ShortWriter:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.body = bytearray()

    def write(self, body):
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        if type(reply) is int and 0 < reply <= len(body):
            self.body.extend(body[:reply])
        return reply


class BInitialNativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_module(PRODUCTION, 'b1_production_under_independent_test')
        cls.owner = load_module(OWNER, 'b1_genuine_fixture_owner')
        cls.utility = load_module(UTILITY, 'b1_fixed_bounded_pump')
        cls.literal = json.loads((HERE / 'literal_oracles_v1.json').read_bytes())
        cls.api = json.loads((HERE / 'api_oracles_v2.json').read_bytes())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix='solarsindy-b1-independent-', dir=os.path.realpath(tempfile.gettempdir()))
        self.root = Path(self.temporary.name).resolve(strict=True)

    def tearDown(self):
        result = getattr(self._outcome, 'result', None)
        failures = [] if result is None else list(result.failures) + list(result.errors)
        if any(getattr(test, 'test_case', test) is self for test, _ in failures):
            # Preserve the complete genuine child raws on an unexpected test
            # failure; successful generated fixtures are removed normally.
            self.temporary._finalizer.detach()
            print('B1_FAILURE_FIXTURE_ROOT=' + self.temporary.name, file=sys.stderr)
        else:
            self.temporary.cleanup()

    def timing(self):
        # All program pins are current immutable input bytes, not future output
        # pins or an authority. This object never admits an actual B source.
        return dict(schema='b_initial_l0_timing_v2', request_sha256='a' * 64,
                    intent_sha256='b' * 64, execution_directory=str(self.root),
                    head='c' * 40, parent_pid=os.getpid(),
                    python_sha256=file_sha(sys.executable), start_ns=1000000000,
                    deadline_ns=7201000000000, driver_sha256=file_sha(PRODUCTION),
                    entry_sha256=file_sha(PRODUCTION.with_suffix('.jl')),
                    owner_sha256='ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21',
                    legacy_owner_sha256='15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8')

    def owned_child(self, mode):
        directory = self.root / mode
        directory.mkdir()
        native = self.owner.run_owned(
            [sys.executable, '-I', '-S', '-u', '-B', '-X', 'utf8', str(CHILD),
             mode, str(directory)], cwd=str(ROOT),
            environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            logpath=directory / 'startup.log', timeout=12)
        self.assertIs(type(native['native_exit']), int)
        self.assertIs(native['timed_out'], False)
        self.assertIs(native['leader_reaped'], True)
        self.assertIs(native['ownership_boundary_closed'], True)
        self.assertIs(native['all_descendants_os_reaped'], False)
        return directory, native

    def diagnostic(self, directory):
        body = (directory / 'SUPERVISOR_DIAGNOSTIC_WAIT.json').read_bytes()
        self.assertLessEqual(len(body), 16384)
        result = json.loads(body)
        self.assertEqual(set(result), {'schema', 'capture_complete', 'truncated',
                         'overflow_observed', 'reason', 'retained_bytes',
                         'observed_bytes', 'live_limit_bytes'})
        self.assertEqual(result['schema'], 'b_supervisor_diagnostic_wait_v1')
        self.assertEqual(result['live_limit_bytes'], 8388608)
        for key in ('capture_complete', 'truncated', 'overflow_observed'):
            self.assertIs(type(result[key]), bool)
        for key in ('retained_bytes', 'observed_bytes', 'live_limit_bytes'):
            self.assertIs(type(result[key]), int)
        return result

    def negative_parent(self, *, inner_exit=0, outer_exit=7, missing_inner=False,
                        lose_launch=False, callback_write_failure=False,
                        startup_bytes=0, late=False):
        """Control-flow-only negative: no authority/Julia/native success exists.

        The admission dependency is stubbed solely to reach failure accounting.
        Every produced final receipt must remain complete=false; genuine owner
        exits are tested separately using fixed_child and the unchanged owner.
        """
        root = self.root / 'negative-parent'
        root.mkdir()
        ref = {'path': str(root / 'not-an-authority.json'), 'sha256': 'd' * 64}
        request = dict(schema='b_initial_l0_native_request_v2',
                       evidence_mode='archival_assumed_completion_research',
                       operation='initial-l0', execution_directory=str(root),
                       intent=dict(ref), permit=dict(ref), source_proof=dict(ref),
                       root_qualification=dict(ref), b_execution_qualification=dict(ref),
                       julia={'path': sys.executable, 'sha256': file_sha(sys.executable)},
                       runtime=dict(ref))
        path = root / 'invalid-fixture-request.json'
        exclusive(path, request)
        request_sha = file_sha(path)
        returned = [False]
        def invoke(argv, **kw):
            if lose_launch:
                raise OSError('fixed lost launch observation')
            kw['on_launch'](12345)
            with Path(kw['logpath']).open('xb') as stream:
                block = b'B' * 65536
                left = startup_bytes
                while left:
                    size = min(left, len(block))
                    stream.write(block[:size])
                    left -= size
            (root / 'supervisor.raw').write_bytes(b'')
            exclusive(root / 'SUPERVISOR_DIAGNOSTIC_WAIT.json', dict(
                schema='b_supervisor_diagnostic_wait_v1', capture_complete=True,
                truncated=False, overflow_observed=False, reason='eof', retained_bytes=0, observed_bytes=0,
                live_limit_bytes=8388608))
            if not missing_inner:
                (root / 'julia.log').write_bytes(b'generated-negative-only\n')
                exclusive(root / 'JULIA_NATIVE_WAIT.json', dict(
                    pid=23456, native_exit=inner_exit, complete=True, timed_out=False,
                    raw_byte_limit_exceeded=False, reader_joined=True,
                    reader_pipe_close_pending=False, exact_child_reaped=True,
                    raw_bytes=(root / 'julia.log').stat().st_size,
                    raw_sha256=file_sha(root / 'julia.log')))
            returned[0] = True
            return dict(pid=12345, native_exit=outer_exit, timed_out=False,
                        leader_reaped=True, ownership_boundary_closed=True,
                        descendant_evidence='mock-only negative control')
        original_write = self.driver.write
        def publish(path, value):
            if callback_write_failure and Path(path).name == 'B1_OWNER_LAUNCH.json':
                raise OSError('fixed launch publication failure')
            return original_write(path, value)
        def observed_clock():
            return 7201000000001 if late and returned[0] else 1000000000
        clock = mock.patch.object(self.driver.time, 'monotonic_ns', side_effect=observed_clock)
        with mock.patch.object(self.driver, 'admit_request', return_value=(
                request, {}, {'head': 'c' * 40}, {}, {}, 8589934592,
                'generated mock-only threshold', 17179869184)), \
                mock.patch.object(self.driver, 'utility', return_value=SimpleNamespace(
                    source_guard=lambda proof: None)), \
                mock.patch.object(self.driver, 'module', return_value=SimpleNamespace(
                    run_owned=invoke)), mock.patch.object(self.driver, 'write',
                                                         side_effect=publish), clock:
            code = self.driver.main(path, request_sha)
        receipt = json.loads((root / 'B1_NATIVE_RECEIPT.json').read_bytes())
        self.assertEqual(code, 1)
        self.assertIs(receipt['complete'], False)
        self.assertIs(receipt['scientific_qualification'], False)
        self.assertIs(receipt['serving_promotion'], False)
        self.assertTrue((root / 'B1_NATIVE_ATTEMPT.json').is_file())
        return root, receipt

    def test_independent_literals_and_fixed_contract(self):
        self.assertEqual(file_sha(HERE / 'literal_oracles_v1.json'),
                         'a7a206f6792c89ebb8c8e8db0fa61d05f295d2f9d39c34407789eaa4e5817401')
        self.assertEqual(file_sha(HERE / 'api_oracles_v2.json'),
                         '5033cdea62da25a27449744a69fff6316daf7545a23dc488cbfb831525c53ff8')
        d = self.driver
        self.assertEqual(list(d.REQUEST_FIELDS), self.api['request_exact_keys'])
        self.assertEqual(list(d.INTENT_FIELDS), self.api['intent_exact_keys'])
        self.assertEqual(list(d.PERMIT_FIELDS), self.api['old_PERMIT_FIELDS_exact'] +
                         self.api['v2_additional_fields_exact'])
        self.assertEqual((d.SECONDS, d.RAM, d.DISK, d.RAW_LIMIT),
                         (7200, 8589934592, 17179869184, 67108864))
        self.assertEqual(d.OWNER_SHA, self.api['fixed_owner_sha256'])
        self.assertEqual(d.LEGACY_SHA, self.api['fixed_preserved_helper_sha256'])
        self.assertEqual(file_sha(OWNER), self.api['fixed_owner_sha256'])
        self.assertEqual(len(self.literal['generated_numerical']['setting_rows']), 36)
        self.assertEqual(file_sha(HERE / 'closure_oracles_v3.json'),
                         '63c62843c33c6945c35d84689214d1be7453351e6bc72fc85b58260ce8e9f0bc')
        closure = json.loads((HERE / 'closure_oracles_v3.json').read_bytes())
        names = (closure['old_training_literal19'] + closure['old_dependencies10'] +
                 closure['approved_additions7'])
        self.assertEqual(len(names), 36)
        self.assertEqual(len(set(names)), 36)
        self.assertEqual(set(d.B_SOURCE_FILES), set(names))
        source = {name: file_sha(ROOT / name) for name in names}
        rows = [dict(name=name, sha256=source[name]) for name in sorted(source)]
        proof = {'source_sha256': dict(source)}
        self.assertEqual(d._sources(rows, proof), source)
        for omitted in names:
            # Coherently dropping a name from BOTH untrusted metadata records
            # must not redefine the required old29+new7 implementation closure.
            partial_rows = [row for row in rows if row['name'] != omitted]
            partial_proof = {'source_sha256': {name: sha for name, sha in source.items()
                                               if name != omitted}}
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                d._sources(partial_rows, partial_proof)
        with self.assertRaises(ValueError):
            d._sources([rows[1], rows[0], *rows[2:]], proof)
        with self.assertRaises(ValueError):
            d._sources([*rows, dict(rows[0])], proof)
        added = 'test/generated-unapproved.jl'
        with self.assertRaises(ValueError):
            d._sources(sorted([*rows, dict(name=added, sha256='0' * 64)],
                              key=lambda row: row['name']),
                       {'source_sha256': dict(source, **{added: '0' * 64})})

    def test_canonical_JSON_positive_and_structural_negatives(self):
        path = self.root / 'metadata.json'
        path.write_bytes(b'{"x":1}\n')
        self.assertEqual(self.driver.load(path, canonical_required=True), {'x': 1})
        bad = [b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n',
               b'{"x":1}{"x":2}\n', b'{"x": 1}\n', b'[' * 14 + b'0' + b']' * 14,
               b' ' * (2097152 + 1)]
        for body in bad:
            with self.subTest(body=body[:40]):
                path.write_bytes(body)
                with self.assertRaises((ValueError, UnicodeError)):
                    self.driver.load(path, canonical_required=True)

    def test_write_all_positive_partial_and_failures(self):
        stream = ShortWriter([3, 1])
        self.driver.write_all(stream, b'ABCD')
        self.assertEqual(stream.body, b'ABCD')
        for replies in ([3, 0], [3, OSError('fixture')], [5], [True], [-1]):
            with self.subTest(replies=replies):
                stream = ShortWriter(replies)
                with self.assertRaises((ValueError, OSError)):
                    self.driver.write_all(stream, b'ABCD')

    def test_absolute_deadline_remaining_and_no_reset(self):
        context = self.timing()
        with mock.patch.object(self.driver.time, 'monotonic_ns', return_value=6000000000):
            self.assertEqual(self.driver.validate_timing(
                context, 'a' * 64, self.root, parent_pid=os.getpid()), context)
            self.assertEqual(self.driver.remaining_seconds(context), 7195)
        for observed in (7201000000000, 7201000000001, 999999999):
            with self.subTest(observed=observed), mock.patch.object(
                    self.driver.time, 'monotonic_ns', return_value=observed):
                with self.assertRaises(ValueError):
                    self.driver.remaining_seconds(context)
        self.assertNotEqual(7195, 7200)

    def test_timing_context_identity_and_integer_grammar(self):
        valid = self.timing()
        mutations = [('schema', 'other'), ('request_sha256', 'd' * 64),
                     ('execution_directory', str(self.root / 'other')),
                     ('parent_pid', True), ('parent_pid', os.getpid() + 1),
                     ('start_ns', True), ('deadline_ns', float(valid['deadline_ns'])),
                     ('deadline_ns', valid['deadline_ns'] + 1),
                     ('driver_sha256', '0' * 64), ('entry_sha256', '0' * 64),
                     ('python_sha256', '0' * 64),
                     ('owner_sha256', self.api['legacy_v1_owner_sha256_not_v2']),
                     ('legacy_owner_sha256', '0' * 64)]
        with mock.patch.object(self.driver.time, 'monotonic_ns', return_value=6000000000):
            for key, value in mutations:
                candidate = dict(valid, **{key: value})
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.driver.validate_timing(candidate, 'a' * 64, self.root,
                                                parent_pid=os.getpid())
            for name in valid:
                candidate = dict(valid)
                del candidate[name]
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    self.driver.validate_timing(candidate, 'a' * 64, self.root)

    def test_request_shape_rejects_before_source_or_owner(self):
        ref = {'path': str(self.root / 'missing.json'), 'sha256': '0' * 64}
        base = {name: dict(ref) for name in self.api['request_exact_keys']}
        base.update(schema='b_initial_l0_native_request_v2',
                    evidence_mode='archival_assumed_completion_research',
                    operation='initial-l0', execution_directory=str(self.root))
        cases = [dict(base, operation='initial-l1'), dict(base, operation='selected-refit'),
                 dict(base, evidence_mode='synthetic_only'), dict(base, unexpected=True)]
        for name in self.api['request_exact_keys']:
            value = dict(base)
            del value[name]
            cases.append(value)
        with mock.patch.object(self.driver, 'utility') as utility:
            for index, request in enumerate(cases):
                path = self.root / ('wrong-request-%d.json' % index)
                exclusive(path, request)
                with self.subTest(index=index), self.assertRaises((ValueError, KeyError)):
                    self.driver.admit_request(path, file_sha(path))
            utility.assert_not_called()
        self.assertFalse((self.root / 'B1_NATIVE_ATTEMPT.json').exists())

    def test_every_stale_output_is_rejected_before_source_body(self):
        ref = {'path': str(self.root / 'missing.json'), 'sha256': '0' * 64}
        for index, name in enumerate(('B1_NATIVE_ATTEMPT.json', 'B1_OWNER_LAUNCH.json',
                'B1_TIMING.json', 'b1-argv.json', 'CONTROLS_BEFORE.json',
                'CONTROLS_AFTER.json', 'CONTROL_READ_ERRORS.json', 'supervisor-startup.log',
                'supervisor.raw', 'SUPERVISOR_DIAGNOSTIC_WAIT.json', 'SUPERVISOR_FAILURE.json',
                'julia.log', 'JULIA_NATIVE_WAIT.json', 'B1_JULIA_RESULT.json',
                'B1_NATIVE_RECEIPT.json', 'output')):
            root = self.root / ('stale%d' % index)
            root.mkdir()
            request = {key: dict(ref) for key in self.api['request_exact_keys']}
            request.update(schema='b_initial_l0_native_request_v2',
                           evidence_mode='archival_assumed_completion_research',
                           operation='initial-l0', execution_directory=str(root))
            path = root / 'request.json'
            exclusive(path, request)
            marker = root / name
            marker.mkdir() if name == 'output' else marker.write_bytes(b'prior evidence')
            with self.subTest(name=name), mock.patch.object(self.driver, 'utility') as utility:
                with self.assertRaisesRegex(ValueError, 'spent'):
                    self.driver.admit_request(path, file_sha(path))
                utility.assert_not_called()
            self.assertTrue(marker.exists())

    def test_fixed_scope_order_and_Boolean_integer_aliases(self):
        scope = dict(training_semantics='option_T_in_sample_two_stage_batch_v1',
                     training_boundary='63713520000000', validation_start='63713520000000',
                     validation_end='63808214400000', embargo_hours=168, delay_hours=0,
                     beta_order=[5, 20], b_setting_indices=list(range(1, 37)),
                     initial_hazard_fit_count=2, initial_gain_fit_count=36,
                     state_schedule_count=9, beta_design_count=18,
                     final_fit_count=0, calibration_fit_count=0,
                     exposed_development_scoring=False, joint_primary_selection=False,
                     scientific_verdict=False, serving_promotion=False)
        self.assertEqual(self.driver._scope(scope), scope)
        for key, value in (('delay_hours', True), ('initial_gain_fit_count', 35),
                           ('initial_gain_fit_count', 36.0), ('beta_order', [20, 5]),
                           ('b_setting_indices', list(range(2, 38))),
                           ('final_fit_count', 1), ('calibration_fit_count', 1),
                           ('joint_primary_selection', True), ('scientific_verdict', True),
                           ('serving_promotion', True), ('embargo_hours', 167)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.driver._scope(dict(scope, **{key: value}))
        runtime = dict(julia_version='1.12.6', executable_sha256='0' * 64,
                       architecture='generated-metadata-only', argv=['generated-metadata-only-no-execution'],
                       julia_project_sha256='1' * 64, julia_manifest_sha256='2' * 64,
                       blas_vendor='generated', blas_version='generated',
                       blas_library_sha256='3' * 64, blas_threads=1, omp_threads=1,
                       rng_kind='none', rng_seed=0, compiled_modules=False, pkgimages=False)
        self.assertEqual(self.driver.validate_runtime(runtime, '0' * 64), runtime)
        with self.assertRaises(ValueError):
            self.driver.validate_runtime(dict(runtime, argv=[]), '0' * 64)
        for key in ('blas_threads', 'omp_threads', 'rng_seed'):
            for value in (True, False):
                with self.subTest(runtime_key=key, value=value), self.assertRaises(ValueError):
                    self.driver.validate_runtime(dict(runtime, **{key: value}), '0' * 64)
        for key in runtime:
            wrong = dict(runtime)
            del wrong[key]
            with self.subTest(missing_runtime_key=key), self.assertRaises(ValueError):
                self.driver.validate_runtime(wrong, '0' * 64)
        with self.assertRaises(ValueError):
            self.driver.validate_runtime(dict(runtime, extra=True), '0' * 64)

    def test_resource_threshold_rejection_is_before_attempt(self):
        for memory, disk in ((8589934591, 17179869184), (8589934592, 17179869183),
                             (True, 17179869184), (8589934592, True)):
            hardware = SimpleNamespace(available_memory=lambda: (memory, 'fixed fixture'))
            def admit(*args):
                return self.driver.resource_admission(self.root)
            with self.subTest(memory=memory, disk=disk), \
                    mock.patch.object(self.driver, 'module', return_value=hardware), \
                    mock.patch.object(self.driver.shutil, 'disk_usage',
                                      return_value=SimpleNamespace(free=disk)), \
                    mock.patch.object(self.driver, 'admit_request', side_effect=admit), \
                    mock.patch.object(self.driver, 'utility') as utility:
                with self.assertRaises(ValueError):
                    self.driver.main(self.root / 'never-read.json', '0' * 64)
                utility.assert_not_called()
            self.assertFalse((self.root / 'B1_NATIVE_ATTEMPT.json').exists())
        with mock.patch.object(self.driver, 'module', return_value=SimpleNamespace(
                available_memory=lambda: (8589934592, 'fixed fixture'))), \
                mock.patch.object(self.driver.shutil, 'disk_usage',
                                  return_value=SimpleNamespace(free=17179869184)):
            self.assertEqual(self.driver.resource_admission(self.root),
                             (8589934592, 'fixed fixture', 17179869184))

    def test_negative_parent_preserves_separate_codes_and_unknown_inner(self):
        for index, (inner, outer, missing) in enumerate(((0, 7, False), (7, 0, False),
                                                       (0, 0, True))):
            # Separate generated roots keep every failure/spent attempt immutable.
            original = self.root
            self.root = original / ('case%d' % index)
            self.root.mkdir()
            try:
                _, receipt = self.negative_parent(inner_exit=inner, outer_exit=outer,
                                                  missing_inner=missing)
                self.assertEqual(receipt['supervisor_native_exit'], outer)
                self.assertIsNone(receipt['native_exit']) if missing else self.assertEqual(
                    receipt['native_exit'], inner)
                self.assertEqual(receipt['owner_invocations'], 1)
                self.assertEqual(receipt['observed_owner_launches'], 1)
                self.assertIs(receipt['owner_launch_count_known'], True)
                self.assertIs(receipt['julia_launch_count_known'], not missing)
            finally:
                self.root = original

    def test_lost_launch_and_callback_write_failures_remain_spent(self):
        for index, callback in enumerate((False, True)):
            original = self.root
            self.root = original / ('case%d' % index)
            self.root.mkdir()
            try:
                _, receipt = self.negative_parent(lose_launch=not callback,
                                                  callback_write_failure=callback)
                self.assertEqual(receipt['owner_invocations'], 1)
                self.assertIsNone(receipt['supervisor_native_exit'])
                self.assertIsNone(receipt['native_exit'])
                self.assertIs(receipt['ownership_boundary_closed'], False)
                self.assertIs(receipt['owner_launch_count_known'], callback)
                self.assertEqual(receipt['observed_owner_launches'], int(callback))
                if callback:
                    self.assertEqual(receipt['observed_owner_pid'], 12345)
            finally:
                self.root = original

    def test_startup_transport_overcap_is_post_failure_retaining_all_bytes(self):
        root, receipt = self.negative_parent(inner_exit=0, outer_exit=0,
                                             startup_bytes=8388609)
        self.assertEqual((root / 'supervisor-startup.log').stat().st_size, 8388609)
        self.assertEqual(receipt['startup_transport_bytes'], 8388609)
        self.assertIs(receipt['startup_transport_has_live_quota'], False)
        self.assertEqual(receipt['supervisor_native_exit'], 0)
        self.assertEqual(receipt['native_exit'], 0)

    def test_late_zero_preserves_both_zero_codes_and_fails(self):
        _, receipt = self.negative_parent(inner_exit=0, outer_exit=0, late=True)
        self.assertEqual(receipt['native_exit'], 0)
        self.assertEqual(receipt['supervisor_native_exit'], 0)
        self.assertIs(receipt['whole_owned_phase_timed_out'], True)

    def test_supervisor_resource_failure_has_existing_spent_markers_no_Julia(self):
        context = self.timing()
        context['parent_pid'] = os.getppid()
        timing = self.root / 'B1_TIMING.json'
        exclusive(timing, context)
        timing_sha = file_sha(timing)
        exclusive(self.root / 'B1_OWNER_LAUNCH.json', dict(
            pid=os.getpid(), owner_pid=os.getppid(), request_sha256='a' * 64,
            timing_sha256=timing_sha))
        exclusive(self.root / 'B1_NATIVE_ATTEMPT.json', dict(
            request_sha256='a' * 64, timing_sha256=timing_sha))
        request = dict(execution_directory=str(self.root), intent={'sha256': 'b' * 64},
                       julia={'path': sys.executable})
        path = self.root / 'never-read-request.json'
        exclusive(self.root / 'b1-argv.json', self.driver.julia_argv(request, path, 'a' * 64))
        provider = SimpleNamespace(pump_process=mock.Mock())
        with mock.patch.object(self.driver.sys, 'argv', ['fixed', '--owned-supervisor',
                '--execution-root', str(self.root)]), \
                mock.patch.object(self.driver.time, 'monotonic_ns', return_value=6000000000), \
                mock.patch.object(self.driver, 'admit_request', return_value=(
                    request, {}, {'head': 'c' * 40}, {}, {}, 8589934592,
                    'generated fixture', 17179869184)) as admission, \
                mock.patch.object(self.driver, 'utility', return_value=provider), \
                mock.patch.object(self.driver, 'resource_admission',
                                  side_effect=ValueError('fixed supervisor RAM drop')):
            with self.assertRaisesRegex(ValueError, 'RAM drop'):
                self.driver.supervisor(path, 'a' * 64, timing_sha)
            admission.assert_called_once_with(path, 'a' * 64, started=True)
            provider.pump_process.assert_not_called()
        self.assertTrue((self.root / 'B1_NATIVE_ATTEMPT.json').is_file())
        self.assertFalse((self.root / 'JULIA_NATIVE_WAIT.json').exists())
        self.assertFalse((self.root / 'julia.log').exists())

    def test_generated_empty_tree_result_metadata_is_not_model_qualification(self):
        path = ROOT / 'test/b_point_training/tree_inventory_oracles_v1.json'
        self.assertEqual(file_sha(path),
                         'e829beee61d48333689fea77527c5f6593e86412a1a21d66ab84ab917a4ebc96')
        oracle = json.loads(path.read_bytes())
        output = self.root / 'empty-inventory-only'
        output.mkdir()
        for name in oracle['directories']:
            output.joinpath(*name.split('/')).mkdir(parents=True, exist_ok=True)
        for name in oracle['files']:
            output.joinpath(*name.split('/')).write_bytes(b'')
        completion = dict(schema='source_bound_b_initial_l0_point_stage_v1',
                          complete=True, stage='initial_l0', evidence_mode='synthetic_only',
                          output_sha256='11c371b483f09bb37935d526b5dc1336c82a718d44872e66fe45f9374472bc9b',
                          output_file_count=423, output_directory_count=116, output_bytes=0,
                          provisional_selection={'selected_index': None},
                          initial_hazard_fit_count=2, initial_gain_fit_count=36,
                          state_schedule_count=9, beta_design_count=18, final_fit_count=0,
                          calibration_fit_count=0, exposed_development_scoring=False,
                          joint_primary_selection=False, scientific_verdict=False,
                          serving_promotion=False)
        exclusive(output / 'completion.json', completion)
        result = dict(schema='b_initial_l0_internal_result_v2', complete=True,
                      evidence_mode='synthetic_only', actual_source_access=False,
                      initial_hazard_fit_count=2, initial_gain_fit_count=36,
                      state_schedule_count=9, beta_design_count=18, final_fit_count=0,
                      calibration_fit_count=0, selected_index=None, scientific_verdict=False,
                      serving_promotion=False, stage_directory=str(output),
                      completion_sha256=file_sha(output / 'completion.json'),
                      output_sha256=completion['output_sha256'], output_file_count=423,
                      output_directory_count=116, verified_model_count=36,
                      verified_certificate_count=36, verified_hazard_fit_calls=0,
                      verified_gain_fit_calls=0, verified_raw_replay_A_streams=9,
                      fit_elapsed_ns=1, readback_elapsed_ns=1, worker_elapsed_ns=2,
                      julia_peak_rss_bytes=1, julia_peak_rss_measured=True)
        # This validator checks data-only geometry/checksums, not fitted model
        # validity. Only the real generated Julia replay below can prove that.
        self.assertEqual(self.driver.validate_result(result, None, output, actual=False), result)
        for key, value in (('initial_gain_fit_count', 35), ('verified_certificate_count', 35),
                           ('verified_hazard_fit_calls', 1), ('verified_gain_fit_calls', 1),
                           ('verified_model_count', True), ('selected_index', 1),
                           ('final_fit_count', 1), ('calibration_fit_count', 1),
                           ('actual_source_access', True), ('scientific_verdict', True),
                           ('serving_promotion', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.driver.validate_result(dict(result, **{key: value}), None, output, actual=False)
        member = output / 'points/setting-36.bin'
        member.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.driver.validate_result(result, None, output, actual=False)
        member.write_bytes(b'')
        extra = output / 'unexpected-empty-directory'
        extra.mkdir()
        with self.assertRaises(ValueError):
            self.driver.validate_result(result, None, output, actual=False)

    def test_fixed_runner_summary_LF_CRLF_and_malformed_controls(self):
        runner = load_module(HERE / 'run_controls.py', 'b1_summary_parser_only')
        for newline in (b'\n', b'\r\n'):
            valid = newline.join((b'generated controls', b'Ran 24 tests in 0.001s', b'', b'OK', b''))
            runner.validate_summary(valid)
        bad = [b'Ran 23 tests in 0.001s\nOK\n', b'Ran 24 tests in 0.001s\nOK\nOK\n',
               b'Ran 24 tests in 0.001s\nOK (skipped=1)\n',
               b'Ran 24 tests in 0.001s\nFAILED (failures=1)\n',
               b'Ran 24 tests in 0.001s\nOK\nERROR: hidden\n',
               b'Ran 24 tests in 0.001s\nOK\nRan 24 tests in 0.001s\n',
               b'Ran 24 tests in 0.001s\nSKIPPED\nOK\n', b'OK\n']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises((ValueError, RuntimeError)):
                runner.validate_summary(raw)

    def test_real_owned_exit_codes_are_separate_observations(self):
        for mode, expected in (('exit0', 0), ('exit7', 7)):
            with self.subTest(mode=mode):
                directory, native = self.owned_child(mode)
                self.assertEqual(native['native_exit'], expected)
                self.assertEqual((directory / 'startup.log').read_bytes(),
                                 ('B1_FIXED_EXIT%d\n' % expected).encode())
                self.assertFalse((directory / 'JULIA_NATIVE_WAIT.json').exists())

    def test_diagnostic_below_cap_real_EOF_and_receipt(self):
        directory, native = self.owned_child('diagnostic-below')
        self.assertEqual(native['native_exit'], 0)
        record = self.diagnostic(directory)
        self.assertIs(record['capture_complete'], True)
        self.assertIs(record['truncated'], False)
        self.assertIs(record['overflow_observed'], False)
        self.assertEqual((record['retained_bytes'], record['observed_bytes']), (17, 17))
        self.assertEqual((directory / 'supervisor.raw').read_bytes(), b'B' * 17)
        state = json.loads((directory / 'FIXTURE_RESULT.json').read_bytes())
        self.assertEqual(set(state), {'reader_joined', 'capture_complete', 'retained_bytes',
                         'observed_bytes', 'overflow_observed', 'truncated', 'reason'})
        self.assertIs(state['reader_joined'], True)
        self.assertIs(state['capture_complete'], True)

    def test_diagnostic_at_cap_and_mixed_fd_capture(self):
        for mode in ('diagnostic-at', 'diagnostic-mixed'):
            with self.subTest(mode=mode):
                directory, native = self.owned_child(mode)
                self.assertEqual(native['native_exit'], 0)
                record = self.diagnostic(directory)
                self.assertIs(record['capture_complete'], True)
                self.assertIs(record['truncated'], False)
                self.assertIs(record['overflow_observed'], False)
                self.assertEqual(record['retained_bytes'], 8388608)
                self.assertEqual(record['observed_bytes'], 8388608)
                self.assertEqual(file_sha(directory / 'supervisor.raw'),
                                 hashlib.sha256(b'B' * 8388608).hexdigest())
                self.assertEqual((directory / 'startup.log').stat().st_size, 0)

    def test_diagnostic_live_overflow_preserves_true_outer1(self):
        for mode in ('diagnostic-overflow', 'diagnostic-overflow-fsync-failure',
                     'diagnostic-overflow-close-failure'):
            with self.subTest(mode=mode):
                directory, native = self.owned_child(mode)
                self.assertEqual(native['native_exit'], 1)
                record = self.diagnostic(directory)
                self.assertIs(record['capture_complete'], False)
                self.assertIs(record['truncated'], True)
                self.assertIs(record['overflow_observed'], True)
                self.assertEqual(record['retained_bytes'], 8388608)
                self.assertGreater(record['observed_bytes'], 8388608)
                self.assertEqual((directory / 'supervisor.raw').stat().st_size, 8388608)
                if mode == 'diagnostic-overflow':
                    self.assertEqual(record['reason'], 'overflow')
                else:
                    self.assertEqual(record['reason'], 'close_failure')
                self.assertFalse((directory / 'JULIA_NATIVE_WAIT.json').exists())

    def test_diagnostic_record_publication_failure_is_native1(self):
        directory, native = self.owned_child('diagnostic-record-failure')
        self.assertEqual(native['native_exit'], 1)
        self.assertEqual(json.loads((directory / 'SUPERVISOR_DIAGNOSTIC_WAIT.json').read_bytes()),
                         {'fixture_poison': True})
        self.assertFalse((directory / 'JULIA_NATIVE_WAIT.json').exists())

    def test_diagnostic_storage_and_read_failure_are_native1(self):
        for mode, reason in (('diagnostic-write-failure', 'write_failure'),
                             ('diagnostic-read-failure', 'read_failure')):
            with self.subTest(mode=mode):
                directory, native = self.owned_child(mode)
                self.assertEqual(native['native_exit'], 1)
                record = self.diagnostic(directory)
                self.assertIs(record['capture_complete'], False)
                self.assertIs(record['truncated'], reason == 'write_failure')
                self.assertIs(record['overflow_observed'], False)
                self.assertEqual(record['reason'], reason)
                self.assertLess(record['retained_bytes'], 8388608)
                self.assertFalse((directory / 'JULIA_NATIVE_WAIT.json').exists())

    def test_inherited_diagnostic_writer_blocks_positive_EOF(self):
        directory, native = self.owned_child('diagnostic-held-writer')
        self.assertEqual(native['native_exit'], 1)
        self.assertLess(native['wall_seconds'], 12)
        result = json.loads((directory / 'FIXTURE_RESULT.json').read_bytes())
        self.assertIs(result['reader_joined'], False)
        self.assertIs(result['capture_complete'], False)
        self.assertEqual(result['reason'], 'reader_unjoined')
        self.assertFalse((directory / 'JULIA_NATIVE_WAIT.json').exists())

    def test_real_Julia_stream_limit_is_separate_from_diagnostics(self):
        for mode, expected_overflow in (('raw-at64MiB', False), ('raw-over64MiB', True)):
            directory = self.root / mode
            directory.mkdir()
            argv = [sys.executable, '-I', '-S', '-u', '-B', '-X', 'utf8',
                    str(CHILD), mode, str(directory)]
            native = self.utility.pump_process(argv, str(ROOT), dict(os.environ),
                                               directory / 'julia.log',
                                               limit=67108864, timeout=12)
            with self.subTest(mode=mode):
                self.assertIs(native['exact_child_reaped'], True)
                self.assertIs(native['reader_joined'], True)
                self.assertIs(native['raw_byte_limit_exceeded'], expected_overflow)
                self.assertEqual(native['raw_bytes'], 67108864)
                self.assertEqual((directory / 'julia.log').stat().st_size, 67108864)
                self.assertEqual(native['raw_sha256'], file_sha(directory / 'julia.log'))
                self.assertIs(native['complete'], not expected_overflow)
                if not expected_overflow:
                    self.assertIs(type(native['native_exit']), int)
                    self.assertEqual(native['native_exit'], 0)
                self.assertFalse((directory / 'supervisor.raw').exists())


if __name__ == '__main__':
    if sys.flags.optimize or sys.version_info < (3, 11):
        raise SystemExit('unoptimized Python>=3.11 required')
    unittest.main(verbosity=2)
