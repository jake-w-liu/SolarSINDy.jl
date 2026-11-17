"""Independent inert identity evidence probes; no native execution or authority."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
SUBJECT = ROOT / 'tools/actual_b_chain/materialize_source_qualification.py'
ORACLE = HERE / 'current_identity_coverage_oracles_v2.json'
ORACLE_SHA = 'd4e578658cbdcb94fc82cb4f943ae372eef1ef7fd48a7bc81344ede7320fed01'
LEGACY_PROOF_SHA = '1c32f26c06c06c7d9e17850cb4b56bb9f089eed0b4395961109c8e7153f04fa1'
LEGACY_ZIP_SHA = 'cb9775aeebc2acb0b82a9e5edbb31628821e6c79546849e41073b0378697c520'
F01_ORACLE_SHA = '59b1af3cd6cc1c1feeeb51f35b28f5656c1befeffbc8d2cee2e1a0e033b835d0'
METHODS = 23


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jbytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')


def pinned(path, expected, limit=4 * 1024**2):
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or not 0 < len(raw) <= limit or sha(raw) != expected:
        raise RuntimeError('Independent immutable input changed: ' + str(path))
    return raw


def load_subject():
    spec = importlib.util.spec_from_file_location('independent_current_identity_subject', SUBJECT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_members(raw):
    members = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            if entry.filename in members or entry.file_size > 4 * 1024**2:
                raise RuntimeError('Independent fixture namespace/bound changed')
            members[entry.filename] = (entry, archive.read(entry))
    return members


def write_members(path, entries, replacements=None, omit=()):
    replacements = {} if replacements is None else replacements
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (entry, raw) in entries.items():
            if name not in omit:
                archive.writestr(entry, replacements.get(name, raw))


def replace_json(members, name, change):
    value = json.loads(members[name]); change(value); members[name] = jbytes(value)


def map_key(value, suffix):
    matches = [key for key in value if key.replace('\\', '/').endswith('/' + suffix)]
    if len(matches) != 1:
        raise RuntimeError('Independent source-map suffix is not unique: ' + suffix)
    return matches[0]


class CurrentIdentityCoverageControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = json.loads(pinned(ORACLE, ORACLE_SHA))
        if cls.oracle['schema'] != 'sourceq_current_identity_coverage_independent_oracles_v2':
            raise RuntimeError('Exact precode oracle version required')
        cls.fixture_raw = {}
        for name, row in cls.oracle['fixture_pins'].items():
            raw = pinned(FIXTURES / name, row['sha256'])
            if len(raw) != row['bytes']:
                raise RuntimeError('Independent fixture size changed')
            cls.fixture_raw[name] = raw
        cls.proof = json.loads(cls.fixture_raw['r11-source-proof.json'])
        cls.legacy_proof = json.loads(pinned(FIXTURES / 'native69a8-source-proof.json', LEGACY_PROOF_SHA))
        cls.legacy_raw = pinned(FIXTURES / 'native69a8-posix-macos.zip', LEGACY_ZIP_SHA)
        cls.f01 = json.loads(pinned(HERE / 'lifecycle_event_oracles_v1.json', F01_ORACLE_SHA))
        cls.subject = load_subject()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sourceq-identity-inert-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.serial = 0

    def destination(self):
        self.serial += 1
        return self.directory / ('inert-%03d.zip' % self.serial)

    def fixture_path(self, name):
        path = self.destination(); path.write_bytes(self.fixture_raw[name]); return path

    def unqualified(self, value):
        for field in ('source_qualified', 'software_qualified', 'scientific_qualified'):
            self.assertIs(value[field], False)

    def rejects(self, function, *args, code=None, **kwargs):
        with self.assertRaises(self.subject.SourceQualificationError) as caught:
            function(*args, **kwargs)
        if code is not None:
            self.assertEqual(caught.exception.code, code)

    def owner(self, path, platform='macos-26-intel', proof=None):
        return self.subject.verify_owner_artifact(path, platform=platform,
                                                 proof=copy.deepcopy(self.proof if proof is None else proof))

    def diagnostic(self, path, platform='macos-26-intel', proof=None):
        return self.subject.verify_darwin_identity_diagnostic(
            path, platform=platform, proof=copy.deepcopy(self.proof if proof is None else proof))

    def identity_patch(self, change=None, post=None, fixture='r11-posix-macos.zip'):
        entries = read_members(self.fixture_raw[fixture])
        members = {name: raw for name, (_, raw) in entries.items()}
        result = json.loads(members['identity/RESULT.json'])
        receipt = json.loads(members['identity-owner.json'])
        terminal = json.loads(members['TERMINAL.json'])
        launch = json.loads(members['identity-launch.json'])
        if change is not None:
            change(result, receipt, terminal, launch)
        members['identity/RESULT.json'] = jbytes(result)
        members['identity-owner.json'] = jbytes(receipt)
        members['identity-launch.json'] = jbytes(launch)
        # Semantic mutations retain coherent owner/result joins; stale hashes cannot explain rejection.
        terminal['identity'] = dict(copy.deepcopy(receipt), result_sha256=sha(members['identity/RESULT.json']))
        members['TERMINAL.json'] = jbytes(terminal)
        if post is not None:
            post(members)
        path = self.destination(); write_members(path, entries, members); return path

    def diagnostic_patch(self, change=None, post=None):
        entries = read_members(self.fixture_raw['r11-darwin-identity-diagnostic.zip'])
        members = {name: raw for name, (_, raw) in entries.items()}
        result = json.loads(members['RESULT.json'])
        pre = json.loads(members['PREEXECUTION.json'])
        ready = json.loads(members['READY.json'])
        if change is not None:
            change(result, pre, ready)
        members['RESULT.json'] = jbytes(result)
        members['PREEXECUTION.json'] = jbytes(pre)
        members['READY.json'] = jbytes(ready)
        if post is not None:
            post(members)
        path = self.destination(); write_members(path, entries, members); return path

    def test_01_current_literal_stage_is_mandatory_on_all_three_platforms(self):
        for fixture, platform in (('r11-posix-macos.zip', 'macos-26-intel'),
                                  ('r11-posix-linux.zip', 'ubuntu-latest'),
                                  ('r11-posix-windows.zip', 'windows-latest')):
            with self.subTest(platform=platform):
                projection = json.loads(read_members(self.fixture_raw[fixture])['identity/RESULT.json'][1])['rows']
                self.assertEqual(projection, self.oracle['literal_rows_projection'])
                value = self.owner(self.fixture_path(fixture), platform)
                self.assertIs(value['complete_as_owner_evidence'], True)
                self.assertEqual(value['identity_contract'], self.oracle['current_identity_contract'])
                self.assertEqual(value['identity_methods'], 3)
                self.assertEqual(value['identity_literal_rows'], 25)
                self.unqualified(value)

    def test_02_legacy_fixture_retains_its_explicit_historical_scope(self):
        path = self.destination(); path.write_bytes(self.legacy_raw)
        value = self.owner(path, proof=self.legacy_proof)
        self.assertEqual(value['identity_contract'], self.oracle['legacy_identity_contract'])
        self.assertEqual(value['identity_methods'], 0)
        self.assertEqual(value['identity_literal_rows'], 0)
        self.unqualified(value)

    def test_03_unknown_missing_and_mixed_controller_versions_fail_closed(self):
        for case in ('unknown', 'missing', 'boolean', 'bare-legacy-downgrade', 'mixed-observer'):
            with self.subTest(case=case):
                proof = copy.deepcopy(self.proof); pins = proof['source_sha256']
                controller = 'test/run_posix_owner_v3_controls.py'
                if case == 'missing':
                    del pins[controller]; del proof['commit_object_ids'][controller]; proof['source_files'] -= 1
                elif case == 'mixed-observer':
                    pins['test/posix_owner_v3/observations.py'] = self.oracle['legacy_observer_sha256']
                else:
                    pins[controller] = {'unknown': 'f' * 64, 'boolean': True,
                                        'bare-legacy-downgrade': self.oracle['legacy_controller_sha256']}[case]
                self.rejects(self.owner, self.fixture_path('r11-posix-macos.zip'), proof=proof,
                             code=None if case == 'boolean' else 'unsupported_owner_controller_version')

    def test_04_each_original_current_stage_member_is_required(self):
        entries = read_members(self.fixture_raw['r11-posix-macos.zip'])
        for missing in ('identity/RESULT.json', 'identity-owner.json', 'identity-launch.json', 'identity.log'):
            with self.subTest(missing=missing):
                path = self.destination(); write_members(path, entries, omit=(missing,))
                self.rejects(self.owner, path)

    def test_05_terminal_fields_cannot_be_an_optional_downgrade(self):
        for field in ('identity', 'identity_methods', 'identity_literal_rows'):
            with self.subTest(field=field):
                path = self.identity_patch(post=lambda members, field=field:
                    replace_json(members, 'TERMINAL.json', lambda value: value.pop(field)))
                self.rejects(self.owner, path)
        for field, value in (('identity_methods', True), ('identity_methods', 2),
                             ('identity_literal_rows', 24), ('identity_literal_rows', True)):
            with self.subTest(field=field, value=value):
                path = self.identity_patch(post=lambda members, field=field, value=value:
                    replace_json(members, 'TERMINAL.json', lambda row: row.__setitem__(field, value)))
                self.rejects(self.owner, path)

    def test_06_original_identity_owner_result_must_be_positive(self):
        cases = (('native_exit', 1), ('native_exit', False), ('timed_out', True),
                 ('leader_reaped', False), ('ownership_boundary_closed', False),
                 ('all_descendants_os_reaped', True), ('wall_seconds', 30), ('wall_seconds', True),
                 ('pid', 1), ('ownership', 'unowned'))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                    receipt['native'].__setitem__(field, value)))

    def test_07_fixed_stage_allowance_resource_and_launch_are_checked(self):
        for allowance in (0, 31, True, -1):
            with self.subTest(allowance=allowance):
                self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                    receipt.__setitem__('allowance_seconds', allowance)))
        for field, value in (('available_physical_bytes', 512 * 1024**2 - 1),
                             ('free_disk_bytes', 8 * 1024**3 - 1)):
            with self.subTest(resource=field):
                self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                    receipt['admission'].__setitem__(field, value)))
        self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
            launch.__setitem__('pid', receipt['launch_pid'] + 1)))
        self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
            receipt.__setitem__('launch_pid', True)))

    def test_08_raw_receipt_and_result_hash_joins_are_not_summaries(self):
        changes = (lambda members: members.__setitem__('identity.log', b'Ran 3 tests in 0s\nFAILED\n'),
                   lambda members: replace_json(members, 'identity-owner.json',
                                                lambda row: row.__setitem__('raw_bytes', 0)),
                   lambda members: replace_json(members, 'identity-owner.json',
                                                lambda row: row.__setitem__('raw_sha256', '0' * 64)),
                   lambda members: replace_json(members, 'TERMINAL.json',
                                                lambda row: row['identity'].__setitem__('result_sha256', '0' * 64)),
                   lambda members: members.__setitem__('identity-owner.json', b'{"native":{},"native":{}}\n'))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.owner, self.identity_patch(post=change))
        def failed_raw(members):
            members['identity.log'] = b'Ran 3 tests in 0s\nFAILED\n'
            replace_json(members, 'identity-owner.json', lambda row:
                         row.update(raw_bytes=len(members['identity.log']), raw_sha256=sha(members['identity.log'])))
            owner = json.loads(members['identity-owner.json'])
            replace_json(members, 'TERMINAL.json', lambda row: row['identity'].update(owner))
        self.rejects(self.owner, self.identity_patch(post=failed_raw))

    def test_09_literal_counts_schema_and_completion_are_exact(self):
        cases = (('schema', 'wrong'), ('complete', False), ('unittest_methods', True),
                 ('expected_methods', 2), ('literal_rows', 24), ('expected_rows', True),
                 ('recovery_rows', 1), ('negative_rows', 19), ('compatibility_rows', 2),
                 ('failures', 1), ('errors', True), ('skipped', 1), ('source_unchanged', False))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                    result.__setitem__(field, value)))

    def test_10_literal_rows_are_complete_ordered_and_individually_frozen(self):
        changes = (lambda row: row['rows'].pop(), lambda row: row['rows'].reverse(),
                   lambda row: row['rows'][0].__setitem__('id', 'C02'),
                   lambda row: row['rows'][0].__setitem__('passed', False),
                   lambda row: row['rows'][0]['checks'].__setitem__('getter_call_count', False),
                   lambda row: row['rows'][0]['checks'].pop('getter_call_count'),
                   lambda row: row['rows'][0].__setitem__('info_arguments', []),
                   lambda row: row['rows'][0]['returned_identity'].__setitem__('sid', None),
                   lambda row: row['rows'][0].__setitem__('actual_Darwin_processes', 1))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch: change(result)))

    def test_11_literal_source_maps_are_exact_and_join_the_current_proof(self):
        source = 'test/posix_owner_v3/observations.py'
        def alter_after(result, receipt, terminal, launch):
            result['source_after'][map_key(result['source_after'], source)] = '0' * 64
        self.rejects(self.owner, self.identity_patch(alter_after))
        def coherent_wrong_source(result, receipt, terminal, launch):
            for field in ('source_before', 'source_after'):
                result[field][map_key(result[field], source)] = '0' * 64
        self.rejects(self.owner, self.identity_patch(coherent_wrong_source))
        def missing_pin(result, receipt, terminal, launch):
            for field in ('source_before', 'source_after'):
                del result[field][map_key(result[field], source)]
        self.rejects(self.owner, self.identity_patch(missing_pin))
        def ambiguous_alias(result, receipt, terminal, launch):
            for field in ('source_before', 'source_after'):
                result[field]['/other/' + source] = self.oracle['current_source_pins'][source]
        self.rejects(self.owner, self.identity_patch(ambiguous_alias))
        self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
            terminal['test_sources'].pop('test_identity_transition.py')))

    def test_12_literal_scope_never_becomes_native_or_qualified(self):
        zeros = ('actual_Darwin_observations', 'actual_Darwin_processes_requested',
                 'actual_POSIX_test_processes_started', 'actual_native_observations_requested',
                 'actual_native_test_processes_started', 'actual_signals_requested', 'actual_signals_sent')
        for field in zeros:
            for value in (1, False):
                with self.subTest(field=field, value=value):
                    self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                        result.__setitem__(field, value)))
        for field in ('Julia_launched', 'source_qualified', 'software_qualified', 'scientific_qualified'):
            self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
                result.__setitem__(field, True)))
        self.rejects(self.owner, self.identity_patch(lambda result, receipt, terminal, launch:
            result.__setitem__('evidence_kind', 'actual_native_observation')))

    def test_13_original_controller_clock_and_identity_launch_are_bounded(self):
        changes = (lambda result, receipt, terminal, launch:
                       launch.__setitem__('monotonic_ns', terminal['started_ns'] - 1),
                   lambda result, receipt, terminal, launch:
                       result.__setitem__('finished_ns', terminal['deadline_ns']),
                   lambda result, receipt, terminal, launch:
                       result.__setitem__('started_ns', result['finished_ns'] + 1),
                   lambda result, receipt, terminal, launch:
                       terminal.__setitem__('deadline_ns', terminal['started_ns'] + 661 * 10**9))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.owner, self.identity_patch(change))

    def test_14_original_diagnostic_is_separate_observational_evidence(self):
        value = self.diagnostic(self.fixture_path('r11-darwin-identity-diagnostic.zip'))
        self.assertIs(value['complete_as_identity_diagnostic_evidence'], True)
        self.assertIs(value['observation_only'], True)
        self.assertEqual(value['actual_fixed_children'], 1)
        self.assertIs(value['native_owner_qualified'], False)
        self.unqualified(value)

    def test_15_diagnostic_allows_source_permitted_observational_alternatives(self):
        def readers_return(result, pre, ready):
            result['terminal']['getsid'] = dict(returned=True, value=result['actual_popen_pid'])
            result['terminal']['current_leader'] = copy.deepcopy(result['live']['current_leader'])
        def readers_error_even_live(result, pre, ready):
            error = dict(returned=False, error=dict(type='OSError', errno=5, message='recorded reader error'))
            result['live']['current_info'] = copy.deepcopy(error)
            result['live']['current_leader'] = copy.deepcopy(error)
        def signal_terminal(result, pre, ready):
            result['terminal_waitid'].update(si_code=2, si_status=9)
            result['direct_child_native_exit'] = -9
        for name, change in (('terminal-sid-returned', readers_return),
                             ('reader-errors-observational', readers_error_even_live),
                             ('matched-terminal-signal', signal_terminal)):
            with self.subTest(case=name):
                value = self.diagnostic(self.diagnostic_patch(change))
                self.assertIs(value['observation_only'], True)
                self.unqualified(value)

    def test_16_every_original_diagnostic_member_is_required(self):
        entries = read_members(self.fixture_raw['r11-darwin-identity-diagnostic.zip'])
        self.assertEqual(set(entries), {'PREEXECUTION.json', 'READY.json', 'RESULT.json', 'child.raw', 'release'})
        for name in entries:
            with self.subTest(member=name):
                path = self.destination(); write_members(path, entries, omit=(name,))
                self.rejects(self.diagnostic, path)

    def test_17_diagnostic_source_version_and_preexecution_are_bound(self):
        def wrong_pin(result, pre, ready):
            for row in (result, pre):
                name = map_key(row['source_pins'], 'test/initial_reference/diagnose_darwin_identity_v3.py')
                row['source_pins'][name] = '0' * 64
        changes = (wrong_pin,
                   lambda result, pre, ready: result.__setitem__('source_pins_unchanged', False),
                   lambda result, pre, ready: pre.__setitem__('complete', True),
                   lambda result, pre, ready: pre.__setitem__('observer_pid', result['observer_pid'] + 1),
                   lambda result, pre, ready: ready.__setitem__('source_sha256', '0' * 64))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.diagnostic, self.diagnostic_patch(change))
        for platform in ('ubuntu-latest', 'windows-latest'):
            self.rejects(self.diagnostic, self.fixture_path('r11-darwin-identity-diagnostic.zip'), platform=platform)
        proof = copy.deepcopy(self.proof)
        proof['source_sha256']['test/initial_reference/diagnose_darwin_identity_v3.py'] = 'e' * 64
        self.rejects(self.diagnostic, self.fixture_path('r11-darwin-identity-diagnostic.zip'), proof=proof)

    def test_18_diagnostic_native_and_observational_record_schema_is_checked(self):
        changes = (lambda result, pre, ready: result['terminal'].pop('getsid'),
                   lambda result, pre, ready: result['terminal']['getsid'].__setitem__('returned', 0),
                   lambda result, pre, ready: result['terminal']['getsid'].pop('error'),
                   lambda result, pre, ready: result['terminal']['getsid'].__setitem__('value', 1),
                   lambda result, pre, ready: result['terminal']['current_leader']['error'].__setitem__('errno', True),
                   lambda result, pre, ready: result['terminal']['current_leader']['error'].__setitem__('message', 'x' * 1025),
                   lambda result, pre, ready: result['terminal'].__setitem__('current_info', dict(returned=True, value=None)),
                   lambda result, pre, ready: result['terminal']['short'].__setitem__('returned', False))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.diagnostic, self.diagnostic_patch(change))
        for field in self.oracle['diagnostic_native_abi']:
            def abi(result, pre, ready, field=field):
                result['native_abi'][field] += 1; pre['native_abi'][field] += 1
            self.rejects(self.diagnostic, self.diagnostic_patch(abi))

    def test_19_native_pid_live_calibration_and_terminal_wait_are_joined(self):
        changes = (lambda result, pre, ready: ready.__setitem__('sid', ready['pid'] + 1),
                   lambda result, pre, ready: result['live']['short']['value'].__setitem__('pid', 1),
                   lambda result, pre, ready: result['terminal']['full']['value'].__setitem__('pid', 1),
                   lambda result, pre, ready: result['live']['full']['value'].__setitem__('pgid', 1),
                   lambda result, pre, ready: result['live']['full']['value'].__setitem__('ppid', 1),
                   lambda result, pre, ready: result['live']['full']['value'].__setitem__('status', 5),
                   lambda result, pre, ready: result['live']['getsid'].__setitem__('value', 1),
                   lambda result, pre, ready: result['terminal']['short']['value'].__setitem__('session_leader_flag_0x20', True),
                   lambda result, pre, ready: result['terminal_waitid'].__setitem__('si_pid', 1),
                   lambda result, pre, ready: result['terminal_waitid'].__setitem__('si_code', 5),
                   lambda result, pre, ready: result.__setitem__('direct_child_native_exit', 8),
                   lambda result, pre, ready: result.__setitem__('terminal_observed_before_first_wait', False),
                   lambda result, pre, ready: result.__setitem__('actual_popen_wait_returned', False),
                   lambda result, pre, ready: result.__setitem__('direct_child_reaping_observed', False),
                   lambda result, pre, ready: result.__setitem__('waitid_attempts', True))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.diagnostic, self.diagnostic_patch(change))

    def test_20_diagnostic_raw_admission_and_original_clock_are_checked(self):
        self.rejects(self.diagnostic, self.diagnostic_patch(post=lambda members:
            members.__setitem__('child.raw', b'not the pinned child readiness\n')))
        self.rejects(self.diagnostic, self.diagnostic_patch(post=lambda members:
            members.__setitem__('release', b'0\n')))
        changes = (lambda result, pre, ready: result['raw'].__setitem__('sha256', '0' * 64),
                   lambda result, pre, ready: result['raw'].__setitem__('bytes', True),
                   lambda result, pre, ready: result.__setitem__('finished_ns', result['deadline_ns']),
                   lambda result, pre, ready: result['terminal'].__setitem__('monotonic_ns', result['live']['monotonic_ns'] - 1))
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.rejects(self.diagnostic, self.diagnostic_patch(change))
        for field, value in (('available_physical_bytes', 512 * 1024**2 - 1),
                             ('free_disk_bytes', 8 * 1024**3 - 1), ('required_available_bytes', 1),
                             ('required_disk_bytes', 1)):
            def resource(result, pre, ready, field=field, value=value):
                result['admission'][field] = value; pre['admission'][field] = value
            self.rejects(self.diagnostic, self.diagnostic_patch(resource))
        def extend(result, pre, ready):
            result['deadline_ns'] += 10**9; pre['deadline_ns'] += 10**9
        self.rejects(self.diagnostic, self.diagnostic_patch(extend))

    def test_21_diagnostic_cannot_claim_ownership_or_use_failure_cleanup(self):
        false_fields = ('source_qualified', 'software_qualified', 'scientific_qualified',
                        'native_owner_qualified', 'ownership_boundary_closed_observed',
                        'individual_descendant_OS_reaping_observed', 'Julia_launched',
                        'actual_archive_exposed', 'Q_issued')
        for field in false_fields:
            def broaden(result, pre, ready, field=field):
                result[field] = pre[field] = True
            self.rejects(self.diagnostic, self.diagnostic_patch(broaden))
        for field in ('signals_sent', 'descendants_launched'):
            def nonzero(result, pre, ready, field=field):
                result[field] = pre[field] = 1
            self.rejects(self.diagnostic, self.diagnostic_patch(nonzero))
        for field in ('failure_only_wait_attempted', 'failure_only_actual_wait_returned', 'failure'):
            self.rejects(self.diagnostic, self.diagnostic_patch(lambda result, pre, ready:
                result.__setitem__(field, True)))

    def test_22_frozen_F01_oracle_vectors_also_check_current_materializer(self):
        self.assertEqual(self.f01['schema'], 'sourceq_lifecycle_event_oracles_v1')
        self.assertEqual(len(self.f01['rows']), 15)
        for case in self.f01['rows']:
            with self.subTest(case=case['id']):
                row = copy.deepcopy(case['row']); before = copy.deepcopy(row)
                if case['accepted']:
                    value = self.subject.verify_lifecycle_zero_session_events(row)
                    self.assertEqual(value, self.f01['expected_success']); self.unqualified(value)
                else:
                    self.rejects(self.subject.verify_lifecycle_zero_session_events, row)
                self.assertEqual(row, before)

    def test_23_literal_stage_and_seventh_native_diagnostic_are_not_interchangeable(self):
        self.rejects(self.owner, self.fixture_path('r11-darwin-identity-diagnostic.zip'))
        self.rejects(self.diagnostic, self.fixture_path('r11-posix-macos.zip'))


def source_map():
    paths = (Path(__file__).resolve(), SUBJECT, ORACLE, HERE / 'lifecycle_event_oracles_v1.json',
             FIXTURES / 'native69a8-source-proof.json', FIXTURES / 'native69a8-posix-macos.zip',
             *(FIXTURES / name for name in json.loads(pinned(ORACLE, ORACLE_SHA))['fixture_pins']))
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Ordinary independent source/fixture required')
        result[str(path)] = sha(path.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args(); output = args.output_directory
    if sys.flags.optimize or not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise RuntimeError('Unoptimized fresh absolute external test output required')
    if output == ROOT or output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise RuntimeError('External disjoint test output required')
    started = time.monotonic_ns(); before = source_map(); output.mkdir(); raw = io.StringIO()
    result = unittest.TextTestRunner(stream=raw, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CurrentIdentityCoverageControls))
    transcript = raw.getvalue().encode('utf8'); (output / 'unittest.log').write_bytes(transcript)
    after = source_map(); complete = result.wasSuccessful() and result.testsRun == METHODS and before == after
    record = dict(schema='sourceq_current_identity_coverage_independent_tests_v1', complete=complete,
                  unittest_methods=result.testsRun, expected_methods=METHODS,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  raw_bytes=len(transcript), raw_sha256=sha(transcript),
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  actual_OS_observations_or_owner_launches=0, actual_Julia_launches=0,
                  archive_members_extracted_or_executed=False, qualification_issued=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / 'RESULT.json').write_bytes(jbytes(record))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
