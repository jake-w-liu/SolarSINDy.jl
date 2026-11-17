"""Conditional source invariants for recorded successful reader results; inert data only."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'test_current_identity_coverage.py'
BASE_SHA = '762a106281516af6c81b215275a144540c57b7c462ebaf1d96dc400710045acc'
ORACLE = HERE / 'current_identity_returned_value_oracles_v1.json'
ORACLE_SHA = 'ff9220b8808daac8f8676eb3be31367a424a976442b6e1636f1408857e414d75'


def pinned(path, expected):
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Immutable additive test input changed: ' + str(path))
    return raw


def load_base():
    pinned(BASE, BASE_SHA)
    spec = importlib.util.spec_from_file_location('independent_identity_base_controls', BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    pinned(BASE, BASE_SHA)
    return module


class CurrentReaderReturnedValueControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = json.loads(pinned(ORACLE, ORACLE_SHA))
        if cls.oracle['methods'] != 2 or cls.oracle['base_test_sha256'] != BASE_SHA:
            raise RuntimeError('Exact independent returned-value oracle required')
        cls.base = load_base()
        cls.base.CurrentIdentityCoverageControls.setUpClass()

    def setUp(self):
        self.probe = self.base.CurrentIdentityCoverageControls(
            methodName='test_14_original_diagnostic_is_separate_observational_evidence')
        self.probe.setUp()
        self.addCleanup(self.probe.doCleanups)

    def baseline(self):
        value = self.probe.diagnostic(self.probe.fixture_path('r11-darwin-identity-diagnostic.zip'))
        self.assertIs(value['complete_as_identity_diagnostic_evidence'], True)
        self.probe.unqualified(value)

    def test_01_returned_info_has_the_queried_child_PID(self):
        self.baseline()
        for phase in ('live', 'terminal'):
            with self.subTest(phase=phase):
                def wrong_pid(result, pre, ready, phase=phase):
                    self.assertIs(result[phase]['current_info']['returned'], True)
                    result[phase]['current_info']['value']['pid'] = result['actual_popen_pid'] + 1
                self.probe.rejects(self.probe.diagnostic, self.probe.diagnostic_patch(wrong_pid))

    def test_02_returned_leader_obeys_its_own_source_admission(self):
        self.baseline()
        cases = self.oracle['leader_negative_cases']
        self.assertEqual(cases, ['wrong-pid', 'wrong-pgid', 'missing-sid', 'wrong-sid',
                                 'missing-birth', 'wrong-uid', 'wrong-ruid', 'coherent-wrong-uid-pair'])
        for phase in ('live', 'terminal'):
            for case in cases:
                with self.subTest(phase=phase, case=case):
                    def impossible_success(result, pre, ready, phase=phase, case=case):
                        row = copy.deepcopy(result['live']['current_leader']['value'])
                        if case == 'wrong-pid':
                            row['pid'] += 1
                        elif case == 'wrong-pgid':
                            row['pgid'] += 1
                        elif case == 'missing-sid':
                            row['sid'] = None
                        elif case == 'wrong-sid':
                            row['sid'] += 1
                        elif case == 'missing-birth':
                            row['birth'] = None
                        elif case == 'wrong-uid':
                            row['uid'] += 1
                        elif case == 'wrong-ruid':
                            row['ruid'] += 1
                        elif case == 'coherent-wrong-uid-pair':
                            row['uid'] += 1
                            row['ruid'] += 1
                        result[phase]['current_leader'] = dict(returned=True, value=row)
                    self.probe.rejects(self.probe.diagnostic, self.probe.diagnostic_patch(impossible_success))


def source_map(base):
    values = base.source_map()
    values[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    values[str(ORACLE)] = hashlib.sha256(pinned(ORACLE, ORACLE_SHA)).hexdigest()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    output = parser.parse_args().output_directory
    if sys.flags.optimize or not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise RuntimeError('Unoptimized fresh absolute external output required')
    if output == ROOT or output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise RuntimeError('Disjoint external output required')
    started = time.monotonic_ns(); base = load_base(); before = source_map(base); output.mkdir()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CurrentReaderReturnedValueControls))
    raw = stream.getvalue().encode('utf8'); (output / 'unittest.log').write_bytes(raw)
    after = source_map(base)
    complete = result.wasSuccessful() and result.testsRun == 2 and before == after
    record = dict(schema='sourceq_current_identity_returned_value_independent_tests_v1',
        complete=complete, unittest_methods=result.testsRun, expected_methods=2,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_unchanged=before == after,
        raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
        started_ns=started, finished_ns=time.monotonic_ns(),
        actual_OS_observations_or_owner_launches=0, actual_Julia_launches=0,
        archive_members_extracted_or_executed=False, qualification_issued=False,
        source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / 'RESULT.json').write_bytes(base.jbytes(record))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
