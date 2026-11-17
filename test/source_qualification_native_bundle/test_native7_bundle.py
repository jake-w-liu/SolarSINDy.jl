"""Independent original-native7 bundle probes; inert data, no qualification."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / 'fixtures'
SUBJECT = ROOT / 'tools/actual_b_chain/materialize_source_qualification.py'
HELPER = SUBJECT.with_name('sourceq_full_scale_verifier.py')
ORACLE = HERE / 'native7_bundle_oracles_v1.json'
ORACLE_SHA = 'b2a9ed3f734e8315622668c09b9ea61bf7c405ba60d9f7b33b8129a60bf00f1c'
METHODS = 12


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jbytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')


def pinned(path, expected, size=None):
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or len(raw) > 4 * 1024**2 or sha(raw) != expected:
        raise RuntimeError('Immutable independent input changed: ' + str(path))
    if size is not None and len(raw) != size:
        raise RuntimeError('Immutable independent input size changed')
    return raw


def load_subject():
    spec = importlib.util.spec_from_file_location('independent_native7_bundle_subject', SUBJECT)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Native7BundleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = json.loads(pinned(ORACLE, ORACLE_SHA))
        if cls.oracle['schema'] != 'sourceq_native7_bundle_independent_oracles_v1':
            raise RuntimeError('Exact independent bundle oracle required')
        cls.artifacts = {}
        cls.raw = {}
        for name, row in cls.oracle['artifacts'].items():
            path = FIXTURES / row['file']
            cls.raw[name] = pinned(path, row['sha256'], row['bytes'])
            cls.artifacts[name] = path
        cls.proof = json.loads(pinned(FIXTURES / 'r11-source-proof.json', cls.oracle['source_proof_sha256']))
        jobs_raw = pinned(FIXTURES / 'r11-native-jobs.raw', cls.oracle['native_jobs_raw_sha256'],
                          cls.oracle['native_jobs_raw_bytes'])
        document = json.loads(jobs_raw)
        if type(document['total_count']) is not int or document['total_count'] != 3 or len(document['jobs']) != 3:
            raise RuntimeError('Original complete three-job API response required')
        cls.jobs = document['jobs']
        cls.subject = load_subject()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sourceq-native7-inert-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name); self.serial = 0

    def target(self):
        self.serial += 1
        return self.directory / ('inert-%03d.zip' % self.serial)

    def bundle(self, artifacts=None, jobs=None, proof=None, run_id=None):
        return self.subject.verify_native_artifact_bundle(
            dict(self.artifacts) if artifacts is None else artifacts,
            copy.deepcopy(self.jobs) if jobs is None else jobs,
            proof=copy.deepcopy(self.proof) if proof is None else proof,
            run_id=self.oracle['run_id'] if run_id is None else run_id)

    def rejects(self, **kwargs):
        with self.assertRaises(self.subject.SourceQualificationError):
            self.bundle(**kwargs)

    def unqualified(self, value):
        if type(value) is dict:
            for name, child in value.items():
                if name in ('source_qualified', 'software_qualified', 'scientific_qualified'):
                    self.assertIs(child, False)
                self.unqualified(child)
        elif type(value) is list:
            for child in value:
                self.unqualified(child)

    def job(self, jobs, platform):
        return next(row for row in jobs if row['name'] == 'V3 engineering / ' + platform)

    def step(self, job, name):
        return next(row for row in job['steps'] if row['name'] == name)

    def rewrite(self, name, mutate):
        members = {}
        entries = []
        with zipfile.ZipFile(io.BytesIO(self.raw[name])) as archive:
            for entry in archive.infolist():
                if entry.file_size > 4 * 1024**2 or entry.filename in members:
                    raise RuntimeError('Independent original member bound/namespace differs')
                entries.append(entry); members[entry.filename] = archive.read(entry)
        mutate(members)
        path = self.target()
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                archive.writestr(entry, members[entry.filename])
        artifacts = dict(self.artifacts); artifacts[name] = path
        return artifacts

    def patch_json(self, members, name, change):
        value = json.loads(members[name]); change(value); members[name] = jbytes(value)

    def test_01_all_seven_original_semantic_receipts_and_real_jobs(self):
        before_jobs = copy.deepcopy(self.jobs); before_proof = copy.deepcopy(self.proof)
        value = self.bundle()
        self.assertIs(value['complete_as_native_artifact_bundle'], True)
        self.assertEqual((value['run_id'], value['head']), (self.oracle['run_id'], self.oracle['head']))
        self.assertEqual((value['artifact_count'], value['owner_artifacts'], value['diagnostic_artifacts'],
                          value['job_count']), (7, 6, 1, 3))
        self.assertEqual(value['native_job_ids'], self.oracle['job_ids'])
        self.assertEqual(set(value['semantic_receipts']), set(self.artifacts))
        self.assertEqual(value['artifact_sha256'], {name: row['sha256'] for name, row in self.oracle['artifacts'].items()})
        for name, receipt in value['semantic_receipts'].items():
            field = ('complete_as_identity_diagnostic_evidence' if name.startswith('posix-v3-identity-')
                     else 'complete_as_owner_evidence')
            self.assertIs(receipt[field], True)
            for flag in ('source_qualified', 'software_qualified', 'scientific_qualified'):
                self.assertIs(receipt[flag], False)
        self.unqualified(value)
        self.assertEqual(self.jobs, before_jobs); self.assertEqual(self.proof, before_proof)

    def test_02_each_original_artifact_including_seventh_is_mandatory(self):
        for name in self.artifacts:
            with self.subTest(missing=name):
                artifacts = dict(self.artifacts); del artifacts[name]
                self.rejects(artifacts=artifacts)

    def test_03_artifact_namespace_and_attempt_are_exact(self):
        name = next(iter(self.artifacts))
        for replacement in (name.upper(), name.replace('-1', '-2'), 'unknown-native-artifact'):
            with self.subTest(replacement=replacement):
                artifacts = dict(self.artifacts); path = artifacts.pop(name); artifacts[replacement] = path
                self.rejects(artifacts=artifacts)
        artifacts = dict(self.artifacts); artifacts['unexpected-extra'] = artifacts[name]
        self.rejects(artifacts=artifacts)
        self.rejects(artifacts=list(self.artifacts.items()))

    def test_04_duplicate_physical_or_byte_identical_archive_is_not_two_artifacts(self):
        linux = 'darwin-anchor-v1-ubuntu-latest-%d-1' % self.oracle['run_id']
        windows = 'darwin-anchor-v1-windows-latest-%d-1' % self.oracle['run_id']
        artifacts = dict(self.artifacts); artifacts[linux] = artifacts[windows]
        self.rejects(artifacts=artifacts)
        duplicate = self.target(); duplicate.write_bytes(self.raw[windows])
        artifacts = dict(self.artifacts); artifacts[linux] = duplicate
        self.rejects(artifacts=artifacts)
        artifacts = dict(self.artifacts); artifacts[linux] = None
        self.rejects(artifacts=artifacts)

    def test_05_exact_original_three_job_list_and_unique_identity(self):
        cases = ('missing', 'extra-duplicate', 'unknown-name', 'duplicate-ID', 'duplicate-name', 'deduplicated-map')
        for case in cases:
            with self.subTest(case=case):
                jobs = copy.deepcopy(self.jobs)
                if case == 'missing': jobs.pop()
                elif case == 'extra-duplicate': jobs.append(copy.deepcopy(jobs[0]))
                elif case == 'unknown-name': jobs[0]['name'] = 'unknown native job'
                elif case == 'duplicate-ID': jobs[1]['id'] = jobs[0]['id']
                elif case == 'duplicate-name': jobs[1]['name'] = jobs[0]['name']
                else: jobs = {row['name']: row for row in jobs}
                self.rejects(jobs=jobs)

    def test_06_job_run_head_attempt_and_outcome_are_bound(self):
        cases = (('id', True), ('run_id', 1), ('head_sha', 'f' * 40), ('run_attempt', True),
                 ('run_attempt', 2), ('status', 'in_progress'), ('conclusion', 'failure'),
                 ('started_at', 'invalid'), ('completed_at', '2020-01-01T00:00:00Z'))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                jobs = copy.deepcopy(self.jobs); jobs[0][field] = value; self.rejects(jobs=jobs)
        for run_id in (True, 0, self.oracle['run_id'] + 1):
            self.rejects(run_id=run_id)
        proof = copy.deepcopy(self.proof); proof['head'] = 'f' * 40
        self.rejects(proof=proof)

    def test_07_both_current_control_steps_are_required_on_every_platform(self):
        for platform in ('windows-latest', 'ubuntu-latest', 'macos-26-intel'):
            for step_name in self.oracle['required_steps']:
                for defect in ('missing', 'skipped', 'failure', 'in-progress'):
                    with self.subTest(platform=platform, step=step_name, defect=defect):
                        jobs = copy.deepcopy(self.jobs); job = self.job(jobs, platform)
                        step = self.step(job, step_name)
                        if defect == 'missing': job['steps'].remove(step)
                        elif defect == 'in-progress': step['status'] = 'in_progress'
                        else: step['conclusion'] = defect
                        self.rejects(jobs=jobs)

    def test_08_exact_single_successful_Darwin_diagnostic_step_is_mandatory(self):
        step_name = self.oracle['Darwin_diagnostic_step']
        for defect in ('missing', 'skipped', 'failure', 'renamed', 'duplicate', 'elsewhere-only'):
            with self.subTest(defect=defect):
                jobs = copy.deepcopy(self.jobs); mac = self.job(jobs, 'macos-26-intel'); step = self.step(mac, step_name)
                if defect in ('missing', 'elsewhere-only'): mac['steps'].remove(step)
                elif defect == 'renamed': step['name'] += ' (summary)'
                elif defect == 'duplicate':
                    duplicate = copy.deepcopy(step); duplicate['number'] = max(row['number'] for row in mac['steps']) + 1
                    mac['steps'].append(duplicate)
                else: step['conclusion'] = defect
                if defect == 'elsewhere-only': self.step(self.job(jobs, 'windows-latest'), step_name)['conclusion'] = 'success'
                self.rejects(jobs=jobs)

    def test_09_step_types_numbers_and_original_time_join_are_checked(self):
        name = self.oracle['Darwin_diagnostic_step']
        for field, value in (('number', True), ('number', 5), ('status', 'queued'),
                             ('started_at', '2020-01-01T00:00:00Z'), ('completed_at', 'invalid')):
            with self.subTest(field=field, value=value):
                jobs = copy.deepcopy(self.jobs); step = self.step(self.job(jobs, 'macos-26-intel'), name)
                step[field] = value; self.rejects(jobs=jobs)

    def test_10_unknown_or_incoherent_current_controller_cannot_downgrade_bundle(self):
        for source in ('test/run_posix_owner_v3_controls.py', 'test/posix_owner_v3/observations.py'):
            with self.subTest(source=source):
                proof = copy.deepcopy(self.proof); proof['source_sha256'][source] = 'f' * 64
                self.rejects(proof=proof)

    def test_11_every_routed_original_artifact_receives_semantic_verification(self):
        for name in self.artifacts:
            with self.subTest(artifact=name):
                def incomplete(members, name=name):
                    member = 'RESULT.json' if name.startswith('posix-v3-identity-') else 'TERMINAL.json'
                    self.patch_json(members, member, lambda row: row.__setitem__('complete', False))
                self.rejects(artifacts=self.rewrite(name, incomplete))

    def test_12_coherent_transport_does_not_replace_original_semantics(self):
        mac = 'posix-v3-macos-26-intel-%d-1' % self.oracle['run_id']
        diagnostic = 'posix-v3-identity-%d-1' % self.oracle['run_id']
        anchor = 'darwin-anchor-v1-windows-latest-%d-1' % self.oracle['run_id']
        def literal_counterfeit(members):
            self.patch_json(members, 'identity/RESULT.json', lambda row: row.__setitem__('actual_Darwin_observations', 1))
            self.patch_json(members, 'TERMINAL.json', lambda row:
                            row['identity'].__setitem__('result_sha256', sha(members['identity/RESULT.json'])))
        self.rejects(artifacts=self.rewrite(mac, literal_counterfeit))
        def diagnostic_counterfeit(members):
            for name in ('PREEXECUTION.json', 'RESULT.json'):
                self.patch_json(members, name, lambda row: row.__setitem__('native_owner_qualified', True))
        self.rejects(artifacts=self.rewrite(diagnostic, diagnostic_counterfeit))
        def wrong_reader(members):
            self.patch_json(members, 'RESULT.json', lambda row:
                row['live']['current_info']['value'].__setitem__('pid', row['actual_popen_pid'] + 1))
        self.rejects(artifacts=self.rewrite(diagnostic, wrong_reader))
        def state_counterfeit(members):
            self.patch_json(members, 'state/RESULT.json', lambda row: row['rows'][0].__setitem__('passed', False))
            self.patch_json(members, 'TERMINAL.json', lambda row:
                            row['state'].__setitem__('result_sha256', sha(members['state/RESULT.json'])))
        self.rejects(artifacts=self.rewrite(anchor, state_counterfeit))


def source_map():
    oracle = json.loads(pinned(ORACLE, ORACLE_SHA))
    paths = (Path(__file__).resolve(), SUBJECT, HELPER, ORACLE,
             FIXTURES / 'r11-native-jobs.raw', FIXTURES / 'r11-source-proof.json',
             *(FIXTURES / row['file'] for row in oracle['artifacts'].values()))
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file(): raise RuntimeError('Ordinary independent input required')
        result[str(path)] = sha(path.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    output = parser.parse_args().output_directory
    if sys.flags.optimize or not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise RuntimeError('Unoptimized fresh absolute external output required')
    if output == ROOT or output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise RuntimeError('Disjoint external output required')
    started = time.monotonic_ns(); before = source_map(); output.mkdir(); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Native7BundleControls))
    raw = stream.getvalue().encode('utf8'); (output / 'unittest.log').write_bytes(raw)
    after = source_map(); complete = result.wasSuccessful() and result.testsRun == METHODS and before == after
    record = dict(schema='sourceq_native7_bundle_independent_tests_v1', complete=complete,
        unittest_methods=result.testsRun, expected_methods=METHODS, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), source_before=before, source_after=after,
        source_unchanged=before == after, raw_bytes=len(raw), raw_sha256=sha(raw),
        started_ns=started, finished_ns=time.monotonic_ns(), actual_OS_observations_or_owner_launches=0,
        actual_Julia_launches=0, archive_members_extracted_or_executed=False, qualification_issued=False,
        source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / 'RESULT.json').write_bytes(jbytes(record))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
