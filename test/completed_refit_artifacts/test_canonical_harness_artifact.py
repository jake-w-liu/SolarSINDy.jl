from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest,zipfile

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('hosted_data_only_verifier',ROOT/'tools/verify_canonical_harness_artifact.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
def encode(value):return (json.dumps(value,sort_keys=True)+'\n').encode()
class ArtifactTests(unittest.TestCase):
    def fixture(self):
        expected=dict(head='a'*40,tree='b'*40,source_files=1,source_sha256={'source.jl':'c'*64},commit_object_ids={'source.jl':'d'*40})
        source=dict(expected,complete=True,all_clean_and_checkout_filters_match=True);source_bytes=encode(source)
        before=dict(head=expected['head'],source_sha256=expected['source_sha256'],source_proof_sha256=V.digest(source_bytes),
            canonical_audit_sha256=V.AUDIT_SHA,owner_sha256=V.OWNER_SHA,actual_source_access=False,actual_fits=0,
            scientific_performance_claim=False,controls_sha256={'control':'e'*64})
        files={'SOURCE_PROOF.json':source_bytes,'PREEXECUTION.json':encode(before),'run-harness.sh':b'#!/bin/bash\n',
            'harness.log':b'[PASS] Pkg.test() passed\n[PASS] examples/experiments.jl executed\nPASS: 2\nWARN: 0\nFAIL: 0\n',
            'raw/dev-harness-pkg-test.fixture':b'independent package raw\n','raw/dev-harness-experiments.fixture':b'independent experiment raw\n'}
        terminal=dict(before,complete=True,unfiltered_Pkg_test_and_complete_experiments_passed=True,native={'native_exit':0,'timed_out':False},
            available_physical_bytes_at_admission=8*1024**3,available_disk_bytes_at_admission=8*1024**3,
            memory_admission_method='Linux /proc/meminfo MemAvailable',changed_sources=[],changed_controls=[],guard_read_errors=[],
            inventory_errors=[],git_status='',head_changed=False,audit_counts={'PASS':2,'WARN':0,'FAIL':0},
            harness_log_sha256=V.digest(files['harness.log']),raw_and_operational_members={name:{'bytes':len(body),'sha256':V.digest(body)} for name,body in files.items() if name.startswith('raw/')})
        metadata=dict(schema='canonical_harness_special_fixture_metadata_v1',complete=True,ordinary_file_members=2,
            ordinary_file_bytes=sum(row['bytes'] for row in terminal['raw_and_operational_members'].values()),fixtures=[],
            targets_followed=False,fixture_objects_loaded=False)
        files['FIXTURE_METADATA.json']=encode(metadata)
        terminal.update(fixture_metadata_sha256=V.digest(files['FIXTURE_METADATA.json']),special_fixture_objects=0,
            artifact_snapshot_ordinary_files_only=True,fixture_targets_followed=False,snapshot_source_identities_and_bytes_reverified=True)
        files['TERMINAL.json']=encode(terminal)
        return expected,files,terminal
    def run_zip(self,expected,files):
        with tempfile.TemporaryDirectory(prefix='hosted-artifact-verifier-') as directory:
            path=Path(directory)/'fixture.zip'
            with zipfile.ZipFile(path,'w') as archive:
                for name,body in files.items():archive.writestr(name,body)
            return V.verify(path,expected)
    def test_complete_exact_data_only_fixture(self):
        expected,files,_=self.fixture();result=self.run_zip(expected,files)
        self.assertTrue(result['complete']);self.assertEqual(result['all_raw_and_operational_members_verified'],2)
    def test_wrong_source_head_rejected(self):
        expected,files,_=self.fixture();expected['head']='f'*40
        with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_insufficient_physical_memory_rejected(self):
        expected,files,terminal=self.fixture();terminal['available_physical_bytes_at_admission']-=1;files['TERMINAL.json']=encode(terminal)
        with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_missing_or_corrupt_raw_rejected(self):
        for mutation in ('missing','corrupt','extra'):
            expected,files,_=self.fixture();name='raw/dev-harness-pkg-test.fixture'
            if mutation=='missing':del files[name]
            elif mutation=='corrupt':files[name]=b'changed raw\n'
            else:files['raw/unreported.log']=b'unreported\n'
            with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_failed_or_changed_run_rejected(self):
        for key,value in (('complete',False),('changed_sources',['source.jl']),('git_status',' M source.jl'),('head_changed',True),('actual_source_access',True)):
            expected,files,terminal=self.fixture();terminal[key]=value;files['TERMINAL.json']=encode(terminal)
            with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_unsafe_archive_member_rejected(self):
        for name in ('../outside','/absolute','raw\\alias','raw/C:alias','raw//alias','raw/./alias'):
            expected,files,_=self.fixture();files[name]=b'unsafe'
            with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_duplicate_json_fields_rejected(self):
        expected,files,_=self.fixture();files['SOURCE_PROOF.json']=b'{"complete":true,"complete":true}'
        with self.assertRaises(ValueError):self.run_zip(expected,files)

    def metadata_update(self,files,terminal,*,fixtures=None):
        metadata=json.loads(files['FIXTURE_METADATA.json'])
        if fixtures is not None:metadata['fixtures']=fixtures
        metadata['ordinary_file_members']=len(terminal['raw_and_operational_members'])
        metadata['ordinary_file_bytes']=sum(row['bytes'] for row in terminal['raw_and_operational_members'].values())
        files['FIXTURE_METADATA.json']=encode(metadata)
        terminal.update(fixture_metadata_sha256=V.digest(files['FIXTURE_METADATA.json']),special_fixture_objects=len(metadata['fixtures']))
        files['TERMINAL.json']=encode(terminal)
    def test_hidden_regular_raw_is_required_and_verified(self):
        expected,files,terminal=self.fixture();name='raw/.hidden-raw';files[name]=b'necessary hidden evidence'
        terminal['raw_and_operational_members'][name]=dict(bytes=len(files[name]),sha256=V.digest(files[name]))
        self.metadata_update(files,terminal)
        self.assertEqual(self.run_zip(expected,files)['all_raw_and_operational_members_verified'],3)
        del files[name]
        with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_link_metadata_grants_no_target_loading(self):
        expected,files,terminal=self.fixture()
        fixture=dict(path='raw/fixture-link',kind='link',mode=0o120777,target='/outside/never-open')
        self.metadata_update(files,terminal,fixtures=[fixture])
        self.assertEqual(self.run_zip(expected,files)['special_fixture_objects_verified'],1)
    def test_dereferenced_link_bytes_are_rejected_even_if_inventoried(self):
        for name in ('raw/fixture-link','raw/fixture-link/child'):
            expected,files,terminal=self.fixture();files[name]=b'should never be read'
            terminal['raw_and_operational_members'][name]=dict(bytes=len(files[name]),sha256=V.digest(files[name]))
            self.metadata_update(files,terminal,fixtures=[dict(path='raw/fixture-link',kind='link',mode=0o120777,target='/outside')])
            with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_fixture_permissions_and_metadata_corruption_fail(self):
        for key,value in (('fixture_targets_followed',True),('artifact_snapshot_ordinary_files_only',False),
                          ('snapshot_source_identities_and_bytes_reverified',False),('special_fixture_objects',False)):
            expected,files,terminal=self.fixture();terminal[key]=value;files['TERMINAL.json']=encode(terminal)
            with self.assertRaises(AssertionError):self.run_zip(expected,files)
        expected,files,terminal=self.fixture();files['FIXTURE_METADATA.json']+=b' '
        with self.assertRaises(AssertionError):self.run_zip(expected,files)
    def test_unsafe_fixture_path_or_kind_is_rejected(self):
        for path in ('raw/../escape','raw//bad','raw/./bad','raw\\bad','raw/inner\\bad','other/file'):
            expected,files,terminal=self.fixture()
            self.metadata_update(files,terminal,fixtures=[dict(path=path,kind='link',mode=0o120777,target='/outside')])
            with self.assertRaises(AssertionError):self.run_zip(expected,files)

if __name__=='__main__':unittest.main(verbosity=2)
