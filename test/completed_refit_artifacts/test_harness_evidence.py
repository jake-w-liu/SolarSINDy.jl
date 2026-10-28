"""Small independent byte and no-follow fixture snapshot oracles."""
from pathlib import Path
from unittest import mock
import hashlib,importlib.util,json,os,stat,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('audit_evidence_snapshot',ROOT/'tools/harness_evidence.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

class EvidenceSnapshotTests(unittest.TestCase):
    def test_hidden_bytes_and_exact_members_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();raw=root/'raw';raw.mkdir();operational=root/'operational';operational.mkdir()
            (raw/'.hidden').write_bytes(b'\0hidden\xff');(raw/'visible').write_bytes(b'visible')
            (operational/'nested').mkdir();(operational/'nested/.checkpoint').write_bytes(b'checkpoint')
            result=E.snapshot_members(raw,operational,root/'artifact')
            expected={'raw/.hidden':b'\0hidden\xff','raw/visible':b'visible','operational/nested/.checkpoint':b'checkpoint'}
            self.assertEqual(set(result['members']),set(expected))
            for name,body in expected.items():
                self.assertEqual((root/'artifact'/name).read_bytes(),body)
                self.assertEqual(result['members'][name],dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
            metadata=json.loads((root/'artifact/FIXTURE_METADATA.json').read_bytes())
            self.assertEqual(metadata['fixtures'],[]);self.assertFalse(metadata['targets_followed'])

    def test_fixture_link_is_metadata_and_target_is_never_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();raw=root/'raw';raw.mkdir();operational=root/'operational';operational.mkdir()
            (raw/'pretend-link').write_bytes(b'physical test placeholder')
            real_linked=E.linked
            # Model the lstat result independently of OS symlink privileges.
            original_walk=E.walk
            def fake_walk(roots):
                records,files,fixtures=original_walk(roots)
                files.pop('raw/pretend-link')
                return records,files,[dict(path='raw/pretend-link',kind='link',mode=stat.S_IFLNK|0o777,target='/outside/never-open')]
            with mock.patch.object(E,'walk',side_effect=fake_walk),mock.patch.object(E,'checked_copy',side_effect=AssertionError('link target read')):
                result=E.snapshot_members(raw,operational,root/'artifact')
            self.assertEqual(result['members'],{});self.assertEqual(result['special_fixture_objects'],1)
            self.assertFalse((root/'artifact/raw/pretend-link').exists())
            metadata=json.loads((root/'artifact/FIXTURE_METADATA.json').read_bytes())
            self.assertEqual(metadata['fixtures'][0]['target'],'/outside/never-open')

    def test_real_link_and_fifo_are_not_followed_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();raw=root/'raw';raw.mkdir();operational=root/'operational';operational.mkdir()
            outside=root/'outside';outside.write_bytes(b'not artifact data')
            try:os.symlink(outside,raw/'linked')
            except OSError as error:self.skipTest('OS link creation unavailable;independent no-follow model remains mandatory: '+str(error))
            if hasattr(os,'mkfifo'):os.mkfifo(raw/'pipe')
            result=E.snapshot_members(raw,operational,root/'artifact')
            self.assertEqual(result['members'],{})
            metadata=json.loads((root/'artifact/FIXTURE_METADATA.json').read_bytes())
            self.assertEqual({item['kind'] for item in metadata['fixtures']},{'link','fifo'} if hasattr(os,'mkfifo') else {'link'})
            self.assertEqual(outside.read_bytes(),b'not artifact data')

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();raw=root/'raw';raw.mkdir();operational=root/'operational';operational.mkdir()
            source=raw/'source';source.write_bytes(b'initial');original=E.checked_copy
            def changing(path,destination):
                result=original(path,destination);source.write_bytes(b'changed');return result
            with mock.patch.object(E,'checked_copy',side_effect=changing):
                with self.assertRaises(ValueError):E.snapshot_members(raw,operational,root/'artifact')

    def test_existing_or_overlapping_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();raw=root/'raw';raw.mkdir();operational=root/'operational';operational.mkdir()
            with self.assertRaises(ValueError):E.snapshot_members(raw,operational,raw/'artifact')
            with self.assertRaises(ValueError):E.snapshot_members(raw,operational,operational)

    def test_parent_traversal_and_reparse_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'a').write_bytes(b'a')
            with self.assertRaises(ValueError):E.checked_copy(root/'..'/root.name/'a',root/'copy')
        value=mock.Mock(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
        self.assertTrue(E.linked(value))

if __name__=='__main__':unittest.main(verbosity=2)
