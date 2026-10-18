"""Independent file/budget/JSON failure controls for the saved-state reader."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
import artifact_io as io


class ArtifactIO(unittest.TestCase):
    def setUp(self):
        self.owner=tempfile.TemporaryDirectory(prefix='initial-reference-io-')
        self.root=Path(self.owner.name).resolve(strict=True)

    def tearDown(self):
        # The single owned temporary tree is canonical and contains only this
        # test's files. Link controls explicitly remove their links themselves.
        self.assertEqual(self.root,Path(self.owner.name).resolve(strict=True))
        self.owner.cleanup()

    def put(self,name,contents):
        path=self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_regular_bytes_and_early_cap(self):
        path=self.put('data.bin',b'12345678')
        self.assertEqual(io.member(self.root,'data.bin'),path)
        result=io.read_file(path,8,materialize=True)
        self.assertEqual(result['data'],b'12345678')
        self.assertEqual(result['sha256'],hashlib.sha256(b'12345678').hexdigest())
        self.assertEqual(result['bytes'],8)
        with mock.patch.object(Path,'open',side_effect=AssertionError('must reject before opening')):
            with self.assertRaises(io.InvalidArtifact):io.read_file(path,7)
        for limit in (True,0,-1):
            with self.assertRaises(io.InvalidArtifact):io.read_file(path,limit)

    def test_paths_reject_before_escape(self):
        self.put('valid.json',b'{}')
        version=self.put('model+version.json',b'{}')
        self.assertEqual(io.member(self.root,'model+version.json'),version)
        for name in ('../valid.json','/valid.json','a//b','a/./b','a/../b','a\\b','C:/x','NUL','con.txt','x.','x\x00'):
            with self.assertRaises(io.InvalidArtifact):io.member(self.root,name)
        self.assertEqual(io.plain_root(self.root),self.root)
        with self.assertRaises(io.InvalidArtifact):io.plain_root(Path('.'))
        if sys.platform=='win32':
            with self.assertRaises(io.InvalidArtifact):io.member(self.root,'VALID.json')

    def test_json_grammar_and_preparse_limits(self):
        self.assertEqual(io.decode_json(b'{"a":[0,1]}'),{'a':[0,1]})
        self.assertEqual(io.decode_json(b'{"a":[0,1]}',io.Limits(nodes=5)),{'a':[0,1]})
        for raw in (b'{"a":1,"\\u0061":2}',b'{"a":NaN}',b'{"a":1e999}',b'{"a":[1}',b'{"a":"unterminated}'):
            with self.assertRaises((io.InvalidArtifact,ValueError)):io.decode_json(raw)
        for limits in (io.Limits(nodes=4),io.Limits(depth=1),io.Limits(string_characters=1)):
            value=b'{"aa":[0,1]}' if limits.string_characters==1 else b'{"a":[0,1]}'
            with mock.patch.object(json,'loads',side_effect=AssertionError('JSON must not materialize')):
                with self.assertRaises(io.InvalidArtifact):io.decode_json(value,limits)
        path=self.put('small.json',b'{}')
        digest=hashlib.sha256(b'{}').hexdigest()
        self.assertEqual(io.json_file(path,digest),{})
        with self.assertRaises(io.InvalidArtifact):io.json_file(path,digest+'\n')
        with mock.patch.object(Path,'open',side_effect=AssertionError('invalid budgets must precede I/O')):
            with self.assertRaises(io.InvalidArtifact):io.json_file(path,digest,io.Limits(file_bytes=1,json_bytes=2))

    def test_inventory_preflights_counts_and_bytes(self):
        self.put('one.bin',b'1234')
        self.put('sub/two.bin',b'5678')
        expected={'one.bin':hashlib.sha256(b'1234').hexdigest(),'sub/two.bin':hashlib.sha256(b'5678').hexdigest()}
        limits=io.Limits(file_bytes=4,total_bytes=8,json_bytes=4,files=2,directories=2)
        self.assertEqual(io.inventory(self.root,limits),expected)
        for bad in (io.Limits(file_bytes=4,total_bytes=7,json_bytes=4),
                    io.Limits(files=1),io.Limits(directories=1)):
            with mock.patch.object(Path,'open',side_effect=AssertionError('inventory budgets must precede hashing')):
                with self.assertRaises(io.InvalidArtifact):io.inventory(self.root,bad)
        self.assertEqual(io.inventory(self.root,limits),expected)

    def test_case_alias_inventory_and_links(self):
        self.put('same.bin',b'x')
        other=self.root/'SAME.bin'
        if not other.exists():
            other.write_bytes(b'y')
            with self.assertRaises(io.InvalidArtifact):io.inventory(self.root)
            other.unlink()
        link=self.root/'linked.bin'
        try:
            link.symlink_to(self.root/'same.bin')
        except OSError:
            # Capability is reported; no link-control pass is inferred on hosts
            # where creating a symbolic link requires unavailable privileges.
            self.skipTest('file symlink creation unavailable')
        try:
            with self.assertRaises(io.InvalidArtifact):io.member(self.root,'linked.bin')
            with self.assertRaises(io.InvalidArtifact):io.inventory(self.root)
        finally:
            link.unlink()


if __name__=='__main__':unittest.main()
