"""Invalid configuration must fail before a bounded BSON reader opens a file."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
from bounded_bson import read_bson,InvalidArtifact


class BSONByteCap(unittest.TestCase):
    def test_invalid_cap_precedes_file_io(self):
        with patch.object(Path,'open',side_effect=AssertionError('invalid cap must not open a file')):
            for cap in (-2,-1,0,True,False,1.0,'1',None):
                with self.subTest(cap=cap):
                    with self.assertRaisesRegex(InvalidArtifact,'positive integer BSON byte budget'):
                        read_bson(Path('owned-inert-nonexistent-file'),max_bytes=cap)


if __name__=='__main__':unittest.main()
