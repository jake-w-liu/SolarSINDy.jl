"""Malformed tables must remain bounded before Julia column parsing."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
from bounded_tables import bounded_table,InvalidArtifact


class BoundedTables(unittest.TestCase):
    def test_valid_quotes_and_rejections(self):
        with tempfile.TemporaryDirectory(prefix='solarsindy-table-limits-') as folder:
            root=Path(folder).resolve(strict=True);path=root/'table.csv'
            path.write_bytes(b'a,b\n"quoted, comma",2\n')
            self.assertEqual(bounded_table(path,columns=2,maximum_rows=1,exact_rows=1),1)
            for data in (b'a,a\n1,2\n',b'a,b,c\n1,2,3\n',b'a,b\n1,2\n3,4\n',
                         b'a,b\n"bad\nquote",2\n',b'a,b\n'+b'x'*4097+b',2\n',b'a,b\n\0,2\n'):
                path.write_bytes(data)
                with self.subTest(bytes=len(data)):
                    with self.assertRaises(InvalidArtifact):bounded_table(path,columns=2,maximum_rows=1)
            path.write_bytes(b'a'*65537)
            with patch.object(csv,'reader',side_effect=AssertionError('oversized row must precede column parsing')):
                with self.assertRaisesRegex(InvalidArtifact,'before parsing'):
                    bounded_table(path,columns=2,maximum_rows=1)

    def test_invalid_dimensions_precede_io(self):
        with patch.object(Path,'open',side_effect=AssertionError('invalid dimensions must precede I/O')):
            for columns,rows,exact in ((True,1,None),(18,1,None),(2,0,None),(2,1,2),(2,1,True)):
                with self.assertRaises(InvalidArtifact):
                    bounded_table(Path('inert'),columns=columns,maximum_rows=rows,exact_rows=exact)


if __name__=='__main__':unittest.main()
