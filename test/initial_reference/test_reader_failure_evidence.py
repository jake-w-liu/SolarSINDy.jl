"""A failed native child must expose its cause and retain complete owned logs."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import test_full_reader


class FailureEvidence(unittest.TestCase):
    def test_failed_child_retains_raw_log_and_reports_bounded_tail(self):
        observed={};cause=b'controlled-native-exit47-cause\n'
        def fail_native(argv,**options):
            root=Path(options['cwd']);observed['root']=root
            (root/'positive').mkdir()
            Path(options['logpath']).write_bytes(b'x'*70000+cause)
            (root/'positive/numerical.log').write_bytes(b'independent numerical child: '+cause)
            return {'native_exit':47,'timed_out':False}
        captured=io.StringIO()
        case=test_full_reader.FullReader('test_replay_and_rehashed_corruptions')
        with patch.object(test_full_reader,'run_owned',side_effect=fail_native),redirect_stderr(captured):
            with self.assertRaises(AssertionError):case.test_replay_and_rehashed_corruptions()
        root=observed['root'];text=captured.getvalue()
        self.assertIn(cause.decode().strip(),text)
        self.assertIn('Native failure evidence: positive/numerical.log',text)
        self.assertIn('Retained failed independent reader controls:',text)
        self.assertLess(len(text),67000)
        self.assertEqual((root/'public-cli.log').stat().st_size,70000+len(cause))
        self.assertTrue(root.is_dir())
        self.assertEqual(root,root.resolve(strict=True))
        self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
        self.assertTrue(root.name.startswith('solarsindy-independent-reader-'))
        shutil.rmtree(root)


if __name__=='__main__':unittest.main()
