"""Small cross-platform checks for the process owner used by research commands."""
from pathlib import Path
import importlib.util
import os
import sys
import tempfile
import time
import unittest

if sys.argv[1:2] == ['--child']:
    print('child started', flush=True)
    if sys.argv[2] == 'timeout':
        time.sleep(10)
    raise SystemExit(7 if sys.argv[2] == 'failure' else 0)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'native_process_owner', ROOT / 'research/dbdt_completed_source/candidate_native_owner.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class NativeProcessTests(unittest.TestCase):
    def run_child(self, mode, timeout=10):
        with tempfile.TemporaryDirectory(prefix='solarsindy-process-') as directory:
            log = Path(directory) / 'child.log'
            result = owner.run_owned(
                [sys.executable, '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--child', mode],
                cwd=directory, environment=dict(os.environ), logpath=log, timeout=timeout)
            self.assertIn('child started', log.read_text(encoding='utf8'))
            self.assertTrue(result['leader_reaped'])
            self.assertTrue(result['ownership_boundary_closed'])
            return result

    def test_success(self):
        result = self.run_child('success')
        self.assertEqual(result['native_exit'], 0)
        self.assertFalse(result['timed_out'])

    def test_failure_is_reported(self):
        result = self.run_child('failure')
        self.assertEqual(result['native_exit'], 7)
        self.assertFalse(result['timed_out'])

    def test_timeout_stops_and_reaps_child(self):
        started = time.monotonic()
        result = self.run_child('timeout', timeout=1)
        self.assertTrue(result['timed_out'])
        self.assertLess(time.monotonic() - started, 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
