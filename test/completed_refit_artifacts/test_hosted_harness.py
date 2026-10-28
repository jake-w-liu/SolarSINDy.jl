"""Independent physical-admission and shell-quoting controls; no audit execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import importlib.util,os,shlex,unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('canonical_harness_controls',ROOT/'tools/run_canonical_harness.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
class HarnessAdmissionTests(unittest.TestCase):
    def test_linux_uses_available_not_total_or_free(self):
        with mock.patch.object(H,'os',SimpleNamespace(name='posix')),mock.patch.object(H,'sys',SimpleNamespace(platform='linux')),mock.patch.object(H,'Path') as path:
            path.return_value.read_text.return_value='MemTotal: 999999999 kB\nMemFree: 999999998 kB\nMemAvailable: 1234 kB\n'
            value,method=H.available_memory();self.assertEqual(value,1234*1024);self.assertIn('MemAvailable',method)
    def test_linux_missing_or_duplicate_available_fails_closed(self):
        for text in ('MemTotal: 999999 kB\n','MemAvailable: 4 kB\nMemAvailable: 5 kB\n','MemAvailable: -1 kB\n'):
            with mock.patch.object(H,'os',SimpleNamespace(name='posix')),mock.patch.object(H,'sys',SimpleNamespace(platform='linux')),mock.patch.object(H,'Path') as path:
                path.return_value.read_text.return_value=text
                with self.assertRaises(RuntimeError):H.available_memory()
    def test_macos_excludes_inactive_pages(self):
        text=b'Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 20.\nPages speculative: 30.\nPages inactive: 99999999.\n'
        with mock.patch.object(H,'os',SimpleNamespace(name='posix')),mock.patch.object(H,'sys',SimpleNamespace(platform='darwin')),mock.patch.object(H.subprocess,'check_output',return_value=text):
            value,method=H.available_memory();self.assertEqual(value,50*4096);self.assertIn('inactive pages excluded',method)
    def test_macos_incomplete_evidence_fails_closed(self):
        with mock.patch.object(H,'os',SimpleNamespace(name='posix')),mock.patch.object(H,'sys',SimpleNamespace(platform='darwin')),mock.patch.object(H.subprocess,'check_output',return_value=b'Pages free: 2000000.\n'):
            with self.assertRaises(RuntimeError):H.available_memory()
    def test_unsupported_platform_does_not_admit(self):
        with mock.patch.object(H,'os',SimpleNamespace(name='posix')),mock.patch.object(H,'sys',SimpleNamespace(platform='unknown')):
            with self.assertRaises(RuntimeError):H.available_memory()
    def test_quoted_path_remains_one_literal_shell_word(self):
        for name in ("a b's audit",'literal$(never-run)`never-run`'):
            value=H.shell_path(ROOT/name)
            self.assertEqual(shlex.split(shlex.quote(value)),[value])
        with self.assertRaises(ValueError):H.shell_path(ROOT/'bad\npath')
    def test_windows_never_silently_selects_wsl_bash(self):
        with mock.patch.object(H,'os',SimpleNamespace(name='nt',environ={})): 
            with self.assertRaisesRegex(RuntimeError,'Native Git Bash required'):H.bash_path()
    def test_current_platform_memory_probe_is_nonnegative(self):
        value,method=H.available_memory();self.assertIsInstance(value,int);self.assertGreaterEqual(value,0);self.assertTrue(method)
if __name__=='__main__':unittest.main(verbosity=2)
