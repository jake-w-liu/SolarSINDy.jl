"""Generated-only proof-to-raw-prefix integration; no fitting or actual archive."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'initial_reference'))
from test_full_reader import PROJECT,TOOLS,extract_fixture,sha,io,run_owned


class InitialAInputPrefix(unittest.TestCase):
    def test_verified_generated_input_and_corruption_controls(self):
        root=Path(tempfile.mkdtemp(prefix='solarsindy-a-prefix-')).resolve(strict=True)
        try:
            reference,source=extract_fixture(root/'fixture')
            before=io.inventory(root/'fixture')
            chosen=os.environ.get('SOLARSINDY_AUDIT_TEST_JULIA') or shutil.which('julia')
            self.assertTrue(chosen,'explicit Julia required')
            julia=Path(chosen).resolve(strict=True)
            def command(argv,log):
                native=run_owned(argv,cwd=str(root),environment=dict(os.environ,JULIA_PKG_OFFLINE='true',
                    JULIA_LOAD_PATH='@'+os.pathsep+'@stdlib',JULIA_PKG_PRECOMPILE_AUTO='0',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),logpath=root/log,timeout=600)
                self.assertEqual(native['native_exit'],0,(root/log).read_text(encoding='utf-8')[-15000:])
                self.assertFalse(native['timed_out'])
            command([sys.executable,'-B',str(TOOLS/'audit_reference.py'),
                '--reference',str(reference),'--reference-sha256',sha(reference/'completion.json'),
                '--source',str(source),'--source-sha256',sha(source/'completion.json'),
                '--mode','synthetic_only','--output',str(root/'audit'),
                '--julia',str(julia),'--julia-sha256',sha(julia),'--timeout','240'],'audit.log')
            command([sys.executable,'-B',str(PROJECT/'tools/initial_reference_import/export_reference.py'),
                '--audit',str(root/'audit'),'--audit-sha256',sha(root/'audit/completion.json'),
                '--output',str(root/'export')],'export.log')
            command([str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--depwarn=error',
                '--project='+str(PROJECT),str(Path(__file__).with_name('input_prefix_cases.jl')),
                str(root/'export'),sha(root/'export/completion.json'),sha(root/'audit/completion.json')],'prefix.log')
            self.assertEqual(io.inventory(root/'fixture'),before)
        except BaseException:
            print('FAILED_A_PREFIX_EVIDENCE_ROOT='+str(root),file=sys.stderr)
            raise
        else:
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
            self.assertTrue(root.name.startswith('solarsindy-a-prefix-'))
            io.inventory(root)
            shutil.rmtree(root)


if __name__=='__main__':unittest.main()
