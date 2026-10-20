"""Generated-only full initial grid; actual fits require separate qualification."""
import os
from pathlib import Path
import io as text_io
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'initial_reference'))
from test_full_reader import PROJECT,TOOLS,extract_fixture,sha,io,run_owned


def emit_stage_log(text,stream=None):
    """Keep raw logs intact while rendering unsupported console characters safely."""
    stream=sys.stdout if stream is None else stream
    encoding=getattr(stream,'encoding',None) or 'utf-8'
    stream.write(text.encode(encoding,errors='backslashreplace').decode(encoding))
    stream.flush()


class SuccessLogEncoding(unittest.TestCase):
    def test_utf8_and_windows_console_preserve_success_text(self):
        for encoding,expected in [('cp950',r'\u2713 passed'),('utf-8','\u2713 passed')]:
            buffer=text_io.BytesIO()
            stream=text_io.TextIOWrapper(buffer,encoding=encoding,errors='strict',newline='')
            try:
                emit_stage_log('\u2713 passed\n',stream)
                self.assertEqual(buffer.getvalue().decode(encoding),expected+'\n')
            finally:
                stream.close()


class InitialAStage(unittest.TestCase):
    def test_generated_initial_grid_and_direct_oracles(self):
        root=Path(tempfile.mkdtemp(prefix='solarsindy-a-initial-')).resolve(strict=True)
        try:
            reference,source=extract_fixture(root/'fixture')
            before=io.inventory(root/'fixture')
            chosen=os.environ.get('SOLARSINDY_AUDIT_TEST_JULIA') or shutil.which('julia')
            self.assertTrue(chosen,'explicit Julia required')
            julia=Path(chosen).resolve(strict=True)
            def command(argv,log,timeout=600):
                native=run_owned(argv,cwd=str(root),environment=dict(os.environ,JULIA_PKG_OFFLINE='true',
                    JULIA_LOAD_PATH='@'+os.pathsep+'@stdlib',JULIA_PKG_PRECOMPILE_AUTO='0',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),logpath=root/log,timeout=timeout)
                self.assertEqual(native['native_exit'],0,(root/log).read_text(encoding='utf-8')[-18000:])
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
                '--project='+str(PROJECT),str(Path(__file__).with_name('initial_stage_cases.jl')),
                str(root/'export'),sha(root/'export/completion.json'),sha(root/'audit/completion.json'),str(root/'stage')],
                'initial-stage.log',timeout=1200)
            command([str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--depwarn=error',
                '--project='+str(PROJECT),str(Path(__file__).with_name('publication_failure_cases.jl')),
                str(PROJECT),str(root/'export'),sha(root/'export/completion.json'),sha(root/'audit/completion.json'),
                str(root/'failed-publication-stage')],'publication-failure.log',timeout=1200)
            self.assertEqual(io.inventory(root/'fixture'),before)
            emit_stage_log((root/'initial-stage.log').read_text(encoding='utf-8'))
            emit_stage_log((root/'publication-failure.log').read_text(encoding='utf-8'))
        except BaseException:
            print('FAILED_A_INITIAL_EVIDENCE_ROOT='+str(root),file=sys.stderr)
            raise
        else:
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
            self.assertTrue(root.name.startswith('solarsindy-a-initial-'))
            io.inventory(root)
            shutil.rmtree(root)


if __name__=='__main__':unittest.main()
