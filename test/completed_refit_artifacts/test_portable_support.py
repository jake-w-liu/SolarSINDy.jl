"""Bounded synthetic filesystem/source-projection controls; no model fitting."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from unittest import mock
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fixed_snapshot_under_test',PACKAGE/'research/dbdt_completed_source/synthetic_checkpoint.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
factoryspec=importlib.util.spec_from_file_location('fixed_source_factory_under_test',PACKAGE/'research/dbdt_completed_source/synthetic_sources.py')
F=importlib.util.module_from_spec(factoryspec);factoryspec.loader.exec_module(F)
OUT=Path(tempfile.mkdtemp(prefix='ss-r1-support-')).resolve()

def command(path,*arguments):
    return subprocess.run(['git','-C',str(path),*arguments],check=True,capture_output=True,timeout=30).stdout.decode().strip()

def source(name):
    path=OUT/name;path.mkdir()
    (path/'Project.toml').write_text('[deps]\n',encoding='utf-8')
    (path/'Manifest.toml').write_text('julia_version = "1.12.7"\nmanifest_format = "2.0"\n',encoding='utf-8')
    (path/'algorithm.jl').write_text('identity(x)=x\n',encoding='utf-8')
    command(path,'init');command(path,'config','user.name','Synthetic source fixture')
    command(path,'config','user.email','fixture@invalid');command(path,'add','--','Project.toml','Manifest.toml','algorithm.jl')
    command(path,'commit','-m','Synthetic source fixture')
    return path

class ProjectionTests(unittest.TestCase):
    def test_owned_capture_rejects_partial_and_other_stream_io_failures(self):
        command=[sys.executable,'-B','-c',"import os;os.write(1,b'complete expected stdout\\n');os.write(2,b'complete expected stderr\\n')"]
        positive=S.captured(command,timeout=30)
        self.assertEqual(positive.returncode,0);self.assertEqual(positive.stdout,b'complete expected stdout\n')
        original=S._CAPTURE_CODE
        for mode in ('write_error','short_write','open_error','flush_error','file_close_error','read_error','reader_close_error'):
            prefix='''import builtins,subprocess,sys
MODE=FAULT_MODE
_original_open=builtins.open
_original_popen=subprocess.Popen
class FaultingWrite:
    def __init__(self,file):self.file=file
    def __enter__(self):self.file.__enter__();return self
    def __exit__(self,*args):
        result=self.file.__exit__(*args)
        if MODE=="file_close_error":raise OSError("controlled file close fault")
        return result
    def write(self,data):
        if MODE in ("write_error","short_write"):
            self.file.write(data[:1])
            if MODE=="write_error":raise OSError("controlled partial stdout write fault")
            return 1
        return self.file.write(data)
    def flush(self):
        if MODE=="flush_error":raise OSError("controlled flush fault")
        return self.file.flush()
class FaultingRead:
    def __init__(self,pipe):self.pipe=pipe;self.called=False
    def read(self,size):
        if MODE=="read_error":
            if self.called:raise OSError("controlled reader fault")
            self.called=True;return self.pipe.read(1)
        return self.pipe.read(size)
    def close(self):
        self.pipe.close()
        if MODE=="reader_close_error":raise OSError("controlled reader close fault")
def injected_open(path,*args,**kwargs):
    if str(path)==sys.argv[1] and MODE=="open_error":raise OSError("controlled open fault")
    file=_original_open(path,*args,**kwargs)
    return FaultingWrite(file) if str(path)==sys.argv[1] else file
def injected_popen(*args,**kwargs):
    process=_original_popen(*args,**kwargs)
    if MODE in ("read_error","reader_close_error"):process.stdout=FaultingRead(process.stdout)
    return process
builtins.open=injected_open
subprocess.Popen=injected_popen
'''.replace('FAULT_MODE',repr(mode))
            with mock.patch.object(S,'_CAPTURE_CODE',prefix+original):
                with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
                    S.captured(command,timeout=30)
            directory=Path(str(context.exception).split(': ',1)[1])
            receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
            self.assertEqual(receipt['native_exit'],88,mode);self.assertFalse(receipt['timed_out'],mode)
            self.assertIn(b'failure:',(directory/'capture.log').read_bytes(),mode)
            for name in ('stdout.log','stderr.log'):
                if (directory/name).exists():self.assertLessEqual((directory/name).stat().st_size,8388608,mode)

    def test_owned_capture_distinguishes_internal_launch_failure(self):
        with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
            S.captured([str(OUT/'nonexistent-control-child')],timeout=30)
        directory=Path(str(context.exception).split(': ',1)[1])
        receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
        self.assertEqual(receipt['native_exit'],88);self.assertFalse(receipt['timed_out'])
        self.assertIn(b'capture main failure:',(directory/'capture.log').read_bytes())

    def test_optimized_entrypoints_reject_before_arguments_or_actions(self):
        for name in ('synthetic_checkpoint.py','run_synthetic_chain.py','synthetic_sources.py'):
            path=PACKAGE/'research/dbdt_completed_source'/name
            for flag,environment in ((['-O'],dict(os.environ)),([],dict(os.environ,PYTHONOPTIMIZE='1'))):
                result=subprocess.run([sys.executable,'-B',*flag,str(path)],env=environment,
                    capture_output=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(b'optimized Python is unsupported',result.stderr)
                self.assertNotIn(b'usage:',result.stderr)

    def test_manifest_oversize_rejects_before_body_or_git(self):
        src=source('oversize-manifest')
        with (src/'Manifest.toml').open('wb') as stream:
            stream.truncate(8388609)
        with mock.patch.object(S,'git',side_effect=AssertionError('Git must not run')), \
             mock.patch.object(Path,'open',side_effect=AssertionError('body must not open')):
            with self.assertRaisesRegex(ValueError,'Manifest exceeds bounded source size'):
                S.state(src)
        self.assertFalse((OUT/'oversize-child').exists())

    def test_bounded_file_accepts_exact_limit_and_rejects_one_more(self):
        path=OUT/'read-boundary';path.write_bytes(b'12345678')
        self.assertEqual(S.bounded_file(path,8),b'12345678')
        path.write_bytes(b'123456789')
        with mock.patch.object(Path,'open',side_effect=AssertionError('oversize body must not open')):
            with self.assertRaisesRegex(ValueError,'control file exceeds bounded read limit'):
                S.bounded_file(path,8)

    def test_owned_capture_drains_both_streams_at_exact_boundary(self):
        code="import os; os.write(1,b'a'*8388608); os.write(2,b'b'*8388608)"
        value=S.captured([sys.executable,'-B','-c',code],timeout=30)
        self.assertEqual(value.returncode,0)
        self.assertEqual(len(value.stdout),8388608);self.assertEqual(len(value.stderr),8388608)
        self.assertEqual(value.stdout[:1],b'a');self.assertEqual(value.stderr[-1:],b'b')

    def test_owned_capture_preserves_nonzero_exit(self):
        value=S.captured([sys.executable,'-B','-c',"import sys;sys.stderr.write('ordinary failure');sys.exit(2)"],timeout=30)
        self.assertEqual(value.returncode,2);self.assertEqual(value.stderr,b'ordinary failure')

    def test_owned_capture_overflow_retains_bounded_prefix(self):
        code="import os; os.write(1,b'a'*8388609); os.write(2,b'b'*8388609)"
        with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
            S.captured([sys.executable,'-B','-c',code],timeout=30)
        directory=Path(str(context.exception).split(': ',1)[1])
        receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
        self.assertEqual(receipt['native_exit'],86);self.assertFalse(receipt['timed_out'])
        self.assertLessEqual((directory/'stdout.log').stat().st_size,8388608)
        self.assertLessEqual((directory/'stderr.log').stat().st_size,8388608)
        self.assertEqual((directory/'stdout.log').stat().st_size,8388608)

    def test_owned_capture_timeout_cleans_descendant(self):
        pidfile=OUT/'owned-descendant.pid'
        child="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(60)"
        code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-B','-c',"+repr(child)+"]);time.sleep(60)"
        with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
            S.captured([sys.executable,'-B','-c',code],timeout=2)
        directory=Path(str(context.exception).split(': ',1)[1])
        receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
        self.assertTrue(receipt['timed_out']);self.assertTrue(pidfile.is_file())
        pid=int(pidfile.read_text())
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
            handle=kernel.OpenProcess(0x1000,False,pid)
            if handle:
                try:
                    exitcode=wintypes.DWORD();self.assertTrue(kernel.GetExitCodeProcess(handle,ctypes.byref(exitcode)))
                    self.assertNotEqual(exitcode.value,259)
                finally:kernel.CloseHandle(handle)
        else:
            # A killed process can briefly remain a zombie until its parent is
            # reaped; neither a live process nor a live owned session may remain.
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:
                result=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,timeout=10)
                self.assertTrue(result.stdout.strip().startswith(b'Z'))

    def test_git_diff_output_overflow_rejects_before_buffering(self):
        src=source('git-overflow')
        # Each valid source file remains below the source-file cap; their exact
        # textual before/after diff exceeds the separate Git-output cap.
        (src/'Manifest.toml').write_bytes((b'a'*1024+b'\n')*4500)
        command(src,'add','--','Manifest.toml');command(src,'commit','-m','Bounded generated text fixture')
        (src/'Manifest.toml').write_bytes((b'b'*1024+b'\n')*4500)
        self.assertLess((src/'Manifest.toml').stat().st_size,8388608)
        with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
            S.git(src,'diff','HEAD','--','Manifest.toml',strip_output=False)
        directory=Path(str(context.exception).split(': ',1)[1])
        receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
        self.assertEqual(receipt['native_exit'],86);self.assertFalse(receipt['timed_out'])
        self.assertEqual((directory/'stdout.log').stat().st_size,8388608)
        self.assertLessEqual((directory/'stderr.log').stat().st_size,8388608)

    def test_fixed_native_entry_git_uses_same_bounded_owner(self):
        src=source('entry-git-overflow')
        (src/'Manifest.toml').write_bytes((b'a'*1024+b'\n')*4500)
        command(src,'add','--','Manifest.toml');command(src,'commit','-m','Bounded generated entry fixture')
        (src/'Manifest.toml').write_bytes((b'b'*1024+b'\n')*4500)
        entry=PACKAGE/'research/dbdt_completed_source/run_synthetic_chain.py'
        parsed=ast.parse(entry.read_bytes())
        function=next(node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=='git')
        scope={'checkpointmodule':S,'pkg':src}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(entry),'exec'),scope)
        with self.assertRaisesRegex(RuntimeError,'bounded Git capture failed') as context:
            scope['git']('diff','HEAD','--','Manifest.toml')
        directory=Path(str(context.exception).split(': ',1)[1])
        receipt=json.loads((directory/'OWNED_NATIVE_RAW.json').read_text())
        self.assertEqual(receipt['native_exit'],86);self.assertFalse(receipt['timed_out'])
        self.assertEqual((directory/'stdout.log').stat().st_size,8388608)

    def test_source_change_during_snapshot_rejects(self):
        src=source('concurrent');original_git=S.git;changed=False
        def guarded_git(path,*arguments,**keywords):
            nonlocal changed
            result=original_git(path,*arguments,**keywords)
            if not changed and Path(path)==OUT/'concurrent-child' and arguments[:2]==('checkout','--detach'):
                (src/'algorithm.jl').write_text('identity(x)=0\n',encoding='utf-8');changed=True
            return result
        with mock.patch.object(S,'git',side_effect=guarded_git):
            with self.assertRaisesRegex(AssertionError,'only an explicit tracked Manifest delta'):
                S.prepare_snapshot(src,OUT/'concurrent-child')
        self.assertTrue(changed);self.assertTrue((OUT/'concurrent-child').is_dir())
        self.assertEqual((src/'algorithm.jl').read_text(),'identity(x)=0\n')

    def test_manifest_link_flag_rejects_before_open(self):
        src=source('link-flag');manifest=src/'Manifest.toml';before=manifest.read_bytes()
        original=Path.is_symlink
        def flagged(path):return path==manifest or original(path)
        with mock.patch.object(Path,'is_symlink',flagged), \
             mock.patch.object(Path,'open',side_effect=AssertionError('link target must not open')):
            with self.assertRaisesRegex(AssertionError,'reparse/symlink input rejected'):
                S.state(src)
        self.assertEqual(before,manifest.read_bytes())

    def test_clean_snapshot_preserves_every_byte(self):
        src=source('clean');before=S.state(src)
        independent={name:(src/name).read_bytes() for name in ('Project.toml','Manifest.toml','algorithm.jl')}
        parent=command(src,'rev-parse','HEAD')
        proof=S.prepare_snapshot(src,OUT/'clean-child')
        self.assertEqual(before,S.state(src));self.assertTrue(proof['original_checkout_unchanged'])
        self.assertEqual(before['head'],proof['child_head'])
        self.assertEqual(before['files_sha256'],proof['child']['files_sha256'])
        self.assertFalse((OUT/'clean-child/.git/objects/info/alternates').exists())
        self.assertEqual(parent,command(src,'rev-parse','HEAD'))
        for name,value in independent.items():
            self.assertEqual(value,(src/name).read_bytes())
            self.assertEqual(value,(OUT/'clean-child'/name).read_bytes())

    def test_exact_manifest_delta_is_truthful_child(self):
        src=source('manifest');parent=command(src,'rev-parse','HEAD')
        (src/'Manifest.toml').write_text('julia_version = "1.13.1"\nmanifest_format = "2.1"\n',encoding='utf-8')
        independent={name:(src/name).read_bytes() for name in ('Project.toml','Manifest.toml','algorithm.jl')}
        before=S.state(src);proof=S.prepare_snapshot(src,OUT/'manifest-child')
        self.assertEqual(before,S.state(src));self.assertNotEqual(parent,proof['child_head'])
        self.assertEqual(proof['parent_head'],parent);self.assertEqual(proof['child']['files_sha256'],before['files_sha256'])
        self.assertEqual(command(OUT/'manifest-child','diff','HEAD^','--name-only'),'Manifest.toml')
        self.assertIn('1.13.1',proof['exact_manifest_diff'])
        self.assertEqual(parent,command(src,'rev-parse','HEAD'))
        for name,value in independent.items():
            self.assertEqual(value,(src/name).read_bytes())
            self.assertEqual(value,(OUT/'manifest-child'/name).read_bytes())

    def test_project_change_rejects(self):
        src=source('project');(src/'Project.toml').write_text('[deps]\n# changed\n',encoding='utf-8')
        with self.assertRaises(AssertionError):S.prepare_snapshot(src,OUT/'project-child')
        self.assertFalse((OUT/'project-child').exists())

    def test_algorithm_change_rejects(self):
        src=source('algorithm');(src/'algorithm.jl').write_text('identity(x)=0\n',encoding='utf-8')
        with self.assertRaises(AssertionError):S.prepare_snapshot(src,OUT/'algorithm-child')

    def test_untracked_input_rejects(self):
        src=source('untracked');(src/'unapproved.json').write_text('{}\n',encoding='utf-8')
        with self.assertRaises(AssertionError):S.prepare_snapshot(src,OUT/'untracked-child')

    def test_existing_destination_preserved(self):
        src=source('existing');target=OUT/'existing-child';target.mkdir();(target/'foreign').write_bytes(b'foreign')
        with self.assertRaises(AssertionError):S.prepare_snapshot(src,target)
        self.assertEqual((target/'foreign').read_bytes(),b'foreign')

    def test_inside_source_destination_rejects(self):
        src=source('inside')
        with self.assertRaises(AssertionError):S.prepare_snapshot(src,src/'child')

    def test_canonical_projection_uses_bounded_pair_arrays(self):
        src=source('projection');proof=S.prepare_snapshot(src,OUT/'projection-child');value=S.bounded_projection(proof)
        self.assertIsInstance(value['source_before']['files_sha256'],list)
        self.assertEqual(dict(value['source_before']['files_sha256']),proof['source_before']['files_sha256'])
        self.assertNotIn('exact_manifest_diff',value)

    def test_generated_ground_has_no_wind_body(self):
        root=OUT/'factory-ground';root.mkdir();result=F.create_sources(PACKAGE,root,'ground11')
        protocol=json.loads(Path(result['protocol_path']).read_text())
        self.assertFalse((root/'synthetic-inputs/synthetic-arrival-wind.json').exists())
        self.assertEqual([x['body_id'] for x in protocol['source_permissions'] if x['disposition']=='approved'],['ground'])
        self.assertEqual(len(result['issues']),20)

    def test_generated_mixed_has_exact_wind_and_ground_rows(self):
        root=OUT/'factory-mixed';root.mkdir();result=F.create_sources(PACKAGE,root,'mixed8')
        ground=json.loads((root/'synthetic-inputs/synthetic-ground-vectors.json').read_text())
        wind=json.loads((root/'synthetic-inputs/synthetic-arrival-wind.json').read_text())
        self.assertEqual(len(ground['rows']),1240);self.assertEqual(len(wind['rows']),20)
        self.assertEqual(len(result['issues']),20)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'schema':'synthetic_portable_support_controls_v1','complete':result.wasSuccessful(),'tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'output_directory':str(OUT),'native_fit_claim':False,
        'source_bound_provenance_verified':False}
    (OUT/'TERMINAL.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
