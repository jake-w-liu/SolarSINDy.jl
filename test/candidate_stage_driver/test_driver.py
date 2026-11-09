"""Offline driver controls: toy native children and mocked metadata are explicit."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import ctypes,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time,unittest
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'research/dbdt_completed_source/run_candidate_stage.py'
spec=importlib.util.spec_from_file_location('tested_candidate_native_driver',PATH);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
FIXTURE=Path(__file__).with_name('native_fixture.py')
OUT=Path(os.environ['CANDIDATE_DRIVER_CONTROLS_ROOT'] if 'CANDIDATE_DRIVER_CONTROLS_ROOT' in os.environ else tempfile.mkdtemp(prefix='candidate-native-controls-')).resolve()
OUT.mkdir(exist_ok=True)
class DriverControls(unittest.TestCase):
    def setUp(self):self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=OUT)).resolve()
    def command(self,mode):return [sys.executable,'-B','-X','utf8',str(FIXTURE),'--mode',mode,'--root',str(self.root)]
    def metadata_bound_oracles(self):
        # Independent literal, prepared before this edit. These are generated
        # JSON bytes only; no actual source/qualification authority is created.
        original=b'{"x":1}\n';changed=b'{"x":2}\n'
        digest='bb157861a164e35cdde9d726b0af9ce2765a8f530c35d9e45732b94ee65e9557'
        self.assertEqual(hashlib.sha256(original).hexdigest(),digest)
        file=self.root/'bounded-metadata.json';file.write_bytes(original)
        reference=dict(path=str(file),sha256=digest)
        original_sha=hashlib.sha256;original_loads=json.loads
        original_bounded=D.bounded_bytes;original_open=Path.open
        for kind in ('pinned','initial-request'):
            limit=31 if kind=='pinned' else 1048576
            huge=self.root/(kind+'-oversize.json');huge.write_bytes(b'x'*(limit+1))
            with self.subTest(kind=kind),mock.patch.object(D,'pin',side_effect=AssertionError('unbounded pin called')),\
                mock.patch.object(D.hashlib,'file_digest',side_effect=AssertionError('streaming hash called')) as stream_hash,\
                mock.patch.object(D.hashlib,'sha256',wraps=original_sha) as sha,\
                mock.patch.object(D.json,'loads',wraps=original_loads) as parser,\
                mock.patch.object(Path,'open',side_effect=AssertionError('oversized metadata opened')) as opened,\
                mock.patch.object(subprocess,'Popen') as launched:
                with self.assertRaisesRegex(ValueError,'metadata byte bound'):
                    if kind=='pinned':D.pinned(dict(path=str(huge),sha256='0'*64),limit)
                    else:D.admit_request(huge,'0'*64)
                opened.assert_not_called();stream_hash.assert_not_called();sha.assert_not_called();parser.assert_not_called();launched.assert_not_called()
        # Exact bound accepts a single captured snapshot. Mutating the path
        # AFTER that read must not change which bytes are hashed or parsed.
        def capture(path,limit):
            raw=original_bounded(path,limit);file.write_bytes(changed);return raw
        with mock.patch.object(D,'bounded_bytes',side_effect=capture) as bounded,\
            mock.patch.object(D,'pin',side_effect=AssertionError('unbounded pin called')),\
            mock.patch.object(D.hashlib,'file_digest',side_effect=AssertionError('streaming hash called')),\
            mock.patch.object(D.hashlib,'sha256',wraps=original_sha) as sha,\
            mock.patch.object(D.json,'loads',wraps=original_loads) as parser,\
            mock.patch.object(D,'load',side_effect=AssertionError('metadata reopened')):
            self.assertEqual(D.pinned(reference,8),{'x':1})
            bounded.assert_called_once_with(str(file),8)
            sha.assert_called_once_with(original)
            self.assertEqual(parser.call_count,1);self.assertEqual(parser.call_args.args[0],original)
        self.assertEqual(file.read_bytes(),changed)
        # A wrong digest is rejected before parsing the otherwise-valid blob.
        file.write_bytes(original)
        with mock.patch.object(D.json,'loads',wraps=original_loads) as parser,\
            mock.patch.object(D.hashlib,'sha256',wraps=original_sha) as sha:
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):D.pinned(dict(path=str(file),sha256='0'*64),8)
            sha.assert_called_once_with(original);parser.assert_not_called()
        # Simulate growth DURING the real bounded read. No hash/JSON work can
        # run after the size/identity check detects the changing snapshot.
        read_sizes=[]
        class GrowingReader:
            def __init__(self,stream):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close();return False
            def read(self,count):
                read_sizes.append(count);raw=self.stream.read(count)
                with original_open(file,'ab') as out:out.write(b'x')
                return raw
        def growing_open(path,*args,**kwargs):
            stream=original_open(path,*args,**kwargs)
            return GrowingReader(stream) if Path(path)==file and args==('rb',) else stream
        with mock.patch.object(Path,'open',new=growing_open),\
            mock.patch.object(D.hashlib,'sha256',wraps=original_sha) as sha,\
            mock.patch.object(D.json,'loads',wraps=original_loads) as parser:
            with self.assertRaisesRegex(ValueError,'metadata changed'):D.pinned(reference,8)
            self.assertEqual(read_sizes,[9]);sha.assert_not_called();parser.assert_not_called()
        # Preserve the original duplicate/finite/depth/list/text rejection
        # semantics after the byte-read/decode factorization.
        invalid=[b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":1e999}\n',
            b'['*25+b'0'+b']'*25,json.dumps(['x']*8193).encode(),
            json.dumps({'x':'x'*65537}).encode()]
        for index,raw in enumerate(invalid):
            file.write_bytes(raw);ref=dict(path=str(file),sha256=original_sha(raw).hexdigest())
            with self.subTest(invalid=index),self.assertRaises(ValueError):D.pinned(ref)
        # Initial request admission must use the same bounded snapshot too.
        # The literal intentionally has no request fields, so it stops before
        # authority/source reads or launches even after its digest is accepted.
        file.write_bytes(original)
        with mock.patch.object(D,'bounded_bytes',side_effect=capture) as bounded,\
            mock.patch.object(D,'pin',side_effect=AssertionError('unbounded request pin called')),\
            mock.patch.object(D.json,'loads',wraps=original_loads) as parser,\
            mock.patch.object(subprocess,'Popen') as launched:
            with self.assertRaisesRegex(ValueError,'fixed request fields'):D.admit_request(file,digest)
            bounded.assert_called_once_with(str(file),1048576)
            self.assertEqual(parser.call_args.args[0],original);self.assertEqual(parser.call_count,1);launched.assert_not_called()
        # grant_ref has three known metadata roles. Sourceproof/qualification
        # were hash-only here before; their non-JSON bytes must remain legal at
        # this layer, with their later admission grammar preserved elsewhere.
        roles=(('grant',131072,original),('source_proof',4194304,b'not-JSON-source-proof'),
            ('qualification',1048576,b'not-JSON-qualification'))
        authority={}
        for role,limit,raw in roles:
            path=self.root/(role+'-metadata');path.write_bytes(raw)
            authority[role+'_path']=str(path);authority[role+'_sha256']=original_sha(raw).hexdigest()
        with mock.patch.object(D,'bounded_bytes',wraps=original_bounded) as bounded,\
            mock.patch.object(D,'decode_metadata',wraps=D.decode_metadata) as decode,\
            mock.patch.object(D,'pin',side_effect=AssertionError('unbounded authority pin called')),\
            mock.patch.object(D.hashlib,'file_digest',side_effect=AssertionError('streaming authority hash called')):
            self.assertEqual(D.grant_ref(authority),{'x':1})
            self.assertEqual([call.args for call in bounded.call_args_list],
                [(authority[role+'_path'],limit) for role,limit,raw in roles])
            decode.assert_called_once_with(original)
        for offset,(role,limit,raw) in enumerate(roles):
            candidate=dict(authority);oversize=self.root/(role+'-oversize-metadata');oversize.write_bytes(b'x'*(limit+1))
            candidate[role+'_path']=str(oversize);candidate[role+'_sha256']='0'*64
            def guarded_open(path,*args,**kwargs):
                if Path(path)==oversize:raise AssertionError('oversized role opened')
                return original_open(path,*args,**kwargs)
            with self.subTest(role=role),mock.patch.object(Path,'open',new=guarded_open),\
                mock.patch.object(D,'pin',side_effect=AssertionError('unbounded authority pin called')),\
                mock.patch.object(D.hashlib,'file_digest',side_effect=AssertionError('streaming authority hash called')),\
                mock.patch.object(D.hashlib,'sha256',wraps=original_sha) as sha,\
                mock.patch.object(D,'decode_metadata',wraps=D.decode_metadata) as decode:
                with self.assertRaisesRegex(ValueError,'metadata byte bound'):D.grant_ref(candidate)
                self.assertEqual(sha.call_count,offset);decode.assert_not_called()
    def directory_bound_oracles(self):
        # Actual flat metadata fixture, with an independently pinned8-byte
        # literal. No models, observations or source authorities are created.
        directory=self.root/'bounded-flat-files';directory.mkdir()
        expected={}
        for index in range(1024):
            path=directory/('%04d.json'%index);path.write_bytes(b'{"x":1}\n')
            expected[str(path)]='bb157861a164e35cdde9d726b0af9ce2765a8f530c35d9e45732b94ee65e9557'
        self.assertEqual(D.input_controls(dict(directory=str(directory))),expected)
        (directory/'overflow.json').write_bytes(b'{"x":1}\n')
        with mock.patch.object(D,'pin',side_effect=AssertionError('hash before1025-entry bound')) as hashed:
            with self.assertRaisesRegex(ValueError,'artifact fileset bound'):D.input_controls(dict(directory=str(directory)))
            hashed.assert_not_called()
        class Scan:
            def __init__(self,count,error_at=None):self.count=count;self.error_at=error_at;self.visits=0;self.closed=False
            def __enter__(self):return self
            def __exit__(self,*args):self.closed=True;return False
            def __iter__(self):return self
            def __next__(self):
                if self.visits>=self.count:raise StopIteration
                self.visits+=1
                if self.visits>1025:raise AssertionError('eager enumeration exceeded safe fixture bound')
                if self.error_at==self.visits:raise OSError('fixed scandir iteration error')
                return SimpleNamespace(name='%04d.json'%(self.visits-1))
        # A conceptual huge directory remains safe even for a broken eager
        # implementation: the fixture itself stops at1026 rather than allocate
        # unbounded names. Correct code visits exactly1025 and closes it.
        scan=Scan(1000000000)
        with mock.patch.object(D.os,'scandir',return_value=scan),\
            mock.patch.object(Path,'iterdir',side_effect=AssertionError('Path.iterdir eager fallback forbidden')),\
            mock.patch.object(D,'pin',side_effect=AssertionError('hash before bounded collection')) as hashed:
            with self.assertRaisesRegex(ValueError,'artifact fileset bound'):D.input_controls(dict(directory=str(directory)))
            self.assertEqual(scan.visits,1025);self.assertTrue(scan.closed);hashed.assert_not_called()
        scan=Scan(10,error_at=5)
        with mock.patch.object(D.os,'scandir',return_value=scan),mock.patch.object(D,'pin') as hashed:
            with self.assertRaisesRegex(OSError,'scandir iteration error'):D.input_controls(dict(directory=str(directory)))
            self.assertEqual(scan.visits,5);self.assertTrue(scan.closed);hashed.assert_not_called()
        scan=Scan(2)
        def hash_failure(path):
            self.assertTrue(scan.closed)
            raise OSError('fixed add/hash failure after scan close')
        with mock.patch.object(D.os,'scandir',return_value=scan),mock.patch.object(D,'pin',side_effect=hash_failure):
            with self.assertRaisesRegex(OSError,'add/hash failure after scan close'):D.input_controls(dict(directory=str(directory)))
            self.assertEqual(scan.visits,2);self.assertTrue(scan.closed)
        # Preserve flat-file admission. A subdirectory containing a valid
        # generated file must still reject before any recursive child hash.
        flat=self.root/'reject-subdirectory';flat.mkdir();inner=flat/'inner';inner.mkdir()
        (inner/'not-recursively-admitted.json').write_bytes(b'{"x":1}\n')
        with mock.patch.object(D,'pin') as hashed:
            with self.assertRaisesRegex(ValueError,'ordinary path kind'):D.input_controls(dict(directory=str(flat)))
            hashed.assert_not_called()
    def test_real_native_bytes_and_actual_zero_wait(self):
        result=D.pump_process(self.command('emit'),str(self.root),dict(os.environ),self.root/'raw.log',limit=128,timeout=10)
        self.assertTrue(result['complete']);self.assertEqual(result['native_exit'],0);self.assertTrue(result['exact_child_reaped'])
        self.assertEqual((self.root/'raw.log').read_bytes(),b'native-byte-oracle\n');self.assertEqual(result['raw_bytes'],19)
    def test_real_blocking_child_killed_and_reaped(self):
        started=time.monotonic();result=D.pump_process(self.command('block'),str(self.root),dict(os.environ),self.root/'raw.log',limit=128,timeout=.2)
        self.assertFalse(result['complete']);self.assertTrue(result['timed_out']);self.assertTrue(result['exact_child_reaped']);self.assertLess(time.monotonic()-started,5)
    def test_real_flood_is_retained_to_exact_cap_and_failed(self):
        result=D.pump_process(self.command('flood'),str(self.root),dict(os.environ),self.root/'raw.log',limit=257,timeout=10)
        self.assertFalse(result['complete']);self.assertTrue(result['raw_byte_limit_exceeded']);self.assertEqual(result['raw_bytes'],257)
        self.assertEqual((self.root/'raw.log').stat().st_size,257);self.assertTrue(result['exact_child_reaped'])
    def test_partial_writes_are_retried_and_zero_progress_rejected(self):
        class Partial:
            def __init__(self):self.bytes=bytearray()
            def write(self,body):part=bytes(body[:2]);self.bytes.extend(part);return len(part)
        stream=Partial();D.write_all(stream,b'abcdef');self.assertEqual(stream.bytes,b'abcdef')
        with self.assertRaises(ValueError):D.write_all(SimpleNamespace(write=lambda body:0),b'x')
    def test_original_owner_closes_descendant_boundary(self):
        owner=D.module(D.OWNER,'toy_candidate_original_native_owner')
        result=owner.run_owned(self.command('pump-wrapper'),cwd=str(self.root),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
            logpath=self.root/'outer.log',timeout=10)
        self.assertEqual(result['native_exit'],0);self.assertFalse(result['timed_out'])
        inner=D.load(self.root/'inner-wait.json');self.assertEqual(inner['native_exit'],0);self.assertFalse(inner['complete']);self.assertTrue(inner['timed_out'])
        self.assertFalse(inner['reader_joined']);self.assertTrue(inner['reader_pipe_close_pending'])
        pid=D.load(self.root/'descendant-pid.json')['pid']
        def running():
            if os.name=='nt':
                k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint];k.OpenProcess.restype=ctypes.c_void_p
                k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)];k.CloseHandle.argtypes=[ctypes.c_void_p]
                handle=k.OpenProcess(0x1000,False,pid)
                if not handle:return False
                try:
                    code=ctypes.c_uint();self.assertTrue(k.GetExitCodeProcess(handle,ctypes.byref(code)));return code.value==259
                finally:k.CloseHandle(handle)
            try:os.kill(pid,0)
            except ProcessLookupError:return False
            statpath=Path('/proc')/str(pid)/'stat'
            if statpath.exists():return statpath.read_text().split(') ',1)[1][0]!='Z'
            state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,timeout=2).stdout.strip()
            return bool(state) and not state.startswith('Z')
        deadline=time.monotonic()+5
        while running() and time.monotonic()<deadline:time.sleep(.05)
        self.assertFalse(running())
    def test_post_native_runtime_publication_failure_stays_failed(self):
        root=self.root;request_path=root/'request.json';D.write(request_path,dict(mock_only=True))
        request=dict(operation='inspect-runtime',execution_directory=str(root),source_proof=dict(sha256='a'*64),qualification=None,
            qualification_authority=None,evidence_mode='generated',julia=dict(path=sys.executable,sha256=D.pin(sys.executable)),runtime=None)
        proof=dict(head='1'*40)
        original_write=D.write
        def native(argv,**kwargs):
            kwargs['on_launch'](123)
            julia_argv=D.julia_argv(request,request_path,'b'*64)
            runtime=dict(argv=julia_argv,executable_sha256=D.pin(sys.executable),julia_version='1.12.6',blas_threads=1,omp_threads=1,compiled_modules=False,pkgimages=False)
            (root/'julia.log').write_bytes(b'mocked raw is not actual Julia\n');(root/'supervisor.log').write_bytes(b'mocked supervisor\n')
            original_write(root/'JULIA_NATIVE_WAIT.json',dict(pid=456,native_exit=0,timed_out=False,complete=True,raw_byte_limit_exceeded=False,
                raw_sha256=D.pin(root/'julia.log'),raw_bytes=(root/'julia.log').stat().st_size,exact_child_reaped=True,
                reader_joined=True,reader_pipe_close_pending=False))
            original_write(root/'JULIA_STAGE_RESULT.json',dict(complete=True,request_sha256='b'*64,operation='inspect-runtime',runtime=runtime,job_sha256='c'*64))
            return dict(pid=123,native_exit=0,timed_out=False,ownership='mock-only',descendant_cleanup='mock-only',
                ownership_boundary_closed=True,leader_reaped=True,all_descendants_os_reaped=False)
        def failing_write(path,value):
            if Path(path).name=='RUNTIME.json':raise OSError('injected publication failure')
            return original_write(path,value)
        with mock.patch.object(D,'admit_request',return_value=(request,proof,8*1024**3,'mock',16*1024**3)),mock.patch.object(D,'input_controls',return_value={}),\
            mock.patch.object(D,'source_guard'),mock.patch.object(D,'module',return_value=SimpleNamespace(run_owned=native)),mock.patch.object(D,'write',side_effect=failing_write):
            self.assertEqual(D.main(request_path,'b'*64),1)
        receipt=D.load(root/'CANDIDATE_NATIVE_RECEIPT.json');self.assertFalse(receipt['complete']);self.assertEqual(receipt['failure']['type'],'OSError')
    def test_CLI_has_no_timeout_log_limit_or_script_override(self):
        source=PATH.read_text(encoding='utf8')
        self.assertNotIn("add_argument('--timeout'",source);self.assertNotIn("add_argument('--script'",source)
        self.assertEqual((D.RAM,D.DISK,D.SECONDS,D.RAW_LIMIT,D.QUEUE_CHUNKS),(8*1024**3,16*1024**3,7200,64*1024**2,8))
    def test_wrong_operation_rejected_before_launch(self):
        request=dict(operation='candidate-input',execution_directory=str(self.root))
        ref=dict(grant_path=str(self.root/'permit.json'))
        grant=dict(schema='candidate_input_only_authority_v1',operation='fixed_candidate_2017_validation_only',
            actual_source_access_authorized=True,actual_model_fit_authorized=False,new_case_generation_authorized=False,
            dataset_discovery=False,provider_receipts_available=False,execution_directory=str(self.root))
        with mock.patch.object(subprocess,'Popen') as start,self.assertRaisesRegex(ValueError,'stage operation/permission'):
            D.preflight_grant(request,ref,grant)
        start.assert_not_called()
    def test_fixed_runtime_and_binding_outputs_reject_before_owner_launch(self):
        self.metadata_bound_oracles()
        for filename in ('RUNTIME.json','selected-binding.json'):
            root=self.root/filename.replace('.','-');root.mkdir();(root/filename).write_bytes(b'occupied')
            request=dict(schema='candidate_native_stage_request_v1',evidence_mode='generated',operation='inspect-runtime',execution_directory=str(root),
                source_proof=dict(path=str(root/'absent-proof'),sha256='0'*64),qualification=None,qualification_authority=None,
                julia=dict(path=sys.executable,sha256=D.pin(sys.executable)),runtime=None,case=None,table_index=None,candidate_id=None,inputs={})
            path=root/'request.json';D.write(path,request)
            with mock.patch.object(subprocess,'Popen') as start,self.assertRaisesRegex(ValueError,'stage path already spent: '+filename):D.admit_request(path,D.pin(path))
            start.assert_not_called()
    def test_unknown_inputs_or_grid_rejected_before_closure(self):
        self.directory_bound_oracles()
        request=dict(operation='candidate-input',candidate_id='ridge-0.01',evidence_mode='generated',inputs=dict(authority={},arbitrary_path='forbidden'))
        with self.assertRaisesRegex(ValueError,'fixed stage input fields'):D.validate_headers(request)
        request['inputs']=dict(authority={});request['candidate_id']='arbitrary-model'
        with self.assertRaisesRegex(ValueError,'fixed candidate ID'):D.validate_headers(request)
    def test_new_owner_genuine_zero_and_precise_cleanup_claim(self):
        owner=D.module(D.OWNER,'candidate_precise_native_owner')
        result=owner.run_owned(self.command('emit'),cwd=str(self.root),environment=dict(os.environ),logpath=self.root/'new-owner.log',timeout=10)
        self.assertEqual(result['native_exit'],0);self.assertTrue(result['leader_reaped']);self.assertTrue(result['ownership_boundary_closed'])
        self.assertFalse(result['all_descendants_os_reaped'])
    def test_POSIX_normal_observe_signal_reap_order_without_native_claim(self):
        owner=D.module(D.OWNER,'candidate_mock_POSIX_owner');events=[]
        class Process:
            pid=123
            def poll(self):raise AssertionError('reaping poll forbidden before group signal')
            def wait(self,timeout):self_check.assertEqual(events[-1],'signal');events.append('reap');return 0
        self_check=self
        def waitid(kind,pid,flags):
            if pid==999:events.append('feature');raise ChildProcessError()
            events.append('observe');return SimpleNamespace(si_pid=123)
        fake_os=SimpleNamespace(name='posix',waitid=waitid,getpid=lambda:999,P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,
            killpg=lambda pid,sig:events.append('signal'))
        with mock.patch.object(owner,'os',fake_os),mock.patch.object(owner,'signal',SimpleNamespace(SIGKILL=9)),\
            mock.patch.object(owner.subprocess,'Popen',return_value=Process()):
            result=owner.run_owned(['fixed-mock'],cwd=str(self.root),environment={},logpath=self.root/'order.log',timeout=10)
        self.assertEqual(events,['feature','observe','signal','reap']);self.assertEqual(result['native_exit'],0)
        # Independent literal spy: an on-time exit can finish cleanup at the
        # deadline. The actual zero survives, but the whole stage is late.
        literal=D.module(ROOT/'test/initial_reference/posix_lifetime_oracles_v2.py','candidate_cleanup_deadline_oracle')
        for mode in ('cleanup_late_zero','direct_kill_error'):
            with self.subTest(mode=mode):
                root=self.root/mode;root.mkdir();literal.exercise(owner,mode,root)
    def test_POSIX_callback_exception_still_signals_before_reap(self):
        owner=D.module(D.OWNER,'candidate_mock_exception_owner');events=[]
        class Process:
            pid=123
            def wait(self,timeout):events.append('reap');return -9
        fake_os=SimpleNamespace(name='posix',waitid=lambda *a:(_ for _ in ()).throw(ChildProcessError()),getpid=lambda:999,
            P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,killpg=lambda *a:events.append('signal'))
        with mock.patch.object(owner,'os',fake_os),mock.patch.object(owner,'signal',SimpleNamespace(SIGKILL=9)),\
            mock.patch.object(owner.subprocess,'Popen',return_value=Process()),self.assertRaisesRegex(RuntimeError,'injected'):
            owner.run_owned(['fixed-mock'],cwd=str(self.root),environment={},logpath=self.root/'exception.log',timeout=10,
                on_launch=lambda pid:(_ for _ in ()).throw(RuntimeError('injected')))
        self.assertEqual(events,['signal','reap'])
    def test_missing_POSIX_feature_fails_before_launch(self):
        owner=D.module(D.OWNER,'candidate_mock_feature_gate')
        with mock.patch.object(owner,'os',SimpleNamespace(name='posix')),mock.patch.object(owner.subprocess,'Popen') as start,self.assertRaises(RuntimeError):
            owner.run_owned(['fixed-mock'],cwd=str(self.root),environment={},logpath=self.root/'never.log',timeout=10)
        start.assert_not_called();self.assertFalse((self.root/'never.log').exists())
    def test_late_zero_EOF_cannot_complete_without_native_claim(self):
        class Pipe:
            def fileno(self):return 123
            def close(self):pass
        process=SimpleNamespace(pid=123,stdout=Pipe(),returncode=0,poll=lambda:0,wait=lambda timeout:0)
        times=iter((0.,0.,1.,1.))
        fake_os=SimpleNamespace(name=os.name,read=lambda *a:b'')
        with mock.patch.object(D,'os',fake_os),mock.patch.object(D.subprocess,'Popen',return_value=process),\
            mock.patch.object(D.time,'monotonic',side_effect=lambda:next(times,1.)):
            result=D.pump_process(['fixed-mock'],str(self.root),{},self.root/'late.log',limit=128,timeout=.5)
        self.assertEqual(result['native_exit'],0);self.assertTrue(result['timed_out']);self.assertFalse(result['complete'])
    def test_raw_write_failure_retains_observed_native_wait(self):
        with mock.patch.object(D,'write_all',side_effect=OSError('injected raw write failure')):
            result=D.pump_process(self.command('emit'),str(self.root),dict(os.environ),self.root/'raw.log',limit=128,timeout=10)
        self.assertFalse(result['complete']);self.assertEqual(result['failure']['type'],'OSError')
        self.assertIs(type(result['native_exit']),int);self.assertTrue(result['exact_child_reaped'])
        self.assertEqual(result['raw_bytes'],(self.root/'raw.log').stat().st_size)
    def test_Windows_late_zero_is_timeout_without_rewriting_native_exit(self):
        owner=D.module(D.OWNER,'candidate_mock_Windows_deadline');spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
        previous=SimpleNamespace(run_owned=lambda *a,**k:dict(pid=123,native_exit=0,timed_out=False,wall_seconds=.02))
        with mock.patch.object(owner,'os',SimpleNamespace(name='nt')),\
            mock.patch.object(owner.importlib.util,'spec_from_file_location',return_value=spec),\
            mock.patch.object(owner.importlib.util,'module_from_spec',return_value=previous):
            result=owner.run_owned(['fixed-mock'],cwd=str(self.root),environment={},logpath=self.root/'not-used.log',timeout=.001)
        self.assertEqual(result['native_exit'],0);self.assertTrue(result['timed_out'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DriverControls))
    D.write(OUT/'RESULT.json',dict(complete=result.wasSuccessful(),methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        Julia_executed=False,actual_data_access=False,mocked_metadata_is_not_native_Julia_evidence=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
