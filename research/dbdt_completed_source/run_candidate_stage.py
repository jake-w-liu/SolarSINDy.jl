"""Fixed owned candidate stages; supplied grants only, no automatic grants/data discovery.

The original native_stage owns a fixed Python supervisor. Julia inherits that
private Job/session. Its actual wait status is retained separately from the
supervisor's actual wait status; neither is inferred from an output file.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,math,os,queue,re,shutil,stat,subprocess,sys,threading,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
ENTRY=Path(__file__).with_suffix('.jl')
OWNER=Path(__file__).with_name('candidate_native_owner.py')
OWNER_SHA='c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00'
LEGACY_OWNER=Path(__file__).with_name('native_stage.py')
LEGACY_OWNER_SHA='30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c'
RAM=8*1024**3;DISK=16*1024**3;SECONDS=7200;RAW_LIMIT=64*1024**2;CHUNK=65536;QUEUE_CHUNKS=8
OPERATIONS=('inspect-runtime','generated-chain','candidate-input','candidate-fit','candidate-validation','independent-verify','select-R2','selected-refit-input','selected-refit')
CANDIDATES=tuple('ridge-'+p for p in ('0.01','0.1','1','10','100'))+tuple(f'boost-{d}-{n}' for d in (2,3,4) for n in (50,100,200))+('forest-6-64','forest-9-64')
def need(ok,message):
    if not ok:raise ValueError(message)
def ordinary(path,directory=False):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute ordinary path required')
    for item in (p,*p.parents):
        s=item.lstat();need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,'linked path forbidden')
    need(p.is_dir() if directory else p.is_file(),'ordinary path kind differs');return p
def pin(path):
    with ordinary(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def write_all(stream,body):
    view=memoryview(body)
    while view:
        count=stream.write(view)
        need(type(count) is int and 0<count<=len(view),'short or failed raw write made no progress')
        view=view[count:]
def write(path,value):
    with Path(path).open('xb') as f:write_all(f,canonical(value))
def bounded_bytes(path,limit=1048576):
    p=ordinary(path);before=p.stat();need(before.st_size<=limit,'metadata byte bound')
    with p.open('rb') as f:raw=f.read(limit+1)
    after=p.stat();need(len(raw)<=limit and len(raw)==before.st_size==after.st_size and
        (before.st_dev,before.st_ino,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_mtime_ns),'metadata changed')
    return raw
def decode_metadata(raw):
    def pairs(values):
        out={}
        for k,v in values:need(k not in out,'duplicate metadata key');out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    def shape(v,depth=0):
        need(depth<=24,'metadata nesting bound')
        if isinstance(v,dict):
            need(len(v)<=8192 and all(isinstance(k,str) for k in v),'metadata object bound')
            for k,x in v.items():shape(k,depth+1);shape(x,depth+1)
        elif isinstance(v,list):
            need(len(v)<=8192,'metadata list bound')
            for x in v:shape(x,depth+1)
        elif isinstance(v,str):need(len(v.encode('utf8'))<=65536,'metadata text bound')
        elif isinstance(v,float):need(math.isfinite(v),'finite metadata number required')
        else:need(v is None or type(v) in (bool,int),'metadata atom type')
    shape(value);return value
def load(path,limit=1048576):return decode_metadata(bounded_bytes(path,limit))
def pinned_bytes(ref,limit=1048576):
    need(isinstance(ref,dict) and set(ref)=={'path','sha256'},'outside pin reference')
    need(isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'outside SHA mismatch')
    raw=bounded_bytes(ref['path'],limit)
    need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'outside SHA mismatch')
    return raw
def pinned(ref,limit=1048576):return decode_metadata(pinned_bytes(ref,limit))
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],stderr=subprocess.PIPE,timeout=120).decode().strip()
def source_guard(proof):
    need(set(proof)=={'complete','head','tree','source_files','source_sha256','commit_object_ids','all_clean_and_checkout_filters_match'},'seven-field source proof required')
    need(proof['complete'] is True and proof['all_clean_and_checkout_filters_match'] is True,'source proof incomplete')
    need(type(proof['source_files']) is int and 0<proof['source_files']<=8192 and len(proof['source_sha256'])==proof['source_files'],'source count')
    need(git('rev-parse','HEAD')==proof['head'] and git('rev-parse','HEAD^{tree}')==proof['tree'] and not git('status','--porcelain'),'clean current source required')
    names=git('ls-files','-z').split('\0');need({n for n in names if n}==set(proof['source_sha256']),'source fileset changed')
    objects={}
    for row in git('ls-tree','-r','-z','HEAD').split('\0'):
        if row:
            fields,name=row.split('\t',1);mode,kind,obj=fields.split();need(kind=='blob','ordinary Git blob required');objects[name]=obj
    need(objects==proof['commit_object_ids'],'source proof Git objects differ')
    for name,sha in proof['source_sha256'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and pin(ROOT/name)==sha,'source bytes changed: '+name)
    for path in (Path(__file__),ENTRY,OWNER,LEGACY_OWNER):need(proof['source_sha256'].get(path.relative_to(ROOT).as_posix())==pin(path),'native worker absent from proof')
    need(pin(OWNER)==OWNER_SHA and pin(LEGACY_OWNER)==LEGACY_OWNER_SHA,'versioned or preserved owner changed')
def expected_h(proof):
    fixed=['Julia executable found','Project.toml is present and nonempty','src/ contains Julia source','test/runtests.jl exists',
        'CODE_NOTES.md records implementation correctness context','TEST_REPORT.md records test coverage, tolerances, oracles, or anti-false-test checks',
        'tests contain @testset and test assertions','no obvious tautological tests found','randomness appears to use explicit seed/RNG',
        'no TODO/FIXME/HACK markers in source/tests','Pkg.test() passed','examples/experiments.jl executed']
    return fixed[:6]+['parse ok: '+n for n in sorted(proof['source_sha256']) if n.endswith('.jl')]+fixed[6:]
def qualification(ref,proof,proof_sha):
    q=pinned(ref);need(q['complete'] is True and q['head']==proof['head'] and q['source_proof']['sha256']==proof_sha and
        q['software_qualification_only'] is True and q['unfiltered_package_and_full_experiments_executed'] is True and
        q['strict_docs_both_complete'] is True and q['all_eight_latest_ci_jobs_passed'] is True and q['all_eight_raw_logs_retained'] is True,'current qualification required')
    e=q['candidate_pipeline_full18'];jobs=e['jobs'];expected={f'Julia {v} / {s}-latest' for v in ('1.12.6','1.13.1') for s in ('ubuntu','windows','macos')}|{
        'Documentation build','Julia 1.12 deployment images','Canonical development harness','Full-scale cross-runtime agreement'}|{
        f'Full-scale {v} / {r} / {l}' for v in ('1.12.6','1.13.1') for r in ('primary','independent') for l in ('ground-only','mixed')}
    need(e['complete'] is True and e['head']==proof['head'] and len(jobs)==18 and {j['name'] for j in jobs}==expected and len({j['job_id'] for j in jobs})==18,'all18 native jobs required')
    for j in jobs:
        need(type(j['native_exit']) is int and j['native_exit']==0 and j['timed_out'] is False and j['head']==proof['head'] and j['conclusion']=='success' and
            pin(j['raw_path'])==j['raw_sha256'],'native CI evidence differs')
    for key in ('both_actual_strict_docs_passed','unfiltered_Pkg_and_complete_examples_passed','eight_full_resources_verified','agreement48_verified'):need(e[key] is True,'required software gate missing')
    need(pin(e['harness_raw_path'])==e['harness_raw_sha256'],'H raw SHA differs')
    with ordinary(e['harness_raw_path']).open('rb') as f:raw=f.read(RAW_LIMIT+1)
    need(len(raw)<=RAW_LIMIT,'H raw bound');text=raw.decode('utf8');names=re.findall(r'^\s*\[PASS\] ([^\r\n]+)\r?$',text,re.M)
    need(names==expected_h(proof) and e['harness_counts']=={'PASS':len(names),'WARN':3,'FAIL':0} and '[FAIL] ' not in text,'complete named H checks required')
    need(hashlib.sha256(canonical(names)).hexdigest()==e['harness_named_checks_sha256'],'H name pin mismatch')
    return q
def grant_ref(value):
    fields={'grant_path','grant_sha256','source_proof_path','source_proof_sha256','qualification_path','qualification_sha256'}
    need(isinstance(value,dict) and set(value)==fields,'six-field supplied authority required')
    grant_raw=None
    for suffix,limit in (('grant',131072),('source_proof',4194304),('qualification',1048576)):
        raw=pinned_bytes(dict(path=value[suffix+'_path'],sha256=value[suffix+'_sha256']),limit)
        if suffix=='grant':grant_raw=raw
    return decode_metadata(grant_raw)
def preflight_grant(request,ref,grant):
    op=request['operation'];is_fit=op in ('candidate-fit','selected-refit')
    expected={
        'candidate-input':('candidate_input_only_authority_v1','fixed_candidate_2017_fit_input_only'),
        'candidate-fit':('archive_source_bound_point_fit_grant_v1','fixed_recipe_2017_point_fit_only'),
        'candidate-validation':('candidate_input_only_authority_v1','fixed_candidate_2017_validation_only'),
        'independent-verify':('candidate_input_only_authority_v1','fixed_candidate_2017_independent_read_only'),
        'select-R2':('candidate_input_only_authority_v1','fixed_candidate_2017_independent_read_only'),
        'selected-refit-input':('candidate_input_only_authority_v1','selected_R2_2017_refit_input_only'),
        'selected-refit':('producer_checked_selected_refit_grant_v1','selected_R2_2017_refit_only')}[op]
    need((grant['schema'],grant['operation'])==expected and grant['actual_source_access_authorized'] is True and
        grant['actual_model_fit_authorized'] is is_fit and grant['new_case_generation_authorized'] is False and
        grant['dataset_discovery'] is False and grant['provider_receipts_available'] is False,'stage operation/permission differs before launch')
    directory=ordinary(grant['execution_directory'],True);grant_parent=ordinary(Path(ref['grant_path']).parent,True)
    need(os.path.normcase(str(directory))==os.path.normcase(str(grant_parent)),'grant outside its exact execution directory')
    if op!='select-R2':need(os.path.normcase(str(directory))==os.path.normcase(request['execution_directory']),'stage grant execution directory differs')
    if is_fit:need(type(grant['expected_model_fits']) is int and grant['expected_model_fits']==1,'one supplied fit required')
def exact(value,fields):need(isinstance(value,dict) and set(value)==set(fields),'fixed stage input fields differ')
def validate_headers(r):
    op=r['operation'];x=r['inputs']
    schemas={
        'inspect-runtime':(),'generated-chain':(),'candidate-input':('authority',),'candidate-fit':('input','fit_authority'),
        'candidate-validation':('fit','authority','fit_authority','fit_native'),
        'independent-verify':('fit','validation','authority','fit_authority','fit_native','validation_authority','validation_native'),
        'select-R2':('candidates',),'selected-refit-input':('candidates','family','authority'),
        'selected-refit':('candidates','family','input','authority','fit_authority')}
    exact(x,schemas[op])
    if op in ('candidate-input','candidate-fit','candidate-validation','independent-verify'):need(r['candidate_id'] in CANDIDATES,'fixed candidate ID required')
    else:need(r['candidate_id'] is None,'unused candidate ID must be null')
    if 'family' in x:need(x['family'] in ('ridge','boost','forest'),'retained family required')
    authfields={prefix+suffix for prefix in ('grant','source_proof','qualification') for suffix in ('_path','_sha256')}
    def dependency(v):
        exact(v,('candidate_id','fit','fit_authority','fit_native','validation','validation_authority','validation_native','verification_authority','verification_native','witness'))
        fields(v)
    def fields(v):
        for key,value in v.items():
            if key in ('authority','fit_authority','validation_authority','verification_authority'):exact(value,authfields)
            elif key in ('fit_native','validation_native','verification_native'):exact(value,('receipt_path','receipt_sha256','raw_path','raw_sha256'))
            elif key in ('input','fit','validation','witness'):
                exact(value,('directory','expected','pins') if key=='fit' else ('directory','pins'))
                counts={'input':('descriptor_sha256','summary_sha256','receipt_sha256','manifest_sha256','complete_sha256'),
                    'fit':('context_sha256','recipe_sha256','model_sha256','receipt_sha256','complete_sha256','actual_fit_performed'),
                    'validation':('request_sha256','context_sha256','metric_sha256','receipt_sha256','manifest_sha256','complete_sha256'),
                    'witness':('manifest_sha256','complete_sha256')}
                exact(value['pins'],counts[key])
                if key=='fit':exact(value['expected'],('context_sha256','recipe_sha256','source_sha256','library_sha256'))
    fields(x)
    if 'candidates' in x:
        rows=x['candidates'];need(isinstance(rows,list) and len(rows)==16,'all16 completed candidates required')
        for id,row in zip(CANDIDATES,rows):dependency(row);need(row['candidate_id']==id,'fixed completed grid order required')
    if r['evidence_mode']=='actual_archival_cache':
        exact(r['case'],('request_path','directory','request_sha256','manifest_sha256','complete_sha256','producer','reader','admission_sha256'))
        exact(r['case']['reader'],authfields)
        exact(r['case']['producer'],{prefix+suffix for prefix in ('grant','plan','agreement','source_proof','qualification','floating_policy') for suffix in ('_path','_sha256')})
def input_controls(request):
    controls={};total=0
    def add(path):
        nonlocal total
        p=ordinary(path)
        if str(p) in controls:return
        total+=p.stat().st_size;need(total<=2*1024**3 and len(controls)<16384,'dependency closure bound')
        controls[str(p)]=pin(p)
    def walk(v):
        if isinstance(v,dict):
            if set(v)=={'path','sha256'}:add(v['path'])
            for k,x in v.items():
                if k.endswith('_path') and isinstance(x,str):add(x)
                elif k=='directory' and isinstance(x,str):
                    d=ordinary(x,True);members=[]
                    with os.scandir(d) as entries:
                        for entry in entries:
                            need(len(members)<1024,'artifact fileset bound')
                            members.append(d/entry.name)
                    for p in members:add(p)
                else:walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(request);return controls
def admit_request(path,sha,*,started=False):
    r=pinned(dict(path=str(path),sha256=sha))
    need(set(r)=={'schema','evidence_mode','operation','execution_directory','source_proof','qualification','qualification_authority','julia','runtime','case','table_index','candidate_id','inputs'},'fixed request fields')
    need(r['schema']=='candidate_native_stage_request_v1' and r['operation'] in OPERATIONS,'fixed operation required')
    actual=r['operation'] not in ('inspect-runtime','generated-chain')
    need(r['evidence_mode']==('actual_archival_cache' if actual else 'generated'),'stage evidence mode differs')
    validate_headers(r)
    root=ordinary(r['execution_directory'],True)
    need(not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'stage must be outside checkout')
    fresh=('JULIA_STAGE_RESULT.json','CANDIDATE_NATIVE_RECEIPT.json','JULIA_NATIVE_WAIT.json','RUNTIME.json','selected-binding.json','julia.log','output','independent-verification','generated',
        'CONTROLS_AFTER.json','CONTROL_READ_ERRORS.json','CANDIDATE_FIT_INPUT_ATTEMPT.json','CANDIDATE_POINT_FIT_ATTEMPT.json',
        'POINT_FIT_ATTEMPT.json','CANDIDATE_VALIDATION_ATTEMPT.json','CANDIDATE_INDEPENDENT_READ_ATTEMPT.json','SELECTED_R2_REFIT_ATTEMPT.json')
    if not started:fresh+=('NATIVE_STAGE_ATTEMPT.json','NATIVE_OWNER_LAUNCH.json','stage-argv.json','supervisor.log','CONTROLS_BEFORE.json')
    for name in fresh:
        need(not (root/name).exists() and not (root/name).is_symlink(),'stage path already spent: '+name)
    proof=pinned(r['source_proof'],4*1024**2);source_guard(proof)
    need(set(r['julia'])=={'path','sha256'} and pin(r['julia']['path'])==r['julia']['sha256'],'Julia executable outside pin')
    if actual:
        qualification(r['qualification'],proof,r['source_proof']['sha256']);g=grant_ref(r['qualification_authority'])
        need(g['head']==proof['head'] and g['source_proof_sha256']==r['source_proof']['sha256'] and g['qualification_sha256']==r['qualification']['sha256'],'stage grant qualification differs')
        need(r['runtime'] is not None and r['case'] is not None and type(r['table_index']) is int and 1<=r['table_index']<=16,'actual stage inputs missing')
        runtime=pinned(r['runtime'],131072);need(runtime['executable_sha256']==r['julia']['sha256'],'stage runtime executable differs')
        for key in ('authority','fit_authority'):
            if key in r['inputs']:grant_ref(r['inputs'][key])
        op=r['operation'];x=r['inputs']
        used=x['fit_authority'] if op=='candidate-fit' else x['candidates'][0]['verification_authority'] if op=='select-R2' else x['authority']
        need(r['qualification_authority']==used,'stage qualification authority must be actual operation permit')
        preflight_grant(r,r['qualification_authority'],g)
        if op=='selected-refit':
            inner=grant_ref(x['fit_authority']);temporary=dict(r,operation='candidate-fit',execution_directory=inner['execution_directory'])
            preflight_grant(temporary,x['fit_authority'],inner)
            need(g['inner_fit_grant_sha256']==x['fit_authority']['grant_sha256'] and not (Path(inner['execution_directory'])/'POINT_FIT_ATTEMPT.json').exists(),'selected inner fit already spent or differs')
    else:
        need(r['qualification'] is None and r['qualification_authority'] is None and r['case'] is None and r['inputs']=={} and r['table_index'] is None and r['candidate_id'] is None,'generated stage cannot accept caller data/authority')
    H=module(ROOT/'tools/run_canonical_harness.py','candidate_stage_physical_memory');memory,method=H.available_memory();disk=shutil.disk_usage(root).free
    need(type(memory) is int and memory>=RAM and disk>=DISK,'8GiB physical/16GiB disk admission required before any Julia launch')
    return r,proof,memory,method,disk
def pump_process(argv,cwd,environment,logpath,*,limit=RAW_LIMIT,timeout=SECONDS):
    """Bounded byte pump; test-only Python children exercise ownership mechanics."""
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    started=time.monotonic();q=queue.Queue(maxsize=QUEUE_CHUNKS);stop=threading.Event()
    process=subprocess.Popen(argv,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        bufsize=0,creationflags=flags)
    def reader():
        try:
            while not stop.is_set():
                block=os.read(process.stdout.fileno(),CHUNK)
                while not stop.is_set():
                    try:q.put(block,timeout=.1);break
                    except queue.Full:pass
                if not block:return
        except BaseException as error:
            while not stop.is_set():
                try:q.put(error,timeout=.1);return
                except queue.Full:pass
        finally:
            # Only this thread closes its descriptor. Closing from the owner
            # thread can deadlock with Windows CRT's blocking os.read lock.
            process.stdout.close()
    t=threading.Thread(target=reader,daemon=True);t.start();size=0;reason=None;eof=False;failure=None
    try:
        with Path(logpath).open('xb',buffering=0) as log:
            while not eof:
                if time.monotonic()-started>=timeout:reason='native_deadline';break
                try:block=q.get(timeout=.1)
                except queue.Empty:continue
                if isinstance(block,BaseException):raise block
                if not block:eof=True;break
                remaining=limit-size
                write_all(log,block[:remaining]);size+=min(len(block),remaining)
                if len(block)>remaining:reason='raw_byte_limit';break
        if reason is None and time.monotonic()-started>=timeout:reason='native_deadline'
        if reason is not None and process.poll() is None:process.kill()
        native=process.wait(timeout=20)
    except BaseException as error:
        if process.poll() is None:process.kill()
        native=process.wait(timeout=20)
        failure=dict(type=type(error).__name__,message=str(error)[:1024])
    finally:
        stop.set()
        if process.poll() is None:process.kill();process.wait(timeout=20)
        t.join(timeout=2)
    reader_joined=not t.is_alive()
    raw_size=Path(logpath).stat().st_size
    if raw_size!=size:
        failure=failure or dict(type='RawLengthMismatch',message='retained partial raw length differs from complete writes')
    return dict(pid=process.pid,native_exit=native,timed_out=reason=='native_deadline',raw_byte_limit_exceeded=reason=='raw_byte_limit',
        complete=reason is None and eof and failure is None and reader_joined,raw_bytes=raw_size,raw_sha256=pin(logpath),failure=failure,
        reader_joined=reader_joined,reader_pipe_close_pending=not reader_joined,
        complete_write_bytes=size,wall_seconds=time.monotonic()-started,exact_child_reaped=process.returncode is not None)
def supervisor(request_path,request_sha):
    r,proof,_,_,_=admit_request(request_path,request_sha,started=True);root=Path(r['execution_directory'])
    deadline=time.monotonic()+5
    while not (root/'NATIVE_OWNER_LAUNCH.json').exists():
        need(time.monotonic()<deadline,'owned launch binding missing');time.sleep(.01)
    launch=load(root/'NATIVE_OWNER_LAUNCH.json',16384);lease=load(root/'NATIVE_STAGE_ATTEMPT.json',16384)
    need(launch['pid']==os.getpid() and launch['owner_pid']==os.getppid() and launch['request_sha256']==request_sha and
        lease['head']==proof['head'] and lease['request_sha256']==request_sha,'owned supervisor identity differs')
    argv=load(root/'stage-argv.json',65536)
    need(argv==julia_argv(r,request_path,request_sha),'fixed Julia argv differs')
    env=dict(os.environ,JULIA_LOAD_PATH=os.pathsep.join(('@','@stdlib')),JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        JULIA_PKG_PRECOMPILE_AUTO='0',SOLARSINDY_NO_OPEN='1',PYTHONDONTWRITEBYTECODE='1')
    try:outcome=pump_process(argv,str(ROOT),env,root/'julia.log')
    except BaseException as error:
        outcome=dict(pid=None,native_exit=None,timed_out=False,complete=False,exact_child_reaped=False,
            raw_sha256=pin(root/'julia.log') if (root/'julia.log').is_file() else None,
            failure=dict(type=type(error).__name__,message=str(error)[:1024]))
    write(root/'JULIA_NATIVE_WAIT.json',outcome)
    if outcome.get('reader_joined') is False:
        # The durable negative wait record is closed above. This dedicated
        # owned child bypasses IO finalization locks; the outer genuine owner
        # closes its private Job/session, including inherited pipe writers.
        os._exit(1)
    return 0 if outcome['native_exit']==0 and outcome['complete'] and not outcome['timed_out'] else 1
def julia_argv(r,path,sha):
    return [r['julia']['path'],'--startup-file=no','--compiled-modules=no','--pkgimages=no','--threads=1','--project='+str(ROOT),str(ENTRY),
        '--request',str(path),'--request-sha256',sha,'--argv',str(Path(r['execution_directory'])/'stage-argv.json')]
def main(path,sha):
    r,proof,memory,method,disk=admit_request(path,sha);root=Path(r['execution_directory']);controls=input_controls(r)
    controls.update({str(p):pin(p) for p in (Path(__file__),ENTRY,OWNER,LEGACY_OWNER,Path(path),Path(sys.executable))})
    if r['qualification'] is not None:
        q=pinned(r['qualification']);e=q['candidate_pipeline_full18']
        controls.update({j['raw_path']:j['raw_sha256'] for j in e['jobs']});controls[e['harness_raw_path']]=e['harness_raw_sha256']
    argv=julia_argv(r,path,sha);write(root/'stage-argv.json',argv);controls[str(root/'stage-argv.json')]=pin(root/'stage-argv.json')
    need(len(canonical(controls))<=8*1024**2,'control manifest byte bound')
    write(root/'CONTROLS_BEFORE.json',controls)
    write(root/'NATIVE_STAGE_ATTEMPT.json',dict(schema='candidate_native_owner_attempt_v1',operation=r['operation'],request_sha256=sha,
        head=proof['head'],started_utc=datetime.now(timezone.utc).isoformat(),stage_seconds_limit=SECONDS))
    record=dict(schema='candidate_native_stage_evidence_v1',complete=False,operation=r['operation'],head=proof['head'],request_sha256=sha,
        source_proof_sha256=r['source_proof']['sha256'],qualification_sha256=r['qualification']['sha256'] if r['qualification'] else None,
        available_physical_bytes=memory,memory_method=method,free_disk_bytes=disk,stage_seconds_limit=SECONDS,raw_bytes_limit=RAW_LIMIT,
        driver_sha256=pin(__file__),entry_sha256=pin(ENTRY),runner_sha256=pin(OWNER),legacy_runner_sha256=pin(LEGACY_OWNER),argv=argv,argv_sha256=hashlib.sha256(canonical(argv)).hexdigest(),
        controls_before=dict(path=str(root/'CONTROLS_BEFORE.json'),sha256=pin(root/'CONTROLS_BEFORE.json')),
        actual_authority=r['evidence_mode']=='actual_archival_cache',scientific_qualification=False,
        request_path=str(path),native_exit=None,timed_out=None,ownership_boundary_closed=False,all_descendants_os_reaped=False,
        python_executable_path=sys.executable,python_executable_sha256=pin(sys.executable),python_version=sys.version)
    try:
        owner=module(OWNER,'candidate_stage_versioned_owner')
        owner_started=time.monotonic()
        outer=owner.run_owned([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--owned-supervisor','--request',str(path),'--request-sha256',sha],
            cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=root/'supervisor.log',timeout=SECONDS,
            on_launch=lambda pid:write(root/'NATIVE_OWNER_LAUNCH.json',dict(schema='candidate_owned_launch_v1',pid=pid,owner_pid=os.getpid(),request_sha256=sha)))
        record['supervisor_native']=outer;record['supervisor_raw_sha256']=pin(root/'supervisor.log')
        record['native_owner_wall_seconds']=time.monotonic()-owner_started
        launch=load(root/'NATIVE_OWNER_LAUNCH.json',16384)
        need(launch['pid']==outer['pid'] and launch['request_sha256']==sha,'owned outer launch mismatch')
        record['owned_launch']=launch
        inner=load(root/'JULIA_NATIVE_WAIT.json',16384);record['julia_native']=inner
        record.update(native_exit=inner['native_exit'],timed_out=inner['timed_out'] or outer['timed_out'],raw_sha256=inner['raw_sha256'],
            ownership=outer['ownership'],descendant_cleanup=outer['descendant_cleanup'],
            ownership_boundary_closed=outer.get('ownership_boundary_closed') is True and inner['exact_child_reaped'],
            julia_leader_reaped=inner['exact_child_reaped'],supervisor_leader_reaped=outer.get('leader_reaped') is True,
            descendant_cleanup_evidence=outer.get('descendant_evidence'),all_descendants_os_reaped=False)
        need(record['native_owner_wall_seconds']<=SECONDS and type(outer['native_exit']) is int and outer['native_exit']==0 and outer['timed_out'] is False and
            type(inner['native_exit']) is int and inner['native_exit']==0 and inner['complete'] is True and inner['raw_byte_limit_exceeded'] is False and
            inner['reader_joined'] is True and inner['reader_pipe_close_pending'] is False and
            outer.get('ownership_boundary_closed') is True and outer.get('leader_reaped') is True and inner['exact_child_reaped'] is True and
            type(inner['raw_bytes']) is int and inner['raw_bytes']==(root/'julia.log').stat().st_size<=RAW_LIMIT and
            pin(root/'julia.log')==inner['raw_sha256'],'native stage failed; raw and wait evidence retained')
        result=load(root/'JULIA_STAGE_RESULT.json');need(result['complete'] is True and result['request_sha256']==sha and result['operation']==r['operation'],'native output request binding')
        runtime=result['runtime'];need(runtime['argv']==argv and runtime['executable_sha256']==r['julia']['sha256'] and
            runtime['julia_version'] in ('1.12.6','1.13.1') and runtime['blas_threads']==runtime['omp_threads']==1 and
            runtime['compiled_modules'] is False and runtime['pkgimages'] is False,'native runtime binding')
        if r['runtime'] is not None:
            expected=pinned(r['runtime'],131072);need({k:v for k,v in runtime.items() if k!='argv'}=={k:v for k,v in expected.items() if k!='argv'},'native inspected runtime changed')
        record.update(runtime=runtime,job_sha256=result['job_sha256'],internal_result_sha256=pin(root/'JULIA_STAGE_RESULT.json'),
            outputs=result,complete=True)
        for name in ('point','output_context_sha256','output_witness_sha256','output_manifest_sha256','output_complete_sha256','input_descriptor_sha256'):
            if name in result:record[name]=result[name]
        q=r['qualification_authority']
        if q is not None:record['grant_sha256']=q['grant_sha256']
        if r['operation']=='inspect-runtime':write(root/'RUNTIME.json',runtime)
    except BaseException as error:
        record['complete']=False
        record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        after={};errors={}
        for p in controls:
            try:after[p]=pin(p)
            except BaseException as error:errors[p]=dict(type=type(error).__name__,message=str(error)[:128],message_may_be_truncated=True)
        write(root/'CONTROLS_AFTER.json',after);write(root/'CONTROL_READ_ERRORS.json',errors)
        changed=[p for p,s in controls.items() if after.get(p)!=s]
        record['controls_after']=dict(path=str(root/'CONTROLS_AFTER.json'),sha256=pin(root/'CONTROLS_AFTER.json'))
        record['control_read_errors']=dict(path=str(root/'CONTROL_READ_ERRORS.json'),sha256=pin(root/'CONTROL_READ_ERRORS.json'))
        record['control_read_error_count']=len(errors);record['changed_control_count']=len(changed)
        record['changed_controls_preview']=changed[:16]
        try:source_guard(proof)
        except BaseException as error:record['source_guard_failure']=str(error);record['complete']=False
        record['complete']=record['complete'] and not changed and not errors and 'failure' not in record;record['finished_utc']=datetime.now(timezone.utc).isoformat()
        if len(canonical(record))>262144:
            record.pop('outputs',None);record['complete']=False;record['failure']=dict(type='ReceiptByteBound',message='large internal result retained separately')
        write(root/'CANDIDATE_NATIVE_RECEIPT.json',record)
    return 0 if record['complete'] else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--owned-supervisor',action='store_true');args=parser.parse_args()
    need(not sys.flags.optimize and re.fullmatch('[0-9a-f]{64}',args.request_sha256),'unoptimized Python and SHA required')
    raise SystemExit(supervisor(args.request,args.request_sha256) if args.owned_supervisor else main(args.request,args.request_sha256))
