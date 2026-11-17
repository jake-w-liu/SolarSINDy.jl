"""Direct Julia owner for generated profiling, post-wait issuance and actual initial32.

Python handles file/authority headers and native ownership only. All source
admission, features, fitting, scalar replay, auditing and issuance run in Julia.
"""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.util,json,math,os,re,shutil,signal,stat,struct,sys,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OWNER=ROOT/'research/dbdt_completed_source/candidate_native_owner.py'
LEGACY=OWNER.with_name('native_stage.py')
OWNER_SHA='c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00'
LEGACY_SHA='30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c'
RAM=8*1024**3;DISK=16*1024**3;META=2*1024**2;RAW=64*1024**2
FIELDS={
 'profile':('schema','evidence_mode','operation','qualified_head','implementation_sources','source_proof','component_qualification','contracts','runtime','fixture_spec','limits','budgets','output_directory','owner_directory','request_id'),
 'issue_resource':('schema','operation','head','source_proof','root_qualification','component_qualification','fixture_spec','profile_result','native','raw','prelaunch_admission','measurement','limits','budgets','output_directory','request_id'),
 'actual_initial32':('schema','evidence_mode','operation','execution_authority','sources','source_proof','runtime','output_directory','owner_directory','request_id')}
HEADERS={'profile':('matched342_full_shape_profile_request_v1','initial32_full_shape_profile'),
 'issue_resource':('matched342_resource_issuance_request_v1','issue_initial32_full_shape_resource_qualification'),
 'actual_initial32':('matched342_initial_actual_request_v1','initial_matched342_32fits')}
ENTRIES={'profile':'run_matched342_full_shape_profile.jl','issue_resource':'issue_matched342_resource_qualification.jl','actual_initial32':'run_matched342_initial_actual.jl'}
LIMIT_FIELDS=('source_rows','model_member_bytes','total_model_member_bytes','required_available_bytes','required_disk_bytes','stage_wall_seconds','fit_wall_seconds')
RUNTIME_FIELDS=('julia_version','julia_binary_sha256','evotrees_version','evotrees_source_sha256','threads','blas_threads','omp_threads')
PERMIT_FIELDS=('schema','complete','evidence_mode','operation','qualified_head','implementation_sources','contracts','sources','source_proof','root_qualification','component_qualification','resource_qualification','runtime','execution','scope','limits','permit_id')
CONTRACTS=dict(protocol_sha256='7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404',
 feature_contract_sha256='2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683',
 feature_schema_sha256='109464af1630d16ac766b4f8791ad9f065a528d0aabac201a11fb68a4156249d',
 loss_contract_sha256='f24c0dd5b9ef4baf58f0595056d5b5a30ab21ff5c278126b5ebefaa9435bc73c',
 audit_contract_sha256='60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a',
 evotrees_source_sha256='f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f')
def need(ok,message):
    if not ok:raise ValueError(message)
def exact(value,names):need(type(value) is dict and set(value)==set(names),'closed fields differ')
def integer(value,low,high):need(type(value) is int and low<=value<=high,'integer bound differs');return value
def digest(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def absolute(path):
    need(isinstance(path,str) and Path(path).is_absolute() and path==os.path.normpath(path) and
        all(x not in ('.','..') for x in path.replace('\\','/').split('/')),'canonical absolute path required')
    return Path(path)
def ordinary(path,directory=False):
    p=absolute(str(path))
    for item in (p,*p.parents):
        s=item.lstat();need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,'path alias forbidden')
    need(p.is_dir() if directory else p.is_file(),'ordinary path kind differs')
    if not directory:need(p.stat().st_nlink==1,'hard-linked control forbidden')
    return p
def ref_grammar(ref):
    exact(ref,('path','sha256'));absolute(ref['path']);need(digest(ref['sha256']),'SHA grammar differs');return ref
def canonical_bytes(value):
    def convert(v):
        if type(v) is float:
            need(math.isfinite(v),'nonfinite metadata');return struct.pack('>d',v).hex()
        if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [convert(x) for x in v]
        need(v is None or type(v) in (str,int,bool),'unsupported canonical metadata');return v
    return (json.dumps(convert(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def identity(data,key):
    need(digest(data.get(key)) and hashlib.sha256(canonical_bytes({k:v for k,v in data.items() if k!=key})).hexdigest()==data[key],'canonical identity differs')
def identified(data,key):return dict(data,**{key:hashlib.sha256(canonical_bytes(data)).hexdigest()})
def pin(path,limit=536870912):
    p=ordinary(path);a=p.stat();need(0<=a.st_size<=limit,'file hash bound');h=hashlib.sha256();total=0
    with p.open('rb') as f:
        while block:=f.read(65536):total+=len(block);need(total<=limit,'file grew beyond bound');h.update(block)
    b=p.stat();need(total==a.st_size==b.st_size and (a.st_dev,a.st_ino,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_mtime_ns),'file changed while hashing')
    return h.hexdigest()
def read_json(path,limit=META):
    p=ordinary(path);a=p.stat();need(0<a.st_size<=limit,'metadata byte bound')
    with p.open('rb') as f:body=f.read(limit+1)
    b=p.stat();need(len(body)==a.st_size==b.st_size and (a.st_dev,a.st_ino,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_mtime_ns),'metadata changed')
    def pairs(rows):
        d={}
        for k,v in rows:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    data=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    def shape(v,depth=0):
        need(depth<=24,'metadata nesting bound')
        if isinstance(v,dict):
            need(len(v)<=8192 and all(isinstance(k,str) for k in v),'metadata object bound')
            for k,x in v.items():shape(k,depth+1);shape(x,depth+1)
        elif isinstance(v,list):
            need(len(v)<=8192,'metadata array bound')
            for x in v:shape(x,depth+1)
        elif isinstance(v,str):need(len(v.encode('utf8'))<=65536,'metadata text bound')
        elif type(v) is float:need(math.isfinite(v),'nonfinite JSON')
        else:need(v is None or type(v) in (int,bool),'metadata atom differs')
    shape(data);return data
def pinned(ref,limit=META,canonical=True):
    ref_grammar(ref);need(pin(ref['path'],limit)==ref['sha256'],'pinned file differs');d=read_json(ref['path'],limit)
    if canonical:need(hashlib.sha256(canonical_bytes(d)).hexdigest()==ref['sha256'],'canonical metadata required')
    return d
def write_json(path,data,native=False):
    p=absolute(str(path));ordinary(p.parent,True);need(not os.path.lexists(p),'exclusive output already exists')
    body=(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8') if native else canonical_bytes(data)
    need(len(body)<=META,'output metadata bound');fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_BINARY',0),0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False) as f:need(f.write(body)==len(body),'short metadata write');f.flush();os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=hashlib.sha256(body).hexdigest())
def validate_budgets(data):
    exact(data,('peak_rss_bytes','julia_gc_allocated_bytes','descriptor_growth'))
    integer(data['peak_rss_bytes'],1,RAM);integer(data['julia_gc_allocated_bytes'],1,2**63-1);integer(data['descriptor_growth'],0,2**63-1);return data
def validate_limits(d):
    exact(d,LIMIT_FIELDS)
    for k,n in (('source_rows',500000),('model_member_bytes',4*1024**2),('total_model_member_bytes',128*1024**2)):integer(d[k],n,n)
    integer(d['required_available_bytes'],RAM,2**63-1);integer(d['required_disk_bytes'],DISK,2**63-1)
    integer(d['stage_wall_seconds'],1,(2**63-1)//10**9);integer(d['fit_wall_seconds'],1,d['stage_wall_seconds']);return d
def validate_runtime(d):
    exact(d,RUNTIME_FIELDS);need(d['julia_version'] in ('1.12.6','1.13.1') and d['evotrees_version']=='0.18.7' and digest(d['julia_binary_sha256']) and d['evotrees_source_sha256']==CONTRACTS['evotrees_source_sha256'],'runtime identity differs')
    for k in ('threads','blas_threads','omp_threads'):integer(d[k],1,1)
    return d
def validate_request_metadata(operation,d):
    need(operation in FIELDS,'fixed operation required');exact(d,FIELDS[operation]);identity(d,'request_id')
    need((d['schema'],d['operation'])==HEADERS[operation],'request schema/operation differs')
    ref_grammar(d['source_proof']);absolute(d['output_directory'])
    if operation!='issue_resource':
        need(d['evidence_mode']==('synthetic_only' if operation=='profile' else 'archival_assumed_completion_research'),'request evidence mode differs')
        absolute(d['owner_directory']);validate_runtime(d['runtime'])
    if operation=='actual_initial32':
        ref_grammar(d['execution_authority']);need(type(d['sources']) is list and len(d['sources'])==2,'two outside sources required')
        for k,s in enumerate(d['sources']):
            exact(s,('delay_hours','source_proof','roots'));integer(s['delay_hours'],k,k);ref_grammar(s['source_proof'])
            need(type(s['roots']) is dict and 1<=len(s['roots'])<=32,'source roots map required')
            for label,path in s['roots'].items():need(isinstance(label,str) and re.fullmatch('[A-Za-z0-9_-]+',label),'root label differs');absolute(path)
    else:
        validate_limits(d['limits']);validate_budgets(d['budgets']);exact(d['fixture_spec'],('member','path','sha256'))
        need(d['fixture_spec']['member']=='tools/actual_b_chain/fixtures/full_shape342_spec_v1.json','fixed fixture spec required')
        ref_grammar({k:d['fixture_spec'][k] for k in ('path','sha256')})
        for key in (('component_qualification',) if operation=='profile' else ('root_qualification','component_qualification','profile_result','native','raw','prelaunch_admission','measurement')):ref_grammar(d[key])
        head=d['qualified_head'] if operation=='profile' else d['head'];need(isinstance(head,str) and re.fullmatch('[0-9a-f]{40}',head),'qualified head grammar differs')
        if operation=='profile':
            need(d['contracts']==CONTRACTS,'fixed component contracts differ');rows=d['implementation_sources']
            need(type(rows) is list and len(rows)==136,'fixed136 component source map required');names=[]
            for row in rows:
                exact(row,('name','sha256'));name=row['name']
                need(isinstance(name,str) and name.isascii() and len(name.encode('ascii'))<=256 and not name.startswith('/') and '\\' not in name and
                    all(x not in ('','.','..') for x in name.split('/')) and digest(row['sha256']),'component source member differs');names.append(name)
            need(names==sorted(names) and len(set(names))==136,'sorted unique component136 map required')
    return d
def attempt_marker_path(execution_directory,permit_sha256):
    p=absolute(execution_directory);need(digest(permit_sha256),'permit SHA required')
    return p.with_name(p.name+'.matched342-'+permit_sha256+'.attempt.json')
def create_attempt_marker(path,payload):return write_json(path,payload)
def write_bridge(owner_directory,request_sha256,prelaunch_data):
    root=ordinary(owner_directory,True);need(digest(request_sha256),'request SHA required')
    prelaunch=write_json(root/'PRELAUNCH.json',prelaunch_data)
    bridge=write_json(root/'ON_LAUNCH.json',dict(schema='matched342_native_on_launch_bridge_v1',request_sha256=request_sha256,prelaunch_admission=prelaunch))
    return bridge,prelaunch
def check_native_result(native,pid,deadline_ns,rawpath):
    base=('pid','native_exit','timed_out','leader_reaped','ownership','descendant_cleanup','ownership_boundary_closed','all_descendants_os_reaped','wall_seconds','descendant_evidence')
    need(type(native) is dict,'primitive native owner result required')
    if native.get('ownership')=='CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread':exact(native,base+('wrapper_observed_wall_seconds',))
    else:
        exact(native,base+('observation_method',));need(native.get('ownership')=='start_new_session=True','fixed native ownership method differs')
    integer(deadline_ns,1,2**63-1)
    need(type(native) is dict and type(native.get('pid')) is int and native.get('pid')==pid and type(pid) is int and pid>0 and type(native.get('native_exit')) is int and native['native_exit']==0 and
        native.get('timed_out') is False and native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and
        native.get('all_descendants_os_reaped') is False,'genuine bounded direct Julia result required')
    wall=native.get('wall_seconds');need(type(wall) in (int,float) and math.isfinite(wall) and wall>=0 and time.monotonic_ns()<deadline_ns,'native time differs')
    if 'wrapper_observed_wall_seconds' in native:need(type(native['wrapper_observed_wall_seconds']) in (int,float) and math.isfinite(native['wrapper_observed_wall_seconds']) and native['wrapper_observed_wall_seconds']>=0,'owner observed time differs')
    need(isinstance(native['descendant_cleanup'],str) and bool(native['descendant_cleanup']) and isinstance(native['descendant_evidence'],str) and bool(native['descendant_evidence']),'native boundary evidence missing')
    pin(rawpath,RAW);return native
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def source_preflight(ref):
    proof=pinned(ref,4*META,canonical=False);exact(proof,('complete','head','tree','source_files','source_sha256','commit_object_ids','all_clean_and_checkout_filters_match'))
    need(proof['complete'] is True and proof['all_clean_and_checkout_filters_match'] is True,'complete current whole proof required')
    count=integer(proof['source_files'],1,8192);need(type(proof['source_sha256']) is dict and len(proof['source_sha256'])==count and set(proof['commit_object_ids'])==set(proof['source_sha256']),'whole proof maps differ')
    for name,sha in proof['source_sha256'].items():
        need(isinstance(name,str) and not Path(name).is_absolute() and all(x not in ('','.','..') for x in name.split('/')) and digest(sha),'source member grammar differs')
        need(pin(ROOT/name)==sha,'whole source worktree changed: '+name)
    for p in (Path(__file__),OWNER,LEGACY,*[Path(__file__).with_name(x) for x in ENTRIES.values()]):
        need(proof['source_sha256'].get(p.relative_to(ROOT).as_posix())==pin(p),'runtime member absent from whole proof')
    need(pin(OWNER)==OWNER_SHA and pin(LEGACY)==LEGACY_SHA,'fixed native owner changed')
    # Julia performs the strict current clean Git/OID/checkout-filter proof before body work.
    return proof
def resource_admission(root,limits):
    helper=module(ROOT/'tools/run_canonical_harness.py','matched342_physical_memory')
    available,method=helper.available_memory();free=shutil.disk_usage(root).free
    integer(available,limits['required_available_bytes'],2**63-1);integer(free,limits['required_disk_bytes'],2**63-1)
    need(isinstance(method,str) and bool(method),'physical memory observation method missing');return available,method,free
def disjoint(a,b):
    x=os.path.normcase(os.path.abspath(a));y=os.path.normcase(os.path.abspath(b))
    try:return os.path.commonpath((x,y)) not in (x,y)
    except ValueError:return True  # Distinct Windows volumes have no common ancestor.
def fresh(path,forbidden):
    p=absolute(str(path));ordinary(p.parent,True);need(not os.path.lexists(p),'fresh directory required')
    for x in forbidden:need(disjoint(p,x),'output overlaps source/control/input')
    p.mkdir();return ordinary(p,True)
def check_component_header(ref,head,proof_sha,mode):
    q=pinned(ref,canonical=False);need(q.get('schema')=='b_reference_component_software_qualification_v1' and q.get('complete') is True and q.get('evidence_mode')==mode and
        q.get('head')==head and q.get('source_proof_sha256')==proof_sha,'current external component software qualification required');return q
def _run_owned_stage(owner,argv,**kwargs):
    if os.name!='posix':
        return owner.run_owned(argv,**kwargs)
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(_signal,_frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        raise InterruptedError('matched342 native owner received SIGTERM')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        return owner.run_owned(argv,**kwargs)
    finally:
        signal.signal(signal.SIGTERM,previous)
def main(argv=None):
    entered=time.monotonic_ns()
    a=argparse.ArgumentParser();a.add_argument('operation',choices=tuple(FIELDS));a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True)
    a.add_argument('--julia',required=True);a.add_argument('--owner-directory');args=a.parse_args(argv)
    need(not sys.flags.optimize and sys.version_info>=(3,11) and (sys.platform!='darwin' or sys.version_info>=(3,13)),'supported unoptimized Python required')
    request=dict(path=str(absolute(args.request)),sha256=args.request_sha256);d=pinned(request);validate_request_metadata(args.operation,d)
    proof=source_preflight(d['source_proof']);permit=None
    if args.operation=='actual_initial32':
        permit=pinned(d['execution_authority']);exact(permit,PERMIT_FIELDS);identity(permit,'permit_id')
        need(permit['schema']=='qualified_actual_initial_matched342_permit_v1' and permit['complete'] is True and permit['evidence_mode']==d['evidence_mode'] and
            permit['operation']==d['operation'] and permit['qualified_head']==proof['head'] and permit['source_proof']==d['source_proof'] and permit['runtime']==d['runtime'] and
            permit['execution']['directory']==d['output_directory'],'outside actual permit/request differs')
        limits=validate_limits(permit['limits']);runtime=validate_runtime(permit['runtime']);head=permit['qualified_head']
        rootq=pinned(permit['root_qualification'],canonical=False);need(rootq.get('complete') is True and rootq.get('head')==head and rootq.get('source_proof',{}).get('sha256')==d['source_proof']['sha256'],'external actual full source qualification required')
        check_component_header(permit['component_qualification'],head,d['source_proof']['sha256'],'software_qualified')
        rq=pinned(permit['resource_qualification']);need(rq.get('schema')=='matched342_initial_full_shape_resource_qualification_v1' and rq.get('complete') is True and rq.get('head')==head and rq.get('limits')==limits,'outside measured resource qualification required')
        spec=rq['fixture_spec'];owner_path=d['owner_directory']
    elif args.operation=='profile':
        limits=d['limits'];runtime=d['runtime'];head=d['qualified_head'];spec=d['fixture_spec'];owner_path=d['owner_directory']
        need(head==proof['head'],'profile source head differs');check_component_header(d['component_qualification'],head,d['source_proof']['sha256'],'synthetic_only')
    else:
        limits=d['limits'];head=d['head'];need(head==proof['head'],'issuer source head differs');spec=d['fixture_spec']
        result=pinned(d['profile_result']);runtime=validate_runtime(result['runtime']);need(result.get('head')==head and result.get('complete') is True,'complete generated profile required')
        need(args.owner_directory is not None,'issuer owner directory is an explicit external argument');owner_path=args.owner_directory
    if args.owner_directory is not None:need(str(absolute(args.owner_directory))==owner_path,'owner directory override differs')
    binary=ordinary(args.julia);need(pin(binary)==runtime['julia_binary_sha256'],'fixed Julia binary SHA differs')
    entry=Path(__file__).with_name(ENTRIES[args.operation]);need(proof['source_sha256'].get(entry.relative_to(ROOT).as_posix())==pin(entry),'Julia entry absent from proof')
    out=absolute(d['output_directory']);ordinary(out.parent,True);need(not os.path.lexists(out),'fresh scientific/issuance output required')
    root=fresh(owner_path,[ROOT,request['path'],d['source_proof']['path'],out])
    record=dict(schema='matched342_native_owner_terminal_v1',complete=False,operation=args.operation,request=request,head=head,source_qualified=False,scientific_qualified=False)
    raw=root/'julia.raw';native=None;start=entered;deadline=start+limits['stage_wall_seconds']*10**9;launched_pid=None;prelaunch=None
    try:
        need(time.monotonic_ns()<deadline,'source/header preflight exhausted owner stage budget')
        available,method,free=resource_admission(root,limits)
        if permit is not None:
            marker=attempt_marker_path(permit['execution']['directory'],d['execution_authority']['sha256'])
            create_attempt_marker(marker,identified(dict(schema='matched342_permanent_actual_attempt_v1',execution_directory=permit['execution']['directory'],
                permit_sha256=d['execution_authority']['sha256'],request_sha256=request['sha256'],owner_pid=os.getpid(),created_monotonic_ns=time.monotonic_ns()),'marker_id'))
        env=dict(os.environ)
        for k in tuple(env):
            if k.startswith('JULIA_') or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLAS_NUM_THREADS'):env.pop(k)
        env.update(JULIA_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLAS_NUM_THREADS='1')
        command=[str(binary),'--startup-file=no','--history-file=no','--threads=1','--project='+str(ROOT),str(entry),request['path'],request['sha256']]
        if args.operation!='issue_resource':command.append(str(root/'ON_LAUNCH.json'))
        # No environment, credential or full process-environment dump is persisted.
        write_json(root/'LAUNCH.json',dict(schema='matched342_direct_julia_launch_v1',argv=command,request=request,owner_pid=os.getpid(),
            owner_sha256=OWNER_SHA,preserved_owner_sha256=LEGACY_SHA,julia_binary_sha256=runtime['julia_binary_sha256'],source_proof=d['source_proof']))
        owner=module(OWNER,'matched342_fixed_native_owner');need(time.monotonic_ns()<deadline,'native launch budget exhausted')
        def on_launch(pid):
            nonlocal launched_pid,prelaunch
            launched_pid=integer(pid,1,2**63-1);need(pid!=os.getpid(),'Julia PID cannot be parent owner')
            if args.operation!='issue_resource':
                witness=identified(dict(schema='matched342_native_prelaunch_admission_v1',complete=True,evidence_mode=d['evidence_mode'],head=head,
                    source_proof_sha256=d['source_proof']['sha256'],fixture_spec_sha256=spec['sha256'],runtime_sha256=hashlib.sha256(canonical_bytes(dict(schema='matched342_native_runtime_v1',runtime=runtime))).hexdigest(),
                    owner_pid=os.getpid(),julia_pid=pid,execution_authority_sha256=None if permit is None else d['execution_authority']['sha256'],
                    available_physical_bytes=available,memory_method=method,free_disk_bytes=free,required_available_bytes=limits['required_available_bytes'],
                    required_disk_bytes=limits['required_disk_bytes'],native_owner_sha256=OWNER_SHA,available_observed_before_julia_start=True),'admission_id')
                _,prelaunch=write_bridge(root,request['sha256'],witness)
        native=_run_owned_stage(owner,command,cwd=str(ROOT),environment=env,logpath=str(raw),timeout=max(.001,(deadline-time.monotonic_ns())/1e9),on_launch=on_launch)
        end=time.monotonic_ns();receipt=dict(native=native,start_ns=start,end_ns=end,deadline_ns=deadline,worker_raw_sha256=pin(raw,RAW),individual_descendant_OS_reaping_observed=False)
        nref=write_json(root/'NATIVE.json',receipt,native=True)
        check_native_result(native,launched_pid,deadline,raw)
        need(native['wall_seconds']<limits['stage_wall_seconds'] and ('wrapper_observed_wall_seconds' not in native or native['wrapper_observed_wall_seconds']<limits['stage_wall_seconds']) and end<deadline,'native positive stage budget exceeded')
        source_preflight(d['source_proof']);need(pin(request['path'],META)==request['sha256'],'request changed after native wait')
        record.update(complete=True,native=nref,raw=dict(path=str(raw),sha256=receipt['worker_raw_sha256']),prelaunch_admission=prelaunch,
            actual_julia_pid=launched_pid,start_ns=start,end_ns=end,deadline_ns=deadline)
    except BaseException as error:
        record.update(error_type=type(error).__name__,error=str(error),actual_julia_pid=launched_pid,
            native_wait_result_observed=native is not None,observed_native=native)
        # The fixed native owner owns cleanup/reaping even on callback or wait errors.
        if raw.is_file():
            try:record['raw']=dict(path=str(raw),sha256=pin(raw,RAW))
            except BaseException as raw_error:
                record['raw_unqualified']=dict(path=str(raw),bytes=raw.stat().st_size,reason=str(raw_error),sha256_observed=False)
    finally:
        record['final_monotonic_ns']=time.monotonic_ns()
        if deadline is not None and record['final_monotonic_ns']>=deadline:record['complete']=False;record['late_output']=True
        write_json(root/'TERMINAL.json',record,native=True)
        if deadline is not None and time.monotonic_ns()>=deadline:
            # A post-write overrun must not leave a complete terminal for readers.
            (root/'TERMINAL.json').rename(root/'TERMINAL_LATE_UNQUALIFIED.json')
            write_json(root/'LATE_OUTPUT.json',dict(schema='matched342_native_late_output_v1',complete=False,deadline_ns=deadline,observed_ns=time.monotonic_ns()))
            record['complete']=False
    return 0 if record['complete'] else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
