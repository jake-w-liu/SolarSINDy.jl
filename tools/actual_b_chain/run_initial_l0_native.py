"""Fixed B1 native execution. External permits only; no source/data discovery."""
import os,sys,_thread

DIAGNOSTIC_LIMIT=8*1024**2
DIAGNOSTIC_CHUNK=65536

def _fd_write(fd,body):
    view=memoryview(body)
    while view:
        count=os.write(fd,view)
        if type(count) is not int or not 0<count<=len(view):raise OSError('raw write made no progress')
        view=view[count:]

def _diagnostic_record(root,state,complete):
    # Bootstrap failure evidence uses only fixed ASCII fields and integer counts.
    reason=state.get('reason') or 'eof'
    allowed=('eof','overflow','read_failure','write_failure','close_failure','reader_unjoined','writer_close_failure')
    if reason not in allowed:reason='read_failure'
    body=('{}\n'.format('{"schema":"b_supervisor_diagnostic_wait_v1","capture_complete":'+str(bool(complete)).lower()+
        ',"truncated":'+str(state.get('overflow_observed',False) or state['observed_bytes']>state['bytes']).lower()+',"overflow_observed":'+str(state.get('overflow_observed',False)).lower()+',"reason":"'+reason+'","retained_bytes":'+str(state['bytes'])+
        ',"observed_bytes":'+str(state['observed_bytes'])+',"live_limit_bytes":8388608}')).encode('ascii')
    fd=os.open(os.path.join(root,'SUPERVISOR_DIAGNOSTIC_WAIT.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_BINARY',0),0o600)
    try:_fd_write(fd,body);os.fsync(fd)
    finally:os.close(fd)

def start_diagnostics(root):
    """Install only in the fixed owned supervisor (or an isolated test child)."""
    if not os.path.isabs(root) or '..' in root.replace('\\','/').split('/'):raise ValueError('ordinary absolute diagnostic root required')
    for item in (root,os.path.dirname(root)):
        st=os.lstat(item)
        if st.st_mode&0o170000==0o120000 or getattr(st,'st_file_attributes',0)&0x400:raise ValueError('diagnostic root alias')
    raw=os.open(os.path.join(root,'supervisor.raw'),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_BINARY',0),0o600)
    readfd,writefd=os.pipe();os.set_inheritable(readfd,False);os.set_inheritable(writefd,False)
    done=_thread.allocate_lock();done.acquire()
    state=dict(root=root,bytes=0,observed_bytes=0,overflow_observed=False,reason=None,reader_joined=False,capture_complete=False,done=done)
    def reader():
        try:
            while True:
                try:block=os.read(readfd,DIAGNOSTIC_CHUNK)
                except BaseException:state['reason']='read_failure';break
                if not block:state['capture_complete']=True;break
                state['observed_bytes']+=len(block)
                state['overflow_observed']=state['overflow_observed'] or state['observed_bytes']>DIAGNOSTIC_LIMIT
                keep=block[:DIAGNOSTIC_LIMIT-state['bytes']]
                try:
                    view=memoryview(keep)
                    while view:
                        count=os.write(raw,view)
                        if type(count) is not int or not 0<count<=len(view):raise OSError('raw write made no progress')
                        state['bytes']+=count;view=view[count:]
                except BaseException:state['reason']='write_failure';break
                if len(block)>len(keep):state['reason']='overflow';break
        finally:
            failed=state['reason'] is not None
            try:
                try:os.fsync(raw)
                finally:os.close(raw)
            except BaseException:state['reason']='close_failure';failed=True
            try:os.close(readfd)
            except BaseException:state['reason']='close_failure';failed=True
            try:_diagnostic_record(root,state,state['capture_complete'] and not failed)
            except BaseException:state['reason']='write_failure';state['capture_complete']=False;failed=True
            finally:
                state['reader_joined']=True;done.release()
                if failed:os._exit(1)
    try:
        os.dup2(writefd,1);os.dup2(writefd,2);os.close(writefd)
        _thread.start_new_thread(reader,())
    except BaseException:
        for fd in (readfd,writefd,raw):
            try:os.close(fd)
            except OSError:pass
        raise
    return state

# Capture precedes general imports, request decoding and all application writes.
_BOOTSTRAP=None
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='--owned-supervisor':
    try:
        if len(sys.argv)!=10 or sys.argv[2]!='--execution-root':raise ValueError('fixed bootstrap argv')
        _BOOTSTRAP=start_diagnostics(sys.argv[3])
    except BaseException:
        try:os.write(2,b'fixed B1 diagnostic bootstrap failed\n')
        finally:os._exit(1)

from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,math,re,shutil,stat,subprocess,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
ENTRY=Path(__file__).with_suffix('.jl')
GATE=Path(__file__).with_name('BInitialNativePermitV2.jl')
UTILITY=ROOT/'research/dbdt_completed_source/run_candidate_stage.py'
OWNER=UTILITY.with_name('candidate_native_owner.py')
LEGACY=UTILITY.with_name('native_stage.py')
OWNER_SHA='3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1'
LEGACY_SHA='8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836'
UTILITY_SHA='6af1ab4748ac315fc511c938def27067fcadc85a7ea96adfdb0be5c36dfa4a1d'
RAM=8*1024**3;DISK=16*1024**3;SECONDS=7200;METADATA_SECONDS=900;RAW_LIMIT=64*1024**2
CONTRACT='f0fbbbd3bd7fea2fac983a232c3cbf7265af9524cfb5ad80e45d3ac2610a148b'
ACTUAL_SOURCE='593354e93e8f6c1486dc46de3ae96d4281d1129f9d5d39933347c2702122d248'
LIMITS=dict(chunk_rows=4096,fit_scalar_rows=1000000,hash_read_chunk_bytes=65536,horizon=6,json_depth=12,metadata_bytes=2097152,min_informative_joint_origins=100,ordinary_file_bytes=536870912,source_map_entries=2048,source_name_bytes=256,source_rows=500000,stage_bytes=17179869184,stage_directories=256,stage_files=2048)
REQUEST_FIELDS=('schema','evidence_mode','operation','execution_directory','intent','permit','source_proof','root_qualification','b_execution_qualification','julia','runtime')
INTENT_FIELDS=('schema','evidence_mode','stage','execution_directory','contract_sha256','implementation_sources','input','reference','runtime','programs','limits','scope')
SCOPE_FIELDS=('training_semantics','training_boundary','validation_start','validation_end','embargo_hours','delay_hours','beta_order','b_setting_indices','initial_hazard_fit_count','initial_gain_fit_count','state_schedule_count','beta_design_count','final_fit_count','calibration_fit_count','exposed_development_scoring','joint_primary_selection','scientific_verdict','serving_promotion')
PERMIT_FIELDS=('schema','complete','stage','qualified_head','implementation_sources','contract_sha256','training_semantics','source_sha256','input_prefix_sha256','input_view_id','input_envelope_sha256','admitted_input_seal','reference_completion_sha256','reference_audit_sha256','qualification_file','qualification_sha256','julia_version','julia_binary_sha256','native_owner_helper_sha256','training_boundary','validation_start','validation_end','embargo_hours','delay_hours','beta_order','b_setting_indices','initial_hazard_fit_count','initial_gain_fit_count','state_schedule_count','beta_design_count','final_fit_count','calibration_fit_count','exposed_development_scoring','joint_primary_selection','scientific_verdict','serving_promotion','limits','execution_directory','execution_intent_sha256','native_driver_sha256','native_entry_sha256','preserved_native_owner_sha256')
B_SOURCE_FILES=(
    'src/duration_point_training.jl','src/duration_point_values.jl','src/duration_point_contexts.jl','src/duration_point_coordinator.jl','src/duration_point_fit.jl',
    'src/duration_point_artifacts.jl','src/duration_point_codec.jl','src/duration_point_basis.jl','src/duration_point_fit_artifacts.jl','src/duration_point_bundle.jl','src/duration_point_schema.jl',
    'tools/actual_b_chain/ActualBChain.jl','tools/actual_b_chain/BInputs.jl','tools/actual_b_chain/BInitialPointFits.jl','tools/actual_b_chain/BInitialNativePermitV2.jl',
    'tools/actual_b_chain/run_initial_l0_native.py','tools/actual_b_chain/run_initial_l0_native.jl','research/dbdt_completed_source/candidate_native_owner.py',
    'research/dbdt_completed_source/native_stage.py','research/dbdt_completed_source/run_candidate_stage.py','tools/run_canonical_harness.py',
    'tools/actual_b_chain/BSelection.jl','tools/actual_b_chain/BReferenceAdmission.jl','src/SolarSINDy.jl','Project.toml','Manifest.toml',
    'src/duration_conditioned_forcing.jl','src/duration_current_state_stream.jl',
    'src/interval_mean_forecaster/DstIntervalObserver.jl','src/interval_mean_forecaster/DstIntervalReplay.jl','src/interval_mean_forecaster/sandu_fullchannel_features.jl',
    'src/interval_mean_forecaster/DstIntervalAdapter.jl','src/interval_mean_forecaster/DstIntervalStream.jl','src/interval_mean_forecaster/DstIntervalTraining.jl',
    'src/interval_mean_forecaster/DstIntervalArtifacts.jl','src/interval_mean_forecaster/DstIntervalExperiment.jl')
TIMING_FIELDS=('schema','request_sha256','intent_sha256','execution_directory','head','parent_pid','python_sha256','start_ns','deadline_ns','driver_sha256','entry_sha256','owner_sha256','legacy_owner_sha256')
FRESH=('B1_NATIVE_ATTEMPT.json','B1_OWNER_LAUNCH.json','B1_TIMING.json','b1-argv.json','CONTROLS_BEFORE.json','CONTROLS_AFTER.json','CONTROL_READ_ERRORS.json','supervisor-startup.log','supervisor.raw','SUPERVISOR_DIAGNOSTIC_WAIT.json','SUPERVISOR_FAILURE.json','julia.log','JULIA_NATIVE_WAIT.json','B1_JULIA_RESULT.json','B1_NATIVE_RECEIPT.json','output')


def need(ok,message):
    if not ok:raise ValueError(message)
def exact(value,names):need(isinstance(value,dict) and set(value)==set(names),'fixed field set differs')
def ordinary(path,directory=False):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute ordinary path required')
    for item in (p,*p.parents):
        s=item.lstat();need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,'alias path forbidden')
    need(p.is_dir() if directory else p.is_file(),'ordinary path kind')
    if not directory:need(p.stat().st_nlink==1,'hard-linked file forbidden')
    return p
def pin(path,max_bytes=536870912):
    p=ordinary(path);before=p.stat();need(max_bytes is None or before.st_size<=max_bytes,'hash byte bound')
    digest=hashlib.sha256();total=0
    with p.open('rb') as f:
        while True:
            block=f.read(65536)
            if not block:break
            total+=len(block);need(max_bytes is None or total<=max_bytes,'file grew beyond hash byte bound');digest.update(block)
    after=p.stat();need(total==before.st_size==after.st_size and (before.st_ino,before.st_dev,before.st_mtime_ns)==(after.st_ino,after.st_dev,after.st_mtime_ns),'file changed while hashing')
    return digest.hexdigest()
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def write_all(stream,body):
    view=memoryview(body)
    while view:
        count=stream.write(view);need(type(count) is int and 0<count<=len(view),'raw write made no progress');view=view[count:]
def write(path,value):
    body=canonical(value);need(len(body)<=2*1024**2,'metadata output byte bound')
    with Path(path).open('xb') as f:write_all(f,body);f.flush();os.fsync(f.fileno())
def _metadata(path,limit=2097152,canonical_required=False):
    p=ordinary(path);before=p.stat();need(before.st_size<=limit,'metadata byte bound')
    with p.open('rb') as f:raw=f.read(limit+1)
    after=p.stat();need(len(raw)<=limit and len(raw)==before.st_size==after.st_size and (before.st_ino,before.st_dev,before.st_mtime_ns)==(after.st_ino,after.st_dev,after.st_mtime_ns),'metadata changed')
    def pairs(rows):
        out={}
        for k,v in rows:need(k not in out,'duplicate metadata key');out[k]=v
        return out
    # Limit JSON structural depth before json.loads, including hostile failures.
    depth=0;quoted=False;escape=False
    for b in raw:
        if quoted:
            if escape:escape=False
            elif b==92:escape=True
            elif b==34:quoted=False
        elif b==34:quoted=True
        elif b in (123,91):depth+=1;need(depth<=13,'metadata nesting bound')
        elif b in (125,93):depth-=1;need(depth>=0,'metadata container grammar')
    need(not quoted and depth==0,'metadata grammar')
    v=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite metadata')))
    def shape(x,d=0):
        need(d<=12,'metadata nesting bound')
        if isinstance(x,dict):
            need(len(x)<=8192 and all(isinstance(k,str) for k in x),'metadata object bound')
            for k,y in x.items():shape(k,d+1);shape(y,d+1)
        elif isinstance(x,list):
            need(len(x)<=8192,'metadata list bound')
            for y in x:shape(y,d+1)
        elif isinstance(x,str):need(len(x.encode('utf8'))<=65536,'metadata string bound')
        elif isinstance(x,float):need(math.isfinite(x),'finite metadata required')
        else:need(x is None or type(x) in (int,bool),'metadata atom')
    shape(v)
    if canonical_required:need(canonical(v)==raw,'canonical metadata required')
    return v,hashlib.sha256(raw).hexdigest()
def load(path,limit=2097152,canonical_required=False):return _metadata(path,limit,canonical_required)[0]
def pinned(ref,limit=2097152):
    exact(ref,('path','sha256'));need(isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'outside pin grammar')
    value,digest=_metadata(ref['path'],limit);need(digest==ref['sha256'],'outside pin mismatch');return value
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def utility():
    need(pin(UTILITY)==UTILITY_SHA and pin(OWNER)==OWNER_SHA and pin(LEGACY)==LEGACY_SHA,'fixed owner/utility identity changed');return module(UTILITY,'b1_fixed_candidate_utilities')
def validate_timing(context,request_sha,execution_directory,parent_pid=None):
    exact(context,TIMING_FIELDS);need(context['schema']=='b_initial_l0_timing_v2' and context['request_sha256']==request_sha and context['execution_directory']==str(execution_directory),'timing context request/root differs')
    need(type(context['start_ns']) is int and type(context['deadline_ns']) is int and context['deadline_ns']-context['start_ns']==SECONDS*10**9,'fixed whole phase deadline')
    need(type(context['parent_pid']) is int and context['parent_pid']>0 and (parent_pid is None or context['parent_pid']==parent_pid),'timing parent differs')
    need(context['python_sha256']==pin(sys.executable) and context['driver_sha256']==pin(__file__) and context['entry_sha256']==pin(ENTRY) and context['owner_sha256']==OWNER_SHA and context['legacy_owner_sha256']==LEGACY_SHA,'timing fixed identity differs')
    remaining_seconds(context);return context
def remaining_seconds(context):
    now=time.monotonic_ns();need(type(context['start_ns']) is int and type(context['deadline_ns']) is int and now>=context['start_ns'],'invalid shared monotonic start')
    left=context['deadline_ns']-now;need(0<left<=SECONDS*10**9,'whole owned phase deadline expired');return left/10**9

def finish_diagnostics(state,context):
    try:sys.stdout.flush();sys.stderr.flush()
    except BaseException:state['reason']='writer_close_failure'
    for fd in (1,2):
        try:os.close(fd)
        except OSError:state['reason']='writer_close_failure'
    try:timeout=min(2.,remaining_seconds(context))
    except BaseException:timeout=0.
    joined=state['done'].acquire(timeout=timeout)
    if joined:state['done'].release()
    result=dict(reader_joined=joined,capture_complete=joined and state['capture_complete'] and state['reason'] is None,
        retained_bytes=state['bytes'],observed_bytes=state['observed_bytes'],overflow_observed=state['overflow_observed'],truncated=state['overflow_observed'] or state['observed_bytes']>state['bytes'],reason=state['reason'])
    if not joined:result['reason']='reader_unjoined'
    return result
def same_typed(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(same_typed(actual[k],expected[k]) for k in expected)
    if isinstance(expected,list):return len(actual)==len(expected) and all(same_typed(a,b) for a,b in zip(actual,expected))
    return actual==expected

def validate_runtime(value,julia_sha):
    exact(value,('julia_version','executable_sha256','architecture','argv','julia_project_sha256','julia_manifest_sha256','blas_vendor','blas_version','blas_library_sha256','blas_threads','omp_threads','rng_kind','rng_seed','compiled_modules','pkgimages'))
    need(type(value['blas_threads']) is int and type(value['omp_threads']) is int and value['blas_threads']==value['omp_threads']==1 and type(value['rng_seed']) is int and value['rng_seed']==0 and value['rng_kind']=='none','runtime integer/RNG grammar')
    need(value['executable_sha256']==julia_sha and value['julia_version'] in ('1.12.6','1.13.1') and value['compiled_modules'] is False and value['pkgimages'] is False,'fixed inspected runtime required')
    for k in ('executable_sha256','julia_project_sha256','julia_manifest_sha256','blas_library_sha256'):need(isinstance(value[k],str) and re.fullmatch('[0-9a-f]{64}',value[k]),'runtime SHA grammar')
    for k in ('julia_version','architecture','blas_vendor','blas_version','rng_kind'):need(isinstance(value[k],str) and 0<len(value[k].encode('utf8'))<=65536,'runtime text grammar')
    need(isinstance(value['argv'],list) and 0<len(value['argv'])<=64 and all(isinstance(x,str) and 0<len(x.encode('utf8'))<=8192 for x in value['argv']),'runtime argv bound')
    return value
def _same_path(a,b):return os.path.normcase(str(a))==os.path.normcase(str(b))
def _int(value,lo,hi):need(type(value) is int and lo<=value<=hi,'bounded integer required');return value
def _clock(year,month=1,day=1):return str(datetime(year,month,day).toordinal()*86400000)
def _sources(value,proof):
    need(isinstance(value,list) and 1<=len(value)<=2048,'B source map bound');out={}
    for row in value:
        exact(row,('name','sha256'));name=row['name'];need(isinstance(name,str) and name not in out and len(name)<=256 and name in proof['source_sha256'] and proof['source_sha256'][name]==row['sha256'],'B source map differs');out[name]=row['sha256']
    need(list(out)==sorted(out),'canonical B source order required')
    need(set(out)==set(B_SOURCE_FILES) and len(B_SOURCE_FILES)==len(set(B_SOURCE_FILES)),'exact transitive B implementation map required')
    for p in (Path(__file__),ENTRY,GATE,OWNER,LEGACY,UTILITY):need(out.get(p.relative_to(ROOT).as_posix())==pin(p),'B fixed closure missing')
    return out

def _scope(value):
    exact(value,SCOPE_FIELDS)
    fixed=dict(training_semantics='option_T_in_sample_two_stage_batch_v1',training_boundary=_clock(2020),validation_start=_clock(2020),validation_end=_clock(2023),
        embargo_hours=168,delay_hours=0,beta_order=[5,20],b_setting_indices=list(range(1,37)),initial_hazard_fit_count=2,initial_gain_fit_count=36,state_schedule_count=9,beta_design_count=18,final_fit_count=0,calibration_fit_count=0,
        exposed_development_scoring=False,joint_primary_selection=False,scientific_verdict=False,serving_promotion=False)
    need(canonical(value)==canonical(fixed),'fixed initial B1 scope differs')
    return fixed

def _roots(value,output):
    need(isinstance(value,dict) and 1<=len(value)<=32,'named input root bound');out={}
    for key,path in value.items():
        need(isinstance(key,str) and re.fullmatch('[A-Za-z0-9_-]+',key),'root key grammar');p=ordinary(path,True)
        need(not p.is_relative_to(ROOT) and not ROOT.is_relative_to(p) and not p.is_relative_to(output) and not output.is_relative_to(p),'input/output/repository overlap')
        for prior in out.values():need(not p.is_relative_to(prior) and not prior.is_relative_to(p),'input roots overlap')
        out[key]=p
    return out

def _member(roots,root,name):
    need(root in roots and isinstance(name,str) and len(name)<=256 and name.isascii() and not any(s in name for s in ('\\',':')) and all(x not in ('','..','.') for x in name.split('/')),'fixed relative member')
    p=ordinary(roots[root].joinpath(*name.split('/')))
    current=roots[root]
    for part in name.split('/'):
        need(part in os.listdir(current),'member name/case alias');current=current/part
    return p

SOURCE_V2_COMMON=('complete','evidence_mode','availability_kind','source_sha256','raw_prefix_sha256','prefix_start_ms','prefix_stop_ms','prefix_rows','channel_count','training_boundary_ms','validation_start_ms','validation_end_ms','receipt_delay_hours','implementation_sources')
PROVENANCE_V2_FIELDS=('schema',*SOURCE_V2_COMMON,'source_format_sha256','parser_source_sha256','raw_layout')
SOURCE_AUDIT_V2_FIELDS=('schema',*SOURCE_V2_COMMON,'raw_provenance_sha256')
SOURCE_PROOF_V2_FIELDS=('schema',*SOURCE_V2_COMMON,'raw_prefix_member','raw_prefix_bytes','raw_provenance_sha256','source_audit_sha256','limits','members')
JOIN_AUDIT_V2_FIELDS=('schema',*SOURCE_V2_COMMON,'source_admission_sha256','input_view_id','input_prefix_sha256','reference_kind','reference_completion_sha256','reference_audit_sha256','source_audit_sha256','raw_provenance_sha256')
ENVELOPE_V2_FIELDS=('schema','complete','evidence_mode','availability_kind','source_sha256','prefix_start_ms','prefix_stop_ms','prefix_rows','channel_count','raw_prefix_member','raw_prefix_bytes','raw_prefix_sha256','training_boundary_ms','validation_start_ms','validation_end_ms','receipt_delay_hours','independent_admission_completion_sha256','audit_completion_sha256','reference_kind','reference_completion_sha256','reference_audit_sha256','implementation_sources','limits','members','reference_join_audit_sha256')
SOURCE_V2_ROLES=('raw_prefix','raw_provenance','independent_audit')
JOIN_V2_ROLES=(*SOURCE_V2_ROLES,'source_admission','reference_completion','reference_audit','reference_join_audit')
SOURCE_FORMAT_V2='ab7cc12ea18d178ed3b529d603b867c9e948df419275a4cea56a8da08042feee'
SOURCE_PARSER_V2='src/interval_mean_forecaster/sandu_fullchannel_features.jl'
REFERENCE_COMMON_V2=('complete','evidence_mode','reference_kind','source_sha256','input_prefix_sha256','input_view_id','delay_hours','training_boundary','validation_start','validation_end','training_history_start','feature_contract_sha256','tuning_loss_contract_sha256','baseline_qualification_sha256','baseline_sources')
REFERENCE_COMPLETION_V2_FIELDS=('schema',*REFERENCE_COMMON_V2,'audit_contract_sha256','model_ids','selected_setting_ids','raw_row_identity_sha256','prediction_panel_sha256','grid_sha256','tuning_sha256','audit_sha256','members','completion_id')
REFERENCE_AUDIT_V2_FIELDS=('schema',*REFERENCE_COMMON_V2,'audit_contract_sha256','raw_row_identity_sha256','prediction_panel_sha256','grid_sha256','tuning_sha256','all_eight_settings_checked','full_per_lead_cohorts_checked','feature_receipt_history_checked','selected_prediction_columns_checked','initial_only','audit_id')

def admit_input_metadata_v2(input_spec,reference_spec,expected_sources,execution_root,expected_evidence_mode='archival_assumed_completion_research'):
    """Check the acyclic S/P/SA/J/E graph without reading raw or prediction bodies.

    This supplies no fitting/reference/qualification authority. Julia still
    reconstructs the raw view and performs the unchanged complete reference read.
    """
    need(expected_evidence_mode in ('synthetic_only','archival_assumed_completion_research'),'v2 expected evidence mode')
    need(isinstance(expected_sources,dict) and set(expected_sources)==set(B_SOURCE_FILES) and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in expected_sources.values()),'v2 exact expected source map')
    metadata_controls={}
    def hex64(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
    def metadata(ref):
        exact(ref,('path','sha256'));need(hex64(ref['sha256']),'v2 metadata pin grammar')
        value,digest=_metadata(ref['path'],canonical_required=True);need(digest==ref['sha256'],'v2 metadata pin differs')
        key=str(ordinary(ref['path']));need(key not in metadata_controls or metadata_controls[key]==digest,'v2 conflicting metadata pin')
        metadata_controls[key]=digest;return value
    def clock(value):
        need(isinstance(value,str) and re.fullmatch('0|-?[1-9][0-9]*',value),'v2 canonical clock')
        number=int(value);need(-(2**63)<=number<2**63 and number%3600000==0,'v2 exact-hour clock');return number
    def source_map(rows):
        need(isinstance(rows,list) and len(rows)==len(expected_sources),'v2 source map width')
        mapping={}
        for row in rows:
            exact(row,('name','sha256'));name=row['name']
            need(isinstance(name,str) and name not in mapping and name in expected_sources and hex64(row['sha256']) and expected_sources[name]==row['sha256'],'v2 source map differs')
            mapping[name]=row['sha256']
        need(list(mapping)==sorted(mapping),'v2 source map order')
    def common(value,base=None):
        need(value['complete'] is True and value['evidence_mode']==expected_evidence_mode and value['availability_kind']=='assumed_completed_hour','v2 source mode/completion')
        need(hex64(value['source_sha256']) and hex64(value['raw_prefix_sha256']),'v2 raw/source hash grammar')
        n=_int(value['prefix_rows'],1,500000);need(_int(value['channel_count'],17,17)==17,'v2 source channels');_int(value['receipt_delay_hours'],0,1)
        start,stop,boundary,first,last=[clock(value[k]) for k in ('prefix_start_ms','prefix_stop_ms','training_boundary_ms','validation_start_ms','validation_end_ms')]
        need(stop-start==n*3600000 and start<boundary<=first<last<=stop,'v2 source/split geometry')
        if expected_evidence_mode=='archival_assumed_completion_research':
            need(value['source_sha256']==ACTUAL_SOURCE and value['prefix_start_ms']==_clock(1976,11,1) and value['prefix_stop_ms']==value['validation_end_ms']==_clock(2023) and value['training_boundary_ms']==value['validation_start_ms']==_clock(2020),'v2 actual source/split')
        source_map(value['implementation_sources'])
        if base is not None:
            need(all(same_typed(value[k],base[k]) for k in SOURCE_V2_COMMON),'v2 repeated source facts')
    def members(rows,roots,roles):
        need(isinstance(rows,list) and len(rows)==len(roles),'v2 exact member count')
        found={};identities=set()
        for row in rows:
            exact(row,('root','path','bytes','sha256','role'));role=row['role']
            need(isinstance(row['root'],str) and isinstance(row['path'],str) and isinstance(role,str) and role in roles and role not in found and hex64(row['sha256']),'v2 member grammar/role')
            identity=(row['root']+'/'+row['path']).lower();need(identity not in identities,'v2 duplicate member');identities.add(identity)
            path=_member(roots,row['root'],row['path']);need(path.stat().st_size==_int(row['bytes'],1,536870912),'v2 member size')
            if role!='raw_prefix':
                need(row['bytes']<=2097152,'v2 metadata member bound')
            found[role]=dict(row,absolute=path)
        need(set(found)==set(roles),'v2 complete roles');return found
    exact(input_spec,('envelope','roots','independent_admission','expected_view_id','expected_admitted_seal'))
    exact(reference_spec,('completion_path','roots','pins'))
    roots=_roots(input_spec['roots'],Path(execution_root));refroots=_roots(reference_spec['roots'],Path(execution_root))
    env=metadata(input_spec['envelope']);exact(env,ENVELOPE_V2_FIELDS);need(env['schema']=='b_point_input_envelope_v2','v2 joined envelope required');common(env)
    need(same_typed(env['limits'],LIMITS),'v2 joined limits')
    source=metadata(input_spec['independent_admission']);exact(source,SOURCE_PROOF_V2_FIELDS)
    need(source['schema']=='independently_audited_b_source_prefix_v2','v2 source-only proof required');common(source,env)
    need(same_typed(source['limits'],LIMITS),'v2 source limits')
    source_members=members(source['members'],roots,SOURCE_V2_ROLES);joined=members(env['members'],roots,JOIN_V2_ROLES)
    for role,row in source_members.items():need(row==joined[role],'v2 joined upstream member differs')
    raw=source_members['raw_prefix'];p=source_members['raw_provenance'];sa=source_members['independent_audit'];sp=joined['source_admission'];j=joined['reference_join_audit']
    need(source['raw_prefix_member']==env['raw_prefix_member']==raw['root']+'/'+raw['path'] and _int(source['raw_prefix_bytes'],1,536870912)==raw['bytes']==env['raw_prefix_bytes']==136*source['prefix_rows'] and raw['sha256']==source['raw_prefix_sha256'],'v2 raw prefix geometry')
    need(str(sp['absolute'])==str(Path(input_spec['independent_admission']['path'])) and sp['sha256']==input_spec['independent_admission']['sha256']==env['independent_admission_completion_sha256'],'v2 source proof role')
    need(source['raw_provenance_sha256']==p['sha256'] and source['source_audit_sha256']==sa['sha256']==env['audit_completion_sha256'],'v2 upstream source pins')
    provenance=metadata(dict(path=str(p['absolute']),sha256=p['sha256']));exact(provenance,PROVENANCE_V2_FIELDS)
    source_audit=metadata(dict(path=str(sa['absolute']),sha256=sa['sha256']));exact(source_audit,SOURCE_AUDIT_V2_FIELDS)
    need(provenance['schema']=='b_raw_prefix_provenance_v2' and source_audit['schema']=='independently_audited_b_raw_source_v2','v2 upstream source schemas');common(provenance,source);common(source_audit,source)
    need(source_audit['raw_provenance_sha256']==p['sha256'] and hex64(provenance['source_format_sha256']) and hex64(provenance['parser_source_sha256']) and provenance['parser_source_sha256']==expected_sources.get(SOURCE_PARSER_V2) and provenance['raw_layout']=='source17_le_f64_column_major_v1','v2 parser/provenance facts')
    if expected_evidence_mode=='archival_assumed_completion_research':need(provenance['source_format_sha256']==SOURCE_FORMAT_V2,'v2 actual source format')
    join=metadata(dict(path=str(j['absolute']),sha256=j['sha256']));exact(join,JOIN_AUDIT_V2_FIELDS)
    need(join['schema']=='independently_audited_b_reference_join_v2' and env['reference_join_audit_sha256']==j['sha256'],'v2 independent join audit');common(join,source)
    links=dict(source_admission_sha256=sp['sha256'],source_audit_sha256=sa['sha256'],raw_provenance_sha256=p['sha256'],reference_kind=env['reference_kind'],reference_completion_sha256=joined['reference_completion']['sha256'],reference_audit_sha256=joined['reference_audit']['sha256'])
    need(all(same_typed(join[k],v) for k,v in links.items()),'v2 join upstream pins')
    need(env['reference_kind'] in ('original_reference_diagnostic_only','matched_information_gbm_initial_v1') and (env['reference_kind']!='original_reference_diagnostic_only' or env['receipt_delay_hours']==0),'v2 reference kind/delay')
    need(hex64(join['input_view_id']) and hex64(join['input_prefix_sha256']) and join['input_view_id']==input_spec['expected_view_id'] and hex64(input_spec['expected_admitted_seal']),'v2 expected stable view/seal')
    rp=metadata(reference_spec['pins']);exact(rp,('completion_sha256','audit_sha256','grid_sha256','tuning_sha256','prediction_sha256','feature_contract_sha256','tuning_loss_contract_sha256','baseline_qualification_sha256','baseline_sources','baseline_qualified_head','feature_contract_member','tuning_loss_contract_member'))
    r=joined['reference_completion'];ra=joined['reference_audit']
    need(str(r['absolute'])==str(Path(reference_spec['completion_path'])) and r['absolute'].name=='completion.json' and r['absolute'].parent in refroots.values() and ra['absolute']==r['absolute'].parent/'audit.json','v2 reference roots/paths')
    need(r['sha256']==rp['completion_sha256']==env['reference_completion_sha256'] and ra['sha256']==rp['audit_sha256']==env['reference_audit_sha256'],'v2 outside reference pins')
    completion=metadata(dict(path=str(r['absolute']),sha256=r['sha256']));exact(completion,REFERENCE_COMPLETION_V2_FIELDS)
    audit=metadata(dict(path=str(ra['absolute']),sha256=ra['sha256']));exact(audit,REFERENCE_AUDIT_V2_FIELDS)
    matched=env['reference_kind']=='matched_information_gbm_initial_v1'
    need(completion['schema']==('b_matched_reference_completion_v2' if matched else 'b_original_reference_completion_v1') and audit['schema']==('b_matched_reference_audit_v2' if matched else 'b_original_reference_audit_v1') and completion['complete'] is True and audit['complete'] is True,'v2 reference header schema/completion')
    for value,key in ((completion,'completion_id'),(audit,'audit_id')):
        need(hex64(value[key]) and hashlib.sha256(canonical({k:v for k,v in value.items() if k!=key})).hexdigest()==value[key],'v2 reference header identity')
    need(all(same_typed(completion[k],audit[k]) for k in REFERENCE_COMMON_V2),'v2 reference/audit repeated fields')
    header_links=dict(evidence_mode=expected_evidence_mode,reference_kind=env['reference_kind'],source_sha256=env['source_sha256'],input_view_id=join['input_view_id'],input_prefix_sha256=join['input_prefix_sha256'],training_boundary=env['training_boundary_ms'],validation_start=env['validation_start_ms'],validation_end=env['validation_end_ms'],training_history_start=env['prefix_start_ms'])
    need(all(same_typed(completion[k],v) for k,v in header_links.items()) and _int(completion['delay_hours'],0,1)==env['receipt_delay_hours'],'v2 reference header source/clock/view')
    for key in ('feature_contract_sha256','tuning_loss_contract_sha256','baseline_qualification_sha256','audit_contract_sha256','raw_row_identity_sha256','prediction_panel_sha256','grid_sha256','tuning_sha256','audit_sha256'):need(hex64(completion[key]),'v2 reference header hash grammar')
    need(all(same_typed(audit[k],completion[k]) for k in ('audit_contract_sha256','raw_row_identity_sha256','prediction_panel_sha256','grid_sha256','tuning_sha256')),'v2 repeated audit header hashes')
    need(all(audit[k] is True for k in ('full_per_lead_cohorts_checked','feature_receipt_history_checked','selected_prediction_columns_checked','initial_only')) and audit['all_eight_settings_checked'] is matched and completion['audit_sha256']==ra['sha256'],'v2 reference audit scope/header')
    need(isinstance(completion['model_ids'],list) and isinstance(completion['selected_setting_ids'],list) and len(completion['model_ids'])==len(completion['selected_setting_ids'])==2 and all(hex64(v) for v in completion['model_ids']),'v2 reference model IDs')
    for value in completion['selected_setting_ids']:_int(value,1,8 if matched else 10000)
    for key in ('completion_sha256','audit_sha256','grid_sha256','tuning_sha256','prediction_sha256','feature_contract_sha256','tuning_loss_contract_sha256','baseline_qualification_sha256'):need(hex64(rp[key]),'v2 outside reference pin grammar')
    need(isinstance(rp['baseline_qualified_head'],str) and re.fullmatch('[0-9a-f]{40}',rp['baseline_qualified_head']),'v2 baseline head grammar')
    rows=rp['baseline_sources'];need(isinstance(rows,list) and 1<=len(rows)<=2048,'v2 baseline source bound');baseline={}
    for row in rows:
        exact(row,('name','sha256'));name=row['name']
        need(isinstance(name,str) and name.isascii() and 0<len(name)<=256 and not name.startswith('/') and '\\' not in name and all(p not in ('','.','..') for p in name.split('/')) and name not in baseline and hex64(row['sha256']),'v2 baseline source grammar');baseline[name]=row['sha256']
    need(list(baseline)==sorted(baseline) and isinstance(rp['feature_contract_member'],str) and isinstance(rp['tuning_loss_contract_member'],str) and baseline.get(rp['feature_contract_member'])==rp['feature_contract_sha256'] and baseline.get(rp['tuning_loss_contract_member'])==rp['tuning_loss_contract_sha256'],'v2 outside baseline contract members')
    need(same_typed(completion['baseline_sources'],rows) and all(completion[k]==rp[k] for k in ('grid_sha256','tuning_sha256','feature_contract_sha256','tuning_loss_contract_sha256','baseline_qualification_sha256')) and completion['prediction_panel_sha256']==rp['prediction_sha256'],'v2 outside reference component headers')
    names=['audit.json','predictions.bin','qualification.json']
    if matched:names+=['grid.json','tuning.json']+[f'candidates/h{h}/setting-{s}.bin' for h in (4,6) for s in range(1,9)]
    names.sort();rows=completion['members'];need(isinstance(rows,list) and len(rows)==len(names),'v2 reference member ledger count')
    ledger={};total=r['bytes']
    for row,name in zip(rows,names):
        exact(row,('name','bytes','sha256'));need(row['name']==name and hex64(row['sha256']),'v2 reference member ledger')
        size=_int(row['bytes'],1,536870912)
        if name.endswith('.json'):need(size<=2097152,'v2 reference metadata bound')
        member=_member({'reference':r['absolute'].parent},'reference',name)
        need(member.stat().st_size==size,'v2 reference member stat/size');total+=size;need(total<=128*1024**2,'v2 reference total bound');ledger[name]=row
    pred=ledger['predictions.bin'];need(pred['bytes']>=32 and (pred['bytes']-16)%16==0 and 1<=(pred['bytes']-16)//16<=source['prefix_rows'],'v2 selected prediction header geometry')
    need(ledger['audit.json']['sha256']==ra['sha256'] and pred['sha256']==completion['prediction_panel_sha256'] and ledger['qualification.json']['sha256']==completion['baseline_qualification_sha256'],'v2 reference member/header pins')
    if matched:
        need(ledger['grid.json']['sha256']==completion['grid_sha256'] and ledger['tuning.json']['sha256']==completion['tuning_sha256'],'v2 grid/tuning member headers')
        count=(pred['bytes']-16)//16
        need(all(ledger[f'candidates/h{h}/setting-{s}.bin']['bytes']==16+8*count for h in (4,6) for s in range(1,9)),'v2 candidate header geometry')
    # Only the unchanged full Julia reader can establish numeric/reference authority.
    return dict(env=env,roots=roots,reference_roots=refroots,reference_pins=rp,source=source,provenance=provenance,source_audit=source_audit,join=join,metadata_controls=metadata_controls)

def resource_admission(root):
    H=module(ROOT/'tools/run_canonical_harness.py','b1_physical_memory');memory,method=H.available_memory();disk=shutil.disk_usage(root).free
    need(type(memory) is int and memory>=RAM and type(disk) is int and disk>=DISK,'8GiB available physical/16GiB disk required before Julia')
    return memory,method,disk

def admit_request(path,sha,started=False):
    began=time.monotonic();need(not sys.flags.optimize and sys.version_info>=(3,11) and (sys.platform!='darwin' or sys.version_info>=(3,13)),'supported unoptimized Python required')
    r,request_digest=_metadata(path,canonical_required=True);need(request_digest==sha,'request pin differs');exact(r,REQUEST_FIELDS)
    need(r['schema']=='b_initial_l0_native_request_v2' and r['operation']=='initial-l0' and r['evidence_mode']=='archival_assumed_completion_research','fixed actual B1 operation required')
    root=ordinary(r['execution_directory'],True);need(not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'execution outside checkout required')
    if not started:
        for name in FRESH:need(not os.path.lexists(root/name),'fixed path already spent: '+name)
    D=utility();proof=pinned(r['source_proof'],4*1024**2)
    need(isinstance(proof['source_sha256'],dict) and len(proof['source_sha256'])<=8192,'whole source map bound')
    for name in proof['source_sha256']:
        need(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts and ordinary(ROOT/name).stat().st_size<=536870912,'whole source byte bound')
    D.source_guard(proof)
    rootq=pinned(r['root_qualification'],1048576);evidence=rootq['candidate_pipeline_full18']
    need(isinstance(evidence['jobs'],list) and len(evidence['jobs'])==18,'full18 preflight cardinality')
    for row in evidence['jobs']:need(ordinary(row['raw_path']).stat().st_size<=RAW_LIMIT,'qualification raw byte bound')
    need(ordinary(evidence['harness_raw_path']).stat().st_size<=RAW_LIMIT,'H raw byte bound')
    for p in (Path(__file__),ENTRY,GATE):need(proof['source_sha256'].get(p.relative_to(ROOT).as_posix())==pin(p),'new B native file absent from source proof')
    D.qualification(r['root_qualification'],proof,r['source_proof']['sha256'])
    need(time.monotonic()-began<METADATA_SECONDS,'metadata preflight deadline')
    intent=pinned(r['intent']);exact(intent,INTENT_FIELDS)
    need(intent['schema']=='b_initial_l0_execution_intent_v2' and intent['stage']=='initial_l0' and intent['evidence_mode']==r['evidence_mode'] and intent['execution_directory']==str(root) and intent['contract_sha256']==CONTRACT,'B intent scope differs')
    implementation_sources=_sources(intent['implementation_sources'],proof);scope=_scope(intent['scope']);need(canonical(intent['limits'])==canonical(LIMITS),'fixed B limits differ')
    need(intent['runtime']==r['runtime'],'runtime reference differs');runtime=pinned(r['runtime'],131072)
    validate_runtime(runtime,r['julia']['sha256'])
    exact(r['julia'],('path','sha256'));need(pin(r['julia']['path'])==r['julia']['sha256'],'Julia executable pin differs')
    programs=intent['programs'];exact(programs,('driver','entry','versioned_owner','preserved_owner','utility_driver'))
    for name,p in (('driver',Path(__file__)),('entry',ENTRY),('versioned_owner',OWNER),('preserved_owner',LEGACY),('utility_driver',UTILITY)):
        exact(programs[name],('path','sha256'));need(_same_path(programs[name]['path'],p) and pin(p)==programs[name]['sha256'],'fixed program binding differs')
    permit=pinned(r['permit']);exact(permit,PERMIT_FIELDS)
    need(permit['schema']=='qualified_actual_initial_b_l0_owned_permit_v2' and permit['complete'] is True and permit['stage']=='initial_l0' and permit['contract_sha256']==CONTRACT and permit['qualified_head']==proof['head'],'v2 B permit/head required')
    for k,v in scope.items():need(canonical(permit[k])==canonical(v),'permit fixed scope differs: '+k)
    need(permit['implementation_sources']==intent['implementation_sources'] and same_typed(permit['limits'],intent['limits']) and permit['execution_directory']==str(root) and permit['execution_intent_sha256']==r['intent']['sha256'],'permit input intent differs')
    need(permit['native_owner_helper_sha256']==OWNER_SHA and permit['preserved_native_owner_sha256']==LEGACY_SHA and permit['native_driver_sha256']==pin(__file__) and permit['native_entry_sha256']==pin(ENTRY),'actual v2 native program identity differs')
    need(permit['qualification_file']==r['b_execution_qualification']['path'] and permit['qualification_sha256']==r['b_execution_qualification']['sha256'] and permit['julia_binary_sha256']==r['julia']['sha256'] and permit['julia_version']==runtime['julia_version'],'permit qualification/runtime differs')
    q=pinned(r['b_execution_qualification'])
    exact(q,('schema','complete','evidence_mode','head','source_sha256','harness','native_jobs','docs_and_containers','root_qualification_sha256','source_proof_sha256','canonical_harness_sha256','software_qualification_only','actual_panel_fit_or_outcome_qualified','serving_promotion'))
    exact(q['harness'],('counts','all17_independent_workers_complete','unfiltered_pkg_test_complete','experiments_entry_complete'))
    need(q['evidence_mode']=='software_qualified' and all(q['harness'][k] is True for k in ('all17_independent_workers_complete','unfiltered_pkg_test_complete','experiments_entry_complete')),'B full harness flags required')
    expected_native={f'Julia {v} / {s}-latest' for v in ('1.12.6','1.13.1') for s in ('ubuntu','windows','macos')}
    for cells,names in ((q['native_jobs'],expected_native),(q['docs_and_containers'],{'Documentation build','Julia 1.12 deployment images'})):
        need(isinstance(cells,list) and len(cells)==len(names) and {cell['name'] for cell in cells}==names,'B qualification cells differ')
        for cell in cells:
            exact(cell,('name','conclusion','receipt_sha256','log_sha256'));need(cell['conclusion']=='success' and all(isinstance(cell[k],str) and re.fullmatch('[0-9a-f]{64}',cell[k]) and cell[k]!='0'*64 for k in ('receipt_sha256','log_sha256')),'B native qualification identity differs')
    for counts in (q['harness']['counts'],pinned(r['root_qualification'])['candidate_pipeline_full18']['harness_counts']):
        exact(counts,('PASS','WARN','FAIL'));need(all(type(counts[k]) is int and counts[k]>=0 for k in counts),'qualification integer counter grammar')
    need(same_typed(q['harness']['counts'],evidence['harness_counts']),'B H raw count linkage differs')
    need(q['schema']=='b_point_execution_software_qualification_v1' and q['complete'] is True and q['head']==proof['head'] and q['source_sha256']=={row['name']:row['sha256'] for row in intent['implementation_sources']} and q['root_qualification_sha256']==r['root_qualification']['sha256'] and q['source_proof_sha256']==r['source_proof']['sha256'] and q['canonical_harness_sha256']==pinned(r['root_qualification'])['candidate_pipeline_full18']['harness_raw_sha256'] and q['software_qualification_only'] is True and q['actual_panel_fit_or_outcome_qualified'] is False and q['serving_promotion'] is False,'B execution wrapper/root linkage differs')
    x=intent['input'];ref=intent['reference']
    admitted=admit_input_metadata_v2(x,ref,implementation_sources,root,r['evidence_mode'])
    env=admitted['env'];roots=admitted['roots'];refroots=admitted['reference_roots'];rp=admitted['reference_pins'];n=env['prefix_rows']
    need(env['receipt_delay_hours']==0,'actual B1 remains L0 only')
    for a,b in (('training_boundary_ms','training_boundary'),('validation_start_ms','validation_start'),('validation_end_ms','validation_end')):need(env[a]==scope[b],'actual split differs')
    need(permit['source_sha256']==env['source_sha256'] and permit['input_prefix_sha256']==admitted['join']['input_prefix_sha256'] and permit['input_envelope_sha256']==x['envelope']['sha256'] and permit['input_view_id']==x['expected_view_id'] and permit['admitted_input_seal']==x['expected_admitted_seal'],'permit v2 joined input identity differs')
    need(rp['completion_sha256']==permit['reference_completion_sha256'] and rp['audit_sha256']==permit['reference_audit_sha256'],'permit reference pins differ')
    # Nothing below may decode observations; current native admission precedes body hashing.
    memory,method,disk=resource_admission(root)
    if os.name=='posix':module(OWNER,'b1_preflight_owner_features').posix_features()
    controls={};total=0
    def add(p,expected=None):
        nonlocal total
        p=ordinary(p);key=str(p)
        if key in controls:
            if expected is not None:need(controls[key]==expected,'duplicate control differs')
            return
        size=p.stat().st_size;total+=size;need(size<=536870912 and total<=2*1024**3 and len(controls)<16384,'control closure bound')
        h=pin(p);need(expected is None or h==expected,'control SHA differs');controls[key]=h
        need(time.monotonic()-began<METADATA_SECONDS,'metadata preflight deadline')
    for refvalue in (r['intent'],r['permit'],r['source_proof'],r['root_qualification'],r['b_execution_qualification'],r['julia'],r['runtime'],x['envelope'],x['independent_admission'],ref['pins']):add(refvalue['path'],refvalue['sha256'])
    for p in (Path(__file__),ENTRY,GATE,OWNER,LEGACY,UTILITY,Path(path),Path(sys.executable)):add(p)
    for metadata_path,metadata_sha in admitted['metadata_controls'].items():add(metadata_path,metadata_sha)
    for row in env['members']:add(_member(roots,row['root'],row['path']),row['sha256'])
    refdir=ordinary(Path(ref['completion_path']).parent,True);stack=[refdir];visited=0;refnames=set();refdirs=set();refbytes=0
    while stack:
        directory=stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                visited+=1;need(visited<=256,'reference tree enumeration bound')
                p=Path(entry.path);isdir=entry.is_dir(follow_symlinks=False);ordinary(p,isdir)
                relative=p.relative_to(refdir).as_posix()
                if isdir:refdirs.add(relative);stack.append(p)
                else:
                    refnames.add(relative);refbytes+=p.stat().st_size;need(refbytes<=128*1024**2,'reference128MiB bound');add(p)
    matched=env['reference_kind']=='matched_information_gbm_initial_v1'
    need(matched or env['reference_kind']=='original_reference_diagnostic_only','reference role differs')
    expected_names={'completion.json','audit.json','predictions.bin','qualification.json'}
    if matched:expected_names|={'grid.json','tuning.json'}|{f'candidates/h{h}/setting-{s}.bin' for h in (4,6) for s in range(1,9)}
    need(refnames==expected_names and refdirs==({'candidates','candidates/h4','candidates/h6'} if matched else set()),'exact reference member tree differs')
    e=pinned(r['root_qualification'])['candidate_pipeline_full18']
    for j in e['jobs']:add(j['raw_path'],j['raw_sha256'])
    add(e['harness_raw_path'],e['harness_raw_sha256']);need(time.monotonic()-began<METADATA_SECONDS,'metadata preflight deadline')
    return r,intent,proof,permit,controls,memory,method,disk

def validate_result(result,request_sha,output,actual=True):
    began=time.monotonic()
    need(result['schema']=='b_initial_l0_internal_result_v2' and result['complete'] is True and result['actual_source_access'] is actual and result['evidence_mode']==('archival_assumed_completion_research' if actual else 'synthetic_only'),'B result schema/mode differs')
    for k,v in dict(initial_hazard_fit_count=2,initial_gain_fit_count=36,state_schedule_count=9,beta_design_count=18,final_fit_count=0,calibration_fit_count=0,verified_model_count=36,verified_certificate_count=36,verified_hazard_fit_calls=0,verified_gain_fit_calls=0,verified_raw_replay_A_streams=9,output_file_count=423).items():need(type(result[k]) is int and result[k]==v,'B result fixed count differs: '+k)
    _int(result['output_directory_count'],0,256)
    need(result['selected_index'] is None and result['scientific_verdict'] is False and result['serving_promotion'] is False and _same_path(result['stage_directory'],output),'B result scope/output differs')
    if actual:need(result['request_sha256']==request_sha,'B result request differs')
    completion=load(Path(output)/'completion.json');_int(completion['output_directory_count'],0,256);_int(completion['output_bytes'],0,LIMITS['stage_bytes']);need(pin(Path(output)/'completion.json')==result['completion_sha256'],'B completion outside pin differs')
    need(completion['schema']=='source_bound_b_initial_l0_point_stage_v1' and completion['complete'] is True and completion['stage']=='initial_l0' and completion['evidence_mode']==result['evidence_mode'] and completion['output_sha256']==result['output_sha256'] and completion['output_file_count']==423 and type(completion['output_file_count']) is int and completion['output_directory_count']==result['output_directory_count'],'B completion metadata differs')
    if actual:
        need(completion['training_semantics']=='option_T_in_sample_two_stage_batch_v1' and completion['training_boundary']==_clock(2020) and completion['validation_start']==_clock(2020) and completion['validation_end']==_clock(2023) and type(completion['delay_hours']) is int and completion['delay_hours']==0 and type(completion['embargo_hours']) is int and completion['embargo_hours']==168 and completion['source_sha256']==ACTUAL_SOURCE,'B actual completion split/source differs')
    for k in SCOPE_FIELDS:
        if k in completion:
            expected={'exposed_development_scoring':False,'joint_primary_selection':False,'scientific_verdict':False,'serving_promotion':False}.get(k,result.get(k))
            if expected is not None:need(type(completion[k]) is type(expected) and completion[k]==expected,'B completion scope differs')
    need(completion['provisional_selection']['selected_index'] is None,'B1 cannot finalize primary selection')
    expected={'support.json','provisional-selection.json','basis/manifest.json'}|{f'basis/basis-{i}.bin' for i in range(1,19)}
    expected|={f'laws/beta-{b}/{n}' for b in (5,20) for n in ('model.json','MANIFEST.json','COMMITTED.json')}
    for i in range(1,37):
        expected|={f'fits/setting-{i}/{n}' for n in ('joint-fit.bin','fit-mask.ascii','manifest.json')}
        expected|={f'models/setting-{i}/{n}' for n in ('model.json','checkpoints.json','provenance.json','manifest.json','hazard/model.json','hazard/manifest.json')}
        expected|={f'points/setting-{i}.bin',f'gain-journal/setting-{i}.json'}
    files={};directories=set();total=0
    for parent,dirs,names in os.walk(ordinary(output,True),followlinks=False):
        ordinary(parent,True)
        for d in dirs:ordinary(Path(parent)/d,True);directories.add((Path(parent)/d).relative_to(output).as_posix())
        for name in names:
            need(time.monotonic()-began<METADATA_SECONDS,'result metadata deadline')
            p=ordinary(Path(parent)/name);relative=p.relative_to(output).as_posix()
            if relative=='completion.json':continue
            size=p.stat().st_size;need(size<=536870912,'B ordinary output byte bound');total+=size;need(total<=LIMITS['stage_bytes'] and len(files)<2048 and len(directories)<=256,'B output tree bound')
            files[relative]=dict(name=relative,bytes=size,sha256=pin(p))
    wantdirs={str(parent).replace('\\','/') for n in expected for parent in Path(n).parents if str(parent)!='.'}
    need(set(files)==expected and directories==wantdirs and len(directories)==result['output_directory_count'] and total==completion['output_bytes'],'B exact output tree differs')
    tree=[files[n] for n in sorted(files)];need(hashlib.sha256(canonical(tree)).hexdigest()==result['output_sha256'],'B output tree digest differs')
    return result
def julia_argv(r,path,sha):
    return [r['julia']['path'],'--startup-file=no','--compiled-modules=no','--pkgimages=no','--threads=1','--project='+str(ROOT),str(ENTRY),'--request',str(path),'--request-sha256',sha,'--argv',str(Path(r['execution_directory'])/'b1-argv.json')]

_SUPERVISOR_JULIA_ATTEMPTED=False

def supervisor(path,sha,timing_sha):
    global _SUPERVISOR_JULIA_ATTEMPTED
    root=Path(sys.argv[3]);context=load(root/'B1_TIMING.json',16384);need(pin(root/'B1_TIMING.json')==timing_sha,'timing outside pin differs')
    validate_timing(context,sha,root,parent_pid=os.getppid())
    while not (root/'B1_OWNER_LAUNCH.json').is_file():
        remaining_seconds(context);time.sleep(.01)
    launch=load(root/'B1_OWNER_LAUNCH.json',16384);attempt=load(root/'B1_NATIVE_ATTEMPT.json',16384)
    need(launch['pid']==os.getpid() and launch['owner_pid']==os.getppid() and launch['request_sha256']==sha and launch['timing_sha256']==timing_sha and attempt['request_sha256']==sha and attempt['timing_sha256']==timing_sha,'exact owned attempt/launch differs')
    # This validates existing parent markers; it never applies fresh checks to them.
    r,intent,proof,permit,controls,_,_,_=admit_request(path,sha,started=True)
    need(r['execution_directory']==str(root) and context['intent_sha256']==r['intent']['sha256'] and context['head']==proof['head'],'owned context intent/head differs')
    argv=load(root/'b1-argv.json',65536);need(argv==julia_argv(r,path,sha),'fixed B1 Julia argv differs')
    D=utility();env=dict(os.environ,JULIA_LOAD_PATH=os.pathsep.join(('@','@stdlib')),JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JULIA_PKG_PRECOMPILE_AUTO='0',SOLARSINDY_NO_OPEN='1',PYTHONDONTWRITEBYTECODE='1',
        SOLARSINDY_B_V2_NATIVE_OWNER_SHA256=OWNER_SHA,SOLARSINDY_B_V2_REQUEST_SHA256=sha,SOLARSINDY_B_V2_REQUEST_PATH=str(path),SOLARSINDY_B_V2_SUPERVISOR_PID=str(os.getpid()),SOLARSINDY_B_V2_TIMING_SHA256=timing_sha)
    outcome=None
    try:
        resource_admission(root)
        left=remaining_seconds(context);_SUPERVISOR_JULIA_ATTEMPTED=True
        outcome=D.pump_process(argv,str(ROOT),env,root/'julia.log',limit=RAW_LIMIT,timeout=left)
        write(root/'JULIA_NATIVE_WAIT.json',outcome)
    finally:
        if outcome is not None and outcome.get('reader_joined') is False:os._exit(1)
    remaining_seconds(context)
    return 0 if outcome['native_exit']==0 and outcome['complete'] is True and outcome['timed_out'] is False else 1

def _supervisor_main():
    context=None;status=1
    try:
        need(_BOOTSTRAP is not None and len(sys.argv)==10 and sys.argv[4]=='--request' and sys.argv[6]=='--request-sha256' and sys.argv[8]=='--timing-sha256','fixed supervisor argv')
        context=load(Path(sys.argv[3])/'B1_TIMING.json',16384)
        status=supervisor(sys.argv[5],sys.argv[7],sys.argv[9])
    except BaseException as error:
        try:write(Path(sys.argv[3])/'SUPERVISOR_FAILURE.json',dict(schema='b_supervisor_failure_v1',complete=False,error_type=type(error).__name__,message=str(error)[:1024],julia_launch_attempted=_SUPERVISOR_JULIA_ATTEMPTED,julia_launch_count_known=not _SUPERVISOR_JULIA_ATTEMPTED,observed_julia_launches=0))
        except BaseException:pass
    finally:
        try:
            if context is None:context=dict(start_ns=time.monotonic_ns(),deadline_ns=time.monotonic_ns())
            diagnostic=finish_diagnostics(_BOOTSTRAP,context)
            if not diagnostic['capture_complete']:status=1
            remaining_seconds(context)
        except BaseException:status=1
        # Only this fixed owned child uses _exit, after explicit evidence closes.
        os._exit(status)

def main(path,sha):
    preflight_began=time.monotonic_ns()
    r,intent,proof,permit,controls,memory,method,disk=admit_request(path,sha)
    root=Path(r['execution_directory']);D=utility();argv=julia_argv(r,path,sha)
    write(root/'b1-argv.json',argv);controls[str(root/'b1-argv.json')]=pin(root/'b1-argv.json')
    write(root/'CONTROLS_BEFORE.json',controls)
    controls[str(root/'CONTROLS_BEFORE.json')]=pin(root/'CONTROLS_BEFORE.json')
    record=dict(schema='b_initial_l0_native_evidence_v2',complete=False,operation='initial-l0',evidence_mode=r['evidence_mode'],head=proof['head'],request_sha256=sha,intent_sha256=r['intent']['sha256'],permit_sha256=r['permit']['sha256'],
        source_proof_sha256=r['source_proof']['sha256'],root_qualification_sha256=r['root_qualification']['sha256'],available_physical_bytes=memory,memory_method=method,free_disk_bytes=disk,
        whole_owned_phase_limit_seconds=SECONDS,metadata_limit_seconds=METADATA_SECONDS,julia_raw_limit_bytes=RAW_LIMIT,supervisor_diagnostic_live_limit_bytes=DIAGNOSTIC_LIMIT,startup_transport_positive_bound_bytes=DIAGNOSTIC_LIMIT,startup_transport_has_live_quota=False,
        native_exit=None,supervisor_native_exit=None,owner_invocations=0,observed_owner_launches=0,owner_launch_count_known=True,julia_launch_count_known=False,observed_julia_launches=0,
        ownership_boundary_closed=False,all_descendants_os_reaped=False,actual_source_access_authorized=True,scientific_qualification=False,serving_promotion=False)
    start=time.monotonic_ns();preflight_elapsed=start-preflight_began;need(0<=preflight_elapsed<METADATA_SECONDS*10**9,'parent metadata preflight deadline')
    record['metadata_preflight_elapsed_ns']=preflight_elapsed
    context=dict(schema='b_initial_l0_timing_v2',request_sha256=sha,intent_sha256=r['intent']['sha256'],execution_directory=str(root),head=proof['head'],parent_pid=os.getpid(),python_sha256=pin(sys.executable),start_ns=start,deadline_ns=start+SECONDS*10**9,
        driver_sha256=pin(__file__),entry_sha256=pin(ENTRY),owner_sha256=OWNER_SHA,legacy_owner_sha256=LEGACY_SHA)
    write(root/'B1_TIMING.json',context);timing_sha=pin(root/'B1_TIMING.json');controls[str(root/'B1_TIMING.json')]=timing_sha
    write(root/'B1_NATIVE_ATTEMPT.json',dict(schema='b_initial_l0_native_attempt_v2',request_sha256=sha,intent_sha256=r['intent']['sha256'],permit_sha256=r['permit']['sha256'],timing_sha256=timing_sha,head=proof['head'],parent_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()))
    post_started=None
    def launched(pid):
        record['observed_owner_launches']=1;record['owner_launch_count_known']=True;record['observed_owner_pid']=pid
        write(root/'B1_OWNER_LAUNCH.json',dict(schema='b_initial_l0_owner_launch_v2',pid=pid,owner_pid=os.getpid(),request_sha256=sha,timing_sha256=timing_sha))
    try:
        validate_timing(context,sha,root,parent_pid=os.getpid())
        owner=module(OWNER,'b1_fixed_versioned_owner');record['owner_invocations']=1;record['owner_launch_count_known']=False
        outer=owner.run_owned([sys.executable,'-I','-S','-u','-B','-X','utf8',str(Path(__file__)),'--owned-supervisor','--execution-root',str(root),'--request',str(path),'--request-sha256',sha,'--timing-sha256',timing_sha],
            cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=root/'supervisor-startup.log',timeout=remaining_seconds(context),on_launch=launched)
        end=time.monotonic_ns();post_started=time.monotonic();record.update(supervisor_native=outer,supervisor_native_exit=outer['native_exit'],owned_phase_end_ns=end,owned_phase_elapsed_ns=end-start,whole_owned_phase_timed_out=end>=context['deadline_ns'] or outer['timed_out'],ownership_boundary_closed=outer.get('ownership_boundary_closed') is True,
            supervisor_leader_reaped=outer.get('leader_reaped') is True,descendant_cleanup_evidence=outer.get('descendant_evidence'))
        record['startup_transport_bytes']=(root/'supervisor-startup.log').stat().st_size;record['startup_transport_sha256']=pin(root/'supervisor-startup.log',max_bytes=None)
        if (root/'JULIA_NATIVE_WAIT.json').is_file():
            inner=load(root/'JULIA_NATIVE_WAIT.json',16384);record['julia_native']=inner;record['native_exit']=inner['native_exit'];record['observed_julia_launches']=1 if type(inner.get('pid')) is int else 0;record['julia_launch_count_known']=type(inner.get('pid')) is int
        else:
            inner=None
            if (root/'SUPERVISOR_FAILURE.json').is_file():
                failed=load(root/'SUPERVISOR_FAILURE.json',16384)
                if failed.get('julia_launch_count_known') is True and failed.get('julia_launch_attempted') is False and failed.get('observed_julia_launches')==0:
                    record['julia_launch_count_known']=True;record['observed_julia_launches']=0
        diagnostic=load(root/'SUPERVISOR_DIAGNOSTIC_WAIT.json',16384);exact(diagnostic,('schema','capture_complete','truncated','overflow_observed','reason','retained_bytes','observed_bytes','live_limit_bytes'));record['supervisor_diagnostic']=diagnostic
        need(diagnostic['schema']=='b_supervisor_diagnostic_wait_v1' and diagnostic['capture_complete'] is True and diagnostic['truncated'] is False and diagnostic['overflow_observed'] is False and diagnostic['reason']=='eof' and type(diagnostic['live_limit_bytes']) is int and diagnostic['live_limit_bytes']==DIAGNOSTIC_LIMIT and type(diagnostic['retained_bytes']) is int and diagnostic['retained_bytes']==(root/'supervisor.raw').stat().st_size<=DIAGNOSTIC_LIMIT and type(diagnostic['observed_bytes']) is int and diagnostic['observed_bytes']==diagnostic['retained_bytes'],'supervisor diagnostic capture incomplete')
        record['supervisor_raw_sha256']=pin(root/'supervisor.raw')
        need(record['startup_transport_bytes']<=DIAGNOSTIC_LIMIT and type(outer['native_exit']) is int and outer['native_exit']==0 and outer['timed_out'] is False and outer.get('leader_reaped') is True and outer.get('ownership_boundary_closed') is True and end<context['deadline_ns'],'owned phase failed/late')
        need(inner is not None and type(inner['native_exit']) is int and inner['native_exit']==0 and inner['complete'] is True and inner['timed_out'] is False and inner['raw_byte_limit_exceeded'] is False and inner['reader_joined'] is True and inner['reader_pipe_close_pending'] is False and inner['exact_child_reaped'] is True and type(inner['raw_bytes']) is int and inner['raw_bytes']==(root/'julia.log').stat().st_size<=RAW_LIMIT and pin(root/'julia.log')==inner['raw_sha256'],'Julia wait/raw incomplete')
        need(time.monotonic()-post_started<METADATA_SECONDS,'metadata postflight deadline')
        result=load(root/'B1_JULIA_RESULT.json');validate_result(result,sha,root/'output')
        need(time.monotonic()-post_started<METADATA_SECONDS,'metadata postflight deadline')
        need(type(result['worker_pid']) is int and result['worker_pid']==inner['pid'],'B1 result actual worker differs')
        for field in ('fit_elapsed_ns','readback_elapsed_ns','worker_elapsed_ns','julia_peak_rss_bytes'):need(type(result[field]) is int and result[field]>=0,'B1 measured resource grammar')
        need(result['julia_peak_rss_measured'] is True and result['fit_elapsed_ns']+result['readback_elapsed_ns']<=result['worker_elapsed_ns'],'B1 resource measurement differs')
        inspected=pinned(r['runtime'],131072);runtime=result['runtime'];need(runtime['argv']==argv and {k:v for k,v in runtime.items() if k!='argv'}=={k:v for k,v in inspected.items() if k!='argv'},'actual inspected runtime differs')
        need(result['job_sha256']==hashlib.sha256(canonical(dict(driver_sha256=pin(__file__),entry_sha256=pin(ENTRY),owner_sha256=OWNER_SHA,legacy_owner_sha256=LEGACY_SHA,argv=argv,request_sha256=sha))).hexdigest(),'fixed B1 job identity differs')
        record.update(outputs=result,result_sha256=pin(root/'B1_JULIA_RESULT.json'),complete=True)
    except BaseException as error:record.update(complete=False,failure=dict(type=type(error).__name__,message=str(error)[:2048]))
    finally:
        post=post_started if post_started is not None else time.monotonic();after={};errors={}
        for p,expected in controls.items():
            try:need(time.monotonic()-post<METADATA_SECONDS,'postflight metadata deadline');after[p]=pin(p)
            except BaseException as error:errors[p]=type(error).__name__
        write(root/'CONTROLS_AFTER.json',after);write(root/'CONTROL_READ_ERRORS.json',errors)
        changed=[p for p,v in controls.items() if after.get(p)!=v];record.update(changed_control_count=len(changed),control_read_error_count=len(errors),controls_before_sha256=pin(root/'CONTROLS_BEFORE.json'),controls_after_sha256=pin(root/'CONTROLS_AFTER.json'),control_read_errors_sha256=pin(root/'CONTROL_READ_ERRORS.json'))
        try:need(time.monotonic()-post<METADATA_SECONDS,'postflight metadata deadline');D.source_guard(proof);need(time.monotonic()-post<METADATA_SECONDS,'postflight metadata deadline')
        except BaseException as error:record['source_guard_failure']=str(error)[:1024];record['complete']=False
        for name,key in (('supervisor.raw','retained_supervisor_raw_sha256'),('julia.log','retained_julia_raw_sha256'),('JULIA_NATIVE_WAIT.json','retained_julia_wait_sha256'),('SUPERVISOR_DIAGNOSTIC_WAIT.json','retained_diagnostic_wait_sha256')):
            try:
                if (root/name).is_file():record[key]=pin(root/name)
            except BaseException as error:record['complete']=False;record[key+'_error']=type(error).__name__
        record['complete']=record['complete'] and not changed and not errors and 'failure' not in record
        record['finished_utc']=datetime.now(timezone.utc).isoformat();record['metadata_postflight_elapsed_ns']=int((time.monotonic()-post)*10**9)
        write(root/'B1_NATIVE_RECEIPT.json',record)
    return 0 if record['complete'] else 1

if __name__=='__main__':
    if _BOOTSTRAP is not None:_supervisor_main()
    parser=argparse.ArgumentParser(description='Fixed actual B1 L0 native stage; supplied permits only')
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args()
    try:code=main(args.request,args.request_sha256)
    except BaseException as error:
        sys.stderr.write('B1 preflight failed: '+type(error).__name__+'\n');code=1
    raise SystemExit(code)