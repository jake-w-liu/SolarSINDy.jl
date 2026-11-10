"""Fixed bounded campaign tests/plan entries, without Julia or archive access."""
from pathlib import Path
import argparse,ast,collections,hashlib,importlib.util,json,math,os,re,stat,sys,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SELF=Path(__file__).resolve(strict=True)
TEST=Path(__file__).with_name('test_campaign.py')
PLAN=ROOT/'examples/candidate_campaign_plan_synthetic.py'
OWNER=ROOT/'research/dbdt_completed_source/candidate_native_owner.py'
LEGACY=ROOT/'research/dbdt_completed_source/native_stage.py'
LITERALS=Path(__file__).with_name('descriptor_literals.json')
ORIGINAL=ROOT/'test/selected_point_bindings/full_codec_oracles.json'
PINS={
 'test/candidate_campaign/test_campaign.py':'e4399df43a566f80423553316beee9a9fbdf441e614d7eb20197d1d672f0449a',
 'examples/candidate_campaign_plan_synthetic.py':'06d7528ee3c7c85187ff599492ae027e6dcb2b4afc7b2c7dfafe276a0610b3c7',
 'research/dbdt_completed_source/candidate_native_owner.py':'c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00',
 'research/dbdt_completed_source/native_stage.py':'30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
 'research/dbdt_completed_source/run_candidate_campaign.py':'9993b20e1a3dd9c2f7232e4c2e44bd281ad1953072c869e3815ed3f2ad58f2da',
 'research/dbdt_completed_source/run_candidate_descriptor_preview.jl':'6c10f28c46d3040f2e37afff6a67a346430388451c288148cfe8a0f7cdc10909',
 'research/dbdt_completed_source/candidate_descriptor_metadata.jl':'87d7c4561336badb780c5f6e83d111e358d88a557730d116d11f3ec8bcd72f39',
 'research/dbdt_completed_source/candidate_validation.jl':'e409bd7baecd6db780191f93526f590d85cfd196fdcab3535c6a9ccad01f7676',
 'test/candidate_campaign/descriptor_literals.json':'7d692a65e05abd4eac8e2e5e6e35b398ab3169f81acd8f5c589eaca34c474424',
 'test/candidate_campaign/descriptor_preview_tests.jl':'9b2307de7f69cc252f336548dec48e56e07521a523004f33ac7bca3afe63708e',
 'examples/candidate_descriptor_preview_synthetic.jl':'0c33ff087df7f079fb8d5bf560dc642d0da1cbec8a5af19759ce1d2cc20ef3cd',
 'test/selected_point_bindings/full_codec_oracles.json':'55949343d67e4776ace7c1be90f572ad7dc3c8af4b66e53178a19e8f43fd0b0f'}
def need(ok,message):
    if not ok:raise ValueError(message)
def ordinary(path,directory=False):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute ordinary path required')
    for part in (p,*p.parents):
        info=part.lstat();need(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400,'linked path forbidden')
    need(p.is_dir() if directory else p.is_file(),'ordinary file/directory required');return p
def pin(path):
    with ordinary(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    with path.open('xb') as f:
        view=memoryview(raw)
        while view:
            count=f.write(view);need(type(count) is int and 0<count<=len(view),'metadata write made no progress');view=view[count:]
def raw(path,limit):
    p=ordinary(path);before=p.stat();need(before.st_size<=limit,'raw/metadata byte limit')
    with p.open('rb') as f:body=f.read(limit+1)
    after=p.stat();need(len(body)==before.st_size==after.st_size and len(body)<=limit and
        (before.st_dev,before.st_ino,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_mtime_ns),'raw/metadata changed')
    return body
def decode(body):
    def pairs(values):
        out={}
        for k,v in values:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    result=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    def check(v,depth=0):
        need(depth<=24,'metadata nesting limit')
        if isinstance(v,dict):
            need(len(v)<=8192 and all(isinstance(k,str) for k in v),'metadata object limit')
            for k,x in v.items():check(k,depth+1);check(x,depth+1)
        elif isinstance(v,list):
            need(len(v)<=8192,'metadata array limit')
            for x in v:check(x,depth+1)
        elif isinstance(v,str):need(len(v.encode('utf8'))<=65536,'metadata string limit')
        elif type(v) is float:need(math.isfinite(v),'finite JSON atom required')
        else:need(v is None or type(v) in (bool,int),'metadata atom type')
    check(result);return result
def admit_output(mode,output):
    need(mode in ('tests','plan'),'fixed runner mode required')
    need(output.is_absolute() and not output.exists() and not output.is_symlink(),'fresh absolute output required')
    ordinary(output.parent,True)
    need(output.parent.resolve(strict=True)==output.parent,'aliased output parent forbidden')
    need(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output),'output must stay outside checkout')
def source_controls():
    need(all(pin(ROOT/name)==sha for name,sha in PINS.items()),'frozen campaign input source differs')
    return {str(ROOT/name):sha for name,sha in PINS.items()}|{str(SELF):pin(SELF)}
def verify_tests(path,report_path):
    tree=ast.parse(raw(TEST,1048576));expected=sorted(c.name+'.'+f.name for c in tree.body if isinstance(c,ast.ClassDef)
        for f in c.body if isinstance(f,ast.FunctionDef) and f.name.startswith('test_'))
    need(len(expected)==17,'frozen17 discovery differs')
    text=raw(path,4194304).decode('utf8')
    names=re.findall(r'^test_[^ \r\n]+ \((?:__main__\.)?([^\r\n)]+)\) \.\.\.',text,re.M)
    need(sorted(names)==expected and len(names)==17 and
        re.findall(r'^Ran ([0-9]+) tests in [^\r\n]+\r?$',text,re.M)==['17'] and
        re.findall(r'^(OK)\r?$',text,re.M)==['OK'] and '... skipped ' not in text,'all fixed17 methods must pass without skip')
    report=decode(raw(report_path,1048576))
    need(report['complete'] is True and type(report['methods']) is int and report['methods']==17 and
        report['failures']==report['errors']==0 and report['Julia_executed'] is False and report['actual_data_access'] is False and
        type(report['actual_fits']) is int and report['actual_fits']==0,'campaign control report differs')
    return dict(methods=17,skipped=0,discovered_names=expected,result_sha256=pin(report_path))
def verify_plan(path):
    value=decode(raw(path,1048576));need(isinstance(value,dict) and set(value)=={'schema','stages','stage_count','fit_operations',
        'worker_launches','actual_fits','actual_source_access','authority_granted','scientific_qualification'},'fixed plan result fields')
    need(value['schema']=='unqualified_campaign_plan_example_v1' and type(value['stage_count']) is int and value['stage_count']==142 and
        type(value['fit_operations']) is int and value['fit_operations']==38 and type(value['worker_launches']) is int and value['worker_launches']==0 and
        type(value['actual_fits']) is int and value['actual_fits']==0 and value['actual_source_access'] is False and
        value['authority_granted'] is False and value['scientific_qualification'] is False,'plan example must remain unqualified/zero work')
    original=decode(raw(ORIGINAL,1048576))['decisions']['R2']['request']
    ids=[row['candidate_key'] for row in original['producer_expectations']];need(len(ids)==16 and len(set(ids))==16,'frozen original16 grid')
    expected=[]
    for station in ('CMO','FRD'):
        for id in ids:
            for op in ('candidate-input','candidate-fit','candidate-validation','independent-verify'):
                expected.append(dict(station=station,candidate_id=id,family=None,operation=op,key=station+'/'+id+'/'+op))
        expected.append(dict(station=station,candidate_id=None,family=None,operation='select-R2',key=station+'/select-R2'))
        for family in ('ridge','boost','forest'):
            for op in ('selected-refit-input','selected-refit'):
                expected.append(dict(station=station,candidate_id=None,family=family,operation=op,key=station+'/'+family+'/'+op))
    expected=[dict(row,number=i+1,directory_name=f'{i+1:03d}-'+row['key'].replace('/','-')) for i,row in enumerate(expected)]
    need(value['stages']==expected,'every fixed142 plan row/order/field must match original contract')
    literals=decode(raw(LITERALS,1048576));counts=dict(collections.Counter(row['operation'] for row in expected))
    need(counts==literals['operation_counts'] and all(expected[int(i)-1]['key']==key for i,key in literals['positions_one_based'].items()),'independent queue literals differ')
    return dict(stage_count=142,fit_operations=38,verified_stage_rows=142,worker_launches=0,actual_fits=0,
        actual_source_access=False,authority_granted=False,scientific_qualification=False)
def main(mode,output):
    need(not sys.flags.optimize,'unoptimized Python required before runner admission')
    need(sys.version_info>=(3,11),'Python>=3.11 required')
    need(sys.platform!='darwin' or sys.version_info>=(3,13),'macOS requires Python>=3.13 waitid/WNOWAIT; no fallback')
    admit_output(mode,output);controls=source_controls()
    python=ordinary(Path(sys.executable).resolve(strict=True));controls[str(python)]=pin(python)
    output.mkdir();fixtures=output/'fixtures';fixtures.mkdir()
    record=dict(schema='fixed_campaign_package_controls_v1',mode=mode,complete=False,Julia_executed=False,actual_data_access=False,
        actual_fits=0,source_qualified=False,scientific_qualified=False,controls_sha256=controls,native=None)
    try:
        spec=importlib.util.spec_from_file_location('fixed_campaign_package_native_owner',OWNER)
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        probes=[]
        for label,flags,selected,code in (('optimized',['-O'],mode,1),('unsupported-mode',[],'arbitrary-script',2)):
            denied=output/(label+'-denied');log=output/(label+'.log')
            outcome=owner.run_owned([str(python),*flags,'-B','-X','utf8',str(SELF),'--mode',selected,'--output',str(denied)],
                cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=log,timeout=10)
            probes.append(dict(name=label,native=outcome,raw_sha256=pin(log),expected_native_exit=code,output_created=denied.exists()))
            need(type(outcome['native_exit']) is int and outcome['native_exit']==code and outcome['timed_out'] is False and not denied.exists(),'runner admission negative failed')
        record['runner_guard_native_probes']=probes
        script=TEST if mode=='tests' else PLAN
        argv=[str(python),'-B','-X','utf8','-W','error::ResourceWarning',str(script)]
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CANDIDATE_CAMPAIGN_CONTROLS_ROOT=str(fixtures),
            TMP=str(fixtures),TEMP=str(fixtures),TMPDIR=str(fixtures))
        record['argv']=argv
        outcome=owner.run_owned(argv,cwd=str(ROOT),environment=environment,logpath=output/'native.log',timeout=900)
        record.update(native=outcome,raw_sha256=pin(output/'native.log'))
        need(type(outcome['native_exit']) is int and outcome['native_exit']==0 and outcome['timed_out'] is False and
            outcome['leader_reaped'] is True and outcome['ownership_boundary_closed'] is True,'fixed package child failed; native/raw retained')
        verified=verify_tests(output/'native.log',fixtures/'RESULT.json') if mode=='tests' else verify_plan(output/'native.log')
        record.update(verified);record['complete']=True
    except BaseException as error:
        record['complete']=False;record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        after={};errors={}
        for path,sha in controls.items():
            try:after[path]=pin(path)
            except BaseException as error:errors[path]=dict(type=type(error).__name__,message=str(error)[:128])
        record['changed_controls']=[path for path,sha in controls.items() if after.get(path)!=sha];record['control_read_errors']=errors
        record['complete']=record['complete'] and not record['changed_controls'] and not errors and 'failure' not in record
        write(output/'TERMINAL.json',record)
        print(json.dumps(dict(mode=mode,complete=record['complete'],methods=record.get('methods'),stage_count=record.get('stage_count'),
            actual_fits=0,Julia_executed=False,actual_data_access=False,terminal=str(output/'TERMINAL.json'))),flush=True)
    return 0 if record['complete'] else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('tests','plan'),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    raise SystemExit(main(args.mode,args.output))
