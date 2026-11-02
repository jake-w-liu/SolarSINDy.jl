"""Fixed bounded owner for all17 offline driver controls; no Julia subprocess."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,sys,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).with_name('test_driver.py')
FIXTURE=Path(__file__).with_name('native_fixture.py')
OWNER=ROOT/'research/dbdt_completed_source/candidate_native_owner.py'
PINS={SCRIPT:'3284bb932958ba36e80b5475c8cc159624f9d679af279f50a92d08380b0a6f91',
    FIXTURE:'d855e765adf0f1820edf0a25d1c96ed8cb59cf9c3edb5dd6ae2d395bc1cd4f57',
    OWNER:'ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21'}
def pin(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def admit_output(output):
    assert output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir()
    assert output.parent.resolve(strict=True)==output.parent,'aliased output parent is forbidden'
    assert not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output)

def validate_raw(raw):
    assert len(raw)<=4*1024**2 and re.findall(rb'^Ran (\d+) tests in [^\r\n]+\r?$',raw,re.M)==[b'17']
    assert re.findall(rb'^(OK)\r?$',raw,re.M)==[b'OK']

def main(output):
    if sys.flags.optimize:raise RuntimeError('unoptimized Python is required; assertion removal is forbidden')
    if sys.version_info<(3,11):raise RuntimeError('Python>=3.11 required')
    if sys.platform=='darwin' and sys.version_info<(3,13):raise RuntimeError('macOS requires Python>=3.13 for waitid/WNOWAIT; no skip or unsafe fallback')
    admit_output(output)
    output.mkdir();fixtures=output/'fixtures';fixtures.mkdir()
    controls={str(p):pin(p) for p in (Path(__file__),*PINS,ROOT/'research/dbdt_completed_source/run_candidate_stage.py',
        ROOT/'research/dbdt_completed_source/run_candidate_stage.jl',ROOT/'research/dbdt_completed_source/native_stage.py')}
    record=dict(complete=False,Julia_executed=False,actual_data_access=False,mocked_metadata_is_not_native_Julia_evidence=True,
        controls_sha256=controls,methods_expected=17)
    try:
        assert all(pin(p)==sha for p,sha in PINS.items())
        spec=importlib.util.spec_from_file_location('package_candidate_driver_control_owner',OWNER)
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        denied=output/'optimized-denied'
        rejected=owner.run_owned([sys.executable,'-O','-B','-X','utf8',str(Path(__file__)),'--output',str(denied)],cwd=str(ROOT),
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=output/'optimized-native.log',timeout=10)
        record['optimized_negative']=dict(native=rejected,raw_sha256=pin(output/'optimized-native.log'),expected_failure=True)
        assert type(rejected['native_exit']) is int and rejected['native_exit']==1 and rejected['timed_out'] is False and not denied.exists()
        result=owner.run_owned([sys.executable,'-B','-X','utf8','-W','error::ResourceWarning',str(SCRIPT)],cwd=str(ROOT),
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CANDIDATE_DRIVER_CONTROLS_ROOT=str(fixtures)),
            logpath=output/'native.log',timeout=900)
        record.update(native=result,raw_sha256=pin(output/'native.log'))
        assert type(result['native_exit']) is int and result['native_exit']==0 and result['timed_out'] is False
        with (output/'native.log').open('rb') as f:raw=f.read(4*1024**2+1)
        validate_raw(raw)
        report=json.loads((fixtures/'RESULT.json').read_bytes())
        assert report['complete'] is True and report['methods']==17 and report['failures']==report['errors']==0
        assert report['Julia_executed'] is False and report['actual_data_access'] is False
        record.update(complete=True,methods=17,result_sha256=pin(fixtures/'RESULT.json'))
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        record['changed_controls']=[p for p,sha in controls.items() if pin(p)!=sha]
        record['complete']=record['complete'] and not record['changed_controls']
        write(output/'TERMINAL.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='controls_sha256'}),flush=True)
    return 0 if record['complete'] else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    raise SystemExit(main(args.output))
