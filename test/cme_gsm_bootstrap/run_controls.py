"""Fixed owned bootstrap pin regressions: Python/Git metadata only, never Julia."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import re,stat,sys,traceback

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).with_name('test_owner_pins.py')
OWNER=ROOT/'research/dbdt_completed_source/native_stage.py'
SCRIPT_SHA='401aad1bd271f3b81c03f97b705157c8cb6b62b49032180b5951310d54a732d7'
OWNER_SHA='15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8'
SECONDS=180
NAMES=(
 'research/cme_gsm_bootstrap/bootstrap_indices.jl','research/cme_gsm_bootstrap/bootstrap_entry.jl',
 'research/cme_gsm_bootstrap/test_bootstrap_indices.jl','research/cme_gsm_bootstrap/test_native_generated_artifact.jl',
 'research/cme_gsm_bootstrap/frozen_population_years.toml','research/cme_gsm_bootstrap/run_native_bootstrap.py',
 'research/dbdt_completed_source/native_stage.py','tools/run_canonical_harness.py','tools/canonical_dev_harness.sh',
 'tools/run_full_scale_generated.py','tools/verify_canonical_harness_artifact.py',
 'research/dbdt_completed_source/candidate_native_owner.py','research/dbdt_completed_source/run_candidate_stage.py',
 'test/candidate_stage_driver/run_controls.py','research/dbdt_completed_source/synthetic_checkpoint.py',
 'research/dbdt_completed_source/run_synthetic_chain.py','research/dbdt_completed_source/candidate_validation_authority.jl',
 'src/completed_source_dbdt/refit_artifacts/runtime.jl','test/cme_gsm_bootstrap_tests.jl')
def ordinary(path,directory=False):
    info=path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('reparse/link control refused')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):raise ValueError('ordinary control required')
def pin(path):
    ordinary(path)
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def validate_summary(raw):
    if not isinstance(raw,bytes) or len(raw)>4*1024**2:
        raise RuntimeError('bounded native transcript bytes required')
    if re.findall(rb'^Ran (\d+) tests in [^\r\n]+\r?$',raw,re.M)!=[b'9'] or re.findall(rb'^(OK)\r?$',raw,re.M)!=[b'OK']:
        raise RuntimeError('exact nine-method native transcript required')
    if re.search(rb'^(?:FAILED\b|FAIL:|ERROR:|SKIPPED\b)',raw,re.M):
        raise RuntimeError('failed or skipped native transcript rejected')
def main(output):
    if sys.flags.optimize:raise RuntimeError('unoptimized Python is required')
    if sys.version_info<(3,11):raise RuntimeError('Python>=3.11 required')
    if sys.platform=='darwin' and sys.version_info<(3,14):raise RuntimeError('macOS Python>=3.14 required; no feature skip')
    if not output.is_absolute() or output.exists() or output.is_symlink():raise ValueError('fresh absolute output required')
    if output.parent.resolve(strict=True)!=output.parent:raise ValueError('canonical output parent required')
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):raise ValueError('output/source overlap forbidden')
    for path in (ROOT,*ROOT.parents,output.parent,*output.parent.parents):ordinary(path,directory=True)
    output.mkdir();fixtures=output/'fixtures';fixtures.mkdir()
    executable=Path(sys.executable).resolve(strict=True)
    controls={str(path):pin(path) for path in (Path(__file__),SCRIPT,executable,*(ROOT/name for name in NAMES))}
    record=dict(complete=False,controls_before_sha256=controls,methods_expected=9,
        Julia_executed=False,actual_GSM_values=0,source_authority_granted=False,scientific_qualification=False,
        fixture_scope='Eight-file generated clean Git repositories only; no canonical/P/M/public mutations')
    try:
        if pin(SCRIPT)!=SCRIPT_SHA or pin(OWNER)!=OWNER_SHA:raise RuntimeError('fixed regression/owner pin differs')
        spec=importlib.util.spec_from_file_location('bootstrap_control_owned_native',OWNER)
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        native=owner.run_owned([sys.executable,'-B','-X','utf8',str(SCRIPT)],cwd=str(ROOT),
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMP=str(fixtures),TEMP=str(fixtures),TMPDIR=str(fixtures)),
            logpath=output/'native.log',timeout=SECONDS)
        record.update(native=native,raw_sha256=pin(output/'native.log'))
        if type(native['native_exit']) is not int or native['native_exit']!=0 or native['timed_out'] is not False:
            raise RuntimeError('native pin regression failed or timed out')
        with (output/'native.log').open('rb') as f:raw=f.read(4*1024**2+1)
        validate_summary(raw)
        record.update(complete=True,methods=9)
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        after={}
        for path in controls:
            try:after[path]=pin(Path(path))
            except BaseException:after[path]=None
        record.update(controls_after_sha256=after,changed_controls=[p for p,v in controls.items() if after[p]!=v])
        record['complete']=record['complete'] and not record['changed_controls']
        write(output/'TERMINAL.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('controls_before_sha256','controls_after_sha256')},sort_keys=True),flush=True)
    return 0 if record['complete'] else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().output))
