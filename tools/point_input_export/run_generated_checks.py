"""Bounded generated-only input verification with retained independent raw logs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[2]
def pin(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def worker(report_path,report_sha256,output):
    reader=module(Path(__file__).with_name('bundle_reader.py'),'portable_generated_input_reader')
    reader.C.V._sha(report_sha256);body=reader.C.read(report_path,1048576)
    assert hashlib.sha256(body).hexdigest()==report_sha256
    report=reader.C.decoded(body)
    reader.C.keys(report,('schema','complete','runtime','exports','actual_source_access','actual_fits'))
    assert report['schema']=='generated_portable_point_input_exports_v1' and report['complete'] is True
    assert report['actual_source_access'] is False and type(report['actual_fits']) is int and report['actual_fits']==0
    assert report['runtime'] in ('1.12.6','1.13.1') and isinstance(report['exports'],list) and len(report['exports'])==8
    expected={(role,layout,station) for role in ('primary','independent') for layout in ('ground-only','mixed') for station in ('CMO','FRD')}
    seen=set();checks=[];exporter_pin=pin(ROOT/'research/dbdt_completed_source/point_input_export.jl')
    fields=('directory','case_directory','case_request_path','case_request_sha256','case_manifest_sha256','case_complete_sha256',
        'table_index','start','stop','asof','entry_sha256','numeric_source_sha256','library_sha256',
        'context_sha256','manifest_sha256','complete_sha256','receipt_sha256','input_sha256','role','layout','station')
    for entry in report['exports']:
        reader.C.keys(entry,fields);key=entry['role'],entry['layout'],entry['station']
        assert key in expected and key not in seen;seen.add(key)
        assert entry['entry_sha256']==exporter_pin and entry['table_index']==('CMO','FRD').index(entry['station'])+1
        args={name:entry[name] for name in fields if name not in ('directory','role','layout','station')}
        result=reader.verify_bundle(entry['directory'],**args)
        context=result['case_verification'];assert context['runtime']==report['runtime'] and context['role']==entry['role'] and context['layout']==entry['layout']
        assert result['rows']==50 and result['first_row']==41 and result['last_row']==90 and result['scheduled_rows_scanned']==181
        assert result['columns']==(11 if entry['layout']=='ground-only' else 8) and not result['actual_source_access'] and not result['fit_authority_granted']
        checks.append(dict(role=key[0],layout=key[1],station=key[2],verification=result))
    assert seen==expected and pin(report_path)==report_sha256
    write(output,dict(complete=True,created_utc=now(),runtime=report['runtime'],report_sha256=report_sha256,
        independently_verified_exports=len(checks),total_verified_input_rows=sum(c['verification']['rows'] for c in checks),
        checks=checks,actual_source_access=False,actual_fits=0,software_qualification=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();report=args.report.resolve(strict=True);out=args.output.resolve()
    if args.worker:worker(report,args.report_sha256,out);return 0
    assert sys.version_info>=(3,11)
    assert not out.exists() and out.parent.is_dir() and not out.is_relative_to(ROOT)
    assert pin(report)==args.report_sha256;out.mkdir()
    controls={str(p):pin(p) for p in (report,*sorted(Path(__file__).parent.glob('*.py')),
        ROOT/'research/dbdt_completed_source/point_input_export.jl',*sorted((ROOT/'test/point_input_export').glob('*.py')))}
    record=dict(complete=False,started_utc=now(),controls_sha256=controls,stages={},actual_source_access=False,actual_fits=0,software_qualification=False)
    write(out/'PREEXECUTION.json',record)
    try:
        commands=[('input-values',[sys.executable,'-B',str(ROOT/'test/point_input_export/test_input_values.py')]),
            ('bundle-reader',[sys.executable,'-B',str(ROOT/'test/point_input_export/test_bundle_reader.py')]),
            ('native-export-reader',[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--report',str(report),'--report-sha256',args.report_sha256,'--output',str(out/'INDEPENDENT_EXPORTS.json')])]
        for name,command in commands:
            assert all(pin(p)==sha for p,sha in controls.items())
            stage=out/name;stage.mkdir();tmp=stage/'tmp';tmp.mkdir()
            # These fixed Python children spawn no descendants. Inherit the
            # outer Julia owner's group/Job rather than escaping it into a new
            # session. subprocess.run kills and waits for its child on timeout.
            started=time.monotonic()
            with (stage/'native.log').open('xb') as log:
                try:
                    child=subprocess.run(command,cwd=ROOT,env=dict(os.environ,TMP=str(tmp),TEMP=str(tmp),TMPDIR=str(tmp)),stdout=log,stderr=subprocess.STDOUT,timeout=900)
                    result=dict(native_exit=child.returncode,timed_out=False,wall_seconds=time.monotonic()-started)
                except subprocess.TimeoutExpired:
                    result=dict(native_exit=None,timed_out=True,wall_seconds=time.monotonic()-started)
            record['stages'][name]=dict(native=result,raw_sha256=pin(stage/'native.log'))
            assert result['native_exit']==0 and not result['timed_out'],name+' failed;raw='+str(stage/'native.log')
        independent=json.loads((out/'INDEPENDENT_EXPORTS.json').read_bytes())
        assert independent['complete'] and independent['independently_verified_exports']==8 and independent['total_verified_input_rows']==400
        record.update(complete=True,independently_verified_exports=8,total_verified_input_rows=400,independent_checks_sha256=pin(out/'INDEPENDENT_EXPORTS.json'))
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        record['changed_controls']=[p for p,sha in controls.items() if pin(p)!=sha];record['complete']=record['complete'] and not record['changed_controls'];record['finished_utc']=now()
        write(out/'TERMINAL.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('controls_sha256','stages')}),flush=True)
    return 0 if record['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
