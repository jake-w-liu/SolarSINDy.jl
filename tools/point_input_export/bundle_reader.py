"""Independent data-only reader for bounded canonical native input exports."""
from pathlib import Path
from hashlib import sha256
import importlib.util
BASE=Path(__file__).with_name('input_values.py')
assert sha256(BASE.read_bytes()).hexdigest()=='0926891a2904a4bdd2476a4dee4fbc0be8c0120e4b63bc547c32be3841e84671'
spec=importlib.util.spec_from_file_location('point_input_bundle_value_oracle',BASE)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
C=V.C;need=V.need
CHUNK_BYTES=8388608

def verify_bundle(directory,case_directory,case_request_path,*,manifest_sha256,complete_sha256,context_sha256,receipt_sha256,input_sha256,
                  case_request_sha256,case_manifest_sha256,case_complete_sha256,table_index,start,stop,asof,
                  entry_sha256,numeric_source_sha256,library_sha256,expected_case_authority=None,expected_planning_authorization=None):
    directory=C.ordinary(Path(directory),True);case_directory=Path(case_directory);case_request_path=Path(case_request_path)
    identities={}
    def read(name,limit,expected):
        path=directory/name;before=C.V._identity(C.ordinary(path));body=C.read(path,limit)
        C.V._sha(expected);need(sha256(body).hexdigest()==expected,'input export member pin differs')
        identities[path]=before;return body
    manifest_bytes=read('manifest.json',65536,manifest_sha256);marker_bytes=read('COMPLETE.json',4096,complete_sha256)
    manifest=C.decoded(manifest_bytes);marker=C.decoded(marker_bytes)
    C.keys(manifest,('schema','context_sha256','input_sha256','input_bytes','chunk_bytes','chunks','members'))
    C.keys(marker,('schema','manifest_sha256','context_sha256','receipt_sha256','input_sha256','members','actual_fits'))
    need(manifest['schema']=='source_bound_point_input_export_manifest_v1' and marker['schema']=='source_bound_point_input_export_complete_v1','input export schema differs')
    need(manifest['context_sha256']==context_sha256 and manifest['input_sha256']==input_sha256,'input export external context differs')
    total=C.integer(manifest['input_bytes'],V.MAX_BYTES,1);count=C.integer(manifest['chunks'],16,1)
    need(C.integer(manifest['chunk_bytes'],CHUNK_BYTES)==CHUNK_BYTES and count==(total+CHUNK_BYTES-1)//CHUNK_BYTES,'input export chunk geometry differs')
    names=['context.json','receipt.json']+[f'input-{i:04d}.bin' for i in range(1,count+1)]
    need(isinstance(manifest['members'],list) and len(manifest['members'])==len(names),'input export member list differs')
    expected_files=set(names+['manifest.json','COMPLETE.json'])
    actual_names=sorted(path.name for path in directory.iterdir())
    need(set(actual_names)==expected_files and len(actual_names)==len(expected_files),'input export fileset differs')
    bodies={}
    for index,(name,entry) in enumerate(zip(names,manifest['members'])):
        C.keys(entry,('name','bytes','sha256'));need(entry['name']==name,'input export member order differs')
        limit=262144 if index==0 else 65536 if index==1 else CHUNK_BYTES
        size=C.integer(entry['bytes'],limit,1);body=read(name,limit,entry['sha256'])
        need(len(body)==size,'input export member size differs')
        if index>=2:need(size==min(CHUNK_BYTES,total-(index-2)*CHUNK_BYTES),'input export exact chunk size differs')
        bodies[name]=body
    need(marker['manifest_sha256']==manifest_sha256 and marker['context_sha256']==context_sha256 and marker['receipt_sha256']==receipt_sha256 and
         marker['input_sha256']==input_sha256 and C.integer(marker['members'],20)==len(expected_files) and C.integer(marker['actual_fits'],0)==0,'input export completion binding differs')
    need(sha256(bodies['context.json']).hexdigest()==context_sha256 and sha256(bodies['receipt.json']).hexdigest()==receipt_sha256,'input export context/receipt pin differs')
    context=C.decoded(bodies['context.json']);receipt=C.decoded(bodies['receipt.json'])
    C.keys(context,('schema','request','rows','columns','first_row','last_row','input_sha256','input_bytes','strict_label_ready_before','target_transform','availability_kind','measured_receipts_available'))
    case_report=C.verify_cases(case_directory,case_request_path,request_sha256=case_request_sha256,manifest_sha256=case_manifest_sha256,
        complete_sha256=case_complete_sha256,expected_authority=expected_case_authority)
    case_request=C.decoded(C.read(case_request_path,131072));case_manifest=C.decoded(C.read(case_directory/'manifest.json',1048576))
    V.integer(table_index,1,len(case_request['tables']),'input export case index differs')
    declaration=case_request['tables'][table_index-1];table=case_manifest['tables'][table_index-1]
    begin,end,cutoff=map(V.minute,(start,stop,asof))
    need(V.minute(declaration['start_inclusive'])<=begin<end<=V.minute(declaration['stop_exclusive']) and cutoff<=V.minute(declaration['label_asof']),'input export split exceeds case table')
    actual=case_request['evidence_mode']=='actual_archival_cache'
    if actual:need(V.minute('2017-01-01T00:00:00.000Z')<=begin<end<=V.minute('2018-01-01T00:00:00.000Z'),'input export outside archival2017 scope')
    expected_request=dict(schema='source_bound_point_input_planning_request_v1',evidence_mode=case_request['evidence_mode'],
        case_request_sha256=case_request_sha256,case_manifest_sha256=case_manifest_sha256,case_complete_sha256=case_complete_sha256,
        table_index=table_index,table=declaration,runtime=case_request['runtime'],role=case_request['role'],layout=case_request['layout'],
        start_inclusive=start,stop_exclusive=stop,target_asof=asof,entry_sha256=entry_sha256,numeric_source_sha256=numeric_source_sha256,
        library_sha256=library_sha256,feature_payload_sha256=table['feature_payload_sha256'],label_payload_sha256=table['label_payload_sha256'],
        actual_model_fit_authorized=False,provider_receipts_available=False)
    for value in (entry_sha256,numeric_source_sha256,library_sha256):C.V._sha(value)
    if actual:
        need(isinstance(expected_planning_authorization,dict),'archival input export needs separate external planning admission')
        expected_request['case_read_admission_sha256']=expected_planning_authorization['case_read_admission_sha256']
    else:need(expected_planning_authorization is None,'generated input export cannot consume actual planning admission')
    C.keys(context['request'],expected_request)
    V.integer(context['request']['table_index'],1,len(case_request['tables']),'input export request index type differs')
    need(context['request']==expected_request,'input export planning request differs')
    need(context['request']['actual_model_fit_authorized'] is False and context['request']['provider_receipts_available'] is False,'input export request permissions differ')
    width=11 if case_request['layout']=='ground-only' else 8
    rows=C.integer(context['rows'],V.MAX_ROWS,2);need(C.integer(context['columns'],11)==width,'input export context width differs')
    need(context['schema']=='source_bound_point_input_export_context_v1' and context['input_sha256']==input_sha256 and
        C.integer(context['input_bytes'],V.MAX_BYTES)==total and context['strict_label_ready_before']==stop and
        context['target_transform']=='log1p_rate_reference_1_nt_per_min' and context['availability_kind']=='assumed_completed_source_intervals' and
        context['measured_receipts_available'] is False,'input export context convention differs')
    receipt_fields={'schema','complete','context_sha256','input_sha256','actual_source_access','actual_model_fit_performed','fit_authority_granted','provider_receipts_available','software_qualification'}
    if actual:receipt_fields.add('authorization')
    C.keys(receipt,receipt_fields)
    need(receipt['schema']=='source_bound_point_input_export_receipt_v1' and receipt['complete'] is True and receipt['context_sha256']==context_sha256 and
        receipt['input_sha256']==input_sha256 and receipt['actual_source_access'] is actual and receipt['actual_model_fit_performed'] is False and
        receipt['fit_authority_granted'] is False and receipt['provider_receipts_available'] is False and receipt['software_qualification'] is False,'input export receipt differs')
    if actual:need(receipt['authorization']==expected_planning_authorization,'input export separate planning receipt differs')
    data=b''.join(bodies[name] for name in names[2:]);need(len(data)==total and sha256(data).hexdigest()==input_sha256,'input export complete stream differs')
    descriptors={entry['name']:entry for entry in case_manifest['members']};case_identities={}
    def records():
        row=0
        for fn,ln in zip(table['feature_shards'],table['label_shards']):
            payloads=[]
            for name,width_bytes in ((fn,160),(ln,64)):
                path=case_directory/name;before=C.V._identity(C.ordinary(path));body=C.read(path,8192*width_bytes)
                need(len(body)==descriptors[name]['bytes'] and sha256(body).hexdigest()==descriptors[name]['sha256'],'input export case shard changed')
                case_identities[path]=before;payloads.append(body)
            fb,lb=payloads;need(len(fb)//160==len(lb)//64,'input export case shard pairing differs')
            for local in range(len(fb)//160):
                row+=1;yield row,fb[160*local:160*(local+1)],lb[64*local:64*(local+1)]
    report=V.check_input_bytes(data,records(),start_ms=begin,stop_ms=end,asof_ms=cutoff,width=width)
    need(report['rows']==rows and report['scheduled_rows_scanned']==table['scheduled'] and
        C.integer(context['first_row'],V.MAX_ROWS,1)==report['first_row'] and C.integer(context['last_row'],V.MAX_ROWS,1)==report['last_row'],'input export independent row identity differs')
    need(sorted(path.name for path in directory.iterdir())==actual_names and all(C.V._identity(C.ordinary(path))==before for path,before in identities.items()),'input export changed during read')
    need(all(C.V._identity(C.ordinary(path))==before for path,before in case_identities.items()) and V.pin(case_request_path)==case_request_sha256 and
        V.pin(case_directory/'manifest.json')==case_manifest_sha256 and V.pin(case_directory/'COMPLETE.json')==case_complete_sha256,'input export source cases changed during read')
    report.update(case_verification=case_report,context_sha256=context_sha256,manifest_sha256=manifest_sha256,complete_sha256=complete_sha256,
        receipt_sha256=receipt_sha256,chunk_count=count,all_members_bounded=True,actual_source_access=actual,
        actual_model_fits=0,fit_authority_granted=False,verifier_sha256=V.pin(__file__))
    return report
