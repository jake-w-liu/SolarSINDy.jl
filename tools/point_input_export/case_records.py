"""Independent bounded structural case-artifact verification; never opens source paths."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util,math,re,struct
SOURCE=Path(__file__).with_name('bounded_json.py')
assert sha256(SOURCE.read_bytes()).hexdigest()=='1912a60ab9e014437f54ae133caec353fcf2839972344a126b27836b23f0aa5f'
spec=importlib.util.spec_from_file_location('case_artifact_bounded_json',SOURCE)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
need=V.need;canonical=V.canonical;decoded=V.decoded
FEATURE_REASONS=('complete','missing_ground_history','invalid_ground_history','missing_omni_history','invalid_omni_history','invalid_feature_summary')
LABEL_REASONS=('complete','split_not_strictly_ready','label_not_ready','missing_ground_target','invalid_ground_target')
def ordinary(path,directory=False):
    path=V._ordinary(path,directory)
    for part in (path,*path.parents):
        info=part.lstat()
        need(not getattr(part,'is_junction',lambda:False)() and not (getattr(info,'st_file_attributes',0)&0x400),'reparse path component')
    return path
def read(path,limit):
    ordinary(path);return V.bounded_read(path,limit)
def integer(value,maximum,minimum=0):
    V._integer(value,maximum);need(value>=minimum,'integer below minimum');return value
def keys(value,names):need(isinstance(value,dict) and set(value)==set(names),'object fields differ')
def _millis(text):return V._minute(text)
def _header(data,at,issue,reasons):
    stamp,ready=struct.unpack_from('<qq',data,at);status,reason=struct.unpack_from('BB',data,at+16)
    need(stamp==issue and ready==issue+1_845_000,'record issue/readiness differs')
    need(status in (0,1) and reason<len(reasons) and (status==1)==(reason==0),'record admission reason differs')
    need(data[at+18:at+24]==bytes(6),'record padding differs')
    return status,reason,ready
def _numeric(data,at,mode,accepted):
    if not accepted:
        need(data[at+24:at+128]==bytes(104),'rejected feature payload differs');return
    values=struct.unpack_from('<13d',data,at+24)
    need(all(math.isfinite(x) for x in values) and values[0]>=0 and values[1]>=1,'accepted feature domain differs')
    if mode=='ground11':
        need(all(x>=0 for x in values[2:9]) and all(-1<=x<=1 for x in values[9:13]),'ground feature domain differs')
        need(values[6]<=1 and values[7]<=1 and struct.pack('<d',values[1])==struct.pack('<d',1+values[4]),'ground scale/ratio differs')
    else:
        need(data[at+104:at+128]==bytes(24),'mixed unused feature bytes differ')
        need(all(x>=0 for x in values[2:6]) and 100<values[6]<90000 and abs(values[7])<9000 and 0<=values[8]<9000,'mixed feature domain differs')
        need(struct.pack('<d',values[0])==struct.pack('<d',values[2]) and values[8]==max(-values[7],0.) and values[9]==values[6]*values[8],'mixed direct feature identity differs')
def _splits():
    out={}
    for name,start,stop in (
        ('point_fit','2017-01-01','2017-10-01'),('adaptation_calibration','2017-10-01','2017-11-01'),
        ('selection_validation','2017-11-01','2018-01-01'),('final_score_calibration','2017-10-01','2018-01-01')):
        out[name]=(_millis(start+'T00:00:00.000Z'),_millis(stop+'T00:00:00.000Z'))
    return out
def _authority(value,request,request_sha256):
    keys(value,('operation','request_sha256','grant_sha256','plan_sha256','agreement_sha256','R0_context',
        'source_proof_sha256','qualification_sha256','floating_policy_sha256','head','actual_model_fit_authorized'))
    need(value['operation']=='sealed_features_labels_only' and value['request_sha256']==request_sha256 and value['actual_model_fit_authorized'] is False,'feature-only authority context differs')
    need(value['R0_context']==request['runtime']+'-'+request['role']+'-'+request['layout'],'authority R0 context differs')
    need(isinstance(value['head'],str) and re.fullmatch('[0-9a-f]{40}',value['head']) is not None,'authority head syntax differs')
    for k in ('request_sha256','grant_sha256','plan_sha256','agreement_sha256','source_proof_sha256','qualification_sha256','floating_policy_sha256'):V._sha(value[k])
def verify_cases(directory,request_path,*,request_sha256,manifest_sha256,complete_sha256,expected_authority=None):
    directory=ordinary(directory,True);ordinary(request_path)
    identities={}
    def pinned(path,limit,expected):
        V._sha(expected);before=V._identity(path);data=read(path,limit)
        need(sha256(data).hexdigest()==expected,'external metadata pin differs');identities[path]=before
        return decoded(data),data
    request,rb=pinned(Path(request_path),131072,request_sha256)
    keys(request,('schema','evidence_mode','layout','role','runtime','source','tables'))
    actual=request['evidence_mode']=='actual_archival_cache'
    need((request['schema'],request['evidence_mode']) in (
        ('generated_catalog_case_request_v1','generated'),('archive_catalog_case_request_v1','actual_archival_cache')),'request schema differs')
    need(request['layout'] in ('ground-only','mixed') and request['role'] in ('primary','independent') and request['runtime'] in ('1.12.6','1.13.1'),'request context differs')
    keys(request['source'],('directory','manifest_sha256','complete_sha256'))
    # This path is metadata only. In particular, never resolve or open it here.
    need(isinstance(request['source']['directory'],str) and 1<=len(request['source']['directory'])<=4096,'source path text differs')
    V._sha(request['source']['manifest_sha256']);V._sha(request['source']['complete_sha256'])
    manifest,mb=pinned(directory/'manifest.json',1048576,manifest_sha256)
    marker,cb=pinned(directory/'COMPLETE.json',4096,complete_sha256)
    metadata_keys={'schema','request_sha256','source','runtime','role','layout','tables','members','actual_source_authority_granted','provider_receipts_available','actual_fits'}
    if actual:metadata_keys.add('authority')
    keys(manifest,metadata_keys);keys(marker,('schema','manifest_sha256','members','snapshot_bytes_without_marker','actual_fits'))
    stem='archive' if actual else 'generated'
    need(manifest['schema']==stem+'_catalog_case_artifact_v1' and marker['schema']==stem+'_catalog_case_complete_v1','artifact schema differs')
    need(manifest['request_sha256']==request_sha256 and marker['manifest_sha256']==manifest_sha256 and manifest['source']==request['source'],'artifact request/source differs')
    need(all(manifest[k]==request[k] for k in ('runtime','role','layout')),'artifact native context differs')
    need(manifest['actual_source_authority_granted'] is actual and manifest['provider_receipts_available'] is False,'artifact evidence flags differ')
    integer(manifest['actual_fits'],0);integer(marker['actual_fits'],0)
    if actual:
        _authority(manifest['authority'],request,request_sha256)
        need(isinstance(expected_authority,dict) and manifest['authority']==expected_authority,'actual feature authority binding differs')
    else:need(expected_authority is None,'generated artifact actual authority forbidden')
    entries=manifest['members'];need(isinstance(entries,list) and 1<=len(entries)<=1022,'artifact member count differs')
    descriptors={};total=len(mb)
    for entry in entries:
        keys(entry,('name','bytes','sha256'));name=entry['name']
        need(isinstance(name,str) and re.fullmatch(r'table-\d\d-(?:features|labels)-\d{4}\.bin',name) is not None and name not in descriptors,'artifact member name differs')
        count=integer(entry['bytes'],8388608,1);V._sha(entry['sha256']);total+=count
        need(total<=1073741824,'artifact cumulative bytes exceed bound');descriptors[name]=entry
    names=sorted(p.name for p in directory.iterdir())
    need(set(names)=={'manifest.json','COMPLETE.json',*descriptors},'artifact fileset differs')
    need(integer(marker['members'],1024)==len(names) and integer(marker['snapshot_bytes_without_marker'],1073741824)==total,'artifact member/byte count differs')
    need(isinstance(request['tables'],list) and 1<=len(request['tables'])<=16 and isinstance(manifest['tables'],list) and len(manifest['tables'])==len(request['tables']),'artifact table count differs')
    used=set();reports=[];allrows=0;ids=set();splits=_splits()
    for index,(declaration,table) in enumerate(zip(request['tables'],manifest['tables']),1):
        keys(declaration,('id','station','start_inclusive','stop_exclusive','label_asof'))
        name=declaration['id'];need(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) is not None and name.lower() not in ids,'table identity differs');ids.add(name.lower())
        need(declaration['station'] in ('CMO','FRD'),'table station differs')
        start,stop,asof=(_millis(declaration[k]) for k in ('start_inclusive','stop_exclusive','label_asof'))
        n=(stop-start)//60000;need(start<stop<=asof and 1<=n<=600000,'table clock/row bound differs')
        allrows+=n;need(allrows<=1320000,'artifact total row bound differs')
        keys(table,('declaration','scheduled','mode','feature_payload_sha256','feature_schedule_sha256','label_payload_sha256','label_audit_sha256','counts','feature_shards','label_shards'))
        mode='ground11' if request['layout']=='ground-only' else 'mixed8'
        need(table['declaration']==declaration and integer(table['scheduled'],600000,1)==n and table['mode']==mode,'table context differs')
        for field in ('feature_payload_sha256','feature_schedule_sha256','label_payload_sha256','label_audit_sha256'):V._sha(table[field])
        for field in ('feature_shards','label_shards'):need(isinstance(table[field],list) and len(table[field])==(n+8191)//8192,'table shard count differs')
        fhash=sha256();lhash=sha256();fa=la=supervised=0;freasons=[0]*6;lreasons=[0]*5
        split_counts={k:0 for k in splits} if declaration['start_inclusive'].startswith('2017-') else {}
        for ordinal,(fn,ln) in enumerate(zip(table['feature_shards'],table['label_shards']),1):
            count=min(8192,n-(ordinal-1)*8192);payloads=[]
            for kind,filename,width,ctx in (('features',fn,160,fhash),('labels',ln,64,lhash)):
                need(filename==f'table-{index:02d}-{kind}-{ordinal:04d}.bin' and filename in descriptors and filename not in used,'table shard identity differs');used.add(filename)
                entry=descriptors[filename];need(entry['bytes']==count*width,'table shard geometry differs')
                path=directory/filename;before=V._identity(ordinary(path));data=read(path,8388608)
                need(len(data)==count*width and sha256(data).hexdigest()==entry['sha256'],'table shard bytes/hash differ')
                identities[path]=before;ctx.update(data);payloads.append(data)
            fb,lb=payloads
            for local in range(count):
                row=(ordinal-1)*8192+local;issue=start+row*60000
                f,fr,ready=_header(fb,local*160,issue,FEATURE_REASONS);l,lr,lrdy=_header(lb,local*64,issue,LABEL_REASONS)
                _numeric(fb,local*160,mode,f)
                target=struct.unpack_from('<d',lb,local*64+24)[0]
                if l:need(math.isfinite(target) and target>=0 and ready<=asof,'accepted label domain differs')
                else:need(lb[local*64+24:local*64+32]==bytes(8),'rejected label payload differs')
                need(lr!=1 and (lr==2)==(ready>asof),'full-table label readiness reason differs')
                fa+=f;la+=l;supervised+=f*l;freasons[fr]+=1;lreasons[lr]+=1
                if f and l:
                    for split,(begin,end) in splits.items():
                        if split in split_counts and begin<=issue<end and ready<end and ready<=asof:split_counts[split]+=1
        need(fhash.hexdigest()==table['feature_payload_sha256'] and lhash.hexdigest()==table['label_payload_sha256'],'table joined payload hash differs')
        counts=dict(scheduled=n,features_accepted=fa,labels_accepted=la,supervised=supervised,features_rejected=n-fa,labels_rejected=n-la)
        keys(table['counts'],counts)
        for k,v in counts.items():need(integer(table['counts'][k],600000)==v,'table recomputed count differs')
        reports.append(dict(id=name,station=declaration['station'],counts=counts,
            feature_reasons=dict(zip(FEATURE_REASONS,freasons)),label_reasons=dict(zip(LABEL_REASONS,lreasons)),
            strict_split_counts=split_counts,feature_payload_sha256=fhash.hexdigest(),label_payload_sha256=lhash.hexdigest()))
    need(used==set(descriptors),'unused artifact member differs')
    need(sorted(p.name for p in directory.iterdir())==names and all(V._identity(ordinary(p))==identity for p,identity in identities.items()),'artifact changed during verification')
    return dict(complete=True,request_sha256=request_sha256,manifest_sha256=manifest_sha256,complete_sha256=complete_sha256,
        role=request['role'],runtime=request['runtime'],layout=request['layout'],tables=reports,scheduled=sum(t['counts']['scheduled'] for t in reports),
        artifact_members=len(names),snapshot_bytes_without_marker=total,actual_source_authority_granted=actual,
        independent_numeric_source_reconstruction=False,actual_fits=0,forecast_skill_qualified=False)
