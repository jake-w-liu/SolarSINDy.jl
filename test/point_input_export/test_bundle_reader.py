"""Literal data-only case/input oracles; no native exporter or source loader.

Opaque source schedule/audit digests stay externally pinned metadata. These
fixtures test complete record/input binding, not source reconstruction.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from decimal import Decimal,localcontext
from unittest import mock
from types import SimpleNamespace
import hashlib,importlib.util,json,math,stat,struct,tempfile,unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('portable_point_input_bundle',ROOT/'tools/point_input_export/bundle_reader.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(body):return hashlib.sha256(body).hexdigest()
def clock(minute):return (datetime(2020,1,1)+timedelta(minutes=minute)).isoformat(timespec='milliseconds')+'Z'

def fixture(root,role='primary',layout='ground-only',runtime='1.12.6',corrupt=None):
    cases=root/'cases';cases.mkdir();bundle=root/'input';bundle.mkdir()
    width=11 if layout=='ground-only' else 8;features=[];labels=[];epoch=1577836800000
    for k in range(121):
        feature=bytearray(160);label=bytearray(64);accepted=k not in (37,52);label_ok=k!=41
        struct.pack_into('<qqBB',feature,0,epoch+k*60000,epoch+k*60000+1845000,int(accepted),0 if accepted else 1)
        struct.pack_into('<qqBB',label,0,epoch+k*60000,epoch+k*60000+1845000,int(label_ok),0 if label_ok else 3)
        if accepted:
            raw=5.;scale=1+math.log1p(10.)
            values=([math.log1p(5.),math.log1p(3.),math.log1p(10.),1.,.25,.5,.75,0.,1.,-1.,0.]
                if width==11 else [raw,1.,2.,3.,400.,-2.,2.,800.])
            struct.pack_into('<dd',feature,24,raw,scale)
            for j,value in enumerate(values):struct.pack_into('<d',feature,40+8*j,value)
        if label_ok:struct.pack_into('<d',label,24,5. if k%2 else 10.)
        features.append(bytes(feature));labels.append(bytes(label))
    fb=b''.join(features);lb=b''.join(labels)
    source=dict(directory='never-opened-source-root',manifest_sha256=sha(b'fixed source manifest'),complete_sha256=sha(b'fixed source completion'))
    declaration=dict(id='CMO-literal',station='CMO',start_inclusive=clock(0),stop_exclusive=clock(121),label_asof=clock(151))
    request=dict(schema='generated_catalog_case_request_v1',evidence_mode='generated',layout=layout,role=role,runtime=runtime,source=source,tables=[declaration])
    request_bytes=encode(request);request_path=root/'case-request.json';request_path.write_bytes(request_bytes)
    entries=[]
    for name,body in (('table-01-features-0001.bin',fb),('table-01-labels-0001.bin',lb)):
        (cases/name).write_bytes(body);entries.append(dict(name=name,bytes=len(body),sha256=sha(body)))
    table=dict(declaration=declaration,scheduled=121,mode='ground11' if width==11 else 'mixed8',feature_payload_sha256=sha(fb),
        feature_schedule_sha256=sha(b'opaque fixed feature schedule'),label_payload_sha256=sha(lb),label_audit_sha256=sha(b'opaque fixed label audit'),
        counts=dict(scheduled=121,features_accepted=119,labels_accepted=120,supervised=118,features_rejected=2,labels_rejected=1),
        feature_shards=[entries[0]['name']],label_shards=[entries[1]['name']])
    manifest=dict(schema='generated_catalog_case_artifact_v1',request_sha256=sha(request_bytes),source=source,runtime=runtime,role=role,
        layout=layout,tables=[table],members=entries,actual_source_authority_granted=False,provider_receipts_available=False,actual_fits=0)
    mb=encode(manifest);(cases/'manifest.json').write_bytes(mb)
    marker=encode(dict(schema='generated_catalog_case_complete_v1',manifest_sha256=sha(mb),members=4,snapshot_bytes_without_marker=len(mb)+len(fb)+len(lb),actual_fits=0))
    (cases/'COMPLETE.json').write_bytes(marker)
    # Independently specified strict membership; minute60's label is ready at
    # minute90:45 and must be excluded from a minute90 split stop.
    selected=[k for k in range(35,60) if k not in (37,41,52)];assert len(selected)==22
    rata_die=datetime(1970,1,1).toordinal()*86400000
    parts=[b'SOURCE_BOUND_POINT_INPUT_V1\n',struct.pack('<QQ',22,width),struct.pack('<22Q',*[k+1 for k in selected]),
        struct.pack('<22q',*[rata_die+epoch+k*60000 for k in selected])]
    for j in range(width):parts.append(b''.join(features[k][40+8*j:48+8*j] for k in selected))
    parts.append(b''.join(labels[k][24:32] for k in selected))
    with localcontext() as context:
        context.prec=100;logs=[float(Decimal(6 if k%2 else 11).ln()) for k in selected]
    parts.extend((struct.pack('<22d',*logs),b''.join(features[k][24:32] for k in selected),b''.join(features[k][32:40] for k in selected)))
    data=b''.join(parts)
    if corrupt=='feature':
        changed=bytearray(data);struct.pack_into('<d',changed,len(M.V.PREFIX)+16+16*22,123456.);data=bytes(changed)
    entry=sha((ROOT/'research/dbdt_completed_source/point_input_export.jl').read_bytes())
    args=dict(case_directory=cases,case_request_path=request_path,case_request_sha256=sha(request_bytes),case_manifest_sha256=sha(mb),case_complete_sha256=sha(marker),
        table_index=1,start=clock(35),stop=clock(90),asof=clock(120),entry_sha256=entry,numeric_source_sha256=sha(b'fixed generated numeric source'),library_sha256=sha(b'fixed generated library'))
    planning=dict(schema='source_bound_point_input_planning_request_v1',evidence_mode='generated',case_request_sha256=args['case_request_sha256'],
        case_manifest_sha256=args['case_manifest_sha256'],case_complete_sha256=args['case_complete_sha256'],table_index=1,table=declaration,runtime=runtime,role=role,layout=layout,
        start_inclusive=args['start'],stop_exclusive=args['stop'],target_asof=args['asof'],entry_sha256=entry,numeric_source_sha256=args['numeric_source_sha256'],
        library_sha256=args['library_sha256'],feature_payload_sha256=sha(fb),label_payload_sha256=sha(lb),actual_model_fit_authorized=False,provider_receipts_available=False)
    context=dict(schema='source_bound_point_input_export_context_v1',request=planning,rows=22,columns=width,first_row=36,last_row=60,input_sha256=sha(data),input_bytes=len(data),
        strict_label_ready_before=args['stop'],target_transform='log1p_rate_reference_1_nt_per_min',availability_kind='assumed_completed_source_intervals',measured_receipts_available=False)
    if corrupt=='rows':context['rows']-=1
    if corrupt=='request_index':context['request']['table_index']=True
    cb=encode(context);receipt=dict(schema='source_bound_point_input_export_receipt_v1',complete=True,context_sha256=sha(cb),input_sha256=sha(data),
        actual_source_access=False,actual_model_fit_performed=False,fit_authority_granted=False,provider_receipts_available=False,software_qualification=False)
    if corrupt=='fit_authority':receipt['fit_authority_granted']=True
    rb=encode(receipt);members=[]
    for name,body in (('context.json',cb),('receipt.json',rb),('input-0001.bin',data)):
        (bundle/name).write_bytes(body);members.append(dict(name=name,bytes=len(body),sha256=sha(body)))
    if corrupt=='member_bool':members[0]['bytes']=True
    manifest=encode(dict(schema='source_bound_point_input_export_manifest_v1',context_sha256=sha(cb),input_sha256=sha(data),input_bytes=len(data),chunk_bytes=8388608,chunks=1,members=members))
    marker=encode(dict(schema='source_bound_point_input_export_complete_v1',manifest_sha256=sha(manifest),context_sha256=sha(cb),receipt_sha256=sha(rb),input_sha256=sha(data),members=5,actual_fits=0))
    (bundle/'manifest.json').write_bytes(manifest);(bundle/'COMPLETE.json').write_bytes(marker)
    if corrupt=='extra_file':(bundle/'extra.bin').write_bytes(b'x')
    if corrupt=='missing_chunk':(bundle/'input-0001.bin').unlink()
    args.update(manifest_sha256=sha(manifest),complete_sha256=sha(marker),context_sha256=sha(cb),receipt_sha256=sha(rb),input_sha256=sha(data))
    return bundle,args

class BundleReaderTests(unittest.TestCase):
    def test_all_eight_literal_generated_contexts(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-bundle-') as directory:
            root=Path(directory).resolve()
            for runtime in ('1.12.6','1.13.1'):
                for role in ('primary','independent'):
                    for layout in ('ground-only','mixed'):
                        base=root/(runtime+'-'+role+'-'+layout);base.mkdir();path,args=fixture(base,role,layout,runtime)
                        result=M.verify_bundle(path,**args)
                        self.assertEqual((result['rows'],result['first_row'],result['last_row']),(22,36,60))
                        self.assertEqual(result['scheduled_rows_scanned'],121);self.assertTrue(result['all_members_bounded'])
                        self.assertFalse(result['actual_source_access']);self.assertFalse(result['fit_authority_granted'])
    def test_rehashed_semantic_and_file_corruptions(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-corruption-') as directory:
            root=Path(directory).resolve()
            for name in ('feature','rows','request_index','fit_authority','member_bool','extra_file','missing_chunk'):
                base=root/name;base.mkdir();path,args=fixture(base,corrupt=name)
                with self.subTest(name=name),self.assertRaises(ValueError):M.verify_bundle(path,**args)
    def test_external_pins_and_generated_actual_permissions(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-pins-') as directory:
            path,args=fixture(Path(directory).resolve())
            for name in ('manifest_sha256','complete_sha256','context_sha256','receipt_sha256','input_sha256','entry_sha256','numeric_source_sha256'):
                changed=dict(args);changed[name]='1'*64
                with self.subTest(name=name),self.assertRaises(ValueError):M.verify_bundle(path,**changed)
            with self.assertRaises(ValueError):M.verify_bundle(path,**args,expected_planning_authorization={})
    def test_source_catalogue_path_remains_metadata_only(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-reads-') as directory:
            root=Path(directory).resolve();path,args=fixture(root);original=Path.open;reads=[]
            def checked_open(value,*rest,**kwargs):
                self.assertTrue(value.is_relative_to(root) or value==ROOT/'tools/point_input_export/bundle_reader.py');reads.append(value)
                return original(value,*rest,**kwargs)
            with mock.patch.object(Path,'open',checked_open):result=M.verify_bundle(path,**args)
            self.assertTrue(result['complete']);self.assertTrue(reads)
    def test_linked_input_member_rejected(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-link-') as directory:
            root=Path(directory).resolve();path,args=fixture(root);member=path/'input-0001.bin';saved=root/'original.bin';member.replace(saved)
            try:member.symlink_to(saved)
            except OSError:
                saved.replace(member);original=Path.lstat
                # Filesystems without unprivileged symlinks still exercise the
                # Windows reparse rejection using its native attribute value.
                def reparse(value,*rest,**kwargs):
                    info=original(value,*rest,**kwargs)
                    return SimpleNamespace(st_mode=stat.S_IFREG,st_file_attributes=0x400) if value==member else info
                with mock.patch.object(Path,'lstat',reparse),self.assertRaises(ValueError):M.verify_bundle(path,**args)
            else:
                with self.assertRaises(ValueError):M.verify_bundle(path,**args)
if __name__=='__main__':unittest.main(verbosity=2)
