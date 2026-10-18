"""Source-generated frozen data replay and rehashed corruption controls; no fit."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
TOOLS=PROJECT/'tools/initial_reference_audit'
sys.path.insert(0,str(TOOLS))
import artifact_io as io
import prepare_artifact_audit as prepare
from audit_reference import audit_reference
from owned_process import run_owned

FIXTURES=Path(__file__).resolve().parent/'fixtures'
ARCHIVE_SHA='72a16392bcc2ee9d4280481607a71f18eea8304c5460715c73f1012aa2bf0e1d'
def sha(path):return io.read_file(path,io.Limits().file_bytes)['sha256']
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def put(path,value):path.write_text(json.dumps(value,allow_nan=False,separators=(',',':')),encoding='utf-8')


def extract_fixture(output):
    manifest=load(FIXTURES/'nonconstant-reference-v1.json')
    archive=FIXTURES/'nonconstant-reference-v1.zip'
    io.need(sha(archive)==manifest['archive_sha256']==ARCHIVE_SHA,'frozen archive identity')
    io.need(manifest['synthetic_only'] is True and manifest['actual_data'] is False and
            manifest['execution_lineage_qualification'] is False,'data-only synthetic fixture required')
    with zipfile.ZipFile(archive) as source:
        entries=source.infolist();names=[entry.filename for entry in entries]
        io.need(len(entries)==177==manifest['member_count'] and len(set(names))==len(names) and
                len(set(name.casefold() for name in names))==len(names),'exact unaliased archive member set')
        io.need(set(names)==set(manifest['members_sha256']),'archive inventory differs')
        io.need(sum(entry.file_size for entry in entries)==manifest['expanded_bytes']<=40*1024**2,'expanded archive budget')
        for entry in entries:
            io.relative_parts(entry.filename)
            io.need(stat.S_ISREG(entry.external_attr>>16) and not entry.is_dir() and
                    entry.file_size<=8*1024**2,'ordinary bounded fixture member')
        output.mkdir()
        for entry in entries:
            path=output.joinpath(*io.relative_parts(entry.filename));path.parent.mkdir(parents=True,exist_ok=True)
            with source.open(entry) as stream:data=stream.read(entry.file_size+1)
            io.need(len(data)==entry.file_size and hashlib.sha256(data).hexdigest()==manifest['members_sha256'][entry.filename],
                    'expanded fixture identity differs')
            with path.open('xb') as target:io.need(target.write(data)==len(data),'short fixture extraction')
    io.need(io.inventory(output)==manifest['members_sha256'],'extracted fixture inventory differs')
    return output/'reference',output/'source'


def csv_edit(path,change):
    with path.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);names=reader.fieldnames;rows=list(reader)
    change(rows)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=names,lineterminator='\n');writer.writeheader();writer.writerows(rows)


class FullReader(unittest.TestCase):
    def test_replay_and_rehashed_corruptions(self):
        root=Path(tempfile.mkdtemp(prefix='solarsindy-independent-reader-')).resolve(strict=True)
        # Retain the entire owned tree on any failure; remove only after every
        # assertion, native stage and unchanged-input check has passed.
        try:
            reference,source=extract_fixture(root/'fixture')
            before=io.inventory(root/'fixture')
            selected=os.environ.get('SOLARSINDY_AUDIT_TEST_JULIA') or shutil.which('julia')
            self.assertTrue(selected,'Julia must be explicitly available for the numerical audit')
            julia=Path(selected).resolve(strict=True)
            native=run_owned([sys.executable,'-B',str(TOOLS/'audit_reference.py'),
                '--reference',str(reference),'--reference-sha256',sha(reference/'completion.json'),
                '--source',str(source),'--source-sha256',sha(source/'completion.json'),
                '--mode','synthetic_only','--output',str(root/'positive'),
                '--julia',str(julia),'--julia-sha256',sha(julia),'--timeout','240'],
                cwd=str(root),environment=dict(os.environ),logpath=root/'public-cli.log',timeout=300)
            self.assertEqual(native['native_exit'],0);self.assertFalse(native['timed_out'])
            result=load(root/'positive/completion.json')
            self.assertTrue(result['complete']);self.assertGreaterEqual(result['numerical']['checks'],68220)
            self.assertEqual(result['numerical']['model_points'],1616)
            self.assertEqual(result['numerical']['fallback_points'],1968)
            self.assertFalse(result['scientific_verdict']);self.assertFalse(result['execution_lineage_qualification'])
            self.assertFalse(result['serving_promotion'])
            cases=[];rejections=[]
            def create(name,mutate):
                work=root/name;work.mkdir();ref=work/'reference';shutil.copytree(reference,ref)
                completion=load(ref/'completion.json');original=load(reference/'completion.json')
                mutate(ref,completion)
                for slot,item in enumerate(completion['results']):
                    for index,record in enumerate(item['grid']):
                        for field in ('model','scaling','predictions'):
                            if record[field+'_sha256']==original['results'][slot]['grid'][index][field+'_sha256']:
                                record[field+'_sha256']=sha(ref/record[field])
                completion['output_sha256']=io.inventory(ref)
                del completion['output_sha256']['completion.json']
                put(ref/'completion.json',completion)
                try:
                    prepare.prepare(ref,sha(ref/'completion.json'),source,sha(source/'completion.json'),'synthetic_only',work/'audit')
                except (ValueError,KeyError,TypeError) as error:
                    rejections.append({'name':name,'stage':'preflight','reason':str(error)})
                else:
                    cases.append({'name':name,'plan':str(work/'audit/audit-plan.json'),'sha256':sha(work/'audit/audit-plan.json')})
            def prediction(name,change):
                create(name,lambda r,c:csv_edit(r/c['results'][0]['grid'][0]['predictions'],change))
            prediction('point',lambda rows:rows[0].update(prediction_dst_nt=str(float(rows[0]['prediction_dst_nt'])+1)))
            prediction('anchor',lambda rows:rows[0].update(anchor_dst_nt=str(float(rows[0]['anchor_dst_nt'])+1)))
            prediction('target',lambda rows:rows[0].update(observation_dst_nt=str(float(rows[0]['observation_dst_nt'])+1)))
            prediction('clock-millisecond',lambda rows:rows[0].update(assumed_target_receipt_utc=rows[0]['assumed_target_receipt_utc'].split('.')[0]+'.001'))
            prediction('wrong-source',lambda rows:rows[0].update(source_sha256='0'*64))
            prediction('missing-row',lambda rows:rows.pop())
            prediction('duplicate-row',lambda rows:rows.append(rows[-1].copy()))
            prediction('reordered-rows',lambda rows:rows.reverse())
            prediction('fallback-point',lambda rows:next(x for x in rows if x['status']=='persistence_feature_unavailable').update(prediction_dst_nt='123'))
            prediction('fallback-status',lambda rows:next(x for x in rows if x['status']=='persistence_feature_unavailable').update(status='model'))
            create('fit-future-row',lambda r,c:(r/'fit_h4.txt').write_text((r/'fit_h4.txt').read_text()+'999\n'))
            create('scaling',lambda r,c:csv_edit(r/c['results'][0]['grid'][0]['scaling'],lambda rows:rows[0].update(center=str(float(rows[0]['center'])+1))))
            create('metric',lambda r,c:c['results'][0]['grid'][0]['metric'].update(rmse_nt=999))
            create('choice-index',lambda r,c:c['results'][0]['choice'].update(selected_index=1))
            create('choice-bool',lambda r,c:c['results'][0]['choice'].update(selected_index=True))
            create('choice-configuration',lambda r,c:c['results'][0]['choice']['configuration'].update(max_depth=6))
            create('choice-expected-metric',lambda r,c:c['results'][0]['choice']['expected_metric'].update(rmse_nt=999))
            create('point-digest-binding',lambda r,c:c['results'][0]['grid'][0].update(predictions_sha256='0'*64))
            create('scale-digest-binding',lambda r,c:c['results'][0]['grid'][0].update(scaling_sha256='0'*64))
            create('grid-order',lambda r,c:c['results'][0]['grid'].reverse())
            create('a-join-key',lambda r,c:csv_edit(r/'a_join_h4.csv',lambda rows:rows[0].update(key='wrong')))
            create('a-join-missing',lambda r,c:csv_edit(r/'a_join_h4.csv',lambda rows:rows.pop()))
            create('imputed-prefix',lambda r,c:(r/'prefix_imputed17.f64').write_bytes(b'\0'*8+(r/'prefix_imputed17.f64').read_bytes()[8:]))
            create('mode-relabel',lambda r,c:c.update(evidence_mode='exposed_development_reconstruction'))
            create('bson-truncated',lambda r,c:(r/c['results'][0]['grid'][0]['model']).write_bytes(b'bad'))
            def imputer_version(ref,completion,version):
                replay=load(ref/'initial_imputation_replay.json')
                entry=next(x for x in replay['models'] if x['target']==1)
                path=ref/entry['new_model'];value=load(path);value['version']=version;put(path,value)
                entry['new_model_sha256']=sha(path);put(ref/'initial_imputation_replay.json',replay)
            create('imputer-version-bool',lambda r,c:imputer_version(r,c,[3,2,False]))
            create('imputer-version-float',lambda r,c:imputer_version(r,c,[3,2,0.]))
            self.assertEqual(len(cases)+len(rejections),27)
            put(root/'cases.json',cases)
            driver=Path(__file__).resolve().parent/'reader_corruption_cases.jl'
            env=dict(os.environ,JULIA_LOAD_PATH='@'+os.pathsep+'@stdlib',JULIA_PKG_OFFLINE='true',
                JULIA_PKG_PRECOMPILE_AUTO='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            native=run_owned([str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--depwarn=error',
                '--project='+str(PROJECT),str(driver),str(root/'cases.json'),str(root/'results.json')],
                cwd=str(root),environment=env,logpath=root/'corruptions.log',timeout=300)
            self.assertEqual(native['native_exit'],0);self.assertFalse(native['timed_out'])
            results=load(root/'results.json');self.assertEqual(len(results),len(cases))
            self.assertTrue(all(not x['accepted'] for x in results))
            # Change only an excluded raw Dst observation after the complete
            # validation prefix. Rebind its full-matrix metadata and both
            # completions; all stored initial models/points remain byte exact.
            future=root/'future-tail';future.mkdir()
            future_source=future/'source';future_reference=future/'reference'
            shutil.copytree(source,future_source);shutil.copytree(reference,future_reference)
            original=load(future_source/'completion.json')
            raw=future_source/'input_hourly_channels.f64';data=bytearray(raw.read_bytes())
            offset=(1000-1)*17*8+14*8
            value=struct.unpack_from('<d',data,offset)[0];self.assertTrue(math.isfinite(value))
            struct.pack_into('<d',data,offset,value+123)
            raw.write_bytes(data)
            self.assertNotEqual(sha(raw),original['raw_hourly_matrix_sha256'])
            metadata=load(future_source/'input_hourly_channels.json')
            metadata['raw_matrix_sha256']=sha(raw);put(future_source/'input_hourly_channels.json',metadata)
            original['raw_hourly_matrix_sha256']=sha(raw)
            for name in ('input_hourly_channels.f64','input_hourly_channels.json'):
                original['output_sha256'][name]=sha(future_source/name)
            put(future_source/'completion.json',original)
            completion=load(future_reference/'completion.json')
            completion['actual_full_raw_matrix_sha256']=sha(raw)
            completion['original_completion_sha256']=sha(future_source/'completion.json')
            put(future_reference/'completion.json',completion)
            other=audit_reference(reference=future_reference,reference_sha256=sha(future_reference/'completion.json'),
                source=future_source,source_sha256=sha(future_source/'completion.json'),mode='synthetic_only',
                output=future/'audit',julia=julia,julia_sha256=sha(julia),timeout=240)
            self.assertEqual(other['numerical']['results'],result['numerical']['results'])
            self.assertEqual(other['numerical']['model_points'],result['numerical']['model_points'])
            self.assertEqual(other['numerical']['fallback_points'],result['numerical']['fallback_points'])
            self.assertEqual(io.inventory(root/'fixture'),before)
            print(json.dumps({'independent_reader_complete':True,'rehashed_corruptions':27,
                'numerical_checks':result['numerical']['checks'],'author_or_ml_imported':False}),flush=True)
        except BaseException:
            print('Retained failed independent reader controls: '+str(root),file=sys.stderr,flush=True)
            raise
        self.assertEqual(root,root.resolve(strict=True))
        self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
        self.assertTrue(root.name.startswith('solarsindy-independent-reader-'))
        shutil.rmtree(root)


if __name__=='__main__':unittest.main()
