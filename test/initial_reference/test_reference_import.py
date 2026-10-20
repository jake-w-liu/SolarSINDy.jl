"""Generated serialized reference import, independent rows and rehashed failures.

The perfect candidate points below are a test oracle for selection wiring only.
They are neither fitted forecasts nor evidence of predictive performance.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_full_reader import PROJECT,TOOLS,extract_fixture,sha,load,put,csv_edit,io,run_owned

IMPORT=PROJECT/'tools/initial_reference_import'


class ReferenceImport(unittest.TestCase):
    def test_serialized_union_and_rehashed_failures(self):
        root=Path(tempfile.mkdtemp(prefix='solarsindy-reference-import-')).resolve(strict=True)
        try:
            reference,source=extract_fixture(root/'fixture')
            before=io.inventory(root/'fixture')
            selected=os.environ.get('SOLARSINDY_AUDIT_TEST_JULIA') or shutil.which('julia')
            self.assertTrue(selected,'explicit Julia required for independent replay')
            julia=Path(selected).resolve(strict=True)
            def command(argv,log,timeout=300):
                native=run_owned(argv,cwd=str(root),environment=dict(os.environ,JULIA_PKG_OFFLINE='true',
                    JULIA_LOAD_PATH='@'+os.pathsep+'@stdlib',JULIA_PKG_PRECOMPILE_AUTO='0',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),logpath=root/log,timeout=timeout)
                self.assertEqual(native['native_exit'],0,(root/log).read_text(encoding='utf-8')[-12000:])
                self.assertFalse(native['timed_out'])
            command([sys.executable,'-B',str(TOOLS/'audit_reference.py'),
                '--reference',str(reference),'--reference-sha256',sha(reference/'completion.json'),
                '--source',str(source),'--source-sha256',sha(source/'completion.json'),
                '--mode','synthetic_only','--output',str(root/'audit'),
                '--julia',str(julia),'--julia-sha256',sha(julia),'--timeout','240'],'audit.log')
            audit_sha=sha(root/'audit/completion.json')
            command([sys.executable,'-B',str(IMPORT/'export_reference.py'),
                '--audit',str(root/'audit'),'--audit-sha256',audit_sha,'--output',str(root/'export')],'export.log')
            original=load(root/'export/completion.json')
            self.assertEqual(original['row_count'],226)
            self.assertEqual(original['cohort_counts'],[224,222])
            self.assertFalse(original['new_fitting']);self.assertFalse(original['scientific_verdict'])
            joined=[]
            for h in (4,6):
                with (reference/f'a_join_h{h}.csv').open(newline='',encoding='utf-8') as stream:
                    joined.append(list(csv.DictReader(stream)))
            expected={'counts':[224,222],'storm_counts':[93,91],
                'issues':[[r['issue'].split('.')[0] for r in rows] for rows in joined],
                'predictions':[[float(r['prediction']) for r in rows] for rows in joined]}
            self.assertEqual([len(rows) for rows in joined],expected['counts'])
            self.assertEqual([sum(float(r['target'])<=-50 for r in rows) for rows in joined],expected['storm_counts'])
            cases=[{'name':'positive','directory':str(root/'export'),'sha256':sha(root/'export/completion.json'),
                'audit_sha256':audit_sha,'expected':expected}]
            def create(name,change=None,meta_change=None):
                directory=root/name;shutil.copytree(root/'export',directory)
                meta=load(directory/'completion.json')
                if change:csv_edit(directory/'selection_rows.csv',change)
                meta['output_sha256']['selection_rows.csv']=sha(directory/'selection_rows.csv')
                if meta_change:meta_change(directory,meta)
                put(directory/'completion.json',meta)
                cases.append({'name':name,'directory':str(directory),'sha256':sha(directory/'completion.json'),
                    'audit_sha256':audit_sha,'expected':expected})
            create('relocated-output')
            def field(name,key,value):create(name,lambda rows:rows[0].update({key:value}))
            field('point','h4_prediction','123456')
            field('anchor','anchor','123456')
            field('target','h4_target','123456')
            field('wrong-source','source_sha256','0'*64)
            field('wrong-delay','dst_delay_hours','1')
            field('row-index-bool','origin_row','true')
            field('driver-clock','driver_receipt',expected['issues'][0][0]+'.001')
            field('label-clock','h6_target_receipt','2020-01-01T00:00:00.001')
            field('scoreable','h4_scoreable','0')
            field('eligibility','h4_selection_eligible','0')
            create('missing-row',lambda rows:rows.pop())
            create('duplicate-row',lambda rows:rows.append(rows[-1].copy()))
            create('reordered-rows',lambda rows:rows.reverse())
            create('fallback-point',lambda rows:next(r for r in rows if r['h4_status']=='persistence_feature_unavailable').update(h4_prediction='123456'))
            create('fallback-status',lambda rows:next(r for r in rows if r['h4_status']=='persistence_feature_unavailable').update(h4_status='model'))
            create('opposite-slot-point',lambda rows:next(r for r in rows if r['h6_status']=='outside_selection_cohort').update(h6_prediction='123456'))
            create('opposite-slot-eligibility',lambda rows:next(r for r in rows if r['h6_status']=='outside_selection_cohort').update(h6_selection_eligible='1'))
            for name,key,value in (('row-count-bool','row_count',True),('prefix-count-bool','prefix_rows',True),
                    ('delay-bool','delay_hours',False),('mode-relabel','evidence_mode','exposed_development_reconstruction'),
                    ('fitting-claim','new_fitting',True),('verdict-claim','scientific_verdict',True),
                    ('source-metadata','source_sha256','0'*64),('audit-trust-anchor','audit_completion_sha256','0'*64)):
                create(name,meta_change=lambda d,m,k=key,v=value:m.update({k:v}))
            create('cohort-count-bool',meta_change=lambda d,m:m.update(cohort_counts=[True,222]))
            create('missing-upstream-receipt',meta_change=lambda d,m:m['upstream_sha256'].pop(next(iter(m['upstream_sha256']))))
            create('changed-importer-identity',meta_change=lambda d,m:m['implementation_sources'].update(
                {'tools/initial_reference_import/InitialReferenceImport.jl':'0'*64}))
            create('unexpected-output',meta_change=lambda d,m:(d/'unexpected.txt').write_text('unexpected',encoding='utf-8'))
            create('oversized-record',meta_change=lambda d,m:((d/'selection_rows.csv').write_text('x'*4097+'\n',encoding='utf-8'),
                m['output_sha256'].update({'selection_rows.csv':sha(d/'selection_rows.csv')})))
            create('post-load-mutation')
            put(root/'cases.json',cases)
            command([str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--depwarn=error',
                '--project='+str(PROJECT),str(Path(__file__).with_name('reference_import_cases.jl')),
                str(root/'cases.json'),str(root/'results.json')],'cases.log',600)
            results=load(root/'results.json')
            self.assertEqual(len(results),len(cases))
            self.assertEqual([r['name'] for r in results if r['accepted']],['positive','relocated-output'])
            self.assertTrue(all(r['checks']==18 for r in results if r['accepted']))
            # Simulate an upstream change during CSV serialization: publication
            # must fail before writing completion, even though preflight passed.
            spec=importlib.util.spec_from_file_location('reference_export_failure_oracle',IMPORT/'export_reference.py')
            exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
            write=exporter.write_new;plan=root/'audit/prepared/audit-plan.json';saved=plan.read_bytes()
            def mutation(path,data):
                result=write(path,data)
                if path.name=='selection_rows.csv':plan.write_bytes(saved+b' ')
                return result
            exporter.write_new=mutation
            try:
                with self.assertRaisesRegex(ValueError,'upstream proof changed during serialization'):
                    exporter.export_reference(audit=root/'audit',audit_sha256=audit_sha,output=root/'failed-export')
            finally:plan.write_bytes(saved)
            self.assertFalse((root/'failed-export/completion.json').exists())
            self.assertEqual(io.inventory(root/'fixture'),before)
        except BaseException:
            print('FAILED_IMPORT_EVIDENCE_ROOT='+str(root),file=sys.stderr)
            raise
        else:
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve(strict=True))
            self.assertTrue(root.name.startswith('solarsindy-reference-import-'))
            io.inventory(root) # reject links and over-budget cleanup targets
            shutil.rmtree(root)


if __name__=='__main__':unittest.main()
