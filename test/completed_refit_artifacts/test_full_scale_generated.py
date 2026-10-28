"""Independent full-size contracts and rejection controls, without native execution."""
from pathlib import Path
import copy, hashlib, importlib.util, json, tempfile, unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('full_scale_contracts',ROOT/'tools/run_full_scale_generated.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def literal_report(runtime='1.12.6',role='primary',layout='ground-only'):
    tables=[]
    for station in ('CMO','FRD'):
        for month in ('2017-01','2023-03','2024-05','2024-10'):
            # Independent calendar facts:2017has365days;all three selected months31.
            rows=365*24*60 if month=='2017-01' else 31*24*60
            tables.append(dict(id=station+'-generated-'+month,scheduled=rows,
                features_accepted=rows-31,labels_accepted=rows-30,supervised=rows-61))
    return dict(schema='generated_case_artifact_scale_report_v1',complete=True,runtime=runtime,role=role,layout=layout,
        actual_source_access=False,actual_features=False,actual_fits=0,forecast_skill_qualified=False,
        novelty_qualified=False,artifact_members=334,peak_native_process_RSS_bytes=1024,tables=tables)

class FullScaleContractTests(unittest.TestCase):
    def test_exact_physical_and_disk_boundaries(self):
        H.admit(8589934592,17179869184)
        for memory,disk in ((8589934591,17179869184),(8589934592,17179869183),(True,17179869184),(8589934592,False)):
            with self.assertRaises(ValueError):H.admit(memory,disk)

    def test_all_eight_full_workload_contexts(self):
        for runtime in ('1.12.6','1.13.1'):
            for role in ('primary','independent'):
                for layout in ('ground-only','mixed'):
                    value=literal_report(runtime,role,layout)
                    H.validate_case_report(value,runtime,role,layout)
                    self.assertEqual(sum(table['scheduled'] for table in value['tables']),1319040)
                    self.assertEqual(sum(table['supervised'] for table in value['tables']),1318552)

    def test_compensating_count_mutation_is_rejected(self):
        value=literal_report();value['tables'][0]['supervised']-=1;value['tables'][1]['supervised']+=1
        self.assertEqual(sum(table['supervised'] for table in value['tables']),1318552)
        with self.assertRaises(ValueError):H.validate_case_report(value,'1.12.6','primary','ground-only')

    def test_wrong_clock_counts_order_and_boolean_integer_are_rejected(self):
        mutations=[lambda v:v['tables'][0].update(features_accepted=525570),
            lambda v:v['tables'].reverse(),lambda v:v.update(actual_fits=False),
            lambda v:v.update(artifact_members=True),lambda v:v.update(peak_native_process_RSS_bytes=8589934593),
            lambda v:v.update(actual_source_access=True),lambda v:v.update(forecast_skill_qualified=True)]
        for mutation in mutations:
            value=literal_report();mutation(value)
            with self.assertRaises(ValueError):H.validate_case_report(value,'1.12.6','primary','ground-only')

    def test_partial_or_other_context_does_not_pass(self):
        for key,value in (('complete',False),('runtime','1.13.1'),('role','independent'),('layout','mixed')):
            report=literal_report();report[key]=value
            with self.assertRaises(ValueError):H.validate_case_report(report,'1.12.6','primary','ground-only')

    def test_duplicate_nonfinite_and_oversized_metadata_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();path=root/'metadata.json'
            for body in (b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'{"a":1e999}\n',b'{"a":"'+b'x'*4097+b'"}\n'):
                path.write_bytes(body)
                with self.assertRaises(ValueError):H.load(path)
            path.write_bytes(b'{}\n')
            with self.assertRaises(ValueError):H.load(path,2)

    def test_output_cannot_modify_checkout_or_reuse_directory(self):
        with self.assertRaises(ValueError):H.fresh_output(ROOT/'full-size-should-never-be-created')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            with self.assertRaises(ValueError):H.fresh_output(root)
            self.assertEqual(H.fresh_output(root/'fresh'),root/'fresh')

    def test_parent_alias_cannot_hide_output_inside_checkout(self):
        alias=ROOT.parent/'..'/ROOT.parent.name/ROOT.name/'full-size-should-never-be-created'
        self.assertEqual(alias.resolve(),ROOT/'full-size-should-never-be-created')
        with self.assertRaises(ValueError):H.fresh_output(alias)

    def test_resource_inventory_rejects_escape_and_added_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'a.bin').write_bytes(b'abc');(root/'TERMINAL.json').write_text('{}')
            expected={'a.bin':dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())}
            H.check_inventory(root,expected)
            (root/'extra.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError):H.check_inventory(root,expected)
            for name in ('../escape','/absolute','bad\\name','a/./b'):
                with self.assertRaises(ValueError):H.check_inventory(root,{name:expected['a.bin']})

    def test_inventory_rejects_entry_vanishing_during_walk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'a.bin').write_bytes(b'abc');(root/'TERMINAL.json').write_text('{}')
            class WalkRace:
                def __fspath__(self):return str(root)
                def __truediv__(self,name):return root/name
                def rglob(self,pattern):return iter([root/'a.bin',root/'TERMINAL.json',root/'vanished'])
            expected={'a.bin':dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())}
            with self.assertRaises(OSError):H.check_inventory(WalkRace(),expected)

    def test_original_nine_jobs_strict_docs_and_corrected_evidence_contract(self):
        workflow=(ROOT/'.github/workflows/ci.yml').read_bytes()
        prefix=workflow.split(b'\n  full-scale-generated:',1)[0]
        self.assertEqual(hashlib.sha256(prefix).hexdigest(),'dc262831827339c1eff92ff07d1e16f1942dfb4b53659dbba1a6f117d9312768')
        self.assertEqual(workflow.count(b'\n  full-scale-generated:'),1)
        self.assertEqual(workflow.count(b'\n  full-scale-agreement:'),1)
        self.assertIn(b'needs: full-scale-generated',workflow)
        self.assertNotIn(b'continue-on-error:',workflow)

    def test_maximum_primitive_wrong_rows_rejected_before_chunks(self):
        reader=H.module(ROOT/'tools/point_input_export/resource_values.py','primitive_contract_controls')
        report=dict(complete=True,numeric_primitive_only=True,rows=599999,columns=11,input_sha256='0'*64,
            input_bytes=1,members=[],elapsed_nanoseconds=1,actual_source_access=False,actual_fits=0)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'REPORT.json').write_bytes(reader.C.V.canonical(report))
            with self.assertRaises(ValueError):reader.verify_primitive(root,report,width=11)

if __name__=='__main__':unittest.main(verbosity=2)
