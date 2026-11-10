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
        import re
        workflow=(ROOT/'.github/workflows/ci.yml').read_bytes()
        prefix=workflow.split(b'\n  full-scale-generated:',1)[0]
        fixture_path=ROOT/'test/completed_refit_artifacts/workflow_extensions_v1.json'
        self.assertTrue(fixture_path.is_file() and not fixture_path.is_symlink())
        self.assertLessEqual(fixture_path.stat().st_size,65536)
        fixture_bytes=fixture_path.read_bytes()
        self.assertLessEqual(len(fixture_bytes),65536)
        self.assertEqual(hashlib.sha256(fixture_bytes).hexdigest(),
                         '98c32c023ab746fb666bdab99b2204f60f1f9396371af539bd250a0ac4a56bac')
        fixture=json.loads(fixture_bytes)
        self.assertEqual(set(fixture),{'schema','extensions'})
        self.assertEqual(fixture['schema'],'full_scale_workflow_approved_extensions_v1')
        extensions=fixture['extensions']
        self.assertEqual([item['name'] for item in extensions],
            ['Python_provisioning','hardware','matched_owner_matrix','matched_owner_canonical',
             'Pkg_thread_env','support_retention'])
        self.assertEqual(sum(len(item['replacements']) for item in extensions),7)
        # Require the exact added gates before reversing their independent
        # literals. Both original workflow hash oracles remain unchanged.
        approved=(
            b'      # Some early Julia controls invoke Python, so provision it before them.\n'
            b'      # macOS exposes the required nonreaping waitid/WNOWAIT API in this runtime.\n'
            b'      - uses: actions/setup-python@v5\n'
            b'        with:\n'
            b"          python-version: ${{ runner.os == 'macOS' && '3.14' || '3.12' }}\n")
        checkout=b'      - uses: actions/checkout@v4\n'
        julia_setup=b'      - uses: julia-actions/setup-julia@v3\n'
        old_setup=(b'      - uses: actions/setup-python@v5\n'
                   b'        with:\n'
                   b"          python-version: '3.12'\n")
        old_position=b'      - name: Verify portable source-bound point input exports\n'
        first_consumer=b'      - name: Verify cold ordinary duration API\n'
        def check(candidate):
            self.assertEqual(candidate.count(b'\n  full-scale-generated:'),1)
            current=candidate.split(b'\n  full-scale-generated:',1)[0]
            self.assertEqual(current.count(b'\n  canonical-harness:'),1)
            matrix=current.split(b'\n  canonical-harness:',1)[0]
            self.assertEqual(matrix.count(checkout),1)
            self.assertEqual(current.count(approved),1)
            self.assertEqual(matrix.count(approved),1)
            self.assertEqual(matrix.count(b'actions/setup-python@'),1)
            self.assertIn(checkout+approved+julia_setup,matrix)
            self.assertLess(matrix.index(approved),matrix.index(julia_setup))
            self.assertLess(matrix.index(approved),matrix.index(first_consumer))
            jobs=list(re.finditer(rb'(?m)^  ([A-Za-z0-9_-]+):$',current))
            sections={match[1].decode('ascii'):current[match.start():
                jobs[index+1].start() if index+1<len(jobs) else len(current)]
                for index,match in enumerate(jobs)}
            self.assertEqual(len(sections),len(jobs))
            self.assertIn('test',sections);self.assertIn('canonical-harness',sections)
            scopes={'matrix':sections['test'],'canonical':sections['canonical-harness']}
            restored=candidate
            for extension in extensions:
                self.assertEqual(set(extension),{'name','scope','anchor_before','anchor_after','replacements'})
                self.assertIn(extension['scope'],scopes)
                scope=scopes[extension['scope']]
                for replacement in extension['replacements']:
                    self.assertEqual(set(replacement),{'current','original'})
                    added=replacement['current'].encode('utf-8')
                    self.assertTrue(added)
                    self.assertEqual(candidate.count(added),1)
                    self.assertEqual(scope.count(added),1)
                    self.assertEqual(restored.count(added),1)
                    restored=restored.replace(added,replacement['original'].encode('utf-8'),1)
                first=extension['replacements'][0]['current'].encode('utf-8')
                anchored=extension['anchor_before'].encode('utf-8')+first+extension['anchor_after'].encode('utf-8')
                self.assertEqual(scope.count(anchored),1)
            restored_prefix=restored.split(b'\n  full-scale-generated:',1)[0]
            self.assertEqual(hashlib.sha256(restored_prefix).hexdigest(),'dc262831827339c1eff92ff07d1e16f1942dfb4b53659dbba1a6f117d9312768')
            # Sealed963 SOURCE_INVENTORY2fc9b7fd files_sha256['.github/workflows/ci.yml'].
            # This also freezes all eight workloads and the agreement job.
            self.assertEqual(hashlib.sha256(restored).hexdigest(),'ae8c23a3faad1a0071c8ef4fe75c9ef364f1bde62095cf2590a689471410a680')
        check(workflow)
        def changed(before,after):
            self.assertEqual(prefix.count(before),1)
            return workflow.replace(before,after,1)
        without=workflow.replace(approved,b'',1)
        before_harness,harness_and_later=without.split(b'\n  canonical-harness:',1)
        moved_other_job=before_harness+b'\n  canonical-harness:'+harness_and_later.replace(checkout,checkout+approved,1)
        suffix=workflow.split(b'\n  full-scale-generated:',1)[1]
        def changed_suffix(before,after):
            self.assertEqual(suffix.count(before),1)
            return prefix+b'\n  full-scale-generated:'+suffix.replace(before,after,1)
        mutations={
            'missing provisioning':without,
            'duplicate provisioning':workflow.replace(approved,approved+approved,1),
            'provisioning after early consumer':without.replace(old_position,approved+old_position,1),
            'provisioning in another job':moved_other_job,
            'macOS runtime':workflow.replace(approved,approved.replace(b"'3.14'",b"'3.13'"),1),
            'other OS runtime':workflow.replace(approved,approved.replace(b"'3.12'",b"'3.11'"),1),
            'Python setup action':workflow.replace(approved,approved.replace(b'setup-python@v5',b'setup-python@v4'),1),
            'extra step':changed(b'      - name: Verify the JavaScript test runtime\n',b'      - run: echo unapproved\n      - name: Verify the JavaScript test runtime\n'),
            'missing step':changed(b'      - name: Verify the JavaScript test runtime\n        run: node --version\n',b''),
            'Julia runtime grid':changed(b"julia-version: ['1.12.6', '1.13.1']",b"julia-version: ['1.12.6', '1.13.0']"),
            'fail-fast gate':changed(b'      fail-fast: false\n',b'      fail-fast: true\n'),
            'native cap':changed(b'    timeout-minutes: 180\n',b'    timeout-minutes: 181\n'),
            'checkout action':changed(checkout+approved+julia_setup,checkout.replace(b'@v4',b'@v3')+approved+julia_setup),
            'Pkg test action':changed(b'julia-actions/julia-runtest@v1',b'julia-actions/julia-runtest@v0'),
            'complete experiments':changed(b'examples/experiments.jl',b'examples/incomplete.jl'),
            'canonical harness command':changed(b'python -B tools/run_canonical_harness.py --evidence-dir',b'python -B tools/other_harness.py --evidence-dir'),
            'strict docs runtime assertion':changed(b'VERSION == v"1.13.1"',b'VERSION == v"1.13.0"'),
            'required output assertion':changed(b'            exit 1\n',b'            exit 0\n'),
            'full-scale native cap':changed_suffix(b'    timeout-minutes: 420\n',b'    timeout-minutes: 421\n'),
            'agreement native cap':changed_suffix(b'    timeout-minutes: 180\n',b'    timeout-minutes: 181\n'),
            'full-scale command':changed_suffix(b'tools/run_full_scale_generated.py --mode workload',b'tools/run_full_scale_generated.py --mode preview'),
            'agreement command':changed_suffix(b"--mode agreement --runtime '1.13.1'",b"--mode agreement --runtime '1.12.6'"),
            'agreement artifact action':changed_suffix(b'actions/download-artifact@v4',b'actions/download-artifact@v3'),
            'agreement dependency':changed_suffix(b'    needs: full-scale-generated\n',b'    needs: test\n'),
            'disabled full-scale job':workflow.replace(b'\n  full-scale-generated:\n',b'\n  full-scale-generated:\n    if: false\n',1),
            'disabled agreement job':changed_suffix(b'\n  full-scale-agreement:\n',b'\n  full-scale-agreement:\n    if: false\n'),
            'full-scale role grid':changed_suffix(b'        role: [primary, independent]\n',b'        role: [primary]\n'),
            'full-scale layout grid':changed_suffix(b'        layout: [ground-only, mixed]\n',b'        layout: [ground-only]\n'),
            'full-scale raw retention guard':changed_suffix(b'      - name: Retain full-size generated outputs and every raw stage log\n        if: always()\n',b'      - name: Retain full-size generated outputs and every raw stage log\n        if: success()\n'),
        }
        blocks={item['name']:item['replacements'][0]['current'].encode('utf-8') for item in extensions}
        for key,block in blocks.items():
            mutations['missing extension '+key]=workflow.replace(block,b'',1)
            mutations['duplicated extension '+key]=workflow.replace(block,block+block,1)
        def extension_change(key,before,after):
            block=blocks[key]
            self.assertIn(before,block)
            return workflow.replace(block,block.replace(before,after,1),1)
        matrix_owner=blocks['matched_owner_matrix']
        canonical_owner=blocks['matched_owner_canonical']
        hardware=blocks['hardware']
        no_matrix_owner=workflow.replace(matrix_owner,b'',1)
        no_canonical_owner=workflow.replace(canonical_owner,b'',1)
        no_hardware=workflow.replace(hardware,b'',1)
        mutations.update({
            'matrix owner in canonical job':no_matrix_owner.replace(canonical_owner,matrix_owner+canonical_owner,1),
            'canonical owner in matrix job':no_canonical_owner.replace(matrix_owner,canonical_owner+matrix_owner,1),
            'matrix owner after Pkg.test':no_matrix_owner.replace(blocks['Pkg_thread_env'],blocks['Pkg_thread_env']+matrix_owner,1),
            'canonical owner after harness command':no_canonical_owner.replace(
                b'        run: python -B tools/run_canonical_harness.py --evidence-dir "$RUNNER_TEMP/canonical-harness"\n',
                b'        run: python -B tools/run_canonical_harness.py --evidence-dir "$RUNNER_TEMP/canonical-harness"\n'+canonical_owner,1),
            'hardware moved after checkout':no_hardware.replace(checkout,checkout+hardware,1),
            'hardware uses logical low RAM runner':extension_change('hardware',
                b"${{ matrix.os == 'macos-latest' && 'macos-26-intel' || matrix.os }}",b'${{ matrix.os }}'),
            'hardware wrong macOS target':extension_change('hardware',b'macos-26-intel',b'macos-15-intel'),
            'hardware all Ubuntu':extension_change('hardware',
                b"${{ matrix.os == 'macos-latest' && 'macos-26-intel' || matrix.os }}",b'ubuntu-latest'),
            'hardware conditional changes Windows':extension_change('hardware',b"matrix.os == 'macos-latest'",b"matrix.os == 'windows-latest'"),
            'Pkg thread count':extension_change('Pkg_thread_env',b"JULIA_NUM_THREADS: '1'",b"JULIA_NUM_THREADS: '2'"),
            'support catchall omitted':extension_change('support_retention',b'            ${{ runner.temp }}/matched342-*\n',b''),
            'support exclusion broadened':extension_change('support_retention',b'/test_native_predicates_reject_modified_observations/oversize.raw',b'/**'),
        })
        for key in ('matched_owner_matrix','matched_owner_canonical'):
            for label,before,after in (
                ('contract command',b'profile_owner_tests.py',b'profile_owner_missing.py'),
                ('lifecycle command',b'test_native_owner_lifecycle.py',b'profile_owner_tests.py'),
                ('interpreter isolation',b'python -I -B -X utf8',b'python -B -X utf8'),
                ('oversize byte note',b'fixture_bytes=67108865',b'fixture_bytes=67108864'),
                ('exclusion note',b'excluded_path=matched342-owner-contracts/',b'excluded_path=other/'),
                ('hardlink scope note',b'artifacts retain file bytes, not native hardlink relationships',b'artifacts prove native hardlink relationships'),
                ('ordinary retention note',b'ordinary native evidence is retained unchanged',b'ordinary native evidence may be omitted'),
                ('always notes',b'Record the deliberate owner artifact omission and byte scope\n        if: always()',b'Record the deliberate owner artifact omission and byte scope\n        if: success()'),
                ('always artifact',b'Retain complete matched342 owner engineering evidence\n        if: always()',b'Retain complete matched342 owner engineering evidence\n        if: success()'),
                ('artifact root',b'${{ runner.temp }}/matched342-owner-contracts\n',b'${{ runner.temp }}/other\n'),
                ('artifact exclusion',b'/test_native_predicates_reject_modified_observations/oversize.raw\n',b'/**\n'),
                ('artifact name',b'name: matched342-owner-',b'name: other-owner-'),
                ('missing artifact rejection',b'if-no-files-found: error',b'if-no-files-found: ignore'),
                ('hidden evidence',b'include-hidden-files: true',b'include-hidden-files: false'),
                ('compression',b'compression-level: 6',b'compression-level: 0')):
                mutations[key+' '+label]=extension_change(key,before,after)
        for name,candidate in mutations.items():
            with self.subTest(unapproved_change=name):
                with self.assertRaises(AssertionError):check(candidate)
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
