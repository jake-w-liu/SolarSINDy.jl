"""Independent metadata-only controls; mocked handoffs are NEVER native proof.

These fixtures create no VerifiedCases, grant file, actual model or native-stage
receipt claiming an executed fit. The fixed driver's public admission is not
weakened; only its pure header validator is used on literal request geometry.
"""
from pathlib import Path
from collections import Counter
import copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
PYTHON=str(Path(sys.executable).resolve())
ENTRY=ROOT/'research/dbdt_completed_source/run_candidate_campaign.py'
spec=importlib.util.spec_from_file_location('candidate_campaign_metadata_controls',ENTRY)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
L=json.loads(Path(__file__).with_name('descriptor_literals.json').read_bytes())
def sha(label):return hashlib.sha256(label.encode()).hexdigest()
def authority(label):
    return {prefix+suffix:('/UNQUALIFIED_GENERATED_METADATA/'+label+'/'+prefix+'.json' if suffix=='_path' else sha(label+prefix))
        for prefix in ('grant','source_proof','qualification') for suffix in ('_path','_sha256')}
def configuration(root):
    a=authority('case-reader');producer={prefix+suffix:('/UNQUALIFIED_GENERATED_METADATA/'+prefix+'.json' if suffix=='_path' else sha(prefix))
        for prefix in ('grant','plan','agreement','source_proof','qualification','floating_policy') for suffix in ('_path','_sha256')}
    case=dict(request_path='/UNQUALIFIED_GENERATED_METADATA/request.json',directory='/UNQUALIFIED_GENERATED_METADATA/cases',
        request_sha256=sha('request'),manifest_sha256=sha('manifest'),complete_sha256=sha('complete'),producer=producer,reader=a,admission_sha256=sha('admission'))
    return dict(root=str(root),source_proof=dict(path='/UNQUALIFIED_GENERATED_METADATA/proof.json',sha256=sha('proof')),
        qualification=dict(path='/UNQUALIFIED_GENERATED_METADATA/qualification.json',sha256=sha('qualification')),
        julia=dict(path='/UNQUALIFIED_GENERATED_METADATA/julia',sha256=sha('julia')),runtime=dict(path='/UNQUALIFIED_GENERATED_METADATA/runtime.json',sha256=sha('runtime')),
        stations={station:dict(case=copy.deepcopy(case),table_index=i+1,admission_metadata=dict(path='/UNQUALIFIED_GENERATED_METADATA/admission.json',sha256=sha('admission')))
            for i,station in enumerate(('CMO','FRD'))})
def fake_outputs(stage):
    """Request-builder placeholders only; no file or native receipt is published."""
    label=stage['key'];directory='/UNQUALIFIED_GENERATED_METADATA/'+stage['directory_name']
    if stage['operation'] in ('candidate-input','selected-refit-input'):
        return dict(input=dict(directory=directory,pins={k:sha(label+k) for k in
            ('descriptor_sha256','summary_sha256','receipt_sha256','manifest_sha256','complete_sha256')}))
    if stage['operation'] in ('candidate-fit','selected-refit'):
        return dict(fit=dict(directory=directory,expected={k:sha(label+k) for k in ('context_sha256','recipe_sha256','source_sha256','library_sha256')},
            pins=dict({k:sha(label+k) for k in ('context_sha256','recipe_sha256','model_sha256','receipt_sha256','complete_sha256')},actual_fit_performed=False)),
            point={k:sha(label+k) for k in ('context_sha256','recipe_sha256','model_sha256','receipt_sha256','source_sha256')})
    if stage['operation']=='candidate-validation':return dict(validation=dict(directory=directory,pins={k:sha(label+k) for k in
        ('request_sha256','context_sha256','metric_sha256','receipt_sha256','manifest_sha256','complete_sha256')}))
    if stage['operation']=='independent-verify':return dict(witness=dict(directory=directory,pins={k:sha(label+k) for k in ('manifest_sha256','complete_sha256')}))
    return dict(selection=dict(selection_evidence_sha256=sha(label),family_winners=[dict(family='ridge',candidate_id='ridge-1'),
        dict(family='boost',candidate_id='boost-3-100'),dict(family='forest',candidate_id='forest-9-64')]))
def trace():
    config=configuration(Path('/UNQUALIFIED_GENERATED_METADATA/campaign'));done={};requests=[]
    grants={stage['key']+'#'+slot:authority(stage['directory_name']+slot) for stage in C.QUEUE for slot in C.authority_slots(stage)}
    for stage in C.QUEUE:
        req,missing=C.build_request(config,stage,done,grants);assert not missing
        C.D.validate_headers(req)
        native={key:('/UNQUALIFIED_GENERATED_METADATA/'+stage['directory_name']+'/'+key if key.endswith('_path') else sha(stage['key']+key))
            for key in ('receipt_path','receipt_sha256','raw_path','raw_sha256')}
        done[stage['key']]=dict(stage=stage,request=req,outputs=fake_outputs(stage),native=native,receipt=dict(internal_result_sha256=sha(stage['key']+'internal')))
        requests.append(req)
    return config,done,grants,requests
def actual_header_fixture(root,change=None):
    """Self-consistent intent headers, never qualified data or source grants."""
    def meta(name,value):
        path=root/name;C.write(path,value);return dict(path=str(path),sha256=C.pin(path))
    tables=[dict(id=station+'-unqualified',station=station,start_inclusive='2017-01-01T00:00:00.000Z',
        stop_exclusive='2017-10-01T00:00:00.000Z',label_asof='2018-01-01T00:00:00.000Z') for station in ('CMO','FRD')]
    if change:tables[0][change[0]]=change[1]
    request=meta('request.json',dict(schema='archive_catalog_case_request_v1',evidence_mode='actual_archival_cache',layout='ground-only',
        role='primary',runtime='1.12.6',source=dict(fixture_only=True),tables=tables))
    directory=root/'cases';directory.mkdir()
    manifest=meta('cases/manifest.json',dict(request_sha256=request['sha256'],fixture_only=True))
    complete=meta('cases/COMPLETE.json',dict(manifest_sha256=manifest['sha256'],fixture_only=True))
    admission=meta('admission.json',dict(request_sha256=request['sha256'],artifact_manifest_sha256=manifest['sha256'],artifact_complete_sha256=complete['sha256'],fixture_only=True))
    source={p.relative_to(ROOT).as_posix():C.pin(p) for p in (ENTRY,C.DRIVER,C.PREVIEW_ENTRY,ROOT/'research/dbdt_completed_source/candidate_descriptor_metadata.jl')}
    proof=meta('proof.json',dict(head='0'*40,source_sha256=source,fixture_only=True))
    qualification=meta('qualification.json',dict(fixture_only=True,native_execution_evidence=False))
    runtime=meta('runtime.json',dict(executable_sha256=C.pin(PYTHON),julia_version='1.12.6',fixture_only=True))
    config=configuration(root)
    config.update(schema='fixed_candidate_campaign_v1',source_proof=proof,qualification=qualification,runtime=runtime,
        julia=dict(path=PYTHON,sha256=C.pin(PYTHON)),preview_authority=authority('unqualified-anchor'))
    for value in config['stations'].values():
        value['admission_metadata']=admission
        value['case'].update(request_path=request['path'],directory=str(directory),request_sha256=request['sha256'],manifest_sha256=manifest['sha256'],
            complete_sha256=complete['sha256'],admission_sha256=admission['sha256'])
    return meta('config.json',config)
class CampaignMetadataControls(unittest.TestCase):
    def test_independent_literal_queue_positions_counts_and_grant_slots(self):
        self.assertEqual(len(C.QUEUE),L['stage_count']);self.assertEqual(len({s['key'] for s in C.QUEUE}),142)
        self.assertEqual(Counter(s['operation'] for s in C.QUEUE),L['operation_counts'])
        for i,key in L['positions_one_based'].items():self.assertEqual(C.QUEUE[int(i)-1]['key'],key)
        self.assertEqual(sum(s['operation'] in ('candidate-fit','selected-refit') for s in C.QUEUE),38)
        self.assertEqual(sum(len(C.authority_slots(s)) for s in C.QUEUE),146)
    def test_all142_request_handoffs_keep_every_prior_pin(self):
        config,done,grants,requests=trace();self.assertEqual(len(requests),142)
        for stage,req in zip(C.QUEUE,requests):
            self.assertEqual(req['case'],config['stations'][stage['station']]['case'])
            if stage['operation']=='candidate-validation':
                f=done[stage['station']+'/'+stage['candidate_id']+'/candidate-fit']
                self.assertEqual(req['inputs']['fit_native'],f['native']);self.assertEqual(req['inputs']['fit'],f['outputs']['fit'])
            if stage['operation'] in ('select-R2','selected-refit-input','selected-refit'):
                rows=req['inputs']['candidates'];self.assertEqual([r['candidate_id'] for r in rows],list(C.CANDIDATES))
                for row in rows:self.assertEqual(row['verification_native'],done[stage['station']+'/'+row['candidate_id']+'/independent-verify']['native'])
        self.assertEqual(requests[64]['qualification_authority'],done['CMO/ridge-0.01/independent-verify']['request']['inputs']['authority'])
        self.assertNotIn('CMO/select-R2#authority',grants)
    def test_missing_next_and_inner_grants_return_before_any_launch(self):
        config,done,grants,_=trace()
        with patch.object(C.D,'main',side_effect=AssertionError('launch forbidden')) as launch:
            req,missing=C.build_request(config,C.QUEUE[0],{},{});self.assertIsNone(req);self.assertEqual(len(missing),1)
            stage=C.BY_KEY['CMO/ridge/selected-refit'];bad=copy.deepcopy(grants);del bad[stage['key']+'#fit_authority']
            req,missing=C.build_request(config,stage,done,bad);self.assertIsNone(req);self.assertEqual(missing,[stage['key']+'#fit_authority'])
            launch.assert_not_called()
    def test_public_run_missing_authority_writes_pending_without_preview_or_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();config=configuration(root)
            path=root/'config.json';C.write(path,dict(root=str(root)));ref=dict(path=str(path),sha256=C.pin(path))
            box=dict(path=str(root/'unqualified-inbox.json'),sha256=sha('inbox'))
            with patch.object(C,'validate_config',return_value=config),patch.object(C,'inbox',return_value=({},None)),\
                 patch.object(C.D,'main',side_effect=AssertionError('worker forbidden')) as worker,\
                 patch.object(C,'run_preview',side_effect=AssertionError('preview forbidden')) as preview:
                status=C.main(ref,box)
                self.assertEqual(status,0);worker.assert_not_called();preview.assert_not_called()
            terminal=C.load(root/'invocations/0000.TERMINAL.json')
            self.assertEqual(terminal['status'],'awaiting_authority');self.assertEqual(terminal['completed_study_operations'],0)
            self.assertEqual(terminal['study_launches_this_invocation'],0);self.assertEqual(terminal['auxiliary_previews_this_invocation'],0)
            self.assertFalse(terminal['source_or_fit_permission_from_metadata'])
    def test_unclosed_study_request_cannot_be_relaunched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();config=configuration(root);stage=C.QUEUE[0];directory=C.study_directory(config,stage);directory.mkdir(parents=True)
            C.write(directory/'CAMPAIGN_REQUEST.json',dict(fixture_only=True))
            with patch.object(C.D,'main') as launch:
                with self.assertRaisesRegex(ValueError,'live/failed/unknown'):C.check_closed(config,stage)
                launch.assert_not_called()
    def test_unknown_preview_blocks_both_run_and_preview_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=configuration(Path(temporary).resolve());directory=C.preview_directory(config,C.QUEUE[0]);directory.mkdir(parents=True)
            C.write(directory/'PREVIEW_REQUEST.json',dict(fixture_only=True))
            with patch.object(C.D,'main') as launch:
                with self.assertRaisesRegex(ValueError,'live/unknown auxiliary'):C.closed_previews(config)
                launch.assert_not_called()
    def test_study_call_with_unknown_launch_serializes_and_blocks_history_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();config=configuration(root);stage=C.QUEUE[0]
            path=root/'config.json';C.write(path,dict(root=str(root)));ref=dict(path=str(path),sha256=C.pin(path))
            box=dict(path=str(root/'box.json'),sha256=sha('box'));grants={stage['key']+'#authority':authority('input')}
            with patch.object(C,'validate_config',return_value=config),patch.object(C,'inbox',return_value=(grants,None)),\
                 patch.object(C.D,'grant_ref',return_value={}),patch.object(C.D,'admit_request',return_value=None),\
                 patch.object(C.D,'main',side_effect=OSError('controlled missing launch-publication evidence')):
                self.assertEqual(C.main(ref,box),1)
            terminal_path=root/'invocations/0000.TERMINAL.json';terminal=C.load(terminal_path)
            self.assertEqual(terminal['study_calls_this_invocation'],1);self.assertEqual(terminal['study_launches_this_invocation'],0)
            self.assertIs(terminal['study_launch_count_known'],False);self.assertIs(terminal['launch_counts_are_observed_lower_bounds'],True)
            self.assertGreaterEqual(terminal['cumulative_active_seconds'],0)
            with self.assertRaisesRegex(ValueError,'unknown actual launch count'):
                C.invocation(root,ref,box,grants,'run',dict(path=str(terminal_path),sha256=C.pin(terminal_path)))
    def test_old_grants_are_immutable_and_unknown_invocation_has_no_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();cfg=dict(path=str(root/'config.json'),sha256=sha('cfg'));box=dict(path=str(root/'inbox.json'),sha256=sha('inbox'))
            grants={'CMO/ridge-0.01/candidate-input#authority':authority('one')}
            terminal,seconds,count,start_sha=C.invocation(root,cfg,box,grants,'run',None)
            with self.assertRaisesRegex(ValueError,'lost/unknown'):C.invocation(root,cfg,box,grants,'run',None)
            C.write(terminal,dict(start_sha256=start_sha,invocation_wall_seconds=2.5,auxiliary_previews_this_invocation=0,
                cumulative_active_seconds=2.5,cumulative_auxiliary_previews=0,auxiliary_launch_count_known=True,study_launch_count_known=True))
            ref=dict(path=str(terminal),sha256=C.pin(terminal));changed=copy.deepcopy(grants);changed[next(iter(changed))]=authority('changed')
            with self.assertRaisesRegex(ValueError,'pins changed'):C.invocation(root,cfg,box,changed,'run',ref)
            _,seconds,count,_=C.invocation(root,cfg,box,grants,'run',ref)
            self.assertEqual(seconds,2.5);self.assertEqual(count,0)
    def test_invalid_elapsed_values_never_reset_active_budget(self):
        for elapsed in (True,-1,float('nan'),float('inf')):
            with self.subTest(elapsed=repr(elapsed)),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary).resolve();cfg=dict(path=str(root/'config.json'),sha256=sha('cfg'));box=dict(path=str(root/'inbox.json'),sha256=sha('box'))
                terminal,_,_,start_sha=C.invocation(root,cfg,box,{},'run',None)
                body=dict(start_sha256=start_sha,invocation_wall_seconds=elapsed,auxiliary_previews_this_invocation=0,cumulative_active_seconds=elapsed,cumulative_auxiliary_previews=0)
                body.update(auxiliary_launch_count_known=True,study_launch_count_known=True)
                # Nonfinite JSON is itself inadmissible; write it only as a negative input fixture.
                terminal.write_text(json.dumps(body)+'\n',encoding='utf8')
                with self.assertRaises(ValueError):C.invocation(root,cfg,box,{},'run',dict(path=str(terminal),sha256=C.pin(terminal)))
    def test_fixed_bounds_and_no_cli_runtime_override(self):
        self.assertEqual((C.MAX_INVOCATIONS,C.MAX_PREVIEWS,C.MAX_SECONDS),(1024,142,2048400))
        self.assertEqual(C.D.SECONDS,7200);self.assertEqual(C.D.RAW_LIMIT,67108864)
    def test_duration_rejects_bool_nan_inf_and_negative(self):
        for value in (True,False,-1,float('nan'),float('inf'),float('-inf'),'1'):
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):C.duration(value)
        self.assertEqual(C.duration(0),0);self.assertEqual(C.duration(2.5),2.5)
    def test_actual_table_domains_reject_before_stage_or_preview(self):
        cases=(None,('start_inclusive','2020-01-01T00:00:00.000Z'),('start_inclusive','2017-01-02T00:00:00.000Z'),
            ('stop_exclusive','2017-06-30T00:00:00.000Z'),('label_asof','2017-12-31T23:59:59.999Z'),
            ('label_asof','2018-01-01T00:00:00.000+00:00'),('label_asof','2018-01-01T00:00:00.0000Z'))
        for change in cases:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                ref=actual_header_fixture(Path(temporary).resolve(),change)
                with patch.object(C.D,'source_guard',return_value=None),patch.object(C.D,'qualification',return_value=None),\
                     patch.object(C.D,'main') as worker,patch.object(C,'run_preview') as preview:
                    if change is None:self.assertEqual(C.validate_config(ref)['stations']['CMO']['table_index'],1)
                    else:
                        with self.assertRaises(ValueError):C.validate_config(ref)
                    worker.assert_not_called();preview.assert_not_called()
    def test_wrong_preview_lineage_rejects_before_owner_or_memory_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();config=configuration(root);stage=C.QUEUE[0]
            directory=C.preview_directory(config,stage);directory.mkdir(parents=True)
            meta=C.metadata_for_preview(config,stage,{})
            request=dict(schema='candidate_descriptor_preview_request_v1',campaign=dict(path=str(root/'cfg.json'),sha256=sha('cfg')),
                stage_key=stage['key'],execution_directory=str(directory),metadata=meta)
            request['metadata']['station']='FRD';path=directory/'request.json';C.write(path,request)
            with patch.object(C,'validate_config',return_value=config),patch.object(C,'prefix',return_value=({},stage)),\
                 patch.object(C.D,'module',side_effect=AssertionError('no owner/memory module')) as owner:
                with self.assertRaisesRegex(ValueError,'pending known inputs'):C.preview_admission(path,C.pin(path))
                owner.assert_not_called()
    def test_select_R2_preview_rejects_before_creating_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=configuration(Path(temporary).resolve());stage=C.BY_KEY['CMO/select-R2']
            with patch.object(C.D,'module',side_effect=AssertionError('no owner/memory module')) as owner:
                with self.assertRaisesRegex(ValueError,'needs no grant descriptor'):C.run_preview(config,stage,{},dict(path='unused',sha256=sha('unused')))
                self.assertFalse(C.preview_directory(config,stage).exists());owner.assert_not_called()
    def test_preview_supervisor_real_Python_inner_and_outer_waits(self):
        self.run_owned_preview_probe('emit')
    def test_real_owner_callback_publication_failure_retains_observed_PID_count(self):
        parent=Path(os.environ.get('CANDIDATE_CAMPAIGN_CONTROLS_ROOT',tempfile.gettempdir())).resolve()
        root=Path(tempfile.mkdtemp(dir=parent,prefix='campaign-callback-native-')).resolve()
        actual_owner=C.D.module(C.D.OWNER,'campaign_callback_exact_owner')
        fixture=ROOT/'test/candidate_stage_driver/native_fixture.py';path=root/'private-request.json'
        C.write(path,dict(fixture_only=True));controls={str(path):C.pin(path)}
        record=dict(schema='UNQUALIFIED_PYTHON_CALLBACK_ORACLE_ONLY',complete=False,actual_fits=0,actual_source_bodies_read=False,
            owned_attempts_charged=0,observed_owner_launches=0,launch_count_known=True)
        config=dict(_proof=dict(head='0'*40));original_write=C.write
        class FixedPythonProbeOwner:
            @staticmethod
            def run_owned(argv,**kw):
                return actual_owner.run_owned([sys.executable,'-B','-X','utf8',str(fixture),'--mode','emit','--root',str(root)],
                    cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=kw['logpath'],timeout=5,on_launch=kw['on_launch'])
        def publication(path,value):
            if path.name=='PREVIEW_OWNER_LAUNCH.json':raise OSError('injected callback publication failure')
            original_write(path,value)
        with patch.object(C,'prepare_preview',return_value=(root,path,sha('private'),controls,record)),\
             patch.object(C.D,'module',return_value=FixedPythonProbeOwner),patch.object(C.D,'source_guard',return_value=None),\
             patch.object(C,'write',side_effect=publication):
            result=C.run_preview(config,C.QUEUE[0],{},dict(path='unused',sha256=sha('unused')))
        self.assertFalse(result['complete']);self.assertEqual(result['owned_attempts_charged'],1)
        self.assertEqual(result['observed_owner_launches'],1);self.assertIs(result['launch_count_known'],True)
        self.assertGreater(result['observed_owner_pid'],0);self.assertFalse((root/'PREVIEW_OWNER_LAUNCH.json').exists())
        self.assertNotIn('supervisor_native',result)  # No native exit is inferred from callback PID or missing marker.
        print('UNQUALIFIED_PYTHON_CALLBACK_ORACLE_RAW='+str(root),flush=True)
    def test_preview_wait_publication_fault_with_real_pipeheld_child_is_finite(self):
        self.run_owned_preview_probe('pipe-publication-failure')
    def run_owned_preview_probe(self,mode):
        # Real Python waits verify ownership mechanics, never a Julia/source fallback.
        parent=Path(os.environ.get('CANDIDATE_CAMPAIGN_CONTROLS_ROOT',tempfile.gettempdir())).resolve()
        temporary=tempfile.mkdtemp(dir=parent,prefix='campaign-preview-native-')
        root=Path(temporary).resolve();owner=C.D.module(C.D.OWNER,'campaign_control_exact_owner')
        try:
            result=owner.run_owned([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--owned-preview-probe',mode,str(root)],
                cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=root/'outer.log',timeout=20,
                on_launch=lambda pid:C.write(root/'PREVIEW_OWNER_LAUNCH.json',dict(pid=pid,owner_pid=os.getpid(),request_sha256=sha('probe'))))
            self.assertIs(result['timed_out'],False);self.assertIs(result['leader_reaped'],True);self.assertIs(result['ownership_boundary_closed'],True)
            self.assertEqual(result['native_exit'],0 if mode=='emit' else 1)
            observed=C.load(root/('JULIA_NATIVE_WAIT.json' if mode=='emit' else 'PRIVATE_ORACLE_INNER_WAIT.json'))
            self.assertEqual(observed['native_exit'],0);self.assertIs(observed['exact_child_reaped'],True)
            if mode=='emit':self.assertTrue(observed['complete']);self.assertEqual((root/'julia.log').read_bytes(),b'native-byte-oracle\n')
            else:
                self.assertIs(observed['reader_joined'],False);self.assertFalse((root/'JULIA_NATIVE_WAIT.json').exists())
                # The private oracle is not a production receipt: outside it the missing inner wait stays unknown.
        finally:
            # Retain the two tiny genuine native probes under the caller's one
            # control root, including failures; no old/protected cleanup.
            print('UNQUALIFIED_PYTHON_NATIVE_PROBE_RAW='+str(root),flush=True)
def owned_preview_probe(mode,root):
    C.need(mode in ('emit','pipe-publication-failure') and root.is_absolute(),'fixed control probe')
    path=root/'request.json';request=dict(execution_directory=str(root));config={};stage=C.QUEUE[0]
    fixture=ROOT/'test/candidate_stage_driver/native_fixture.py'
    argv=[sys.executable,'-B','-X','utf8',str(fixture),'--mode','emit' if mode=='emit' else 'descendant','--root',str(root)]
    C.write(root/'preview-argv.json',argv)
    original_pump=C.D.pump_process;original_write=C.write
    def fixed_fixture_pump(args,cwd,env,log):return original_pump(args,cwd,env,log,timeout=4,limit=128)
    def publication(path,value):
        if mode!='emit' and path.name=='JULIA_NATIVE_WAIT.json':
            original_write(root/'PRIVATE_ORACLE_INNER_WAIT.json',value);raise OSError('injected wait-publication fault')
        return original_write(path,value)
    with patch.object(C,'preview_admission',return_value=(request,config,stage,0,'UNQUALIFIED_PYTHON_ONLY',0)),\
         patch.object(C,'preview_argv',return_value=argv),patch.object(C.D,'pump_process',side_effect=fixed_fixture_pump),\
         patch.object(C,'write',side_effect=publication):
        return C.preview_supervisor(path,sha('probe'))
if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--owned-preview-probe':raise SystemExit(owned_preview_probe(sys.argv[2],Path(sys.argv[3])))
    program=unittest.main(verbosity=2,exit=False)
    if 'CANDIDATE_CAMPAIGN_CONTROLS_ROOT' in os.environ:
        C.write(Path(os.environ['CANDIDATE_CAMPAIGN_CONTROLS_ROOT'])/'RESULT.json',dict(complete=program.result.wasSuccessful(),
            methods=program.result.testsRun,failures=len(program.result.failures),errors=len(program.result.errors),
            Julia_executed=False,actual_data_access=False,actual_fits=0,mocked_metadata_is_not_native_qualification=True))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
