from pathlib import Path
from hashlib import sha256
import copy, importlib.util, tempfile, unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('evaluation_binding_oracle_under_test', ROOT/'tools/point_input_export/evaluation_binding.py')
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


def fixture(directory, role='primary', layout='ground-only', runtime='1.13.1', **changes):
    """Literal independent metadata; the deliberately opaque model is never loaded."""
    canon = O.V.canonical
    digest = lambda value: sha256(value).hexdigest()
    stamp = lambda year, minute: f'{year}-01-01T{minute//60:02}:{minute%60:02}:00.000Z'
    old_table = dict(id='CMO-train', station='CMO', start_inclusive=stamp(2020, 0), stop_exclusive=stamp(2020, 181), label_asof=stamp(2020, 211))
    table = dict(id='CMO-future', station=changes.get('station', 'CMO'), start_inclusive=stamp(2021, 0), stop_exclusive=stamp(2021, 181), label_asof=stamp(2021, 211))
    recipe = dict(schema='opaque_recipe_fixture', seed=17)
    pc = dict(schema='source_bound_generated_fit_context_v1', evidence_mode='generated',
        case_request_sha256='1'*64, case_manifest_sha256='2'*64, case_complete_sha256='3'*64,
        table_index=1, table=old_table, role=role, layout=layout, runtime=runtime,
        start_inclusive=stamp(2020, 40), stop_exclusive=stamp(2020, 90),
        input_sha256='4'*64, recipe_sha256=digest(canon(recipe)), source_sha256='5'*64, library_sha256='6'*64)
    bodies = {'point-context.json': canon(pc), 'point-recipe.json': canon(recipe),
        'point-model.bin': b'OPAQUE MODEL -- DO NOT EXECUTE\x00\xff'}
    pr = dict(schema='source_bound_point_fit_receipt_v1', complete=True,
        context_sha256=digest(bodies['point-context.json']), model_sha256=digest(bodies['point-model.bin']),
        **{key: pc[key] for key in ('recipe_sha256', 'source_sha256', 'library_sha256', 'input_sha256')},
        actual_source_access=False, actual_fit_performed=False, software_qualification=False)
    bodies['point-receipt.json'] = canon(pr)
    kc = dict(schema='source_bound_calibration_context_v1', evidence_mode='generated', measured_receipts_available=False,
        **{key: pc[key] for key in ('case_request_sha256', 'case_manifest_sha256', 'case_complete_sha256', 'table_index', 'table', 'role', 'layout', 'runtime')},
        point_context_sha256=pr['context_sha256'], point_model_sha256=pr['model_sha256'],
        point_recipe_sha256=pr['recipe_sha256'], point_receipt_sha256=digest(bodies['point-receipt.json']),
        point_source_sha256=pc['source_sha256'], point_training_stop=pc['stop_exclusive'],
        start_inclusive=stamp(2020, 90), stop_exclusive=stamp(2020, 135),
        first_evaluation_issue=changes.get('first_evaluation_issue', stamp(2020, 150)), input_sha256='7'*64)
    bodies['calibration-context.json'] = canon(kc)
    request = dict(schema='generated_catalog_case_request_v1', evidence_mode='generated',
        role=changes.get('evaluation_role', role), layout=changes.get('evaluation_layout', layout), runtime=runtime,
        source=dict(directory='/must-not-read/observations', manifest_sha256='8'*64, complete_sha256='9'*64), tables=[table])
    bodies['case-request.json'] = canon(request)
    manifest = dict(schema='generated_catalog_case_artifact_v1', request_sha256=digest(bodies['case-request.json']),
        role=request['role'], layout=request['layout'], runtime=runtime, source=request['source'],
        actual_source_authority_granted=False, provider_receipts_available=False, actual_fits=0,
        tables=[dict(declaration=table, mode='ground11' if request['layout']=='ground-only' else 'mixed8', feature_payload_sha256='a'*64)])
    bodies['case-manifest.json'] = canon(manifest)
    bodies['case-complete.json'] = canon(dict(schema='generated_catalog_case_complete_v1', manifest_sha256=digest(bodies['case-manifest.json']), actual_fits=0))
    closure = digest(canon(dict(schema='generated_replay_evaluation_job_closure_v1', files=[dict(path=name, sha256=digest((ROOT/name).read_bytes())) for name in O.FILES])))
    context = dict(schema='generated_source_bound_evaluation_binding_v1', evidence_mode='generated', runtime=runtime,
        role=request['role'], layout=request['layout'], job_sha256=closure,
        point=dict(context_sha256=pr['context_sha256'], model_sha256=pr['model_sha256'], receipt_sha256=digest(bodies['point-receipt.json']),
            recipe_sha256=pr['recipe_sha256'], source_sha256=pc['source_sha256'], library_sha256=pc['library_sha256']),
        calibration=dict(context_sha256=digest(bodies['calibration-context.json']), input_sha256=kc['input_sha256'], stop_exclusive=kc['stop_exclusive'], first_evaluation_issue=kc['first_evaluation_issue']),
        training_case=dict(request_sha256=pc['case_request_sha256'], manifest_sha256=pc['case_manifest_sha256'], complete_sha256=pc['case_complete_sha256'], table_index=1, table=old_table),
        evaluation_case=dict(request_sha256=digest(bodies['case-request.json']), manifest_sha256=digest(bodies['case-manifest.json']), complete_sha256=digest(bodies['case-complete.json']), table_index=1, table=table,
            feature_payload_sha256='a'*64, source=request['source']),
        start_inclusive=stamp(2021, 150), stop_exclusive=stamp(2021, 166), target_asof=stamp(2021, 211),
        actual_source_authority_granted=False, actual_model_fit_authorized=False, measured_receipts_available=False, scientific_coverage_qualified=False)
    bodies['evaluation-binding.json'] = canon(context)
    directory.mkdir()
    for name, body in bodies.items():
        (directory/name).write_bytes(body)
    return dict(pins={name: digest(body) for name, body in bodies.items()}, table_index=1,
        start=context['start_inclusive'], stop=context['stop_exclusive'], asof=context['target_asof'])


class EvaluationBindingTests(unittest.TestCase):
    def verify(self, directory, args):
        return O.verify_binding(directory, ROOT, **args)

    def test_all_eight_native_contexts_without_model_or_source_loading(self):
        with tempfile.TemporaryDirectory() as root:
            for role in ('primary', 'independent'):
                for layout in ('ground-only', 'mixed'):
                    for runtime in ('1.12.6', '1.13.1'):
                        directory = Path(root)/f'{role}-{layout}-{runtime}'
                        args = fixture(directory, role, layout, runtime)
                        allowed = {directory/name for name in O.MEMBERS} | {ROOT/name for name in O.FILES}
                        original = O.V.bounded_read
                        def guarded(path, limit):
                            self.assertIn(Path(path), allowed)
                            return original(path, limit)
                        with patch.object(O.V, 'bounded_read', guarded):
                            result = self.verify(directory, args)
                        self.assertTrue(result['metadata_connection_independently_checked'])
                        self.assertFalse(result['model_loaded'])
                        self.assertFalse(result['source_numbers_reconstructed'])
                        self.assertFalse(result['actual_source_access'])

    def test_self_consistently_rehashed_context_mismatches(self):
        for changes in ({'station':'FRD'}, {'evaluation_role':'independent'}, {'evaluation_layout':'mixed'},
                        {'first_evaluation_issue':'2022-01-01T00:00:00.000Z'}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as root:
                directory = Path(root)/'fixture'
                args = fixture(directory, **changes)
                with self.assertRaises(O.V.SnapshotError):
                    self.verify(directory, args)

    def test_rehashed_binding_semantic_corruptions(self):
        for mutate in (lambda c: c['point'].update(model_sha256='f'*64),
                       lambda c: c['evaluation_case'].update(table_index=True),
                       lambda c: c.update(actual_source_authority_granted=0),
                       lambda c: c.update(scientific_coverage_qualified=True),
                       lambda c: c.update(start_inclusive='2020-01-01T00:00:00.000Z'),
                       lambda c: c.update(job_sha256='f'*64)):
            with tempfile.TemporaryDirectory() as root:
                directory = Path(root)/'fixture'; args = fixture(directory)
                path = directory/'evaluation-binding.json'; value = O.V.decoded(path.read_bytes())
                mutate(value); path.write_bytes(O.V.canonical(value)); args['pins'][path.name] = sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(O.V.SnapshotError):
                    self.verify(directory, args)

    def test_external_pin_index_and_window_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/'fixture'; original = fixture(directory)
            for mutate in (lambda a:a.update(table_index=True), lambda a:a.update(table_index=0),
                           lambda a:a.update(table_index=2), lambda a:a.update(asof='2021-01-01T04:00:00.000Z'),
                           lambda a:a['pins'].update({'point-model.bin':'f'*64})):
                args = copy.deepcopy(original); mutate(args)
                with self.assertRaises(O.V.SnapshotError):
                    self.verify(directory, args)

    def test_fileset_and_noncanonical_bytes_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/'fixture'; args = fixture(directory)
            extra = directory/'unexpected'; extra.write_bytes(b'extra')
            with self.assertRaises(O.V.SnapshotError):self.verify(directory, args)
            extra.unlink()
            path = directory/'evaluation-binding.json'; path.write_bytes(path.read_bytes()+b' ')
            args['pins'][path.name] = sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(O.V.SnapshotError):self.verify(directory, args)


if __name__ == '__main__':
    unittest.main()
