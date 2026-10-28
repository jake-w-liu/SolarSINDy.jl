"""Independent generated evaluation metadata oracle; never loads models or targets."""
from pathlib import Path
from hashlib import sha256
import importlib.util

BASE = Path(__file__).with_name('bounded_json.py')
assert sha256(BASE.read_bytes()).hexdigest() == '1912a60ab9e014437f54ae133caec353fcf2839972344a126b27836b23f0aa5f'
spec = importlib.util.spec_from_file_location('evaluation_bounded_json', BASE)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
need = V.need
FILES = tuple(sorted('research/dbdt_completed_source/' + name for name in (
    'source_bound_shared_replay.jl', 'source_bound_evaluation_binding.jl',
    'source_bound_calibration.jl', 'source_bound_calibration_codec.jl',
    'source_bound_static_bounds.jl', 'publisher_up_projection.jl',
    'publisher_up_values.jl', 'retained_publisher_up.jl')))
MEMBERS = ('point-context.json', 'point-model.bin', 'point-recipe.json',
           'point-receipt.json', 'calibration-context.json', 'case-request.json',
           'case-manifest.json', 'case-complete.json', 'evaluation-binding.json')


def _ordinary(path, directory=False):
    path = V._ordinary(Path(path), directory)
    for part in (path, *path.parents):
        need(not (getattr(part.lstat(), 'st_file_attributes', 0) & 0x400), 'reparse path component')
    return path


def _keys(value, names):
    need(isinstance(value, dict) and set(value) == set(names), 'metadata fields differ')


def verify_binding(directory, package_root, *, pins, table_index, start, stop, asof):
    """Reconstruct a binding from independently pinned opaque model and metadata.

    This checks a metadata connection, not source-number reconstruction, fitting,
    calibration values, scientific coverage, or archival access authorization.
    The caller supplies pins separately from the nine fixture members.
    """
    directory = _ordinary(directory, True)
    package_root = _ordinary(package_root, True)
    _keys(pins, MEMBERS)
    names = sorted(p.name for p in directory.iterdir())
    need(names == sorted(MEMBERS), 'evaluation fixture fileset differs')
    identities = {}

    def read(path, limit):
        path = _ordinary(path)
        before = V._identity(path)
        data = V.bounded_read(path, limit)
        identities[path] = before
        return data

    bodies = {}
    for name in MEMBERS:
        V._sha(pins[name])
        body = read(directory / name, 8_388_608 if name == 'point-model.bin' else 1_048_576)
        need(sha256(body).hexdigest() == pins[name], 'external evaluation member pin differs')
        bodies[name] = body
    # The model is opaque. No author model loader or source path is called.
    decoded = {name: V.decoded(body) for name, body in bodies.items() if name != 'point-model.bin'}
    pc, pr, kc = (decoded[name] for name in ('point-context.json', 'point-receipt.json', 'calibration-context.json'))
    request, manifest, marker = (decoded[name] for name in ('case-request.json', 'case-manifest.json', 'case-complete.json'))
    need(pc['schema'] == 'source_bound_generated_fit_context_v1' and pc['evidence_mode'] == 'generated', 'point is not generated')
    need(pr['schema'] == 'source_bound_point_fit_receipt_v1' and pr['complete'] is True and pr['actual_source_access'] is False and
         pr['actual_fit_performed'] is False and pr['software_qualification'] is False, 'point receipt permissions differ')
    for key, name in (('context_sha256', 'point-context.json'), ('model_sha256', 'point-model.bin'), ('recipe_sha256', 'point-recipe.json')):
        need(pr[key] == pins[name], 'point receipt member binding differs')
    for key in ('recipe_sha256', 'source_sha256', 'library_sha256', 'input_sha256'):
        V._sha(pc[key])
        need(pr[key] == pc[key], 'point receipt context differs')
    need(kc['schema'] == 'source_bound_calibration_context_v1' and kc['evidence_mode'] == 'generated' and
         kc['measured_receipts_available'] is False, 'calibration evidence differs')
    for key, name in (('point_context_sha256', 'point-context.json'), ('point_model_sha256', 'point-model.bin'),
                      ('point_recipe_sha256', 'point-recipe.json'), ('point_receipt_sha256', 'point-receipt.json')):
        need(kc[key] == pins[name], 'calibration point member differs')
    need(kc['point_source_sha256'] == pc['source_sha256'] and kc['point_training_stop'] == pc['stop_exclusive'], 'calibration point source/split differs')
    for key in ('case_request_sha256', 'case_manifest_sha256', 'case_complete_sha256', 'table_index', 'table', 'role', 'layout', 'runtime'):
        need(V.canonical(kc[key]) == V.canonical(pc[key]), 'calibration training case differs')
    V._integer(pc['table_index'], 16)
    need(pc['table_index'] >= 1, 'training table index differs')
    for key in ('case_request_sha256', 'case_manifest_sha256', 'case_complete_sha256'):
        V._sha(pc[key])
    need(request['schema'] == 'generated_catalog_case_request_v1' and request['evidence_mode'] == 'generated', 'evaluation is not generated')
    need(manifest['schema'] == 'generated_catalog_case_artifact_v1' and marker['schema'] == 'generated_catalog_case_complete_v1', 'evaluation artifact schema differs')
    need(manifest['request_sha256'] == pins['case-request.json'] and marker['manifest_sha256'] == pins['case-manifest.json'], 'evaluation case chain differs')
    need(manifest['actual_source_authority_granted'] is False and manifest['provider_receipts_available'] is False and
         type(manifest['actual_fits']) is int and manifest['actual_fits'] == 0 and
         type(marker['actual_fits']) is int and marker['actual_fits'] == 0, 'evaluation permissions differ')
    need(V.canonical(manifest['source']) == V.canonical(request['source']), 'evaluation source metadata differs')
    _keys(request['source'], ('directory', 'manifest_sha256', 'complete_sha256'))
    for key in ('manifest_sha256', 'complete_sha256'):
        V._sha(request['source'][key])
    need(pc['role'] in ('primary', 'independent') and pc['layout'] in ('ground-only', 'mixed') and
         pc['runtime'] in ('1.12.6', '1.13.1'), 'training native context differs')
    for key in ('role', 'layout', 'runtime'):
        need(pc[key] == request[key] == manifest[key], 'evaluation native context differs')
    V._integer(table_index, 16)
    need(isinstance(request['tables'], list) and 1 <= table_index <= len(request['tables']) == len(manifest['tables']) <= 16, 'evaluation index differs')
    table = request['tables'][table_index - 1]
    feature = manifest['tables'][table_index - 1]
    need(V.canonical(feature['declaration']) == V.canonical(table) and table['station'] == pc['table']['station'] and
         table['station'] in ('CMO', 'FRD') and feature['mode'] == ('ground11' if pc['layout'] == 'ground-only' else 'mixed8'), 'evaluation station/table/layout differs')
    V._sha(feature['feature_payload_sha256'])
    begin, end, cutoff = map(V._minute, (start, stop, asof))
    need(V._minute(pc['start_inclusive']) < V._minute(pc['stop_exclusive']) <= V._minute(kc['start_inclusive']) <
         V._minute(kc['stop_exclusive']) <= V._minute(kc['first_evaluation_issue']) <= begin < end <= cutoff, 'training/calibration/evaluation order differs')
    need(V._minute(table['start_inclusive']) <= begin < end <= V._minute(table['stop_exclusive']) and
         cutoff <= V._minute(table['label_asof']), 'evaluation window exceeds sealed case')
    files = [dict(path=name, sha256=sha256(read(package_root / name, 1_048_576)).hexdigest()) for name in FILES]
    closure = sha256(V.canonical(dict(schema='generated_replay_evaluation_job_closure_v1', files=files))).hexdigest()
    wanted = dict(schema='generated_source_bound_evaluation_binding_v1', evidence_mode='generated', runtime=pc['runtime'],
        role=pc['role'], layout=pc['layout'], job_sha256=closure,
        point=dict(context_sha256=pins['point-context.json'], model_sha256=pins['point-model.bin'],
            receipt_sha256=pins['point-receipt.json'], recipe_sha256=pins['point-recipe.json'],
            source_sha256=pc['source_sha256'], library_sha256=pc['library_sha256']),
        calibration=dict(context_sha256=pins['calibration-context.json'], input_sha256=V._sha(kc['input_sha256']),
            stop_exclusive=kc['stop_exclusive'], first_evaluation_issue=kc['first_evaluation_issue']),
        training_case=dict(request_sha256=pc['case_request_sha256'], manifest_sha256=pc['case_manifest_sha256'],
            complete_sha256=pc['case_complete_sha256'], table_index=pc['table_index'], table=pc['table']),
        evaluation_case=dict(request_sha256=pins['case-request.json'], manifest_sha256=pins['case-manifest.json'],
            complete_sha256=pins['case-complete.json'], table_index=table_index, table=table,
            feature_payload_sha256=feature['feature_payload_sha256'], source=request['source']),
        start_inclusive=start, stop_exclusive=stop, target_asof=asof, actual_source_authority_granted=False,
        actual_model_fit_authorized=False, measured_receipts_available=False, scientific_coverage_qualified=False)
    need(V.canonical(wanted) == bodies['evaluation-binding.json'], 'independently reconstructed evaluation binding differs')
    need(sorted(p.name for p in directory.iterdir()) == names and
         all(V._identity(_ordinary(path)) == before for path, before in identities.items()), 'evaluation fixture or code changed during read')
    return dict(context_sha256=pins['evaluation-binding.json'], job_sha256=closure,
        metadata_connection_independently_checked=True, source_numbers_reconstructed=False,
        model_loaded=False, actual_source_access=False, actual_fits=0, scientific_coverage_qualified=False)
