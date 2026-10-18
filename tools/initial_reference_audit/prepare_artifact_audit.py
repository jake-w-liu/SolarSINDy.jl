"""Independent source-bound artifact preflight and initial-imputation replay.

This prepares inputs for the independent Julia GBM/feature/cohort replay. It
does not fit, import author/ML code, authorize an experiment, or establish skill.
"""
from array import array
from dataclasses import replace
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
from artifact_io import (Limits,InvalidArtifact,need,is_integer,is_digest,plain_root,
                         member,read_file,json_file,inventory,relative_parts)
from bounded_bson import decode_root,read_bson,validate_model
from xgboost_scalar import xgb_predict,official_model_version
from bounded_tables import bounded_table

HERE=Path(__file__).resolve().parent
RAW_NAMES=['Vector B Magnitude,nT','Lat. Angle of B (GSE)','Long. Angle of B (GSE)',
    'BX, nT (GSE, GSM)','BY, nT (GSM)','BZ, nT (GSM)','Scalar B, nT',
    'SW Plasma Speed, km/s','SW Proton Density, N/cm^3','SW Plasma Temperature, K',
    'Plasma Beta','Flow pressure','E elecrtric field','Magnetosonic Much num.',
    'Dst-index, nT','AE-index, nT','Alpha/Prot. ratio']
FILL_LIMITS=(999.,999.,999.,999.,999.,999.,999.,9999.,999.,9999999.,999.,99.99,999.,99.,99999.,9999.,9.)
GRID=[dict(max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255)]
CONTRACT='db833f075e82ad5a15e10587e538c9688958a0fac180ca5713405da43d145ff7'
ACTUAL_SOURCE='593354e93e8f6c1486dc46de3ae96d4281d1129f9d5d39933347c2702122d248'


def grid_configuration(actual,expected):
    return isinstance(actual,dict) and set(actual)==set(expected) and all(
        type(actual[key]) is int and actual[key]==value for key,value in expected.items())


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def hour(value):
    need(isinstance(value,str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00:00(?:\.0{1,3})?',value),
         'serialized UTC boundary/tag must be an exact hour')
    return datetime.fromisoformat(value)


def exact_rows(actual,expected,label):
    need(isinstance(actual,list) and all(type(i) is int for i in actual) and actual==expected,label)


def compare_arrays(left,right,label):
    need(len(left)==len(right),label+' dimensions')
    for a,b in zip(left,right):
        need(a==b or math.isnan(a) and math.isnan(b),label)


def write_new(path,data):
    with Path(path).open('xb') as stream:
        need(stream.write(data)==len(data),'short audit artifact write')
    need(Path(path).read_bytes()==data,'audit artifact write readback differs')


def bound_source_files(root,completion,limits):
    values=completion.get('output_sha256')
    need(isinstance(values,dict) and len(values)<=limits.files,'bounded complete source inventory required')
    aliases=set();result={};total=0
    for name,expected in values.items():
        relative_parts(name)
        need(name.casefold() not in aliases and is_digest(expected),'source inventory alias/digest')
        aliases.add(name.casefold())
        path=member(root,name);size=path.stat().st_size
        need(size<=limits.file_bytes and size<=limits.total_bytes-total,'source inventory byte budget')
        total+=size;result[str(path)]=expected
    # All declared original outputs are checked; additional later audit receipts
    # are not followed or parsed. The reference output inventory is exact.
    for path,expected in result.items():
        need(read_file(path,limits.file_bytes)['sha256']==expected,'original source output changed')
    return result


def source_json(root,completion,name,limits):
    need(name in completion['output_sha256'],'source metadata/model missing from original inventory')
    return json_file(member(root,name),completion['output_sha256'][name],limits)


def reference_json(root,completion,name,limits):
    need(name in completion['output_sha256'],'reference metadata/model missing from output inventory')
    return json_file(member(root,name),completion['output_sha256'][name],limits)


def validate_imputer_model(model,features):
    need(official_model_version(model['version']),'official3.2 integer-version JSON state required')
    learner=model['learner']
    need(learner['feature_names']==[RAW_NAMES[j-1] for j in features],'initial imputer model feature order')
    need(learner['objective']['name']=='reg:squarederror','initial scalar objective differs')
    need(learner['learner_model_param']['num_feature']==str(len(features)),'initial model feature dimension')
    base=learner['learner_model_param']['base_score']
    need(isinstance(base,str) and len(base)<=256,'bounded scalar base-score JSON')
    decoded=json.loads(base)
    need(isinstance(decoded,list) and len(decoded)==1 and type(decoded[0]) in (int,float) and math.isfinite(decoded[0]),
         'finite scalar base score required')
    trees=learner['gradient_booster']['model']['trees']
    need(isinstance(trees,list) and len(trees)==100,'initial imputer round count differs')
    for tree in trees:
        n=len(tree['left_children'])
        need(1<=n<=127 and all(len(tree[key])==n for key in
             ('right_children','split_type','split_indices','split_conditions','default_left')),'bounded imputer tree shape')
        need(all(type(x) is int and -1<=x<n for key in ('left_children','right_children') for x in tree[key]),'imputer child index')
        need(all(type(x) is int and 0<=x<len(features) for x in tree['split_indices']),'imputer feature index')
        need(all(type(x) is int and x==0 for x in tree['split_type']),'numeric imputer splits required')
        need(all(type(x) in (int,float) and math.isfinite(x) for x in tree['split_conditions']),'finite imputer values')
        need(all(type(x) is bool or type(x) is int and x in (0,1) for x in tree['default_left']),'imputer default branch')


def initial_imputation(source,source_complete,reference,reference_complete,raw,prefix_rows,start,boundary,limits):
    # The real sidecar contains repeated training-row lists. Its explicit8M cap
    # covers15 lists at the admitted500k planning scale plus bounded metadata.
    metadata_limits=replace(limits,nodes=8_000_000)
    metadata=source_json(source,source_complete,'initial_imputation.json',metadata_limits)
    counts=source_json(source,source_complete,'initial_imputed_counts.json',limits)
    replay=reference_json(reference,reference_complete,'initial_imputation_replay.json',metadata_limits)
    config=metadata['configuration']
    need(config==replay['configuration'],'reconstructed initial imputer configuration differs')
    need(config['target_order']==RAW_NAMES[:14] and config['excluded']==RAW_NAMES[14:],'initial imputer channel order')
    required={'minimum_feature_count':6,'source_dataframe_column_count':20,'minimum_data_pct':30,
              'rounds':100,'objective':'reg:squarederror','max_depth':6,'learning_rate':.1,'seed':42,'nthread':1,
              'availability_policy':'assumed_completed_hour'}
    for name,value in required.items():
        need(type(config[name]) is type(value) and config[name]==value,'initial imputer recipe differs: '+name)
    need(config['source_sha256']==source_complete['input_sha256'] and
         is_digest(config['training_sha256']) and counts['training_sha256']==config['training_sha256'],
         'initial imputer source/training identity differs')
    cutoff=boundary-timedelta(hours=168)
    need(hour(config['training_cutoff_utc'])==cutoff,'initial imputer training clock differs')
    hours=int((cutoff-start).total_seconds()//3600)
    fit=list(range(1,min(prefix_rows,hours)+1)) if hours>=1 else []
    need(fit,'empty initial imputer fit pool')
    exact_rows(config['fit_rows'],fit,'initial imputer complete training pool differs')
    clean=array('d',(value if math.isfinite(value) and value<FILL_LIMITS[j%17] else math.nan for j,value in enumerate(raw)))
    expected=[];training={}
    for target in range(1,15):
        features=[j for j in range(1,15) if j!=target]
        valid=[i for i in fit if math.isfinite(clean[(i-1)*17+target-1]) and
               sum(math.isfinite(clean[(i-1)*17+j-1]) for j in features)>=6]
        if len(valid)>=100:expected.append(target);training[target]=(features,valid)
    originals=metadata['models'];rebuilt=replay['models']
    need([x['target'] for x in originals]==[x['target'] for x in rebuilt]==expected,'exact ordered fitted imputer targets')
    need(all(type(x['target']) is int for x in originals+rebuilt),'integer imputer targets required')
    need(config['fitted_targets']==[RAW_NAMES[t-1] for t in expected],'configured fitted target names differ')
    result=array('d',clean);refitted=array('d',clean);prefix_counts=[0]*17
    for old,new in zip(originals,rebuilt):
        target=old['target'];features,valid=training[target]
        need(old['features']==new['features']==features and all(type(i) is int for i in old['features']+new['features']),
             'initial imputer predictor order differs')
        exact_rows(old['fit_rows'],valid,'original imputer column fit mask differs')
        exact_rows(new['fit_rows'],valid,'reconstructed imputer column fit mask differs')
        old_name=old['file'];new_name=new['new_model']
        need(old_name==new_name==f'initial_imputer_{target}.json','initial imputer filename schema differs')
        need(old['sha256']==source_complete['output_sha256'][old_name]==new['source_model_sha256'],
             'original imputer model binding differs')
        need(new['new_model_sha256']==reference_complete['output_sha256'][new_name] and new['compared_prefix_rows']==prefix_rows,
             'reconstructed imputer model binding or prefix differs')
        old_model=source_json(source,source_complete,old_name,limits)
        new_model=reference_json(reference,reference_complete,new_name,limits)
        validate_imputer_model(old_model,features);validate_imputer_model(new_model,features)
        for i in range(prefix_rows):
            offset=i*17
            if math.isfinite(result[offset+target-1]):continue
            values=[result[offset+j-1] for j in features]
            if sum(math.isfinite(value) for value in values)<6:continue
            estimate=xgb_predict(old_model,values)
            other=xgb_predict(new_model,[refitted[offset+j-1] for j in features])
            need(math.isfinite(estimate) and estimate==other,'original/refitted initial imputer points differ')
            result[offset+target-1]=estimate;refitted[offset+target-1]=other;prefix_counts[target-1]+=1
    need(replay['prefix_rows']==prefix_rows and replay['prefix_imputed_counts']==prefix_counts,
         'independent prefix imputation counts differ')
    need(replay['source_full_counts_not_substituted']==counts['counts'] and len(counts['counts'])==17 and
         all(type(n) is int and n>=p for n,p in zip(counts['counts'],prefix_counts)),
         'full-source versus prefix imputation counts differ')
    for j,value in enumerate(clean):
        if math.isfinite(value):need(result[j]==value,'observed raw cell was imputed')
        if j%17>=14:need(result[j]==value or math.isnan(result[j]) and math.isnan(value),'excluded observation was imputed')
    return result,{'prefix_counts':prefix_counts,'source_full_counts':counts['counts'],'targets':expected,
        'fit_count':len(fit),'training_sha256':config['training_sha256'],'training_parameters_execution_bound':True}


def prepare(reference,expected_reference_sha256,source,expected_source_sha256,expected_mode,output,limits=Limits()):
    limits.validate()
    need(expected_mode in ('synthetic_only','exposed_development_reconstruction'),'explicit reference evidence mode required')
    reference=plain_root(reference);source=plain_root(source)
    need(reference!=source and not reference.is_relative_to(source) and not source.is_relative_to(reference),'separate input roots required')
    complete=json_file(member(reference,'completion.json'),expected_reference_sha256,limits)
    original=json_file(member(source,'completion.json'),expected_source_sha256,limits)
    need(complete['schema']=='all_grid_initial_validation_reference_v1' and complete['complete'] is True and original['complete'] is True,
         'complete original/reference artifacts required')
    need(complete['evidence_mode']==expected_mode and complete['contract_sha256']==CONTRACT,'reference mode/contract differs')
    need(complete['scientific_verdict'] is False and complete['candidate_a_fitting'] is False and
         complete['original_model_byte_restoration_claim'] is False and complete['source_run_modified'] is False,
         'reference claim scope differs')
    need(complete['original_completion_sha256']==expected_source_sha256,'reference original-source completion differs')
    config=original['configuration']
    mode='synthetic_fixture' if expected_mode=='synthetic_only' else 'exposed_development'
    need(config['mode']==mode and config['horizons']==[4,6] and all(type(h) is int for h in config['horizons']) and config['target_imputation'] is False,
         'source mode/leads/target handling differs')
    for name,value in (('extra_dst_delay_hours',0),('extra_driver_delay_hours',0),('embargo_hours',168),('gbm_seed',22022026)):
        need(type(config[name]) is int and config[name]==value,'source clock/GBM recipe differs')
    need(len(config['gbm_grid'])==8 and all(grid_configuration(a,b) for a,b in zip(config['gbm_grid'],GRID)) and
         type(complete['grid_member_count']) is int and complete['grid_member_count']==16 and
         type(complete['extra_dst_delay_hours']) is int and complete['extra_dst_delay_hours']==0,
         'complete initial grid/L0 reference required')
    need(complete['native_live_objects']==0 and type(complete['native_live_objects']) is int,'reference native objects remain')
    ref_inventory=inventory(reference,limits)
    need(ref_inventory==dict(complete['output_sha256'],**{'completion.json':expected_reference_sha256}),
         'exact reference output inventory differs')
    snapshots=bound_source_files(source,original,limits)
    snapshots.update({str(member(reference,name)):value for name,value in ref_inventory.items()})
    snapshots[str(member(source,'completion.json'))]=expected_source_sha256
    meta=source_json(source,original,'input_hourly_channels.json',limits)
    n=meta['rows'];need(is_integer(n,1) and n<=1_000_000,'raw source hour budget')
    start=hour(meta['start']);initial=hour(config['validation_start']);selection=hour(config['test_start'])
    need(initial==hour(complete['initial_boundary']) and selection==hour(complete['original_selection_boundary']),
         'reference initial/selection clocks differ')
    a_start=hour(complete['a_selection_start']);a_end=hour(complete['a_selection_end'])
    need(initial<=a_start<a_end<=selection,'A selection window exceeds original validation')
    if mode=='exposed_development':
        need(set(meta)=={'rows','channels','start','stop','encoding','sha256'} and meta['channels']==RAW_NAMES and
             meta['encoding']=='little-endian Float64 hour-major; provider raw values','original raw header schema differs')
        need((n,start,hour(meta['stop']))==(430248,datetime(1976,11,1),datetime(2025,11,30,23)) and
             original['input_sha256']==ACTUAL_SOURCE,'locked actual source coverage/identity differs')
        matrix_sha=meta['sha256']
    else:
        need(set(meta)=={'rows','start','source_sha256','raw_matrix_sha256','synthetic_only'} and
             meta['synthetic_only'] is True and meta['source_sha256']==original['input_sha256'],'synthetic raw header differs')
        matrix_sha=meta['raw_matrix_sha256']
    need(complete['actual_input_sha256']==original['input_sha256'] and
         complete['actual_full_raw_matrix_sha256']==original['raw_hourly_matrix_sha256']==matrix_sha,
         'reference full raw/source identities differ')
    need(sys.byteorder=='little','independent binary reader requires little-endian execution')
    raw_path=member(source,'input_hourly_channels.f64')
    need(raw_path.stat().st_size==17*n*8 and original['output_sha256'][raw_path.name]==matrix_sha,'raw matrix shape/binding')
    raw=array('d')
    with raw_path.open('rb') as stream:raw.fromfile(stream,17*n)
    prefix=complete['prefix_rows']
    need(is_integer(prefix,1) and prefix<=n and start+timedelta(hours=prefix)==selection,'validation-only raw prefix differs')
    prefix_raw=raw[:17*prefix]
    prefix_meta=reference_json(reference,complete,'prefix.json',limits)
    need(type(prefix_meta['rows']) is int and prefix_meta['rows']==prefix and
         prefix_meta['source_sha256']==original['input_sha256'] and hour(prefix_meta['start'])==start and
         hour(prefix_meta['last'])==selection-timedelta(hours=1),'raw prefix metadata clocks/dimensions differ')
    need(hashlib.sha256(prefix_raw.tobytes()).hexdigest()==complete['output_sha256']['prefix_raw17.f64']==prefix_meta['raw_matrix_sha256'],
         'exported prefix differs from original raw matrix')
    expected_imputed,imputation=initial_imputation(source,original,reference,complete,prefix_raw,prefix,start,initial,limits)
    imputed_digest=hashlib.sha256(expected_imputed.tobytes()).hexdigest()
    need(imputed_digest==complete['output_sha256']['prefix_imputed17.f64']==prefix_meta['imputed_matrix_sha256'],
         'independent initial imputed prefix differs')
    need([x['horizon'] for x in complete['results']]==[4,6] and
         all(type(x['horizon']) is int for x in complete['results']),'ordered separate reference leads required')
    models=[]
    for result in complete['results']:
        need(len(result['grid'])==8 and len(result['ordered_names'])==29,'reference grid/feature dimension')
        need(type(result['choice']['selected_index']) is int and 1<=result['choice']['selected_index']<=8,
             'selected grid index must be an integer in 1:8')
        for index,(record,configuration) in enumerate(zip(result['grid'],GRID),1):
            need(type(record['index']) is int and record['index']==index and grid_configuration(record['configuration'],configuration),'original grid order differs')
            need(record['roundtrip_predictions_exact'] is True,'exact saved/reload point parity absent')
            name=record['model'];need(len(relative_parts(name))==1,'flat public GBM model path required')
            need(record['model_sha256']==complete['output_sha256'][name],'GBM model binding differs')
            for field in ('scaling','predictions'):
                name=record[field]
                need(len(relative_parts(name))==1 and record[field+'_sha256']==complete['output_sha256'][name],
                     'GBM '+field+' binding differs')
            bounded_table(member(reference,record['scaling']),columns=3,maximum_rows=29,exact_rows=29)
            bounded_table(member(reference,record['predictions']),columns=17,maximum_rows=prefix)
            name=record['model']
            edges,trees,_=validate_model(decode_root(read_bson(member(reference,name))),result['ordered_names'],configuration,
                                         len(result['choice']['fit_rows']))
            models.append({'horizon':result['horizon'],'index':index,'configuration':configuration,'model_sha256':record['model_sha256'],
                'edges':edges,'trees':[dict(feature=f,condition=c,prediction=p,split=s) for f,c,p,s in trees]})
        bounded_table(member(reference,f"initial_validation_h{result['horizon']}.csv"),columns=17,maximum_rows=prefix)
        bounded_table(member(reference,f"a_join_h{result['horizon']}.csv"),columns=7,maximum_rows=prefix)
    output=Path(output)
    need(output.is_absolute() and not output.exists(),'exclusive absolute audit output required')
    parent=plain_root(output.parent)
    relative_parts(output.name)
    output=parent/output.name
    for root in (source,reference,HERE.parents[1]):
        need(not output.is_relative_to(root) and not root.is_relative_to(output),'audit output overlaps input or code tree')
    output.mkdir()
    plain={'schema':'independent_plain_initial_gbm16_v1','expected_mode':expected_mode,
           'source_completion_sha256':expected_reference_sha256,'models':models}
    plain_bytes=json.dumps(plain,allow_nan=False,separators=(',',':')).encode()
    need(len(plain_bytes)<=limits.json_bytes,'plain-model JSON output budget')
    write_new(output/'plain-trees.json',plain_bytes)
    plan={'schema':'independent_initial_reference_audit_inputs_v1','preflight_complete':True,'complete':False,
        'expected_mode':expected_mode,'reference':str(reference),'source':str(source),'input_files_sha256':snapshots,
        'reference_completion_sha256':expected_reference_sha256,'source_completion_sha256':expected_source_sha256,
        'full_rows':n,'prefix_rows':prefix,'start':meta['start'],'initial_boundary':complete['initial_boundary'],
        'selection_boundary':complete['original_selection_boundary'],'a_start':complete['a_selection_start'],'a_end':complete['a_selection_end'],
        'plain_models_sha256':hashlib.sha256(plain_bytes).hexdigest(),'independent_imputation':imputation,
        'new_fitting':False,'author_or_ml_imported':False,'scientific_verdict':False,'limits':vars(limits),
        'training_metadata_node_cap':8_000_000}
    write_new(output/'audit-plan.json',json.dumps(plan,allow_nan=False,separators=(',',':')).encode())
    for path,expected in snapshots.items():need(read_file(path,limits.file_bytes)['sha256']==expected,'input changed during preflight')
    return plan
