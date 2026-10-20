"""Export exact separate-lead initial-reference rows after pinned artifact audit.

This never fits a model, fetches observations, scores a test period or authorizes
an actual experiment. The caller must pin a trusted completed independent audit.
"""
import argparse
from array import array
import csv
from datetime import timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
AUDITOR=HERE.parent/'initial_reference_audit'
sys.path.insert(0,str(AUDITOR))
from artifact_io import (Limits,need,is_digest,is_integer,plain_root,member,read_file,json_file,inventory,relative_parts)
from bounded_tables import bounded_table
from prepare_artifact_audit import hour,write_new

CONTRACT='db833f075e82ad5a15e10587e538c9688958a0fac180ca5713405da43d145ff7'
SOURCE_FILES=tuple('src/interval_mean_forecaster/'+name for name in (
    'DstIntervalObserver.jl','DstIntervalReplay.jl','sandu_fullchannel_features.jl','DstIntervalAdapter.jl',
    'DstIntervalStream.jl','DstIntervalTraining.jl','DstIntervalArtifacts.jl','DstIntervalExperiment.jl'))
OWN_FILES=('tools/initial_reference_import/export_reference.py',
    'tools/initial_reference_import/InitialReferenceImport.jl',
    'tools/initial_reference/DstInitialGBMReference.jl','tools/initial_reference/BoundInitialReferenceInputs.jl')
FIELDS=['origin_row','issue','source_sha256','dst_delay_hours','driver_tag','driver_receipt',
    'anchor_tag','anchor_complete','anchor_receipt','anchor']
for h in (4,6):FIELDS += [f'h{h}_{name}' for name in
    ('target_tag','target_complete','target_receipt','target','scoreable','selection_eligible','prediction','status')]

def digest(path):return read_file(path,Limits().file_bytes)['sha256']
def date(t):return t.isoformat(timespec='seconds')
def number(x):return repr(float(x)) if math.isfinite(x) else 'NaN'
def same_number(a,b):return a==b or math.isnan(a) and math.isnan(b)

def code_identity():
    return {name:digest(member(plain_root(PROJECT),name)) for name in (*SOURCE_FILES,*OWN_FILES)}

def verified_audit(audit,expected):
    audit=plain_root(audit)
    proof=json_file(member(audit,'completion.json'),expected)
    need(proof['schema']=='independent_initial_reference_artifact_audit_v1' and proof['complete'] is True,
         'complete pinned independent audit required')
    mode=proof['expected_mode'];need(mode in ('synthetic_only','exposed_development_reconstruction'),'unknown audit evidence mode')
    for name in ('new_fitting','author_or_ml_imported','scientific_verdict','execution_lineage_qualification','serving_promotion'):
        need(proof[name] is False,'independent audit scope differs: '+name)
    need(type(proof['native']['native_exit']) is int and proof['native']['native_exit']==0 and
         proof['native']['timed_out'] is False,'independent numerical stage did not pass')
    need(inventory(audit)==dict(proof['output_sha256'],**{'completion.json':expected}),'independent audit output inventory changed')
    plan_path=member(audit,'prepared/audit-plan.json')
    plan=json_file(plan_path,proof['output_sha256']['prepared/audit-plan.json'])
    numerical=proof['numerical']
    need(plan['schema']=='independent_initial_reference_audit_inputs_v1' and plan['preflight_complete'] is True and
         plan['complete'] is False and numerical['complete'] is True and numerical['expected_mode']==mode and
         numerical['input_plan_sha256']==digest(plan_path),'independent numerical/preflight binding differs')
    need(plan['expected_mode']==mode and plan['reference_completion_sha256']==proof['reference_completion_sha256'] and
         plan['source_completion_sha256']==proof['source_completion_sha256'],'independent input identities differ')
    for object in (plan,numerical):
        for name in ('new_fitting','author_or_ml_imported','scientific_verdict'):
            need(object[name] is False,'independent prepared/numerical scope differs: '+name)
    need(is_integer(numerical['checks'],1) and is_integer(numerical['model_points'],1) and
         is_integer(numerical['fallback_points']),'independent numerical counts differ')
    roots={'reference':plain_root(plan['reference']),'source':plain_root(plan['source']),'audit':audit}
    need(len(set(roots.values()))==3 and all(not a.is_relative_to(b) for a in roots.values() for b in roots.values() if a!=b),
         'separate ordinary proof roots required')
    pinned={str(member(audit,name)):value for name,value in proof['output_sha256'].items()}
    pinned[str(member(audit,'completion.json'))]=expected
    need(isinstance(plan['input_files_sha256'],dict) and 1<=len(plan['input_files_sha256'])<=512,'bounded input proof required')
    for text,value in plan['input_files_sha256'].items():
        path=Path(text)
        need(path.is_absolute() and any(path.is_relative_to(roots[role]) for role in ('source','reference')),
             'independent input escaped supplied proof roots')
        checked=member(plain_root(path.parent),path.name)
        need(is_digest(value) and digest(checked)==value,'independent input changed')
        pinned[str(checked)]=value
    need(isinstance(proof['controls_sha256'],dict) and 1<=len(proof['controls_sha256'])<=64,'bounded independent control proof required')
    controls={}
    for text,value in proof['controls_sha256'].items():
        path=Path(text);need(path.is_absolute(),'absolute independent control required')
        checked=member(plain_root(path.parent),path.name)
        need(is_digest(value) and digest(checked)==value,'independent control changed')
        need(path.name not in controls,'ambiguous independent control basename')
        controls[path.name]=value;pinned[str(checked)]=value
    for path in AUDITOR.iterdir():
        if path.suffix in ('.py','.jl'):
            need(controls.get(path.name)==digest(member(plain_root(AUDITOR),path.name)),
                 'completed audit uses a different maintained reader implementation')
    reference=json_file(member(roots['reference'],'completion.json'),proof['reference_completion_sha256'])
    original=json_file(member(roots['source'],'completion.json'),proof['source_completion_sha256'])
    need(reference['complete'] is True and reference['evidence_mode']==mode and reference['contract_sha256']==CONTRACT,
         'reference contract/mode differs')
    need(reference['original_completion_sha256']==proof['source_completion_sha256'] and
         reference['extra_dst_delay_hours']==0 and type(reference['extra_dst_delay_hours']) is int,'initial L0 reference required')
    need(reference['scientific_verdict'] is False and reference['candidate_a_fitting'] is False,'reference scope differs')
    return proof,plan,roots,reference,original,pinned

def expected_cohorts(dst,start,initial,end,a_start,a_end):
    need(start<initial<=a_start<a_end<=end,'unordered reference selection window')
    result={4:[],6:[]}
    for h in (4,6):
        for i in range(1,len(dst)-h+1):
            issue=start+timedelta(hours=i)
            if (a_start<=issue<a_end and issue+timedelta(hours=h)<=a_end-timedelta(hours=168) and
                    math.isfinite(dst[i-1]) and math.isfinite(dst[i+h-1])):
                result[h].append(i)
        need(result[h],'empty separate lead selection cohort')
    return result

def joined_rows(root,h,expected,dst,start,completion):
    name=f'a_join_h{h}.csv';path=member(root,name)
    need(digest(path)==completion['output_sha256'][name],'joined reference changed')
    bounded_table(path,columns=7,maximum_rows=len(dst),exact_rows=len(expected))
    result={}
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        need(set(reader.fieldnames)=={'key','row','issue','target','anchor','prediction','status'},'joined reference schema differs')
        for position,row in enumerate(reader):
            need(row['row'].isascii() and row['row'].isdigit(),'joined row index grammar')
            i=int(row['row']);need(i==expected[position] and i not in result,'complete ordered per-lead join differs')
            anchor,target,prediction=(float(row[key]) for key in ('anchor','target','prediction'))
            need(all(math.isfinite(x) for x in (anchor,target,prediction)) and anchor==dst[i-1] and
                 target==dst[i+h-1] and hour(row['issue'])==start+timedelta(hours=i),'joined raw value or issue differs')
            need(row['status'] in ('model','persistence_feature_unavailable'),'unknown joined reference prediction status')
            need(row['status']!='persistence_feature_unavailable' or prediction==anchor,'reference fallback is not exact required-anchor persistence')
            result[i]=(prediction,row['status'])
    return result

def export_reference(*,audit,audit_sha256,output):
    need(is_digest(audit_sha256),'caller-pinned audit completion SHA256 required')
    own=code_identity()
    proof,plan,roots,reference,original,pinned=verified_audit(audit,audit_sha256)
    start=hour(plan['start']);initial=hour(plan['initial_boundary']);end=hour(plan['selection_boundary'])
    a_start=hour(plan['a_start']);a_end=hour(plan['a_end']);n=plan['prefix_rows']
    need(is_integer(n,1) and n<=500_000 and start+timedelta(hours=n)==end,'bounded pre-selection prefix required')
    raw_path=member(roots['reference'],'prefix_raw17.f64')
    need(raw_path.stat().st_size==17*n*8 and digest(raw_path)==reference['output_sha256'][raw_path.name],'raw prefix binding differs')
    need(sys.byteorder=='little','little-endian execution required')
    raw=array('d')
    with raw_path.open('rb') as stream:raw.fromfile(stream,17*n)
    dst=[value if math.isfinite(value) and value<99999. else math.nan for value in raw[14::17]]
    del raw
    cohorts=expected_cohorts(dst,start,initial,end,a_start,a_end)
    joined={h:joined_rows(roots['reference'],h,cohorts[h],dst,start,reference) for h in (4,6)}
    union=sorted(set(cohorts[4])|set(cohorts[6]))
    need(union and max(union)+6<=n,'union-row target slots exceed admitted pre-selection prefix')
    source=original['input_sha256'];need(is_digest(source) and reference['actual_input_sha256']==source,'source identity differs')
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=FIELDS,lineterminator='\n');writer.writeheader()
    for i in union:
        issue=start+timedelta(hours=i);tag=issue-timedelta(hours=1);anchor=dst[i-1]
        row=dict(origin_row=i,issue=date(issue),source_sha256=source,dst_delay_hours=0,
            driver_tag=date(tag),driver_receipt=date(issue),anchor_tag=date(tag),
            anchor_complete=date(issue),anchor_receipt=date(issue),anchor=number(anchor))
        for h in (4,6):
            target=dst[i+h-1];complete=issue+timedelta(hours=h);eligible=i in joined[h]
            prediction,status=joined[h][i] if eligible else (math.nan,'outside_selection_cohort')
            row.update({f'h{h}_target_tag':date(complete-timedelta(hours=1)),f'h{h}_target_complete':date(complete),
                f'h{h}_target_receipt':date(complete),f'h{h}_target':number(target),
                f'h{h}_scoreable':int(math.isfinite(anchor) and math.isfinite(target)),
                f'h{h}_selection_eligible':int(eligible),f'h{h}_prediction':number(prediction),f'h{h}_status':status})
        writer.writerow(row)
    data=buffer.getvalue().encode();need(len(data)<=Limits().file_bytes,'selection row byte budget')
    output=Path(output);need(output.is_absolute() and not output.exists(),'new absolute selection output required')
    parent=plain_root(output.parent);relative_parts(output.name);output=parent/output.name
    for root in (*roots.values(),PROJECT):
        need(not output.is_relative_to(root) and not root.is_relative_to(output),'selection output overlaps proof or code')
    for path,value in pinned.items():need(digest(path)==value,'upstream proof changed during export')
    need(code_identity()==own,'export implementation changed')
    output.mkdir();write_new(output/'selection_rows.csv',data)
    # Publication follows serialization and a second input/code check. A failed
    # write or changed upstream proof leaves no completion record to consume.
    need(inventory(output)=={'selection_rows.csv':hashlib.sha256(data).hexdigest()},
         'serialized selection rows differ')
    for path,value in pinned.items():need(digest(path)==value,'upstream proof changed during serialization')
    need(code_identity()==own,'export implementation changed during serialization')
    metadata={'schema':'source_bound_initial_reference_selection_v1','complete':True,'evidence_mode':proof['expected_mode'],
        'contract_sha256':CONTRACT,'source_sha256':source,'delay_hours':0,'training_boundary':date(initial),
        'validation_start':date(a_start),'validation_end':date(a_end),'prefix_start':date(start),'prefix_rows':n,
        'row_count':len(union),'cohort_counts':[len(cohorts[h]) for h in (4,6)],
        'audit_root':str(roots['audit']),'audit_completion_sha256':audit_sha256,
        'reference_root':str(roots['reference']),'reference_completion_sha256':proof['reference_completion_sha256'],
        'source_root':str(roots['source']),'source_completion_sha256':proof['source_completion_sha256'],
        'upstream_sha256':pinned,'implementation_sources':own,
        'output_sha256':{'selection_rows.csv':hashlib.sha256(data).hexdigest()},
        'new_fitting':False,'scientific_verdict':False,'serving_promotion':False}
    write_new(output/'completion.json',json.dumps(metadata,allow_nan=False,separators=(',',':')).encode())
    return metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','audit-sha256','output'):parser.add_argument('--'+name,required=True)
    result=export_reference(**vars(parser.parse_args()))
    print(json.dumps({'complete':result['complete'],'evidence_mode':result['evidence_mode'],
        'row_count':result['row_count'],'cohort_counts':result['cohort_counts'],'scientific_verdict':False}))

if __name__=='__main__':main()
