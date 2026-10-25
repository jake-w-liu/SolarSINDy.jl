"""Fixed synthetic source factory; no caller source/model or numerical payload input."""
from pathlib import Path
import datetime as dt,hashlib,json,struct,sys
sys.dont_write_bytecode=True
if sys.flags.optimize:
 raise RuntimeError('optimized Python is unsupported for synthetic source generation')

def create_sources(pkg,out,layout):
 assert layout in ('ground11','mixed8')
 source=pkg/'test/completed_refit_artifacts/fixtures/nonzero_source_cohorts_v2.json'
 assert source.is_file() and not source.is_symlink() and source.stat().st_size<=1048576
 raw=source.read_bytes()
 assert hashlib.sha256(raw).hexdigest()=='f4817c8f94cf318d08f3088ba0965f6f5439ec52b52acc7ce751ccb8ef1956f7'
 frozen=json.loads(raw)
 out=out/'synthetic-inputs';out.mkdir(exist_ok=False)
 def canon(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
 def sha(b):return hashlib.sha256(b).hexdigest()
 def save(p,x):p.write_bytes(canon(x));return sha(p.read_bytes())
 def clock(s):
  value=dt.datetime.fromisoformat(s);delta=value-dt.datetime(1970,1,1)
  return str(delta.days*86400000+delta.seconds*1000+delta.microseconds//1000)
 def bits(x):return struct.pack('>d',float(x)).hex()
 ground=[];wind=[]
 for window in frozen['source_rows']:
  ground.extend({'time':clock(r['center']),'xy':[bits(x) for x in r['xy']]} for r in window['ground_vectors'])
  if layout=='mixed8':wind.append({'time':clock(window['wind']['arrival_start']),'values':[bits(x) for x in window['wind']['V_Bz_By_n']]})
 ground.sort(key=lambda r:int(r['time']));wind.sort(key=lambda r:int(r['time']))
 assert len(ground)==1240 and len({r['time'] for r in ground})==1240 and len(wind)==(20 if layout=='mixed8' else 0)
 gpath=out/'synthetic-ground-vectors.json';wpath=out/'synthetic-arrival-wind.json'
 gpin=save(gpath,{'schema':'dbdt_synthetic_ground_vectors_v1','station':'SYN','product':frozen['ground_product'],'units':'nT','rows':ground})
 wpin=save(wpath,{'schema':'dbdt_synthetic_arrival_wind_v1','station':'SYN','product':frozen['wind_product'],'units':['km/s','nT','nT','cm^-3'],'rows':wind}) if layout=='mixed8' else None
 ground_names=['log_now','log_mean30','log_max30','log_std30','now_over_1pmax','mean_over_1pmax','std_over_1pmean','sin_ut','cos_ut','sin_doy','cos_doy']
 # The exact archived CF.FEATURE_NAMES is checked by the Julia protocol parser;
 # read this fixed source declaration rather than inferring it from outcome data.
 reference=pkg/'src/completed_source_dbdt/reference/dbdt_causal_forecaster.jl'
 assert hashlib.sha256(reference.read_bytes()).hexdigest()=='b2c46d28551443dc88c031b0e9cd3bf6b0e24bbba05f81d515e355afdb424922'
 import re
 text=reference.read_text(encoding='utf-8');match=re.search(r'const FEATURE_NAMES\s*=\s*\[(.*?)\]',text,re.S)
 assert match is not None
 ground_names=re.findall(r'"([^"]+)"',match.group(1));assert len(ground_names)==11
 plans={}
 for layout in (layout,):
  folder=out/layout;folder.mkdir()
  permissions=[]
  for body,path,pin,fmt,product in [('ground',gpath,gpin,'dbdt_synthetic_ground_vectors_v1',frozen['ground_product']),('wind',wpath,wpin,'dbdt_synthetic_arrival_wind_v1',frozen['wind_product'])]:
   approved=body=='ground' or layout=='mixed8'
   permissions.append({'body_id':body,'path_or_null':str(path) if approved else None,'bytes_or_null':path.stat().st_size if approved else None,'sha256_or_null':pin if approved else None,'format':fmt,'period_start':clock('2017-01-01T00:00:00'),'period_stop':clock('2018-01-01T00:00:00'),'station':'SYN','product':product,'neighbor_role':'primary','disposition':'approved' if approved else 'absent','approval_sha256':sha(b'explicit-synthetic-source-factory-v1')})
  for year in (2018,2023,2024,2025):
   permissions.append({'body_id':'forbidden'+str(year),'path_or_null':None,'bytes_or_null':None,'sha256_or_null':None,'format':'not_opened_actual_source','period_start':clock(f'{year}-01-01T00:00:00'),'period_stop':clock(f'{year+1}-01-01T00:00:00'),'station':'SYN','product':'forbidden_actual_product','neighbor_role':'right','disposition':'unapproved','approval_sha256':sha(b'explicit-forbidden-synthetic-oracle-v1')})
  permissions.sort(key=lambda r:r['body_id'])
  splits=[]
  for name,start,stop,boundary,minimum in [('fit','2017-01-01T00:00:00','2017-10-01T00:00:00','2017-10-01T00:00:00',3),('october','2017-10-01T00:00:00','2017-11-01T00:00:00','2017-11-01T00:00:00',9),('final','2017-10-01T00:00:00','2018-01-01T00:00:00','2018-01-01T00:00:00',9)]:
   splits.append({'name':name,'issue_start_inclusive':clock(start),'issue_stop_exclusive':clock(stop),'label_ready_before':clock(boundary),'target_admit_asof':clock(boundary),'min_supervised_rows':minimum})
  recipes=[{'family':'ridge','role':'fixed_mixed8' if layout=='mixed8' else 'candidate','penalty':bits(penalty),'intercept_penalty':layout=='mixed8','std_rule':'sample_std_plus_1e_minus9' if layout=='mixed8' else 'sample_std_zero_nonfinite_to_one','split':'fit','selection_decision_pin':None} for penalty in ([1.] if layout=='mixed8' else [.01,.1,1.,10.,100.])]
  protocol={'schema':'dbdt_completed_refit_protocol_v1','evidence_mode':'synthetic_only','protocol_id':'nonzero-synthetic-'+layout+'-v2','station':'SYN','ground_product':frozen['ground_product'],'wind_product_or_null':frozen['wind_product'] if layout=='mixed8' else None,'units':'ground_nT_rate_nT_per_min_wind_km_s_nT_cm3' if layout=='mixed8' else 'ground_nT_rate_nT_per_min','layout':layout,'feature_names':['dbdt_now','dbdt_mean30','dbdt_max30','dbdt_std30','V','Bz','Bs','VBs'] if layout=='mixed8' else ground_names,'cohort_kind':'natural','policy':frozen['policy_ms'],'source_permissions':permissions,'neighbor_policy':'explicit_approved_only','splits':splits,'minima_mode':'synthetic_oracle','model_recipes':recipes,'selection_rule':'frozen_2017_mse_first_exact_tie_ridge_boost_forest_v1','runtime_contract':'owned_julia_1_12_6_or_1_13_1_single_threaded_v1','numeric_contract':'directed_ridge_fit_v1_portable_prediction_v2','availability_kind':'assumed_completed_source_intervals','measured_receipts_available':False}
  ppath=folder/'protocol.json';ppin=save(ppath,protocol)
  plans[layout]={'directory':str(folder),'protocol_path':str(ppath),'protocol_sha256':ppin,'issues':[clock(w['issue']) for w in frozen['source_rows']],'source_permission_sha256':sha(canon(permissions)),'recipe_sha256':sha(canon(next(r for r in recipes if r['penalty']==bits(1.0))))}
 return plans[layout]
