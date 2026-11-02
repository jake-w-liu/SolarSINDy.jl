"""Fixed campaign plan only; no supplied grants, source reads or worker launch."""
from pathlib import Path
import importlib.util,json,sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1]
path=root/'research/dbdt_completed_source/run_candidate_campaign.py'
spec=importlib.util.spec_from_file_location('fixed_candidate_campaign_plan_example',path)
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
def run_example():
    rows=campaign.queue()
    if len(rows)!=142 or sum(r['operation'] in ('candidate-fit','selected-refit') for r in rows)!=38:
        raise RuntimeError('fixed campaign plan changed')
    result=dict(schema='unqualified_campaign_plan_example_v1',stages=rows,stage_count=142,fit_operations=38,
        worker_launches=0,actual_fits=0,actual_source_access=False,authority_granted=False,scientific_qualification=False)
    print(json.dumps(result,sort_keys=True));return result
if __name__=='__main__':run_example()
