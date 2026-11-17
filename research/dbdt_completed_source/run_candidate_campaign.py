"""Run the full fixed16 candidate selection and three separate refits for one station.

Uses an ordinary completed_source_candidate_request_v2 with operation=campaign.
Invoke once for each station's admitted table. The process result records the
actual execution; no prior workflow, resource or permission certificate is used.
"""
from pathlib import Path
import importlib.util
path=Path(__file__).with_name('run_candidate_stage.py')
spec=importlib.util.spec_from_file_location('completed_source_candidate_cli',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
if __name__=='__main__':raise SystemExit(driver.main(required_operation='campaign'))
