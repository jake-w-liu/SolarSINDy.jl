"""Run the complete generated input/stage checks; retain failure trees in CI."""
from pathlib import Path
import json,os,sys,tempfile,unittest

sys.dont_write_bytecode=True
here=Path(__file__).resolve().parent
root=Path(os.environ.get('RUNNER_TEMP',tempfile.gettempdir()))/'initial-a-chain'
root.mkdir(exist_ok=False)
temporary=root/'tmp';temporary.mkdir()
tempfile.tempdir=str(temporary)
for name in ('TEMP','TMP','TMPDIR'):os.environ[name]=str(temporary)
suite=unittest.defaultTestLoader.discover(str(here),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
record={'complete':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),
    'errors':len(result.errors),'skips':len(result.skipped),'actual_data':False,'new_actual_fitting':False,
    'scientific_verdict':False,'failure_evidence_root':str(root)}
with (root/'terminal.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
if result.testsRun!=3 or result.skipped:raise SystemExit('All three generated executor checks are required')
raise SystemExit(0 if result.wasSuccessful() else 1)
