"""Run every reader test with CI-owned retained failure evidence; fail first."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--evidence-root',type=Path)
    args=parser.parse_args()
    selected=args.evidence_root
    if selected is None:
        selected=Path(os.environ['RUNNER_TEMP'])/'initial-reference-reader'
    if not selected.is_absolute() or selected.exists():raise ValueError('new absolute CI evidence root required')
    root=selected.parent.resolve(strict=True)/selected.name
    # Canonicalize the trusted host's temporary directory before creating test
    # inputs. Artifact paths themselves retain their strict no-alias contract.
    root.mkdir()
    for index in range(1,args.repeat+1):
        work=root/f'attempt-{index:03d}';work.mkdir()
        for name in ('TMP','TEMP','TMPDIR'):os.environ[name]=str(work)
        tempfile.tempdir=None
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        record={'attempt':index,'required_attempts':args.repeat,'complete':result.wasSuccessful(),
            'tests_run':result.testsRun,'skips':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors)}
        with (work/'test-summary.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
        print(json.dumps(record),flush=True)
        # Every repeated control must pass. A failure is retained and stops the
        # run; this never retries a failing test until it becomes green.
        if not result.wasSuccessful():return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
