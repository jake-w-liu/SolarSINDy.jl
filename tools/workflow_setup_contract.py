"""Normalize the single approved setup extension for frozen workflow oracles.

The caller still verifies the complete original workflow and its job prefix.
No other workflow bytes are normalized or accepted by this helper.
"""
import re

_CURRENT_STEP = (
    b"      - name: Resolve the package environment\n"
    b"        env:\n"
    b"          JULIA_NUM_PRECOMPILE_TASKS: '1'\n"
    b"        # resolve() does not bootstrap registries in a fresh Julia depot.\n"
    b"        run: julia --project=. -e 'using Pkg; Pkg.Registry.add(\"General\"); Pkg.resolve(); Pkg.instantiate()'\n"
)
_ORIGINAL_STEP = (
    b"      - name: Resolve the package environment\n"
    b"        # resolve() does not bootstrap registries in a fresh Julia depot.\n"
    b"        run: julia --project=. -e 'using Pkg; Pkg.Registry.add(\"General\"); Pkg.resolve(); Pkg.instantiate()'\n"
)
_BEFORE = (
    b"      - name: Verify the dashboard frontend directly\n"
    b"        run: node --test --test-reporter=tap app/test/frontend_freshness.cjs\n"
)
_AFTER = b"      - name: Confirm the checkout carries no generated study output\n"
_JOB = re.compile(rb"(?m)^  ([A-Za-z0-9_-]+):$")


def normalize_precompile_setup_extension(workflow: bytes) -> bytes:
    """Remove only the exact, required, step-local precompile-task extension."""
    if type(workflow) is not bytes:
        raise TypeError("Workflow must be bytes")
    if workflow.count(b"JULIA_NUM_PRECOMPILE_TASKS") != 1:
        raise ValueError("Exactly one setup precompile-task setting is required")
    jobs = list(_JOB.finditer(workflow))
    matches = [index for index, job in enumerate(jobs) if job[1] == b"test"]
    if len(matches) != 1:
        raise ValueError("Exactly one test job is required")
    index = matches[0]
    end = jobs[index + 1].start() if index + 1 < len(jobs) else len(workflow)
    matrix = workflow[jobs[index].start():end]
    anchored = _BEFORE + _CURRENT_STEP + _AFTER
    if (workflow.count(_CURRENT_STEP) != 1 or matrix.count(_CURRENT_STEP) != 1
            or workflow.count(anchored) != 1 or matrix.count(anchored) != 1):
        raise ValueError("Exact approved setup step, matrix scope and adjacent gates are required")
    return workflow.replace(_CURRENT_STEP, _ORIGINAL_STEP, 1)
