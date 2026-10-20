# Source-bound initial-reference selection import

This adapter connects an independently replayed initial L0 reference to the
existing 18-setting Candidate A selector. It loads data and proof records;
it does not load an author model, fit a forecaster, fetch observations, score a
test period, or authorize a scientific or operational claim.

First run the independent reader in `../initial_reference_audit`. Pin its
completed audit digest from a trusted execution record. Export to a new,
absolute output directory:

```text
python -B tools/initial_reference_import/export_reference.py --audit AUDIT_ROOT --audit-sha256 TRUSTED_AUDIT_SHA256 --output NEW_EXPORT_ROOT
```

Then use Julia from the package project:

```julia
include(joinpath("tools", "initial_reference_import", "InitialReferenceImport.jl"))
using .InitialReferenceImport
reference = load_reference(export_root, export_completion_sha256;
    audit_completion_sha256=trusted_audit_completion_sha256,
    admitted=original_source_admission)
rows = reference_rows(reference)
selection = SolarSINDy.IntervalMeanForecaster.select_setting(candidate_panels, reference;
    validation_start=validation_start, validation_end=validation_end, delay_hours=0)
```

For generated synthetic artifacts, omit `admitted`. Actual development artifacts
require the matching `BoundInitialReferenceInputs.admit` token. The importer
rechecks that token and the pinned upstream files before and after consumption.
Moving only the export directory is supported; relocating upstream evidence
requires a fresh independent audit and fresh export with explicit new roots.
There is no implicit search for artifacts or automatic model fitting.

The two forecast leads retain separate finite, mature-label cohorts and their
ordered union. An ineligible opposite slot preserves its raw target, clocks and
scoreability, with no reference prediction manufactured for selection. Required
anchor persistence is accepted only for the audited feature-unavailable status.
Selection can start after the original initial-model boundary; the fit-label
embargo remains anchored to that original boundary. The synthetic NamedTuple
entry point retains its existing equal-boundary restriction.

Both export and independent-audit completion digests are explicit trust anchors.
Rehashed export corruption cannot replace the pinned audit. Caller-fabricated
trust anchors are not proof that an independent execution occurred. Native
execution qualification, actual-data admission, and predictive evidence remain
separate requirements.

Generated tests use frozen literal clock/selection oracles, an independently
replayed data-only reference, valid output relocation, 31 rejection cases,
and a simulated input change during serialization. Test-only perfect candidate
points check selection wiring; they are not forecast-performance evidence.
