# Source-bound initial GBM reference

These research modules reconstruct the eight original initial GBM29 settings
for each of the 4-hour and 6-hour Dst leads. They preserve separate per-lead
training/validation masks, train-only transformations, the original first-minimum
choice, exact persistence fallback, all-grid metric parity, saved states and
source/receipt bindings. They do not run Candidate A or change live serving.

Load, in order, `DstInitialGBMReference.jl`, `InitialGBMReplay.jl`,
`BoundInitialReferenceInputs.jl`, and `AllGridInitialGBMReference.jl` in one
namespace. `BoundInitialReferenceInputs.admit(manifest, expected_sha256; roots)`
requires a caller-pinned manifest and explicit existing roots. Its token must be
rechecked before and after consumption. `AllGridInitialGBMReference.reconstruct_reference`
takes the separately loaded original module, parsed full input and exclusively
owned native library, plus `config`, `admitted`, and a new output directory.

Historical reconstruction requires the exact admitted executable location and
digest, original source13, comparator runtime388 and official native3.2.0 library.
Changing Julia versions is an unsupported replacement for that historical run;
the executor rejects a mismatched process before model/input access or output
publication. It records and rechecks the process runtime in its completion.
Portable parsing and fixed-model auditing are separate from retraining parity.

Original archive headers and generated fixture headers have distinct explicit
schemas. The archive header's `sha256` binds its parsed raw17 matrix; the raw-text
source identity comes from the admitted completed run. Fixture digest fields and
its true synthetic flag cannot be relabeled as an archive header. Original
metadata and observations are never rewritten.

Synthetic mode exercises the implementation only. Actual reconstruction also
requires separately pinned full native qualification, unshortened package/
experiments/harness evidence and root execution clearance, supplied through
`execution_permit`. Source admission or a success Boolean alone grants no such
permission. The preserved original and next-chain contracts describe the exact
recipe, evidence boundaries and remaining L0/L1 obligations. Paths in those
historical contracts record provenance; callers supply their current roots.

The selected-only `InitialGBMReplay` entry remains the preserved synthetic
prototype. Only its unchanged imputation/file helpers are reused by the all-grid
executor; it is not an actual-reference entry point. Generated reference outputs
are validation-only deterministic reconstructions, never restoration of absent
original model bytes or evidence of novel/better/operational forecasting.

The package suite includes the original 200 loader/reference assertions, process
identity and metadata-schema regressions, and independent clock/cohort/grid
oracles. The full original native source-generated grid regressions require the
externally admitted original runtime/source/native assets. Do not infer those
native fitting gates from an ordinary package run or invent replacement assets.
