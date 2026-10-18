# Initial GBM reference research tools

The modules under `tools/initial_reference/` support source-bound reconstruction
of the original initial GBM29 validation reference. They preserve all eight fixed
settings for each lead, separate training and validation cohorts, train-only
transforms, the original metric comparison, deterministic first-minimum choice,
and persistence fallbacks. They do not change the operational forecaster.

Load `DstInitialGBMReference.jl`, `InitialGBMReplay.jl`,
`BoundInitialReferenceInputs.jl`, then `AllGridInitialGBMReference.jl` in one
namespace. The loader requires a caller-pinned manifest and explicit roots.
Historical reconstruction requires the exact admitted original source, runtime,
Julia executable and native library. A different process binary is rejected
before model access or output publication. Reading fixed saved states and
refitting historical models have distinct reproducibility requirements.

Original archive and generated fixture headers have separate explicit schemas.
The archive matrix digest comes from its original `sha256` field, while its raw
source identity comes from the admitted completed run. Synthetic digest fields
and flags do not qualify an archive schema. Artifacts are written only to a new
directory, with a completion record last and source/runtime rechecks at exit.

Actual reconstruction additionally requires pinned native qualification, full
package/experiment/harness evidence and execution clearance. Source admission
alone does not establish an independently verified reference. The preserved
selected-only prototype remains synthetic; its unchanged helpers support the
separate all-grid executor. See `tools/initial_reference/README.md` and the two
preserved contracts for arguments and the exact evidence obligations.

`tools/initial_reference_audit/InitialReferenceAudit.jl` supplies independent
clock, cohort, pooled-metric and grid-selection arithmetic. Validation keeps
each lead's full finite-anchor/target cohort, including fallback rows. Feature
availability restricts baseline training; it does not delete validation rows.
Both completion and receipt satisfy the inclusive 168-hour maturity cutoff.

These arithmetic checks are building blocks for a separate full saved-artifact
reader. They do not load artifacts, admit data, authorize fitting, or establish
forecast skill. Candidate A real-data fitting, the L1 reference, independent
saved-state closure, comparative accuracy and novelty remain distinct gates.
