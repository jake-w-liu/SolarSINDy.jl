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

`tools/initial_reference_audit/audit_reference.py` independently reads the saved
artifacts without author prediction code, Julia type resolution or an ML model
loader. It reconstructs initial imputation, decodes all 16 saved trees, replays
features and sample scaling, verifies every validation point and fallback, and
checks metrics, selection and the separate candidate joins. Float64 binning and
per-tree Float32 accumulation are retained. The original imputer fingerprint is
computed independently from cleaned raw channels and eligible hour tags.
Stored means are checked against independent moments before corrected variance
is reconstructed about those stored means. This preserves the source's rounded
state and exact zero-to-unit convention for constant features. The saved scale,
every forecast and the original comparison bound remain unchanged.

The entry point requires Python 3.11 or later, an already instantiated package
environment, explicit canonical paths, caller-pinned completion/executable
digests and an explicit evidence mode. The output is new and separate from code
and inputs. Input sizes, counts and structured data are bounded; the Julia stage
has a timeout and owns its descendant processes on Windows and POSIX. A complete
receipt follows numerical success and input/control rechecks. See the reader
README for command arguments, budgets and the quiescent-input requirement.

The compressed regression fixture contains only source-generated nonconstant
synthetic data. Its unused executable path is normalized as recorded in its
manifest. The tests verify saved-state consistency, rehashed corruption
rejection, future-tail invariance and process cleanup; they do not qualify
execution lineage or real-data performance. Candidate A real-data fitting, the
L1 reference, actual-artifact closure, independent confirmation, comparative
accuracy and novelty remain distinct gates. No reader result authorizes fitting
or changes the operational forecaster.
