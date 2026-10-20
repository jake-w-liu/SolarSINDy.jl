# Initial L0 candidate executor

This separate research tool consumes a pinned selection export from
`initial_reference_import`. It parses the raw prefix, streams nine physical
settings, fits their eighteen fixed gain configurations with the existing
full-matrix disk fitter, saves every initial model and compares all eighteen
panels with the audited reference. A selection can return `none`.

Start Julia with one Julia thread and one BLAS thread. Include
`tools/actual_a_chain/ActualAChain.jl`, then call:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
reference = ActualAChain.load_reference(reference_directory, reference_sha256;
    audit_completion_sha256=audit_sha256, admitted=admitted_source)
result = ActualAChain.run_initial_l0(reference, new_output_directory;
    execution_permit=permit, chunk_hours=24)
closed = ActualAChain.verify_initial_l0(result.directory,
    result.completion_sha256, reference; execution_permit=permit)
```

For generated references, `admitted_source` and `permit` must both be `nothing`.
Actual references require a matching `ReferenceImport.B.admit` token. Actual
fitting additionally requires an `InitialFits.InitialExecutionPermit` whose
caller-pinned JSON binds a complete qualification of the exact executor, the
admitted source, reference audit, Julia executable and stage limits. Constructing
an input token or loading an actual reference grants no fitting permission.

Use an existing ordinary parent and a new output directory outside all input,
proof and code roots. Failures retain partial output and `failed.toml`; a failed
post-publication check invalidates the completion marker. Never resume by
overwriting partial output. Preserve it and diagnose the failure first.

`verify_initial_l0` replays the raw state/design, rechecks the complete joint
mask, saved solver certificates, checkpoints, point bytes and selection without
new gain fits. It shares the independently tested state kernel and checks
artifact closure; it is not an independent optimizer or physical model.

The full raw source parser materializes a matrix before taking the pre-selection
prefix. The subsequent stream uses bounded chunks and retains only the required
selection union, complete disk fit records and final checkpoints. It does not
write eighteen full-history panels or hourly checkpoints.

This stage performs eighteen initial L0 fits and no final refit, L1 fit, test
score, uncertainty calibration or serving promotion. The legacy saved bundle
mode remains unqualified research. Generated checks, release qualification and
actual reconstruction cannot establish forecast superiority or a pristine
holdout. See `EXECUTION_CONTRACT.md` for the fixed scientific scope and gates.
