# Independent initial-reference checks

`InitialReferenceAudit.jl` provides independent exact-hour clocks, per-lead raw
training/validation/A-window cohorts, pooled metrics and strict original-grid
selection. It imports only `Dates` and calls no author prediction or ML library.
It keeps unavailable-feature rows in validation and the A join; only training
uses feature availability. Both completion and receipt clear the inclusive
168-hour cutoff. The 5e-12 metric comparison never changes selection ordering.

`audit_reference.py` adds a data-only saved-artifact reader. It independently
replays the original and reconstructed initial imputers, all 16 stored GBM
models, sample scaling, full per-lead cohorts, fallback predictions, metrics,
strict first-minimum selection and separate candidate joins. It checks the
original imputer training fingerprint and every serialized forecast clock.
Float64 feature binning and per-tree Float32 accumulation are preserved. Saved
Float64 scaling parameters remain model state after independent moment checks.
The stored mean is first checked independently. Corrected variance is then
reconstructed about that verified stored mean, with compensated accumulation
and the source's exact zero-to-unit rule. Recomputing another rounded mean can
change a constant column's tiny positive deviation to zero; applying the unit
replacement there would compare different finite-precision states. Every saved
scale and prediction retains the same 5e-12 comparison bound.

Use Python 3.11 or later and a Julia executable with this package environment
already instantiated. No packages are fetched by the reader. Pass explicit
absolute canonical paths and independently pinned lowercase SHA256 values:

```sh
python -B tools/initial_reference_audit/audit_reference.py \
  --reference /absolute/reference --reference-sha256 REFERENCE_COMPLETION_SHA256 \
  --source /absolute/original --source-sha256 ORIGINAL_COMPLETION_SHA256 \
  --mode synthetic_only --output /absolute/new-audit-directory \
  --julia /absolute/julia --julia-sha256 JULIA_EXECUTABLE_SHA256 --timeout 1800
```

Use `exposed_development_reconstruction` only for the separately admitted,
locked historical source. Synthetic output cannot be relabeled as actual.
The historical reconstruction runtime remains separately execution-bound;
this data-only reader supports both tested Julia versions without retraining.

Input files are read only. Paths cannot escape admitted roots or follow links,
reparse points or aliases. File counts, expanded bytes, JSON bytes, lexical
depth/nodes/strings and BSON structure are bounded before interpretation. The
large original training metadata uses an explicit 8,000,000-node allowance;
other JSON retains the 4,000,000-node default. The original source completion's
declared outputs are all hashed; unlisted later legacy audit receipts are not
followed. The reference inventory is exact. Concurrent hostile file replacement
is outside the contract; inputs and controls must be quiescent during replay.

Julia runs inside a private Windows Job Object or a new POSIX session. Owned
descendants are cleaned on normal exit, timeout and interruption. A completion
is created exclusively after numerical success and input/control rechecks;
failures retain incomplete invocation and failure records. The output must be
new and outside both input roots and the code tree.

This checks numerical/artifact consistency. Independent source admission,
execution lineage, fitting authorization, independent forecast confirmation,
novelty and serving qualification are separate gates. No reader receipt asserts
forecast superiority or authorizes fitting or deployment.

The maintained compressed fixture contains only generated nonconstant source
and reference artifacts. Its manifest records the sole normalization of an
unused executable path. It qualifies numerical replay and corruption rejection,
not execution lineage or real-data performance. Run its no-fit checks with:

```sh
python -B -m unittest discover -s test/initial_reference -p 'test_*.py' -v
```

Set `SOLARSINDY_AUDIT_TEST_JULIA` to an absolute executable to select a runtime;
otherwise tests resolve `julia` from `PATH`. Failed full-reader/process controls
retain their owned temporary evidence and print its location.
