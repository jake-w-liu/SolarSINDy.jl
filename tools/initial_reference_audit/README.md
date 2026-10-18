# Independent initial-reference checks

`InitialReferenceAudit.jl` provides independent exact-hour clocks, per-lead raw
training/validation/A-window cohorts, pooled metrics and strict original-grid
selection. It imports only `Dates` and calls no author prediction or ML library.
It keeps unavailable-feature rows in validation and the A join; only training
uses feature availability. Both completion and receipt clear the inclusive
168-hour cutoff. The 5e-12 metric comparison never changes selection ordering.

These are arithmetic building blocks for a separate saved-artifact reader.
They do not admit a dataset, read model artifacts, qualify a full reference,
authorize fitting, or establish forecast skill. Full artifact/tree/imputation
replay and independent confirmation remain separate required steps.
