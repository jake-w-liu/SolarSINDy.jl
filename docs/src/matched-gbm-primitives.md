# Matched342 numerical primitives

`SolarSINDy.MatchedGBMPrimitives342` provides a receipt-aware feature builder,
explicit training-only finite encoder, fixed-grid generated-data fit capsule and
independent scalar tree replay. It does not implement actual-source admission,
the complete32-fit matched producer, B22 qualification or a serving path.

The342 ordered coordinates retain the maintained29-feature block, append eight
lags for each of17 raw channels, preserve165 pre-imputation missing flags, and
append12 observation/driver/trace fields. Required anchors and targets are never
imputed. Late receipts mask raw features. Missing, physically unusable and
left-censored trace histories retain their explicit status; an initial run's
left-censor provenance persists at49+. Anchor/target-scoreable issue cohorts are not
filtered by feature completeness. Raw infinity/Bool and derived infinity reject.

Every cell uses only its explicit eligible training rows for finite-value mean
imputation and sample-standard-deviation scaling. Constants use scale1. The new
one-row primitive also uses scale1; the maintained scaler's minimum-two-row API
is unchanged. Parameters are immutable. Training row storage has a fixed wrapper
type with checked Int indexing over owned four-byte packed indices, bounded at
500000. The codec stores an ordinary ordered integer array. Max-length storage
checks do not establish a500000-row full-fit memory budget.

Fixed settings are depth4/6, rounds200/400 and bins64/255, bins varying fastest,
with seed22022026 and the maintained pinned CPU MSE recipe. A generated-data fit
captures this recipe before fitting and copies native arrays into immutable
replay data before returning. A capsule writer accepts only that controlled
result and independent expected context. Mutating the separately returned native
model cannot relabel its prior capsule. Synthetic admission never grants actual
fit or release authority.

Model members have21 closed fields and trees have four. Float64 edge and Float32
leaf bits are exact lowercase hex. Replay searches raw Float64 values against
Float64 edges, branches on bin≤condition, accumulates trees in Float32 order and
widens the result to Float64. The original29-feature replay namespace remains
unchanged. File bytes, integer tokens, depth, cumulative values, containers,
strings and duplicate keys are checked before JSON parsing. External content,
cohort, preprocessing, source, cell/grid and admission pins are required.

These primitives carry no held-out skill verdict. The complete matched32-fit
producer, full per-lead evaluation UNION, native evidence and B22 actual
admission remain separate required stages. The diagnostic old29 reference and
all stronger mandatory baseline families remain in scope.

```jldoctest
julia> using SolarSINDy

julia> length(SolarSINDy.MatchedGBMPrimitives342.FEATURE_NAMES)
342
```

```@docs
SolarSINDy.MatchedGBMPrimitives342
SolarSINDy.MatchedGBMPrimitives342.HourlyInputs342
SolarSINDy.MatchedGBMPrimitives342.FeatureRow342
SolarSINDy.MatchedGBMPrimitives342.FEATURE_NAMES
SolarSINDy.MatchedGBMPrimitives342.FEATURE_SCHEMA_SHA256
SolarSINDy.MatchedGBMPrimitives342.foreach_feature
SolarSINDy.MatchedGBMPrimitives342.build_features
SolarSINDy.MatchedGBMPrimitives342.trace_context
SolarSINDy.MatchedGBMPrimitives342.cohorts
SolarSINDy.MatchedGBMPrimitives342.Encoder342
SolarSINDy.MatchedGBMPrimitives342.fit_encoder
SolarSINDy.MatchedGBMPrimitives342.transform
SolarSINDy.MatchedGBMPrimitives342.source_identity
SolarSINDy.MatchedGBMPrimitives342.save_encoder
SolarSINDy.MatchedGBMPrimitives342.load_encoder
SolarSINDy.MatchedGBMPrimitives342.preprocessing_sha256
SolarSINDy.MatchedGBMPrimitives342.SyntheticAdmission342
SolarSINDy.MatchedGBMPrimitives342.ExpectedModelBindings342
SolarSINDy.MatchedGBMPrimitives342.FittedCell342
SolarSINDy.MatchedGBMPrimitives342.GRID342
SolarSINDy.MatchedGBMPrimitives342.prepare_synthetic_bindings
SolarSINDy.MatchedGBMPrimitives342.fit_synthetic_cell
SolarSINDy.MatchedGBMPrimitives342.save_model
SolarSINDy.MatchedGBMPrimitives342.load_model
SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342
SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342.Tree
SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342.Model
SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342.predict_delta
SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342.predict_deltas
```
