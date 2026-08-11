# V2.2 Research Test Report

## Coverage

The focused suite contains 342 assertions across the new V2.2 surfaces:

| Surface | Assertions |
|---|---:|
| Constrained SINDy-dominant stack | 51 |
| Sparse residual | 52 |
| Portable boosted residual | 76 |
| Causal served replay | 100 |
| Leakage-safe primary cross-fit | 63 |

The tests cover constraint projection, pooled fallback, synthetic recovery,
causal feature construction, split embargoes, target maturity, lag and horizon
semantics, exact live-kernel identity, deterministic fitting, portable tree
inference, checksummed artifact round trips, and corruption rejection.

## Independent expectations

- Mixture forecasts are checked against hand-computed convex combinations and
  an exact SINDy-family weight floor.
- Synthetic tables recover known weights and known sparse supports.
- Extracted flat-tree inference is checked against EvoTrees CPU predictions,
  including values exactly on split thresholds.
- Replay predictions are checked against the live V2.1 kernel at all supported
  model steps.
- Cross-fit rows are checked for exact key coverage, whole anchors, and a 168 h
  target-to-evaluation-block gap.

## Tolerances

Exact structural, checksum, key, split, and identity claims use equality.
Well-conditioned floating-point hand calculations use tolerances at or below
`1e-12`. Synthetic constrained-weight recovery uses `2e-8`, commensurate with
the projected-gradient stopping tolerance and tight enough to catch a changed
weight, projection, or component order. The portable tree comparison preserves
EvoTrees' Float32 accumulation and uses exact or near-machine-precision
expectations at threshold edges. Scientific promotion margins are contract
values, not adjustable test tolerances.

## Anti-false-test checks

Mutation tests alter targets, post-issue drivers, split boundaries, artifact
payloads, feature order, lag metadata, and source hashes. Each mutation must
either leave a causal prediction unchanged or trigger the specified fail-closed
error. This makes the tests sensitive to sign, indexing, leakage, schema,
serialization, and boundary errors rather than only checking successful
execution.

## Full verification

- Clean full package suite: 4,896/4,896 assertions passed.
- Deterministic package experiment: `SolarSINDy experiments: V2.1 deterministic
  smoke PASS`.
- Development harness: 147 passes, 3 warnings, and 0 failures. One warning is
  the conservative loose-tolerance scan documented above; the other two report
  that this package-only worktree does not contain the workspace manuscript
  data and figure directories.

## Debug and reverify record

The first full-suite run reached 4,891 passing assertions and one setup error:
the isolated worktree did not contain the ignored OMNI source cache required by
an existing V2.1 replay oracle. Pointing the test at a differently named source
path was correctly rejected by the existing provenance guard. The canonical
workspace source artifact was then mounted at the worktree's expected ignored
cache path, preserving both file content and path identity. No source behavior,
test expectation, or tolerance was weakened. A clean default-path rerun is the
decisive result.
