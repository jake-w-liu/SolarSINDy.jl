# Completed-source dB/dt package port contract

This bounded stage ports the frozen V2 synthetic prototype to an additive
`SolarSINDy.CompletedSourceDBDT` research namespace. The exact parent is
`fb23bdc6ed43d4a40bbf209d3e1085883cbdd337`. No actual raw source, case, fitted
model, outcome, figure, paper, venue or historical evidence is opened or changed.
No commit, push or full heavy qualification is authorized here.

## Contract and immutable inputs

- Copy the qualified feature/model reference as raw bytes under
  `src/completed_source_dbdt/reference/dbdt_causal_forecaster.jl` and retain
  SHA256 `b2c46d28551443dc88c031b0e9cd3bf6b0e24bbba05f81d515e355afdb424922`.
  The local reference is internal. In particular, its legacy raw-tag
  `ground_cases`, loaders and fitters are not imported or re-exported.
- Import sibling `CompletedMinuteReplay`; never depend on an external source
  path or include a second SolarSINDy module. Default whole-minute ground
  anchor is issue minus one minute; target centers are issue+1:issue+30;
  label readiness is final center+45 seconds and the default sufficient
  phase period is31 whole minutes.
- Preserve all four production prototype bodies (adapter, controllers,
  maximum, module scalar/time helpers). Changes permitted by this port are
  namespace/import/location adaptation and public docstrings. No algorithm,
  scalar guard, policy, numeric tolerance, cohort, target or controller change.
- Preserve all479 frozen independent assertions. The module name and scalar
  oracle location may change; all literals, expected values and tolerances
  remain exact. Copy both original oracle JSON byte streams without decoding
  and rewriting them. Add separate package-boundary assertions.
- Export only the existing adapter/calibration/static/phase/FIFO/maximum and
  assumed-receipt API. The module does not enter live serving. Calibration
  identity and `purpose=:refitted` declarations are not verified refit/export
  provenance and must not be described as such.
- Preserve the explicit availability label and unknown outcomes. Default
  complete-label phase/FIFO maps and explicit receipt replays have different
  availability contracts; this port does not prove a matched comparison with
  grace or late receipts. A common event adapter is a subsequent proposal.

## Audit matrix and failure witnesses

| Dimension | Independent check | Failure witness |
|---|---|---|
| Science/clock | Frozen polynomial3/32/61/sqrt(310), target59, strict maturity/split and31-phase/FIFO Fraction traces | Future feature filtering, old30-phase, wrong calendar or sample std |
| Implementation |479 original assertions plus package module/reference boundary | Missing namespace, leaked ground_cases, wrong source path or source hash |
| Scalar/error/atomicity | V2 nonzero-to-zero, signedzero/subnormal, every-state snapshot checks | All7 frozen scalar/atomic mutants |
| Resources | Existing1000-issue bounded pending/source/rolling-state controls | Existing unbounded-state mutation; no production RSS/performance claim |
| Portability | Native Windows Julia1.12.6/1.13.1, @__DIR__/joinpath paths | No platform-only import; Linux/macOS gates remain open |
| Integration/docs | Focused package example plus actual experiments registration; strict Documenter | Undefined public binding or unresolved docs link |

## Planned package changes

`src/SolarSINDy.jl`, new `src/completed_source_dbdt/` module/three bodies/reference;
new `test/completed_source_dbdt/` namespace runner/four test bodies/frozen oracles;
`test/runtests.jl` registration; `examples/completed_source_dbdt_synthetic.jl` and
`examples/experiments.jl` registration; `docs/src/completed-source-dbdt.md`,
`docs/COMPLETED_SOURCE_DBDT_PORT_CONTRACT.md`, `docs/make.jl`;
`CODE_NOTES.md` and `TEST_REPORT.md` additive ledgers.

## Verification and completion boundary

Use the hash-pinned bounded native stage runner, absolute argv, startup off,
pkgimages off, one thread, checked bounds and source guards. Run the exact
focused479 assertions on both runtimes, focused synthetic example on both,
unmodified mutation control and all7 V2 mutants, plus strict documentation.
No tolerance adjustment is allowed. Rejects/failures are retained externally.
Freeze all final source/test/example/doc hashes after these checks.

Root retains unfiltered `Pkg.test()`, the full `examples/experiments.jl`, the
canonical development harness and all8 CI cells at the eventual combined head.
They are outstanding required qualification, not waived by focused success.
No scientific verdict, observational skill, actual fit or paper readiness follows
from this port. After the source freeze, propose (do not implement) the common
`CompletedLabelEvents` contract and independent grace/reuse/expiry/order oracles
for root review.
