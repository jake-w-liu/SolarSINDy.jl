# V2.2 Research Development Ledger

## Development Ledger

| Item | Record |
|---|---|
| Objective | Test whether a product layer anchored to the frozen V2.1 SINDy-20/11 model can beat every same-row local comparator at 1, 2, 3, 4, 6, and 7 h without weakening storm or recovery safety. |
| Contract | Preserve the V2.1 coefficients; use issue-time information only; retain at least 60% combined served/frozen SINDy weight; distinguish anchor-relative model step from issue-relative product horizon; fail closed on missing rows, unsupported leads, lag mismatch, or corrupt artifacts. |
| Evidence | The source replay, primary stack, residual table, cross-fit table, feature schema, and learner version are hash-pinned. The development partitions end before 2023. |
| Independent oracles | Hand-computed constrained blends, synthetic known-weight recovery, exact live-kernel/replay identity, post-issue mutation invariance, whole-anchor embargoes, exact portable-EvoTrees inference, and artifact corruption tests. |
| Test plan | Focused tests cover the primary stack, sparse residual, portable boosted residual, served replay, expanding-window cross-fit, causal sparse-history kernel, M1 cross-fit helpers, and prospective receipt collector. The full package suite and deterministic package experiment are required before handoff. |
| Baseline verification | Every development row carries served and frozen V2.1, raw SINDy, persistence, Burton, Burton full, and O'Brien--McPherron predictions under a common finite-row mask. |
| Data regeneration trigger | Changes to split, feature, primary-stack, or learner code require rebuilding the affected replay and audit tables and rechecking their pinned hashes before interpreting scores. |
| Harness | `julia --project=. -e 'using Pkg; Pkg.test()'`; `julia --project=. examples/experiments.jl`; the repository development-harness audit against this worktree. |
| Risk | The main risks are post-issue leakage, repeated selection on one validation period, lag-zero/lag-one horizon confusion, hidden SINDy removal, artifact drift, and optimistic promotion from a small effect. |

## Implementation correctness gate

The implementation is isolated on `research/v2.2-accuracy`. The production
V2.1 checkout is unchanged. Its monitor, dashboard, and watchdog jobs are
stopped; no research module is wired into serving.

The primary constrained stack, sparse residual, portable boosted residual,
causal served replay, and expanding-window cross-fit have independent focused
tests. The nonlinear development runner reads only the pinned pre-2023 replay,
uses fixed 168 h issue clusters, and records zero exposed-benchmark rows read.

## Scientific outcome

The primary stack improved on the best same-row local comparator by only
0.0216--0.0604 nT on the four primary leads. The sparse residual passed its
regime guards only at 1 h. The frozen nonlinear family had no safe cross-fitted
setting at 2, 3, or 4 h; its final diagnostic gains remained below 0.25 nT at
every supported lead and the 6 and 7 h settings worsened safety cells.

The result is `NO_GO`. These modules are retained as reproducible research
infrastructure, not as a deployable V2.2 artifact.

## M1 mechanism test

The next bounded study inserted one causal exponential coupling-memory state
inside the frozen V2.1 trajectory. The core supports all eight subsets of
`{m, E-m, Dst* m}`, 2/6/12 h memory constants, lag-zero and lag-one anchors,
sign and nonoscillatory-stability constraints, varying-driver rollout, and
checksummed artifact I/O.

The expanding 2013--2017 cross-fit selected a 2 h `{Dst* m}` candidate only as
the least-bad screen. It lost to the strongest same-row comparator by
0.0610--0.6858 nT across the six leads, reached 97.41% availability, and failed
the active/recovery guards at 6 and 7 h. Its 10,000-replicate simultaneous
one-sided lower confidence bound was -0.7720 nT. M1 is therefore research-only
and is not a product upgrade.

## Input-trajectory pivot

The realized-driver oracle retains substantial 2--7 h headroom, whereas M1
adds only 0.0141--0.1181 nT over the conservative core. The next candidate is
therefore receipt-causal L1 advection followed by a stable group-sparse delay
continuation of the solar-wind trajectory, with the frozen SINDy core downstream.

No documented public pre-2023 DSCOVR or ACE holding currently reconstructs the
per-record receipt state required for an exact historical replay. A prospective
raw-response collector is implemented but not started. It writes
content-addressed raw bodies and a durable per-source sequence/hash chain,
records transport failures, rejects path and chronology corruption, and leaves
forecast serving untouched. Ephemeris and documented quality semantics remain
data prerequisites before any M2 fit.
