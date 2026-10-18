# Receipt-report integration test plan

The sealed prototype's independent hand-computed nominal clock and metric cases are ported into the package test entry point. The independent input/metric reader and report mutation controls remain separately source-bound. Tests never use a live endpoint or reconstruct historical receipt times.

New checks exercise source read-only copying and stable inventories, missing legacy evidence versus corrupt existing journals, CSV/head drift before and after copy, lock/budget/static-link rejection, exclusive output creation, completion hashes, fixed empty057 claim support and CLI option errors. The experiment creates only a new synthetic mock journal/report. Test values and strict as-of inequalities are frozen before candidate execution.

Qualification status is held in external source-bound run receipts. A successful earlier prototype or unrelated main-package harness is not qualification of this candidate. Full actual Pkg.test(), examples/experiments.jl and development harness remain required; declared artifact skips and any warnings must be reported without loosening tolerances.

## Coverage and independent oracles

Coverage includes end-minus1ms, exact end and strict verification-as-of equality, wall-time nonpromotion, same-value reconfirmation without CSV rewrite, primary/revision separation, missing journals, corrupt committed heads/bodies, CSV/head drift, foreign identities, explicit source-unqualified status, trusted empty057 input and actual native CLI publication. Windows case aliases and an owned NTFSjunction exercise native path boundaries. Leaf file-symlink coverage remains unavailable after the retainedEPERM fixture failure; it is not reported as passed.

## Tolerance justifications

Integer-millisecond clock boundaries, counts, identities and hand-constructed exactly representable scalar metrics use exact equality. The separate retained-data reader uses fixed1e-13 absolute/relative tolerance only for cross-language floating-point metric serialization/summation, established before its comparisons. This tolerance never decides model selection or maturity. Existing package tolerances remain unchanged and retain their previous ledger obligations.

## Anti-False-Test checks

Negative fixtures mutate counts, pre-end categories, manifest identity, primary event/value/clock, provenance, issuance flags, revised scores, comparator scores and duplicate rows. Output-hash rebinding alone cannot legitimize a nonempty supposedly empty057 cohort or changed snapshot binding. The file-claim competitor writes actual first-writer bytes at the native mkdir boundary; those bytes must survive rejection. Source-copy hooks actually mutate owned inputs before the expected drift failure. Tests do not substitute implementation outputs for the independent hand-computed expectations.

Resource-boundary controls retain an original twelve-directory/four-entry failure and an original stale-stat growth failure. After correction, the directory control rejects before copying and the independent growth control proves its hook executed at the actual IOStream boundary: requested/returned nine bytes for an eight-byte limit, with no unbounded String read. Accepted metadata includes directories and the CSV; parsing and runtime overhead are not represented as a hard memory bound.

## Cross-platform regression qualification (2026-10-18)

The complete declared-environment contract passes 117 assertions on each native
Windows Julia 1.12.6 and 1.13.1. Seven independent workflow mutations reject
missing macOS/runtime coverage, disconnected matrix selection, fail-fast,
omitted full package tests, omitted experiments and continue-on-error.

The daily-review file passes all unchanged 64 assertions on both releases.
A separate lowering/semantic oracle passes 13 assertions per release and
reproduces the old keyword-callback error on Julia 1.12.6. A byte-inverse
comparison confirms that only eight unused argument names changed.
Actionlint passes. Full native OS-matrix execution must pass before claiming
Windows/Linux/macOS support; these focused Windows checks are not that result.
