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

## Physical-temp and shell-prerequisite qualification (2026-10-18)

The original default-parent alias witness failed after all 156 original receipt
assertions passed. The correction adds ten default/explicit-link assertions.
The explicit path guard is unchanged. Independent native timestamp cases pass
10/10 on both Windows Julia releases, including exact future/restored mtimes,
unchanged bytes and closed handles.

The first complete dashboard recheck retained 1613 passed, one failed and one
broken: a bare bash command reached a WSL launcher and never ran the readiness
loop. After selecting a verified GNU Bash, the complete suite passes on both
Windows releases. A separate six-case prerequisite oracle checks positive
execution, explicit valid/invalid choices and Git discovery with a WindowsApps
bash entry at the front of PATH. Initial assumptions about which inaccessible entry
Sys.which would select were rejected before that oracle was frozen.

The final unfiltered native qualification runs all six stages on each of
Windows Julia 1.12.6 and 1.13.1: complete app tests, receipt maturity,
observation evidence, evidence integration, live forecast verification and the
complete experiment script. All 12 stages exit zero with 396 tracked source
controls unchanged. The terminal SHA256 is
f7f6f1be957c246da23928bf699a0196ce30fe2f6a583a77a8125a9cd35bba5c.
No original scientific expectation or numerical tolerance was loosened.

These checks are narrower than Pkg.test(). The unchanged canonical harness and
the six native OS/runtime CI jobs are still running; neither is claimed passed
by this focused qualification. Earlier failures remain retained evidence.

## Interval-mean integration evidence history

The historical preparation snapshots below preserve failures and their fixes.
References to fourteen workers, eight unchanged files or then-pending gates
describe the indicated earlier source. The current numerical candidate retains
sixteen mandatory workers. Its actual final-source gates remain open.

## Interval-mean forecaster preparation checks

The new mandatory package test entry retains all existing tests and invokes 14
native functional workers. It has no optional skip switch and runs no benchmark.
Actual library paths, method definition paths, all eight implementation hashes,
the uncached baseline and all 40 frozen resources are checked before and after
each worker. The frozen-resource table has an exact count and fixed digest, so
an empty table cannot satisfy the guard. Completion requires successful native
exits and valid receipts from every required worker, including the last one.

The prepared narrow chain passed on Windows Julia 1.13.1: 12 namespace smoke
assertions, 19 frozen scanner cases and 2,272 unchanged independent mathematical/
Gaussian/replay assertions, with source/path/hash guards. Explicit Julia 1.12.6
also passed the same 12 smoke and 19 parser cases. Independent peer checks
retain the original scanner false-negative failure and detect six parser
mutations, malformed quoted syntax, a vacuous pin-table regression and invalid
terminal-completion conditions. Preparation failures and raw logs are preserved.

Cross-platform before/after comparisons use the strictly pinned original
uncached implementation to generate expectations on that same platform/runtime.
They do not regenerate expectations from the optimized candidate. Existing fixed
numerical oracle values and tolerances remain unchanged; bit equality within a
platform is separate from universal bit identity across architectures or BLAS.

The complete 14-worker entry, unshortened Pkg.test, full examples/experiments.jl,
strict export/doctest build, canonical development harness and new six-job
Windows/Linux/macOS matrix are still pending. These narrow preparation passes
are not a scientific verdict, actual archive fit, serving promotion or a
substitute for those final-source gates.


## Interval-mean forecaster test-environment repair (2026-10-18)

The first frozen, unshortened Julia 1.13.1 package test attempt terminated
with native exit 1. Its mandatory interval worker entry exposed an undeclared
test-only InteractiveUtils import in the frozen 10,000-hour baseline driver;
the driver stopped before the corresponding candidate run. Two existing pipeline
tests also errored because bare awk was absent from the direct Julia PATH. The
full summary was 289173 passed, 3 failed, 2 errored and 7 declared broken. The failed worker,
complete native package transcript, source controls and exit receipt are
retained in the external final-source-20261018T075547Z run archive.

Identical closed-sandbox tests on Julia 1.12.6 and 1.13.1 fail without the
declaration and pass all three assertions after adding the installed stdlib UUID
to Project.toml extras and the test target. The tests retain code_warntype use.
An independent inventory finds no other missing stdlib imports in the frozen
workers. The eight implementation files, numerical oracles and tolerances are
unchanged. The public target-interval wording now states the existing completed
target convention explicitly. Full final-source package tests, experiments,
canonical development audit and strict documentation remain required for this
repaired freeze; the isolated dependency proof does not replace those gates.

An operational cache appeared under the original checkout's default runtime
output root while its 453 frozen input files remained unchanged. The original
checkout, generated cache and failed run remain intact. The retry uses a fresh
source-only export and records an explicit per-stage operational output root
outside the checkout. Git awk and GNU Bash capabilities are verified and bound
in the direct Julia environment. No test assertions or scientific inputs are
changed by those environment corrections.


## Interval-mean strict-token and publication boundary repair preparation (2026-10-18)

The repaired test environment completed the actual unshortened Julia 1.13.1
package tests on Windows: 289188 passed, 7 declared broken, no failures or
errors, native exit 0. All 453 frozen source inputs and 940 controls remained
unchanged. This completed run is historical; its source has independently
confirmed strict-token and artifact producer/loader boundary defects. Its
examples, canonical development audit and strict docs were therefore held.

Independent preimplementation controls demonstrate terminal-LF acceptance in
64-character source digests and typed tokens, and producer publication of
payloads rejected by the existing loader's token, width, depth or member-size
contract. The proposed correction requires whole-string token matches and
replays the typed codec and existing one-million-byte member limit before
creating the destination. Replay is self-consistency validation, not an
independent numerical oracle or a hard time/memory ceiling.

This source-only preparation retains all original 40 frozen resources and
all 14 existing mandatory jobs, adding a package-loaded boundary regression
job with previously frozen byte/API and producer cases. Literal baseline pins
remain the original eight identities; active pins state the three changed
files truthfully. A narrowly matched AST loading adapter changes only the
historical training oracle's source-provenance name list from its original
two names to the reviewed four; no numerical assertion, formula, tolerance,
setting or valid numeric serialization is changed. Historical A8 timings do
not qualify this new source identity. The new candidate has not yet run:
source/test review, narrow verification, full final-source package tests,
experiments, canonical audit, strict docs and native six-cell CI remain
required before integration or any further capability claim.


The first prepared boundary worker's Julia 1.12.6 native run retained
223 passes and one fixture error: range indexed assignment rejected the
three-string tuple before typed/producer cases ran. All 458 source files and
1403 controls remained exact. The independent digest 215 and AST-binding 8
checks passed. The same statement occurs in the maintained synthetic example.
An independent, identical three-field fixture on both Julia 1.12.6 and 1.13.1
retains the before native error and passes all three checks after changing
only slice assignment to broadcast assignment. That one-character correction
is now prepared in the example and the new worker's generated writer; valid
field strings, numerical recipe and all numerical/oracle expectations remain
unchanged. The failed candidate/run is retained. Actual corrected package
workers and the maintained example still require native execution.


## Interval-mean large-case NNLS repair preparation (2026-10-18)

The preserved original large-case suite passes 73 assertions and fails six
unchanged absolute gain checks at 430248 and 500000 generated origins. The
partial SVD-only correction passes 75 and retains four failures. The complete
isolated numerical source passes all 79 assertions on native Windows Julia
1.12.6 and 1.13.1, retaining the 1e-12 tolerances. The suite includes interior,
rank-one minimum-norm, constrained-axis, zero-design, fallback-loss and extreme
scale controls. The native success receipt is
3b61fa74b5100282d9f19c284dfdd2f7dd9f0e0096a50078cafa50dcabe4195f.

Three separately preserved branch-removal mutations fail the intended gain
assertions: interior without SVD correction, rank-one without exact projection,
and axis without scalar correction. They terminate with native exit 1 after
producing all twelve case outputs, rather than setup or source errors. Their
aggregate receipt is
2903bce31109ea13258c07efe51391fa47961badc97b11a6b0dd741d5e6c2304.

An independent read-only review closes 615 assertions on each native Julia
version over sixteen precision/rounding contexts, private correction triggers,
exact and one-ULP noncollinear directions, input immutability and extreme scales.
Its source/control review receipt is
9882b8a37d8849cc780b401c62241e121868ed49bb81a8395c25925dec058283.
This does not establish a universal absolute-accuracy guarantee. The saved
1000-hour synthetic model panel triggers neither interior nor axis refinement;
those stored-field comparisons are regression coverage, not branch proof.

The preceding boundary-repaired source separately passed both 15-worker direct
package-loaded entries, the grammar/training checks and the actual maintained
synthetic example on Julia 1.12.6 and 1.13.1. Those receipts bind the earlier
Adapter bytes and do not qualify this new combined source automatically.

This preparation retains every old worker and frozen oracle, adding the exact
large-case oracle as a sixteenth mandatory worker with a package-loaded alias.
No package tests or experiments have run on this newly prepared integration
candidate. Root combined-source review and the actual complete package tests,
full experiments, unchanged canonical audit, strict documentation and native
six-cell Windows/Linux/macOS CI remain required. The historical timing results
remain bound to earlier A8 bytes. No actual archive fit, scoring interpretation,
serving change or new performance claim is authorized by these synthetic checks.

## Subsequent focused numerical and disk-fit reverification

The boundary source with the earlier Adapter completed all fifteen direct
package-loaded workers on native Windows Julia 1.12.6 and 1.13.1, including the
four 10,000-hour same-platform L0/L1 before/after payload comparisons. The closure
review digest is 6e0f508a0236f2dcb9af8a30f41d0f4ba68c7857ac43c33cd7d3341cf1a2e201.
This historical closure does not qualify the later numerical Adapter.

The exact repaired Adapter was then used with unchanged generated disk-fit
oracles. All 396 checks passed on each native Julia version; aggregate receipt
93bf4458a69d84fee98a6ebf2b9da40174accb9500722ee5a54f1d16d5c4b1c5.
The unchanged maximum-size oracle passes thirteen checks each at 10000, 430248
and 500000 generated origins on Julia 1.13.1; receipt
7dcdc47a4659d15959aac7f4dbfef12130af2c10cc495164a62f6c72defcbebf.
At 500000 origins, gains are exactly (2,3), weighted SSE is 3.636363636363636
against the literal 3.6363636363636362. The process counters record peak working
set 553926656 bytes and peak private commit 699019264 bytes, including runtime
and warmup; sampled maximum handles are 185. Cumulative allocations are not
peak memory. These external component tests read only generated inputs; the
disk-fit component itself is not part of this package integration or a qualified
full-history importer, archive fit, model-selection pipeline or forecasting result.

The final combined-source package, full experiments, canonical audit, strict
documentation and native six-cell CI have not yet completed. Their eventual
receipts must bind the actual combined source before a main promotion.

## Focused package-loaded numerical qualification, Windows 2026-10-18

The source-only numerical integration candidate, with Adapter source digest
20f68a96ea7edaa88433bec70f1ffddb74128221862e8476f8d2e1704ddda22f,
completed the mandatory sixteen-worker direct package-loaded entry on native
Windows Julia 1.12.6 and 1.13.1. Each outer entry passed 65 assertions, and all
32 actual child processes exited zero without timeout. The existing fifteen
workers remain, followed by the unchanged 79-assertion large numerical oracle.
Both versions completed its twelve independently checked literal case reports.
The 10,000-hour L0 and L1 same-platform baseline/candidate payload comparisons
passed on both versions, with the original baseline bytes retained.

The completed aggregate terminal digest is
fb951848c90d5c050114dbf0dea03b96b31388bffc6a284a3eda450bf083c5c4.
Independent receipt review c997413d97d72f813d6e255a2dbe6b8d1b0ffa3c7818b5b878c28572568159d4
checks the actual exits, job order, package/method source bindings, raw oracle
identities and all 460 source/2333 control hashes. Every worker used one Julia
thread; the independent mathematical worker used one BLAS thread and the
other workers used four. The forty original test resources and eight original
baseline source files remain byte-exact. A separate fresh hash review preserves
the original 52 profiling controls and the earlier measured eight-source A8
implementation; these external historical controls are not part of the 2333
current qualification controls.

These are focused direct entries on the numerical candidate, before the CLI
merge. They are not a completed final combined-source Pkg.test(), full
experiments, canonical development audit, strict documentation build or native
six-cell operating-system matrix. Those gates must bind the final committed
combined source. No actual archive fit, empirical forecasting skill, serving
promotion, cross-architecture bit identity or new performance claim follows
from these synthetic checks. Historical A8 timings apply only to their older
source identities.


## Final v8 focused qualification, 18 October 2026

The frozen CLI candidate subsequently passed 700/700 focused assertions on
Windows Julia 1.12.6 and 1.13.1, each with native exit 0. All 400 admitted source
files and tool controls were unchanged. The independent generation-aware sweep
found all four admitted process generations absent and confirmed 238 exact
recorded direct child waits, with no signals or cleanup warnings. The combined
focused receipt SHA-256 is
82322131683ed76b7d5180cce3e7f88db7e0c47b8d2c688b766a1778c67d1b7c.

Native Linux and macOS CI run 39121627806 also passed the exact v8 FIFO oracle,
33 public PID record/type/directory-search cases with 35 owned children, and
all nine gated-spawn controls. The source was bound to
3c27cdb3cd2b649c9815df7d0ec7c16a943018998288905e68385499fc76316b.
The retained Ubuntu and macOS job logs have SHA-256
ad39a8d66cbe836fa6021c2872dcfbaa72098fa71e5d284635797e0e6333d196 and
77c6289a7c0b6bd2b82d8cbe137677a7d16e1582cd8514c92b8f6706f0191f72.
These results supersede the earlier prepared-only v8 status for the bounded
focused checks. Full tests, experiments, canonical harness, strict documentation
and the six-cell native package matrix remain pending for the combined source.
No archive forecast-skill, novelty or deployment verdict follows from this run.
