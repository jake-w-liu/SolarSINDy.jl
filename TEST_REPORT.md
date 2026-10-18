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


## Disk-backed canonical fitting qualification, 18 October 2026

The boundary-repaired training source is
496af09faaed32580222d29e137716724928b49875770fc51dbc023daa709755.
Both Windows Julia 1.12.6 and 1.13.1 focused workers exited 0 with all 472 source
files and runner controls unchanged. Their displayed testsets total 732 passing
assertions, in addition to mandatory binding/source guards: 390 unchanged
producer/solver assertions, six unchanged malformed-shape assertions, 180 exact
canonical-model/prediction checks, ten real stream/bundle checks, four file/source
guards, 68 independently frozen metadata/loss checks and 74 canonical boundary
regressions. The test bodies and tolerances of the original 396 remain unchanged.
Focused receipt SHA-256: bde12f468ac6a6786364f5973a8acfb189630e60683ab310feaec88a941fc068.

The before witness on source fe1d301a confirmed the public fit/bundle boundary
mismatch on both Julia versions, with 68 positive metadata controls and nine
mismatch checks per run. Current regressions cover both latencies and -1 ms,
+1 ms and +30 minutes; exact-hour models pass the actual bundle validator.
Owned record-to-directory replacement proves the canonical boundary check runs
before numeric file access. Diagnostic input bytes remain preserved.
Before-witness receipt SHA-256: 3d2be1785ecb806dc8bb794dbc1f0c616973b3d34692c2b14ffeab681726cb00.

The unchanged 68-assertion metadata/loss oracle first passed on both native
versions, then caught swapped gains, swapped data/mask hashes, completion time
substituted for receipt time, and a dropped storm count on both versions. These
eight negative controls were run on the earlier fe1d301a implementation; current
496af09f focused workers rerun the exact frozen positive oracle. No negative
control result is relabeled as an execution on the later source.
Mutation receipt SHA-256: 61890887516cd70cdbf0ad92d691eeee8ebb371d8968716269bc7b610a034e63.

The earlier fe1d301a candidate passed all 17 mandatory independent workers on
both native runtimes, with 69/69 parent entry checks and unchanged source/control
maps. Those results remain historical; they do not replace complete verification
of the boundary-repaired source. Prior all17 receipt SHA-256: 9372064d6bb45382ce74ccc1e3e89476c0b1d34d9d5824f471583160bb87b48e.

The actual maintained canonical model API, with source 496af09f, passed the
unchanged 13-assertion generated resource fixture at each of 10000, 430248 and
500000 rows on Windows Julia 1.13.1. The largest recovered gains were exactly
(2,3); weighted SSE was 3.636363636363636 against the literal
3.6363636363636362. Owned OS observations recorded peak working set 781139968
bytes, peak private commit 921698304 bytes and maximum sampled handle count 202.
Cumulative allocation counts are retained separately and are not peak memory.
Timings include their recorded runtime/warmup context and support no new
performance claim. The original insufficient-free-memory admission attempt was
preserved; the successful retry retained the same eight-GiB admission threshold.
Maximum-size receipt SHA-256: 2c24d320bf1833ce96e15aca7fa9b665de333fd84c2ef10dc7af55d54fd0eaaf.

All inputs above were generated. No actual archive fit, empirical comparison,
reference admission, model serving or novelty verdict is authorized by these
component checks. Full package/experiment/harness, strict documentation and the
native six-cell platform qualifications of the final combined source are pending.


## Strict interval manual repair, 18 October 2026

The initial combined CI exposed 25 existing implementation docstrings absent
from the manual. The references were added without changing computational
source, tests, environments or strict Documenter settings. The actual repaired
Windows Julia 1.12.6 build exited 0; no Documenter errors/warnings were reported.
Full dependency warnings and raw output are retained. DOCS_REPORT.md records
the exact bounded coverage and receipt hashes; full computational gates remain
pending until their own actual closures.


## CLI smoke phase repair and historical full gates, 18 October 2026

The previous combined candidate b390755 completed the unchanged canonical
development harness on Windows Julia 1.13.1 with native exit 0: 336 PASS,
3 WARN, 0 FAIL. Its unfiltered package tests passed 289896 assertions with
7 existing broken tests, and full experiments exited 0. The raw harness log
SHA-256 is 90aecb8fcc29f846a237d90d941188df667b1c32b3c0ff31b6c3daf89d1018be.
These results qualify that earlier source only.

The same candidate's full native CI failed strict documentation and Unix CLI
smoke checks. The strict manual repair subsequently passed on Windows and in
Ubuntu CI run 39123287851. Three Unix package cells failed the dead-owner lock
cleanup assertion; Ubuntu Julia 1.13.1 instead failed two earlier contender/lock
assertions after startup exited. Their exact startup/kill phase was not recorded,
so a single historical root cause is not claimed.

Independent controlled Linux and macOS witnesses in run 39125174829 reproduce
the documented UNKNOWN admission after an early kill and successful dead-owner
cleanup after admission release. The fixture now proves actual shim main entry,
a matching committed PID, absent admission and a live launcher before both the
contender and SIGKILL. The original 15-second bound and original dead-owner
assertions remain. Production process-control source is unchanged.

The exact repaired complete smoke file passed on native Ubuntu and macOS with
Julia 1.12.6 and 1.13.1 in CI run 39125478730. Seven relevant caller files match
the package checkout. The repaired file SHA-256 is
78b20a79d74a1b84b0ad33506cec582bef6e34ca4a9f9d4fbe2808c8031eadcb.
The retained native review receipt SHA-256 is
517391f18dca9bee219b3a93f55dcfeff56b898cd010f12ec64a59a03c48e13e.
These focused passes do not replace the final merged source's full six-cell
package/experiment matrix, strict documentation and canonical harness.


## Canonical recipe repair qualification, 18 October 2026

Both native Windows Julia 1.12.6 and 1.13.1 witnesses confirmed successful
fitting with an accepted equal-valued Int32-index recipe followed by actual
save_bundle rejection of its model metadata. Each also passed the original
68 metadata/loss positive controls and 16 mismatch-witness assertions.
Before-witness receipt SHA-256:
0eafb3b639d29938b8ebf4d014ba015bce10857125cfbbba282fe7d38835fba8.

The independently frozen positive regression failed on the old implementation
with 184 PASS and 116 FAIL, no Error During Test, native exit 1 as expected.
Its raw log SHA-256 is
7104093c1e25fd3c0555bd55fe7e08c74c4f43bbcd20a65b4ee558483a6163d4.
The regression file remains byte-exact, SHA-256
f3ab8e8416c5b107a68579d51c905f7e5c2d460b0422c62d8169fcbb9cbd26f9.

On repaired training source acaf046f, both actual focused disk-fit workers
exited 0, with all 474 source files and controls unchanged. Displayed testsets
total 1042 passing assertions each, plus source/binding checks. The original
732 remain, followed by 300 recipe/canonical metadata checks and ten actual
stream bundle roundtrip checks. Variants cover Int32 indices, Float32 dynamics,
combined representations, both delays and two recipes. Off-grid rejection is
verified before callback/output creation. An initial adapter alias omission
was retained as a failed setup run, then corrected without changing any frozen
assertion or production source. Final focused receipt SHA-256:
f9a4b979e0d858b5f56518005d65f0fb1a21dc6042b8d6f56e5e80936efd2d14.

The unchanged 13-assertion generated resource oracle passed at 10000, 430248
and 500000 origins using the actual maintained API on acaf046f. At 500000,
observed peak working set was 752414720 bytes, peak private commit 891686912
bytes and maximum sampled handle count 200. These are process observations,
not a constant-memory or new performance claim. Maximum-size receipt SHA-256:
cc486b04cc2b02f7d4934903c9304e39d94cce9e382b48dd632ee49e09579dbf.

All these inputs were generated. Full unfiltered package tests, all independent
workers, experiments, canonical harness, strict docs and six-cell native CI
remain required on the final committed source. No actual archive fit, skill,
novelty, unopened-cohort or serving verdict follows from these checks.


## Strengthened resource oracle qualification, 18 October 2026

The native Windows Julia 1.12.6 and 1.13.1 independent handle controls each pass
8/8. Two deliberately false helper variants fail assertions with native exit 1,
as required; no setup error is accepted as a fault-control success. Windows
receipt SHA-256: aeffb322683a8e8972ecb4cd2b744de9c268250df7084dd012274759d3c6b5ce.
Actual Linux/macOS by both Julia versions pass the same positive and negative
controls in CI run39129935817. Earlier path-setup and EBADF test-oracle failures
are retained. The corrected oracle propagates all unknown query failures.

Both full focused disk-fit workers subsequently exit 0 with all 475 source
files and tool controls unchanged. Displayed testsets total 1050 passing checks
each: the previous 1042 plus eight independent handle-control checks. The original
390-check body now uses the meaningful closure assertion after failed producers;
all other original source is byte-preserved. Source Training acaf046f is unchanged.
Focused receipt SHA-256:
63993aaed4cba90e8773ee0c15718bf32c801105fd1fe4218bfae21280287940.

The prior 42804b43 unfiltered Pkg.test and full experiments pass, and all17
workers complete, but canonical audit fails the now-replaced tautological
statement (343 PASS/3 WARN/1 FAIL). Its raw log SHA-256 is
c773707670234a6bc7ab2373addcbff0da81ab565ef5eb4e8a13a7de110a3cc0.
That is a historical failed audit, not final release qualification.
The final committed test tree still requires fresh full Pkg/experiments/H,
native six-cell CI, strict documentation and deployment-image qualification.
No real-data skill, novelty, unopened-cohort or serving verdict follows.

## Source-bound initial-reference tools, 18 October 2026

The preceding a48eb163 test tree subsequently completed all listed gates; its
release receipt is bound by tag audit/forecaster-qualified-20261018. The current
addition preserves that full suite and adds mandatory independent reference
tests rather than substituting a narrow run for package or experiment coverage.

The maintained focused suite passes312 displayed assertions on each native
Windows Julia1.12.6 and1.13.1 process, all component/test/control bytes unchanged:
original loader/reference200, binary identity12, metadata schemas36, independent
clock/cohort/grid56, public-entry mismatch4 and source immutability4. Receipt:
893d89262d3bab87b4814af0c9c60eb8aad4cdba4cf77c5e012ea3382f1f5293.

Separate coherent G-b16e native verification has eight successful processes:
the actual1.12-versus-admitted1.13 entrypoint rejection, both binary-control
suites, both36-check metadata suites, seven actual-header-only data-flow checks,
and the unchanged253-check constant/214-check nonconstant all-grid regressions
on the admitted1.13 comparator runtime. Its receipt is
da48bf6072c2c7200ecc163c05c198d2ff9cfeb3ae03591bf59f18afedf2bbdf.
The metadata-only test opens no raw matrix, model, prediction or outcome file.

Anti-false-test evidence retains the original successful constant reconstruction
under the wrong binary, the subsequent nonconstant parity failure, and an
independent before/after entrypoint witness. A first poison-config fixture failed
before function entry; its correction supplies literal boundary defaults while
preserving every assertion. That first fixture is not counted as a valid runtime
witness. The original header independently lacks both keys indexed by the old
executor; the new schema normalizer binds it without adding fields to source.

All previous numerical expectations, masks, grid order, fallback rows and
tolerances remain unchanged. These checks establish software behavior only.
Publication still requires the complete package/examples/canonical-harness,
native matrix, strict docs and image gates on the exact committed candidate.
Actual reconstruction, full independent artifact closure, L1/A fitting and
independent forecast-skill/novelty evidence are separate uncompleted work.

### Case-spelling regression and candidate supersession

The new independent Windows witness confirms both a Python reader prototype and
the B6f4f public resolved API accept ACTUAL.json for actual.json. Both before
processes fail assertions without source changes; original file bytes remain.
Before receipt25d203155b889acbeff983a8d2e427919ae7a5c9809dfd2ab4fb9518ea5d41c1.
The recorded Base.realpath and native long-name results establish the distinction.

After replacing the spelling primitive with realpath, the maintained focused
suite passes322 displayed checks on each Windows Julia version: the prior312
plus10 new canonical/file/root case controls. All source/test/control bytes
remain unchanged during the two runs. Receipt
d0c94e6ac0c2ba9b159394704191834e5f867c86472ded3ea594e8a2e9120f47.
The original200 loader/reference tests are unedited; the B source-identity pin
changes with the reviewed implementation. Case-sensitive platforms exercise
six applicable path controls; four additional existing-alias checks are Windows.

Candidate841's in-progress full harness and native CI are explicitly superseded,
not passed. No final harness terminal exists for the interrupted owned run;
source/start/log/supersession records remain. Its CI conclusion is cancelled.
The healthy independent live services are not stopped or changed. Publication
requires fresh full gates on the coherent corrected tree, including the original
source-generated grid and independent saved-state regressions.

The coherent R906e/E8e7/B4d8c/Gb16e native eight-process rerun passes with all
controls unchanged, including the original253/214 all-grid regressions.
Receiptb0c9a35d6540874b0d92973e7db9924f6ac72c9bca58164cec35248b3f95fd5c.
CI now executes the mandatory focused reference suite before its still-unfiltered
package action and full experiments in all six native cells; no existing step,
platform, tolerance or scientific check is removed.


### Independent artifact-reader pre-commit checkpoint (2026-10-18 UTC)

The Julia package-focused source suite passes361 displayed checks on each local
Windows runtime, Julia1.12.6 and1.13.1. The original322 checks are retained and
39 scalar-tree, invalid-state, ownership and saved-scaling boundary checks are
added. Both native exits are0 with all admitted Julia source bytes unchanged;
receiptc084512555477deebd60983a78d97c18156335b605195b4019073e6947bac62f.

The initial full numerical reader passes68220 checks per runtime. A subsequent
fresh rehashed corruption matrix deliberately fails four rejection assertions
(selected configuration/metric and prediction/scaling digest metadata). Its
failure receipt56481ff8f485bb0313c8d38daad4b4d67987ff592fe7b728b4e06420eb9f124d
and exact before-source are retained. After adding the missing comparisons,
all original rehashed controls reject as required; receipte1a2d3dc79efecd55d2961f34830e78d86b4935dfaa18b33a2f9a3d58ee16964.

The completed maintained Python suite passes25 tests on each local runtime.
Each executes the public CLI,68222 numerical checks across all sixteen models,
25 rehashed corruptions, an actual changed raw future-tail invariance control,
frozen literal BSON/type/bin/Float32 tests, path/count/byte/JSON/CSV guards and
owned descendant listener controls on normal root exit and timeout. The complete
matrix receipt is9852cf3ccda712d6c5bac468217c4b024aa798fff58f09358fc9439a7de4128d;
all pinned source/executable/test bytes remain unchanged. Each local Windows run
explicitly skips one file-symlink creation test because that capability is
unavailable. No file-symlink rejection pass is inferred from that skip. The
other path/case controls and native process lifetime controls execute.

The copied BSON helper's negative-cap witness invokes read(-1) for max_bytes=-2
before rejection. The corrected helper rejects all eight invalid cap values
before file opening, preserving default decoding and the original manual
expectations. CSV preflight additionally bounds physical records before parser
materialization, exact widths, field sizes, unique headers and row counts.
This is parser resource coverage, not a claim of measured peak-memory optimality.

The fixture is entirely generated and its normalized execution path is recorded
in its hash-bound manifest; all numerical/model bytes remain unchanged. No real
archive fit, actual saved-artifact numerical closure, source-admission verdict,
forecast superiority, novelty, fresh confirmation or serving change is claimed.
At this pre-commit checkpoint, complete package/examples/canonical-harness,
strict docs/images and the six native cells for the new reader remain pending.
Those gates must pass on the exact committed tree before main publication;
the immutable release qualification and annotated tag record the later outcome.


A subsequent fresh schema-type witness changes the reconstructed imputer JSON
version from[3,2,0] to[3,2,false], rehashes the model, replay sidecar and reference
completion, and is accepted by preflight. Witnessdd23b50e96089d517654ac444a5084a7080785e949783fa594283c4cf095be9c
confirms Python false/zero equality bypassed the integer-version format guard;
no incorrect numerical point or actual-data effect is demonstrated. Both the
schema validator and scalar JSON reader now require three exact integer version
components. The scalar arithmetic is unchanged, while the copied function's
version guard is deliberately corrected and is no longer claimed byte-identical.
The unchanged original rehashed witness now rejects before any audit output.

Candidate59d9766 is explicitly withheld. Its full harness was stopped through
the verified private-job controller after retaining owner/source/start evidence;
all nine observed owned processes are absent. Its native CI run39147477748 is
cancelled, with only strict docs completed. Neither interrupted run establishes
a full package/native/harness success verdict. Healthy live services continue.
Fresh complete gates must bind the corrected candidate, including Boolean/float
version literals and two additional rehashed imputer-version controls.


The corrected full reader suite subsequently passes26 tests per local Windows
runtime, Julia1.12.6 and1.13.1, including all original literals,68222 numerical
checks,27 rehashed corruptions, the public CLI, changed future-tail invariance
and cleanup/resource controls. Each run reports the same explicit unavailable
file-symlink capability skip. The complete two-runtime receipt is
de860653e09705b483d025e52e28f65bb3b2b1ce8de5508ba237f8a8af1d7239;
all pinned source/test/executable bytes remain unchanged. The original361 Julia
focused assertions retain identical source bytes and their c0845125 receipt.
All eight completed/superseded CI39147477748 raw job logs are retained, with
cancellations preserved. Five native reader steps completed before cancellation;
this is partial evidence only and does not qualify candidate59d9766. Fresh full
canonical and native gates are required on the corrected committed source.


### Retained CI child-failure evidence

The corrected reader's local canonical harness on5ba0433 completes with364
passes,3 warnings and0 failures, with all source/control bytes unchanged.
That is a Windows full package/examples/harness result; it does not replace a
failed native-platform job. Linux/Julia1.13.1 CI job117736884994 in run39148405347
fails at the public reader child exit assertion. Its retained raw job log has
SHA256580ac5ee0883c822060ca10ec1d61ad0dbfb357df07fea1e6df76c9c24b1012f.
Five other native reader steps pass. The failed worker's child log was retained
only under ephemeral/tmp and was absent from the console, so the underlying
numerical or environment cause remains unverified. Candidate5ba0433 is not
qualified for main publication by that incomplete/failed matrix.

Failure reporting now emits bounded tails of known owned child logs while
preserving full raw files. CI uses a dedicated owned temporary root and uploads
it on failure. An independent injected exit47 with a70000-byte raw log confirms
that the original failure is raised, the full log remains, and its trailing
cause is visible within bounded output. The maintained Windows CI entry passes
all27 Python tests, including the original numerical replay and27 rehashed
corruptions; the single unavailable file-symlink capability skip is explicit.
The affected Linux1.13 route requires three consecutive complete iterations;
any failure stops immediately. Neither this diagnostic change nor a subsequent
passing run alone is asserted to explain the original failed child. All full
package/experiment/native gates and numerical tolerances remain mandatory.


### Variance about the verified stored mean

The retained Windows1.13.1 CI artifact from1af9113 reports independent sample
scale mismatch. All177 generated fixture members are byte-identical to the
maintained fixture, and its executable matches the locald8804b0 digest. The
console did not print the exact offending feature/value. Native and generic
CPU diagnostics on the same bytes show different independently recomputed
rounded moments; both local routes otherwise pass. Constant vbs0 at h4 has304
copies of Float641.6, stored center1.6000000000000008 and stored scale
6.672321400300542e-16. Exact256-bit moments give zero variance and the unit
replacement. These observations do not establish the unseen CI value.

A controlled independent exact-moment replay of the same complete16-model
fixture reliably rejects with the original comparison: receipt
b803b0942f9ddcf30da1755f8eece3f73705643f6ee7e9a5c7caf9701850807c.
The reader had independently verified the stored mean but then compared scales
around another rounded mean. Corrected variance is now independently computed
about the verified stored mean, with compensated accumulation and the original
exact zero-to-unit mapping. All finite/type/count/overflow checks are explicit.
Independent mean checks, the5e-12 bound, saved states and every prediction check
remain unchanged. The original independent feature helper retains its bytes.

The same controlled replay now passes all68222 checks, receipt
0ebf841f8877ac63c2e9ce67755fd24283f71f443e14d74cc0e74ef13f61ed21.
The exact-moment generic-CPU route also passes, receiptdce78dc0f8f820899ff392daa48e1844ed408f22167b8c2f114c961f9379dc22,
as does the ordinary generic-CPU route, receipt733299406d2c3394c352b03bcc56ad626f68fd05229903932f561643c50fb830.
Fifteen pre-code literal variance/type/size/overflow expectations pass on both
local Julia versions; frozen oracle396b1e8dcfbe435f2de90694a70574a4ebcd649940eba6b9d3b061ac683e2fc7.
The full focused suite passes376 checks per Windows runtime, receipt
1cc327e7213a537ed92e2e30daddd9952fd04727ba7166ff05ce681d307d7562.

Both native Python suites pass27 tests each, including68222 default-moment
numerical checks, a separate exact-moment positive,29 rehashed corruptions
(including scale value and population-variance errors), future-tail invariance,
CLI/failure evidence and cleanup controls. The same unavailable Windows file-
symlink capability skip remains explicit. Complete matrix receipt
fb82d1b97b958eaf82964ef129923d615e97a0bffd0b95f504e9df02a7131c36
binds unchanged code/tests/executables during those runs. Error messages now
include lead/grid/feature and actual/expected values. The actual prior CI causal
link remains to be checked by fresh cross-platform execution; no tolerance,
model, metric, cohort or scientific acceptance rule is weakened.

The canonical full run on the superseded1af9113 source independently finishes
with364 passes,3 warnings and0 failures. It is retained as evidence for that
source only; it does not qualify this variance correction. New exact-commit
full package/examples/harness, strict docs/images and all native cells remain
mandatory before main publication. Actual-source admission preflight3f86a3af
passes with no numerical raw parsing, fitting or reference output. Actual
baseline fitting, actual artifact closure and the complete A chain are unfinished.
