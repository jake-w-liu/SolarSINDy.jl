# CLI process identity repair: bounded verification

This engineering change is isolated to the ad-hoc CLI, its shipped Windows helper,
test prerequisites and inert process tests. It makes no forecasting/scientific
claim. Actual monitor/dashboard numerical mains and live services were not run.
The parent retains all original probe logs, failed test/controller transcripts,
frozen source copies and native receipts outside the repository.

## Contract and confirmed failures

The contract is [CLI_PROCESS_CONTRACT.md](CLI_PROCESS_CONTRACT.md). The original
Windows Git Bash `ps -o command= -p PID` returned 1 (`unknown option -- o`), and the
exact original `pid_of` deleted a live process's pidfile. Native probes established
that Bash job IDs differ from native PIDs and that a Julia child loses its OS-visible
entrypoint arguments on this host. No cause of that argument loss is asserted.

The replacement uses CLI-owned startup attestation, strict bounded records,
exclusive initial publication, and a per-path Windows named mutex. Native PID,
creation FILETIME and executable are verified through the same handle used to
terminate and wait. Unknown identity never authorizes a spawn or signal. The trust
boundary is user-controlled instance state, not hostile same-user state tampering.
Windows stop is forced termination; it does not claim Unix signal semantics.

An independent exact-function shell witness also confirmed that a simulated read
error deleted Unix pidfile state and reclaimed an existing lock. Read/query errors
now preserve/refuse as unknown. Windows legacy numeric/malformed/foreign/pending
records are preserved; Unix malformed numeric pidfiles retain their prior cleanup
semantics. Unix malformed/unreadable lock ownership is preserved. The legacy smoke
suite retains daemon PID-reuse safety and dead-owner recovery. Its live-unrelated
start-lock expectation follows the conservative compatibility policy below.

## Follow-up admission and readiness repairs

An exact-function witness with an owned filesystem alias showed a live Unix lock
could be reclaimed because flattened argv lacked the physical clone spelling.
The Windows junction/argv observations were native; the Unix ps stream in that
witness was controlled, so it establishes a control-flow failure rather than a
native Linux/macOS pass. Unix start locks now hold any positively live owner,
including unrelated/recycled live PIDs. Strict bounded records and query/read
failures preserve state; a separate successful absence query still permits dead
recovery. Daemon pidfile argv checks and Windows generation checks are unchanged.

A deterministic two-owned-shell barrier at stale removal additionally confirmed
both callers could observe a dead numeric owner, then B could remove A's newly
claimed live lock. Every Unix acquire now atomically creates an admission directory
around the complete check/read/query/remove/claim section. Only its creator may
checked-rmdir its empty directory, including ordinary/EXIT cleanup; errors propagate.
Existing/foreign admission state is UNKNOWN and preserved, with no automatic stale
directory recovery. A crash in this short section requires verified reconciliation;
normal dead numeric-owner recovery is retained. Old CLIs that ignore this directory
cannot safely share instance state. The regression repeats the stale boundary and
simultaneous fresh claims, preserves foreign content, and injects mkdir/rmdir errors.

A second exact-function witness injected mv error 73. The previous helper
returned success and an appendable old banner authorized readiness before the
new inert main's delayed banner. Rotation errors now propagate, and monitor and
dashboard both check directory preparation and rotation before spawn. Real CLI
utility-failure regressions require preserved original logs, no main marker,
no pid/pending record and no leftover start lock. The successful monitor control
waits for an independently marked delayed new banner. Native Unix-only tests add
actual ps plus symlink/relative starts with spaces and a bounded readiness gate.

The earlier full package run stopped with two extraction-test errors because the
external runner omitted the declared awk prerequisite from native Julia's PATH.
All frozen sources/controls remained unchanged. Native Julia then independently
reported no awk on that PATH; adding discovered Git utility directories resolved
awk.exe and executed arithmetic yielding 6. This is an external runner correction,
not a change to extraction tests or scientific source. The failed full run and
its complete logs remain preserved; qualification of this follow-up is pending.

## Source map

| Surface | Responsibility |
|---|---|
| `bin/solarsindy` | OS dispatch, structured helper arguments, tri-state callers, exact Bash owner admission, native bootstrap handshake, Unix reserved/gated publication, readiness caps, stop/status |
| `bin/cli_process_identity.jl` | Bounded TOML, clone/entrypoint bindings, exclusive publication, mutex, native PID generation handle, bootstrap, query/uptime/stop/lock |
| `test/bash_runtime.jl` | Root-supplied shared prerequisite; actual GNU banner and Bash-version roundtrip, bounded probes, explicit override/Git discovery |
| `test/test_cli_process.jl` | Portable inert mains, actual concurrent starts, launcher-exited discovery, identity/error controls, complete child waits/transcripts |
| `test/runtests.jl` | One added include; existing Unix smoke remains |
| `README.md` | Accurate prerequisites and Windows process-state/stop contract |

Records are limited to 16 KiB. Native image queries use a 32768-UTF16-unit buffer.
Mutex admission waits at most 3 s, pending startup at most 30 s, and native forced
termination confirmation at most 5 s. Monitor/dashboard readiness retains its
60/90 iteration caps and adds elapsed caps of 120/180 s; an in-flight helper query
can finish after that deadline. These are bounded-work/wait controls, not hard
wall-clock or RSS guarantees for OS scheduling or filesystem stalls.

## Independent oracles and tests

The original-function witness, native Python WinAPI PID/creation/executable/handle
probes and bootstrap timing probes preceded implementation. The tests use actual
self-expiring inert Julia mains in a relocated clone with spaces, an independently
written main-run counter, separate launcher Bash processes that exit, loopback-only
inert dashboard readiness, and an unrelated Julia decoy. They cover concurrent
start admission, exact native PID recording, generation mismatch, wrong clone,
malformed/oversize/pending/numeric state, missing query executable, unknown locks,
dead-owner reclaim, slash spelling, status uptime and validated native stop.

A controlled error-code seam maps a real failed OpenProcess call to error 5 and
requires UnknownIdentity; it is not an actual permission-denial experiment. Exact
Unix-function read/query failure seams run in Bash on Windows and are shell
control-flow tests, not native Linux/macOS process evidence.

Every direct Bash/helper child records its native PID and confirms completion via
its owned Base.Process wait. Foreground adapters inside the Bash CLI complete
before that Bash can finish. Both admitted main generation records are archived;
an independent final WinAPI sweep queries their absence/generation without signals
and explicitly closes every query handle. No successful finally-cleanup warning
is ignored.

## Qualification boundary

Historical native receipts and source hashes are in the source-only handoff ledger.
For the original bootstrap candidate, Windows Julia 1.12.6 and 1.13.1 each passed
101/101 assertions with native
and controller exit 0 and all five executable/test source pins unchanged. The
final-v7 source is CLI `9df849c8…`, native helper `362a4c5e…`, lifecycle tests
`f4f818d2…`, shared Bash helper `68e56095…`, and runtests `bc0c77b5…`.
Earlier v1/v3 fixture/controller failures and v4 console-encoding failures
remain historical evidence and are not relabeled as final qualification.

The later admission/log/alias candidate passed 297/297 focused assertions on each
native Windows Julia version. Its unfiltered Julia 1.13.1 package run completed
with native exit 0, 289424 passing assertions and 7 broken tests. All 400 frozen
source files and runner/tool controls remained unchanged. The controller was
deliberately held before experiments/harness after a separate owned witness
confirmed that the Unix caller could spawn and report readiness despite failed
PID publication. That old-source package pass does not qualify this gate repair.

The new bounded exact-function inventory covers successful exec/PID continuity,
PID write/replacement failures, no-fork preparation/descriptor failures,
acknowledgement nonregular/replaced/read-only cases, failed cancellation, detected
foreign marker/directory ownership, post-commit cleanup uncertainty, and closure
of every scoped FD5--9 on parent return and before inert main execution. On this
Windows filesystem, retained marker descriptors prevent directory rename; that
blocked injection is not a native Unix replacement pass. A separate controlled
ownership-loss case preserves the replaced directory.

An independent owned native-handle witness leaves the gate running after its
launching Bash exits. It observes natural gate termination around 30 s, no intended
main and closed query handles. A separate inert fork/exec control establishes the
host-specific MSYS Bash wait status 77 versus raw Win32 exit code 19712; the first
over-specific controller assertion remains preserved as a failed witness. These
are Windows/MSYS controls, not native Linux/macOS execution. Native Unix tests
require retained-FD directory replacement, actual symlinks and actual Julia PID
continuity. The new source still needs both focused Windows runs, unfiltered
package tests, full examples and unchanged harness before integration review/CI.

The v5 gate candidate subsequently passed 407/407 focused assertions on native
Windows Julia 1.13.1 with every frozen source/control unchanged; its admitted
fixture generations were independently absent and every recorded direct child
was waited. A new independent native Linux/macOS FIFO witness then confirmed
that the previous start-lock claim blocked without a FIFO reader, wrote its PID
once an owned reader appeared, and returned success for that FIFO. Both before
receipts retained the FIFO inode/type and sent no signals. The new v6 checks type
before every claim attempt and before owner reads; all regular numeric semantics
are unchanged. Shipped native Unix FIFO/symlink tests preserve targets, refuse
queries, bound owned callers and retain a failure-only owned reader to release
the old blocking writer safely. Windows runs the directory case; it does not
claim native Unix FIFO or symlink qualification.

The v6 focused Windows Julia 1.12.6 and 1.13.1 suites each passed 416/416 with all
400 source files and runner controls unchanged. An independent generation-aware
resource check found all four admitted main generations absent and every recorded
direct Bash/helper child waited, without signals. Native Linux/macOS both passed the after-FIFO
oracle, and Linux passed all nine real filesystem/gated-spawn controls. The same
gate's first native macOS success control failed closed before fork. Independent
metadata identified Bash 3.2.57 `-ef` failure: pathname `/dev/fd/9` reported a
different device from the owned path and actual `fstat(9)`, although all were
regular files. Native write-only and read/write diagnostics confirmed that
Apple's no-operand stat on duplicated stdin reports the true retained FD metadata.

The v7 candidate therefore shares one strict device/inode/regular-mode predicate
between all parent bindings and child owner/allow/acknowledgement checks. Darwin
uses native stat/fstat; Linux keeps its original builtin route. Controlled metadata
cases distinguish wrong device with the same inode, wrong inode/type, malformed
fields, trailing newline, partial/nonzero query errors and unknown platform.
Actual closed/different FDs preserve owned bytes. Extraction failure controls
require no fork and scoped cleanup. The v7 exact-function oracle then passed all
nine gated-spawn cases and the after-FIFO oracle on actual native Linux and macOS
in CI run 39120387024. This includes native Darwin exec/PID continuity, retained-FD
foreign-marker/directory replacement and exact owned child waits. These bounded
Unix oracles do not qualify full package integration or the later v8 source.

## v8 public PID record and inaccessible-parent repair

An independent exact-function before witness on the frozen v7 parser admitted
embedded space, leading space and an extra LF as PID 1234 after whitespace removal.
Its controlled process-query stream establishes parser behavior, not operating
system identity. The valid PID and alphabetic stale-cleanup controls behaved as
specified. The v8 parser instead retains bytes through bounded reads and validates
one canonical positive ASCII decimal PID with at most one trailing LF within 128
total bytes, before any query. Invalid regular bytes retain the legacy stale
cleanup policy; read or cleanup failure is unknown and preserved.

Separate native Linux/macOS before probes in CI run 39120464016 ran as UID 1001
and UID 501. An independent Python stat received errno 13 under an owned chmod000
directory, while the exact v7 pid_of returned absence 1. Both probes preserved
bytes, restored modes in finally, and sent no signals. The v8 absence branch now
walks known directory ancestors: unsearchable directories return unknown 2,
including inaccessible nested components, while genuinely missing directory
components remain absent. This is the verified directory-search case, not a claim
to diagnose every ACL or filesystem failure.

Prewritten public-record oracles cover exact 128-byte boundaries, whitespace and
multiline spellings, NUL, zero/leading zero, oversized sparse input, bounded read
failures, cleanup failure and the absence of process queries for malformed bytes.
Native Unix tests additionally require direct/nested unsearchable ancestors,
missing-parent controls, preserved bytes and finally-restored modes. The new v8
source is prepared but not yet dynamically qualified. Both focused Windows runs,
native Unix after controls and the complete package/examples/harness chain remain
pending; prior-source results are not promoted to v8 passes.

The original bootstrap resource sweep confirmed absence of every admitted main and
120 exact owned Bash/direct-helper process waits across the instrumented v6/v7
runs. Its extended history check encountered access denied (5) for one old
pre-bootstrap probe PID, 27460. That current identifier remains UNKNOWN and was
never signaled or classified dead. The original probe controller's retained
successful execution reached `terminate()` plus `communicate(timeout=10)` on its
own Popen handle before writing its receipt; that is prior original-generation
wait evidence, not a fresh identity query of the current identifier. The strict
extended-history query failure remains visible in the handoff, separate from the
passing final-generation checks.

Full package tests, examples/experiments, canonical harness, independent integration
review, and complete native platform qualification of the final combined source
remain root-owned gates. The bounded Unix oracles above do not establish those
full gates or completion of the whole portability audit.

Primary native API contracts: [OpenProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess),
[GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[QueryFullProcessImageNameW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-queryfullprocessimagenamew),
[TerminateProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess),
and [WaitForSingleObject](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject).


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
