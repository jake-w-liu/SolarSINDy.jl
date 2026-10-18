# Isolated CLI process repair contract (before implementation)

Scope: `bin/solarsindy`, its shipped stdlib-only process helper, and network-free inert
process tests. No monitor/dashboard numerical source or live instance is exercised.

Windows uses native PID + creation FILETIME + executable identity. Numeric legacy
pidfiles have an ambiguous PID domain and are UNKNOWN, preserved, and never adopted.
Julia on this host drops its OS-visible arguments after startup (both supported native
versions were probed). The `-L` bootstrap therefore attests its actual `PROGRAM_FILE`,
physical clone, executable and native process identity into a nonce-bound CLI-owned
receipt. This is startup provenance supplied by the CLI, not independently recovered
OS argv and not protection against a malicious same-user writer.

Windows process queries return 0 for verified running, 1 for confirmed absent/stale generation,
2 for unknown (malformed/pending/wrong clone, unavailable/denied query, read errors).
All start/stop/status/lock/readiness callers preserve/refuse on unknown. Status adds
exit 3 for unknown; existing known-state exit codes 0/1/2 remain. An incomplete startup
receipt is deliberately not auto-adopted or discarded. Stop on Windows validates
creation and executable on the same native handle used for forced termination and
bounded wait; it does not claim POSIX SIGTERM semantics.

State and start-lock updates are serialized by a per-path Windows named mutex; lock
owners are native Bash PID generations, never `$!` or `$$` interpreted as a WinPID.
Exclusive initial publication and bounded strict TOML records prevent partial records
from being interpreted as permission to spawn. No query failure means dead. Dead
owner generation may be reclaimed; wrong-root/malformed lock is preserved.

Unix continues to use its single numeric PID format and clone-entrypoint argv
match. Public PID records are bounded to 128 total bytes and contain one canonical
positive ASCII decimal PID, without leading zeroes, followed by at most one LF.
Parsing never removes whitespace or performs arithmetic conversion. Bounded reads
and a byte-count check precede any process query. Malformed regular numeric
pidfiles and confirmed dead/unrelated regular pidfiles are cleaned as before, as
required by the legacy Unix smoke suite. Failed malformed-file cleanup is UNKNOWN.
Symlink and existing nonregular pidfile paths are UNKNOWN and preserved before
numeric parsing; they cannot authorize start, status adoption or signaling. Unix
pidfile read failures and process-query failures now preserve state as unknown.
Before classifying a path as absent, known directory ancestors must be searchable;
an unsearchable ancestor is UNKNOWN, including when deeper path components cannot
be inspected. Genuinely missing directory components remain absent. This guards
the verified directory-search failure and does not claim exhaustive ACL or
arbitrary filesystem-error classification. Existing-directory and symlink tests
precede all reads. In contrast to regular pidfile stale cleanup,
an unreadable, empty, or malformed start-lock owner is also preserved. A failed ps
query proves absence only when a separate successful complete PID query omits it.
Unix start-lock records are bounded to 128 bytes and contain one positive decimal
PID with an optional single trailing newline. Before each exclusive claim attempt
and before owner reads, symlink and existing nonregular lock paths are UNKNOWN2
and preserved. In particular a FIFO cannot become an open/read wait or a successful
lock claim. Regular malformed/numeric/query/dead-owner semantics remain unchanged.
Any positively live owner holds the
lock regardless of its flattened argv: physical, alias and relative spellings are
ambiguous there. This deliberately also preserves a live unrelated/recycled owner;
reconcile it through verified ownership or wait for confirmed exit. Dead owners
remain recoverable. Malformed, unreadable and query-unknown locks are preserved.
This conservative start-lock rule does not weaken daemon pidfile identity checks.

Every Unix acquisition serializes its complete check/read/query/dead-remove/claim
section with an exclusive atomic mkdir admission directory beside the numeric
start lock. Only the caller that successfully creates it may checked-rmdir its
empty directory; ordinary and EXIT cleanup failures propagate. Existing/foreign
admission directories are preserved/refused and are never auto-reclaimed. A crash
inside this short section can leave UNKNOWN admission state requiring verified
reconciliation. Automatic crash recovery of that directory and cooperation with
an older CLI that ignores it are not claimed. Confirmed-dead numeric start-lock
owners remain recoverable when admission is available; the numeric lock continues
to protect the full startup/readiness operation.

Directory preparation and old-log rotation must both succeed before either daemon
is spawned or its pid/pending receipt is prepared. A failed rotation preserves the
old log and fails start; an appendable stale banner cannot authorize readiness.

Unix spawning additionally requires a fresh, exclusively reserved regular PID
destination. Existing/symlink/nonregular destinations are refused. The parent's
FD9 must refer to that same linked regular file before any child is forked;
unsupported/failed identity checks fail closed. On Darwin, native `/usr/bin/stat`
with no file operands inspects `fstat` of stdin duplicated from the retained FD.
Its strict device/inode/regular-octal-mode tuple must exactly match the ordinary
nonsymlink regular path. Linux and Windows Bash control fixtures retain their
original builtin `/dev/fd` regular-file and device/inode checks. Unknown platforms
are refused. A private bounded startup gate
prevents the intended Julia main from executing until the parent's checked PID
write, exact bounded readback and linked-file identity verification succeed.
Before allow, failure cancels and waits the exact direct gated child, without
signals based on a failed pidfile; absent cancellation it self-expires after 30 s.
After allow may have committed, uncertainty/cleanup errors preserve the valid
PID record rather than orphaning a possibly running main. Pending child argv
retains the exact intended clone entrypoint, and exec retains that process PID.
The child closes reservation FD9 before launching the gate and suppresses parent
EXIT cleanup. It ignores HUP before Bash exec, including when the launcher exits.

Gate owner, allow, acknowledgement and cancellation markers retain regular-file
inode bindings through scoped FDs 8, 7, 6 and 5. Existing markers are refused before
opening them, so a preexisting nonregular marker cannot block a redirection. The
child validates owner/allow/acknowledgement identity and writes acknowledgement
through its already-open regular FD6; replacing the path with a directory, FIFO
or another regular file cannot redirect that write. A write error exits the gate.
The parent requires exact acknowledgement and an elapsed/iteration bound, and
returns failure on uncertainty rather than reporting RUNNING. All marker FDs
close before the intended main, and all scoped FDs 5--9 close on parent return.
The original BASH_ENV presence/value is restored only immediately before main
exec. The child uses the same source-defined descriptor predicate, captured through
checked Bash `declare -f`, plus static gate code; path, platform and command values
remain positional arguments, never interpolated into code. Owner, allow and
acknowledgement checks all use the same true retained-FD identity on Darwin.

Cleanup prevalidates every known marker and checks the retained owner/marker
bindings again before deletion. Detected foreign marker or directory replacement,
unknown identity, unexpected contents or cleanup failure is preserved/refused.
No recursive removal is used. This requires private user-controlled instance
state and trusted Bash/POSIX utilities; it is not protection against a hostile
same-user actor substituting paths between individual filesystem operations.
An abandoned private gate is preserved for verified reconciliation. Regular Unix
pidfiles retain their legacy malformed/dead cleanup policy, including empty
pre-allow reservations; a pending child cannot execute main without allow and
self-expires if its parent disappears. After a valid PID commit, cleanup failure retains that PID even
when the CLI returns failure, so a possibly running intended main stays manageable.
The gate's normal 30 s lifetime and parent acknowledgement's 5 s/100-iteration
limits are bounded wait/work controls, not hard deadlines under OS/filesystem stalls.

Primary API contracts: [POSIX2024 exec](https://pubs.opengroup.org/onlinepubs/9799919799/functions/exec.html)
lists Process ID among inherited attributes and retains ignored signals other
than the documented SIGCHLD exception. [GNU Bash exec and trap](https://www.gnu.org/s/bash/manual/bash.html#Bourne-Shell-Builtins)
specify process replacement and ignored-signal inheritance. [GNU Bash redirections](https://www.gnu.org/s/bash/manual/html_node/Redirections.html)
limits noclobber's documented existing-file protection to regular files, hence
the explicit existing-target checks. GNU [nohup source](https://github.com/coreutils/coreutils/blob/master/src/nohup.c)
sets ignored HUP and uses execvp; that source does not prove every Darwin nohup
implementation. The Unix gate instead uses Bash's ignored-HUP/exec semantics
with explicit stdin/stdout/stderr. Windows nohup and its native helper are unchanged.
Direct official-page fetch403/timeouts were retained in the external evidence;
official indexed text and the official raw GNU source verified these narrow facts.
Apple's [stat source](https://raw.githubusercontent.com/apple-oss-distributions/file_cmds/main/stat/stat.c)
uses `fstat(STDIN_FILENO)` with no operands. Apple's [Bash sh_stat source](https://raw.githubusercontent.com/apple-oss-distributions/bash/main/bash-3.2/lib/sh/eaccess.c)
uses pathname stat for configured `/dev/fd`; native Darwin diagnostics establish
that the facade device differs from the actual open file's device. Dropping the
device comparison or using inode alone is forbidden. Actual native Darwin
retained-FD metadata, Linux `/dev/fd/5` through `/dev/fd/9`, Bash3 and exec-PID continuity remain
mandatory CI oracles; documentation alone is not a platform-pass claim.
The exact-function controlled read/query tests run on Windows Bash too; they are
control-flow tests and do not establish native Linux/macOS process compatibility.

Independent oracles: original exact-function Windows ps failure; Python WinAPI PID,
creation and executable probes; actual inert Julia main marker/count; original fixture
PID observed after launching Bash exits; native process handle wait/termination;
wrong-root, unrelated, stale-generation, legacy/malformed, query-tool failure controls;
simultaneous starts and lock-owner generations. Tests must discover and verify Bash,
use runtime Julia paths and disposable directories containing spaces, and retain all
native command transcripts/timings. Unix legacy smoke retains daemon PID-reuse
safety and dead-owner recovery; its live-unrelated start-lock expectation follows
the conservative compatibility rule above. Native Linux
and macOS execution, full Pkg/examples/harness and independent review are separate
root-owned gates; Windows-only evidence does not establish those platform passes.

Official API contracts: OpenProcess, GetProcessTimes, QueryFullProcessImageNameW,
TerminateProcess and WaitForSingleObject (Microsoft Learn). Native probes are under
`C:/Temp/solarsindy-platform-audit-20261018`; those local evidence paths are historical
witnesses, never runtime defaults.
