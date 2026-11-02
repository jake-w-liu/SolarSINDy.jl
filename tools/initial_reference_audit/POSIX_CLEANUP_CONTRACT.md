# Owned POSIX timeout cleanup

Confirmed evidence: macOS Julia 1.13 CI job 118555079345 in run 39400905407
raised `PermissionError` at the second `os.killpg` call in `finally`. Its raw
log has SHA256 6f8e6a57b1c09da27bc5cc4db7ebe6c0e3c35224713b013aac661835b09efb52.
The first kill and parent wait returned; the timeout path then repeated the
group signal. The frozen before implementation is 1c593d90... .

One fresh session is created before the launch callback. Every exit path sends
at most one owned-group SIGKILL and reaps the direct child. A timeout remains a
failed deadline even if the group disappears immediately before cleanup.
Only a missing group is tolerated; genuine permission failures stay fatal.
Normal root exit must still terminate its listening descendant. Preserve all
native Windows Job Object bytes and the two real descendant-listener controls.

The legacy normal-exit path also reaped the leader before its single group
signal. The V2 control contract requires feature admission before launch,
`waitid(P_PID, ..., WEXITED | WNOHANG | WNOWAIT)` observation, one private-group
signal, and then the first reaping wait. Missing nonreaping support fails before
launch; no polling or wait-before-signal fallback is allowed. Permission errors
remain fatal after bounded direct-child cleanup. No actual PGID-reuse event was
observed in the deterministic reproduction.

Both legacy entry points measure the complete stage, including cleanup. An
on-time exit followed by late cleanup remains a failed deadline, with its actual
exit code preserved. The Windows correction wraps the unchanged native
function; its signaled-handle path can otherwise return zero after the deadline
before testing elapsed time. The late-zero controls use fake delegates and
clocks and do not demonstrate a native Windows failure.

`fixtures/posix_cleanup_oracles_v2.json` and the independent V2 process-state
spies enforce these transitions, including separate observation-at-deadline
and cleanup-at-deadline cases. The frozen V1 oracle and before implementation
remain historical bytes. Hosted macOS provisions Python 3.14 before its first
Python consumer; Linux and Windows retain Python 3.12.

The pre-implementation literal events are frozen in
`posix_cleanup_oracles_v1.json`, SHA256
cc6a1d5892321862e1826300600c4f0f46d0dfe9a99c0e1b4106131ead16408c.
Portable simulated controls send no actual signals. Native macOS/Linux and
Windows tests, complete package/examples and the canonical harness remain
required. macOS 1.13 must pass three consecutive reader runs, fail first;
the existing Linux 1.13 three-run requirement remains.

Apple's current XNU `killpg1` filters zombie processes and may return EPERM
when no signalable group member remains (kern_sig.c, lines 1612–1621,
https://raw.githubusercontent.com/apple-oss-distributions/xnu/main/bsd/kern/kern_sig.c).
This supports the redundant-cleanup mechanism. Hypothesis: the failed runner
had only zombie members at its second call. Its kernel process state was not
retained; that precise postmortem state is not a verified claim.
