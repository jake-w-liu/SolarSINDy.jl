# Owned process cleanup

The retained process owner creates a private POSIX session or Windows Job
Object, reports nonzero exits and timeouts, and cleans up its owned process
scope. Process identity and cleanup checks remain part of the implementation.

On POSIX, cleanup retains the leader identity while checking and signaling its
private process group. A successful signal or missing group alone does not
establish session closure. Unknown or live session members fail cleanup; the
shared cleanup allowance is20seconds and counts against the stage deadline.
Windows uses a private Job Object with KILL_ON_JOB_CLOSE and reaps its child.

Essential process tests cover successful exit, nonzero exit and timeout with
cleanup. The current reader-owner tests remain available separately. Historical
source clones, V2 process-state spies, Darwin diagnostic runners and repeated
CI certificate wrappers have been removed from the test tree; their original
versions remain in Git history. They are not development prerequisites.
