using FileWatching: Pidfile

# Does the lock pidfile still have a live local owner? Answering `true` short-circuits the
# `trymkpidlock` call in `_with_forecast_log_lock`, which is the only place Pidfile's own staleness
# rule can run — so this predicate must apply that rule itself, not just a liveness test. The
# contract is exactly `!Pidfile.stale_pidfile(path, stale_age, refresh)` for a local live PID, and
# `test_live_forecast_verify.jl` pins that as a differential oracle against the stdlib.
#
# `Pidfile.stale_pidfile` declares a refreshed lock stale once its mtime is older than
# `5 * stale_age`, EVEN for a live PID, precisely because a PID can be recycled. Without the age
# clause here, a lock left behind by a SIGKILLed daemon whose PID is later reused by an unrelated
# long-lived process is reported as owned forever: every locked step (four issuances, refresh,
# retention, the ACI query) then burns its full 30 s timeout and no restart can clear it.
# `parse_pidfile` already returns the age, so this costs nothing extra.
#
# A FUTURE-dated pidfile (`age < -stale_age`, from a backwards clock step or a skewed filesystem) is
# deliberately NOT treated as stale here, because the stdlib does not treat it as stale either: on
# the pinned Julia, `stale_pidfile` only emits "filesystem time skew detected" for that case and
# returns `false`, so the lock is never reclaimed by age. Reporting it stale would diverge from the
# authority without clearing anything — `trymkpidlock` would still fail. What the short-circuit DOES
# swallow is the stdlib's warning, which would leave the resulting wedge as a silent 30 s stall per
# locked step, so this predicate raises the skew itself and the lock timeout reports the pidfile
# state that caused it.
function _forecast_pidfile_has_local_live_owner(lock_path::AbstractString;
                                                stale_after_sec::Real=900.0)
    isfile(lock_path) && !islink(lock_path) || return false
    try
        pid, hostname, age = Pidfile.parse_pidfile(String(lock_path))
        local_owner = isempty(hostname) || hostname == gethostname()
        (local_owner && Pidfile.isvalidpid(hostname, pid)) || return false
        if age < -Float64(stale_after_sec)
            @warn "forecast log lock is future-dated; neither Pidfile nor this mirror can " *
                  "reclaim it by age until the clock catches up" lock_path age_sec=age maxlog=1
        end
        # stale_age == 0 disables Pidfile's staleness rule entirely; mirror that.
        stale_after_sec > 0 || return true
        return age <= 5 * Float64(stale_after_sec)
    catch error
        # A concurrent release can make the read fail. The subsequent exclusive
        # open remains the authority, so a transient read failure is not ownership.
        error isa IOError || error isa EOFError || rethrow()
        return false
    end
end

# One-line pidfile state for the lock-timeout message, so a wedge names its cause (dead PID, foreign
# host, future-dated mtime) instead of only the path.
function _forecast_pidfile_diagnosis(lock_path::AbstractString)
    isfile(lock_path) || return isdir(lock_path) ? "lock path is a directory" :
                                (islink(lock_path) ? "lock path is a symlink" : "no pidfile present")
    try
        pid, hostname, age = Pidfile.parse_pidfile(String(lock_path))
        owner = isempty(hostname) ? "this host" : hostname
        skew = age < 0 ? "; mtime is $(round(-age; digits=1)) s in the FUTURE (clock skew)" : ""
        return "pid=$(pid) host=$(owner) age=$(round(age; digits=1)) s" *
               " valid_pid=$(Pidfile.isvalidpid(hostname, pid))" * skew
    catch error
        error isa IOError || error isa EOFError || rethrow()
        return "pidfile unreadable"
    end
end

function _with_forecast_log_lock(f, log_path::String; timeout_sec::Float64=30.0,
                                 stale_after_sec::Float64=900.0, poll_sec::Float64=0.05)
    timeout_sec >= 0 || throw(ArgumentError("timeout_sec must be nonnegative"))
    stale_after_sec >= 0 || throw(ArgumentError("stale_after_sec must be nonnegative"))
    poll_sec > 0 || throw(ArgumentError("poll_sec must be positive"))
    lock_path = log_path * ".lock"
    parent = dirname(lock_path)
    !isempty(parent) && mkpath(parent)
    deadline = time() + timeout_sec
    owner = false
    while owner === false
        if !isdir(lock_path) && !islink(lock_path) &&
           !_forecast_pidfile_has_local_live_owner(
               lock_path; stale_after_sec=stale_after_sec,
           )
            owner = Pidfile.trymkpidlock(
                lock_path; stale_age=stale_after_sec,
                refresh=stale_after_sec == 0 ? 0.0 : stale_after_sec / 2,
            )
        end
        owner === false || break
        time() < deadline || error(
            "timed out waiting for forecast log lock: $lock_path " *
            "[$(_forecast_pidfile_diagnosis(lock_path))]",
        )
        sleep(min(poll_sec, max(deadline - time(), 0.0)))
    end
    try
        return f()
    finally
        close(owner)
    end
end
