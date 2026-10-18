# Shared test prerequisite: WindowsApps/bash.exe may be a WSL launcher with no
# distribution. Verify actual GNU Bash execution before running shell oracles.
function _solarsindy_bash_output(command; timeout=5.0)
    output = IOBuffer()
    process = run(pipeline(ignorestatus(command); stdout=output, stderr=devnull); wait=false)
    completed = false
    try
        completed = timedwait(() -> process_exited(process), timeout; pollint=0.02) === :ok
        completed || kill(process)
        wait(process)
        completed && success(process) || error("Bash prerequisite probe failed")
        return String(take!(output))
    finally
        process_running(process) && kill(process)
        wait(process)
    end
end

function solarsindy_test_bash()
    explicit = get(ENV, "SOLARSINDY_TEST_BASH", "")
    candidates = isempty(explicit) ? String[] : [explicit]
    if isempty(explicit)
        found = Sys.which("bash")
        found === nothing || push!(candidates, found)
        git = Sys.which("git")
        if git !== nothing && Sys.iswindows()
            parent = dirname(realpath(git))
            for ancestor in (parent, dirname(parent), dirname(dirname(parent)))
                append!(candidates, [joinpath(ancestor, "bin", "bash.exe"),
                                     joinpath(ancestor, "usr", "bin", "bash.exe")])
            end
        end
    end
    for candidate in unique(candidates)
        try
            isfile(candidate) || continue
            banner = _solarsindy_bash_output(`$candidate --version`)
            startswith(banner, "GNU bash, version ") || continue
            token = _solarsindy_bash_output(`$candidate -c 'printf "BASH_OK:%s" "$BASH_VERSION"'`)
            startswith(token, "BASH_OK:") && length(token) > 8 && return candidate
        catch
            # Try the next known installation; an explicit override has no fallback.
        end
    end
    error("working GNU Bash required; set SOLARSINDY_TEST_BASH to its executable (WSL/app aliases are not sufficient)")
end
