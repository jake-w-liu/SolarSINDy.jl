using Test

const _V22_LAUNCHD_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _V22_LAUNCHD_INSTALLER = joinpath(
    _V22_LAUNCHD_PACKAGE_ROOT, "deploy", "install_launchd.sh",
)

function _v22_fake_launchd_tools(root)
    fake_bin = joinpath(root, "fake-bin")
    event_log = joinpath(root, "launchctl.log")
    julia_bin = joinpath(fake_bin, "julia & # shim")
    real_plutil = Sys.which("plutil")
    real_plutil === nothing && error("plutil is required for launchd tests")
    mkpath(fake_bin)
    write(event_log, "")
    launchctl = joinpath(fake_bin, "launchctl")
    write(launchctl, raw"""#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$SOLARSINDY_TEST_LAUNCHCTL_LOG"
""")
    write(julia_bin, "#!/bin/bash\nexit 0\n")
    plutil = joinpath(fake_bin, "plutil")
    write(plutil, raw"""#!/bin/bash
set -euo pipefail
case "$*" in
  *"${SOLARSINDY_TEST_PLUTIL_FAIL_MATCH:-__never__}"*) exit 9 ;;
esac
exec "$SOLARSINDY_TEST_REAL_PLUTIL" "$@"
""")
    chmod(launchctl, 0o755)
    chmod(julia_bin, 0o755)
    chmod(plutil, 0o755)
    return (; fake_bin, event_log, julia_bin, real_plutil)
end

# Configuration keys the installer reads from its environment. Every test clears them first, so a
# rendered plist depends only on what the test passes and never on the developer's shell.
const _V22_CONFIG_KEYS = (
    "SWM_HOST", "SWM_PORT", "SWM_WEBHOOK_URL",
    "LIVE_MONITOR_INTERVAL_SEC", "LIVE_MONITOR_DEADMAN_CYCLES", "LIVE_MONITOR_MAX_LOG_ROWS",
    "LIVE_MONITOR_LOG_MAX_BYTES", "LIVE_MONITOR_LOG_MAX_FILES",
    "SOLARSINDY_V2_CALIBRATION", "SOLARSINDY_V2_4_DEPLOY_DIR", "SOLARSINDY_V2_3_SHADOW_DIR",
    "SOLARSINDY_V2_2_STACK", "SOLARSINDY_V2_2_STACK_SHA256", "SOLARSINDY_ALLOW_UNPINNED_STACK",
    "SOLARSINDY_WATCHDOG_DASH_URL", "SOLARSINDY_WATCHDOG_STALE_SEC",
    "SOLARSINDY_WATCHDOG_DATA_URL", "SOLARSINDY_WATCHDOG_DATA_TIMEOUT",
    "SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES",
    "LIVE_FUTURE_CLOCK_TOLERANCE_MIN", "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN",
    "SOLARSINDY_JULIA_THREADS", "SOLARSINDY_NO_OPEN", "SOLARSINDY_IGNORE_SERVICE", "JULIA",
)

# Templates the installer ships, and the placeholder names it knows how to substitute. A template
# carrying anything outside this set would install a service whose environment is a literal marker.
const _V22_TEMPLATE_SUFFIXES = ("live-monitor", "dashboard", "watchdog", "v22-receipt-collector")
const _V22_RENDERED_PLACEHOLDERS = Set([
    "__JULIA_BIN__", "__CLONE_DIR__", "__APP_DIR__", "__MONITOR_DIR__",
    "__RECEIPT_DIR__", "__RECEIPT_LOG_DIR__", "__SWM_HOST__", "__SWM_PORT__",
    "__WATCHDOG_DASH_URL__", "__WATCHDOG_STALE_SEC__", "__EXTRA_ENV__",
])

_v22_template_path(suffix) = joinpath(
    _V22_LAUNCHD_PACKAGE_ROOT, "deploy", "com.example.solarsindy.$suffix.plist",
)

# Split a template into its leading XML comment (the manual-install documentation an operator reads)
# and the plist body (the part whose placeholders become live configuration).
function _v22_template_parts(path)
    lines = readlines(path)
    body_start = findfirst(line -> occursin("<plist", line), lines)
    body_start === nothing && error("no <plist> element in $path")
    return (header=join(lines[1:body_start-1], "\n"), body=join(lines[body_start:end], "\n"))
end

_v22_placeholders(text) = sort(unique(String.(m.match for m in eachmatch(r"__[A-Za-z0-9_]+__", text))))

function _v22_launchd_command(
        home, monitor, receipts, logs, tools, services...;
        clone=_V22_LAUNCHD_PACKAGE_ROOT,
        include_clone=true,
        load="0",
        plutil_fail_match=nothing,
        extra_env=Dict{String,String}())
    arguments = String[]
    include_clone && push!(arguments, String(clone))
    append!(arguments, String.(services))
    command = `bash $_V22_LAUNCHD_INSTALLER $arguments`
    command = addenv(command, (key => nothing for key in _V22_CONFIG_KEYS)...)
    command = addenv(
        command,
        "HOME" => home,
        "PATH" => string(tools.fake_bin, ":", get(ENV, "PATH", "")),
        "SOLARSINDY_LOAD" => load,
        "SOLARSINDY_JULIA" => tools.julia_bin,
        "SOLARSINDY_MONITOR_DIR" => monitor,
        "SOLARSINDY_V22_RECEIPT_DIR" => receipts,
        "SOLARSINDY_V22_RECEIPT_LOG_DIR" => logs,
        "SOLARSINDY_ORG" => "empire",
        "SOLARSINDY_TEST_LAUNCHCTL_LOG" => tools.event_log,
        "SOLARSINDY_TEST_REAL_PLUTIL" => tools.real_plutil,
        "SOLARSINDY_TEST_PLUTIL_FAIL_MATCH" => plutil_fail_match,
    )
    isempty(extra_env) && return command
    return addenv(command, (String(k) => String(v) for (k, v) in extra_env)...)
end

function _v22_command_succeeds(command)
    process = run(pipeline(command; stdout=devnull, stderr=devnull); wait=false)
    wait(process)
    return success(process)
end

# Same run, but with the console captured, so install-time reporting is testable.
function _v22_command_output(command)
    out = IOBuffer()
    process = run(pipeline(command; stdout=out, stderr=out); wait=false)
    wait(process)
    return (ok=success(process), text=String(take!(out)))
end

_v22_launchctl_events(tools) = readlines(tools.event_log)

function _v22_xml_text(value)
    return replace(
        String(value),
        "&" => "&amp;",
        "<" => "&lt;",
        ">" => "&gt;",
        "\"" => "&quot;",
        "'" => "&apos;",
    )
end

function _v22_plist(home, suffix)
    return joinpath(
        home, "Library", "LaunchAgents", "com.empire.solarsindy.$suffix.plist",
    )
end

@testset verbose=true "Shipped launchd templates document their own manual procedure" begin
    # Both install paths have to work. The install script substitutes every placeholder; a manual
    # copy-and-fill is only possible if the template's own header names every placeholder its body
    # contains. A placeholder introduced in the body without a matching header entry — which is how
    # __SWM_HOST__/__SWM_PORT__/__WATCHDOG_DASH_URL__/__WATCHDOG_STALE_SEC__ replaced their working
    # literals — renders a service whose environment is the literal marker.
    for suffix in _V22_TEMPLATE_SUFFIXES
        @testset "$suffix" begin
            path = _v22_template_path(suffix)
            @test isfile(path)
            parts = _v22_template_parts(path)
            body_placeholders = _v22_placeholders(parts.body)
            @test !isempty(body_placeholders)
            for placeholder in body_placeholders
                # Documented for the manual procedure. The header names the placeholder without its
                # surrounding underscores where a literal marker would survive substitution into the
                # rendered plist (`__EXTRA_ENV__`), so the name is what has to be present.
                @test occursin(strip(placeholder, '_'), parts.header)
                # ...and known to the renderer, so the scripted procedure fills it too.
                @test placeholder in _V22_RENDERED_PLACEHOLDERS
            end
            # The manual procedure must actually be stated, not merely implied by a list.
            @test occursin("Manual install", parts.header)
            @test occursin("install_launchd.sh", parts.header)
        end
    end
end

# The rendering testsets drive `deploy/install_launchd.sh`, which validates every plist it writes
# with `plutil` — an Apple tool. On a non-Apple runner the installer cannot run at all, so these
# testsets are skipped there with a visible marker rather than erroring; the template-contract
# testset above is pure text and runs everywhere. `test/test_cli_smoke.jl` and `app/test/runtests.jl`
# already guard their launchd sections the same way.
const _V22_LAUNCHD_RUNNABLE = Sys.isapple() && Sys.which("plutil") !== nothing

@testset verbose=true "Explicit-only V2.2 collector launchd rendering" begin
  if !_V22_LAUNCHD_RUNNABLE
    @test_skip "launchd rendering needs macOS and plutil; host is $(Sys.KERNEL) with plutil $(Sys.which("plutil") === nothing ? "absent" : "present")"
  else
    @testset "rendered plists carry no placeholder and leave no temporaries" begin
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            rendered = _v22_launchd_command(
                home, monitor, receipts, logs, tools,
                "monitor", "dashboard", "watchdog", "collector";
                load="0",
                extra_env=Dict("SWM_PORT" => "9311", "SWM_WEBHOOK_URL" => "https://hook.invalid/x"),
            )
            @test _v22_command_succeeds(rendered)

            agents = joinpath(home, "Library", "LaunchAgents")
            installed = sort(readdir(agents))
            # Nothing but the four finished plists: no `.env`, `.pre` or `.tmp` staging file.
            @test installed == sort(["com.empire.solarsindy.$s.plist"
                                     for s in _V22_TEMPLATE_SUFFIXES])
            for suffix in _V22_TEMPLATE_SUFFIXES
                body = _v22_template_parts(_v22_plist(home, suffix)).body
                @test isempty(_v22_placeholders(body))
            end
            # The values the placeholders replaced are the configured ones, not the old literals.
            dashboard = read(_v22_plist(home, "dashboard"), String)
            @test occursin("<key>SWM_PORT</key>\n    <string>9311</string>", dashboard)
            watchdog = read(_v22_plist(home, "watchdog"), String)
            @test occursin("http://127.0.0.1:9311/api/health", watchdog)
            @test occursin("<key>SOLARSINDY_WATCHDOG_STALE_SEC</key>\n    <string>7200</string>",
                           watchdog)
        end
    end

    @testset "a failed render leaves no staging file beside the real plists" begin
        # `set -euo pipefail` aborts the script the moment `plutil -lint` fails. Staging inside
        # ~/Library/LaunchAgents therefore left `.env`/`.pre`/`.tmp` files sitting next to the live
        # plists, where launchd and the operator both read.
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            failed = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor", "collector";
                include_clone=false,
                load="1",
                plutil_fail_match="v22-receipt-collector.plist",
            )
            @test !_v22_command_succeeds(failed)
            agents = joinpath(home, "Library", "LaunchAgents")
            @test sort(readdir(agents)) == ["com.empire.solarsindy.live-monitor.plist"]
            @test isempty(_v22_launchctl_events(tools))
        end
    end

    @testset "the live clock-skew band is validated at install time" begin
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            reject = extra -> _v22_command_output(_v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor";
                load="0", extra_env=extra,
            ))

            # The engine falls back to its documented default rather than refusing to start, so the
            # typo has to be fatal HERE, before it can be written into a plist.
            bad_tolerance = reject(Dict("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "two"))
            @test !bad_tolerance.ok
            @test occursin("LIVE_FUTURE_CLOCK_TOLERANCE_MIN", bad_tolerance.text)

            negative_skew = reject(Dict("LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "-5"))
            @test !negative_skew.ok
            @test occursin("LIVE_MAX_FUTURE_CLOCK_SKEW_MIN", negative_skew.text)

            # An inconsistent pair leaves the "report and exclude" band empty.
            inconsistent = reject(Dict("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "30",
                                       "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "10"))
            @test !inconsistent.ok
            @test occursin("must be at least", inconsistent.text)

            @test !ispath(joinpath(home, "Library", "LaunchAgents",
                                   "com.empire.solarsindy.live-monitor.plist"))
            @test isempty(_v22_launchctl_events(tools))

            accepted = _v22_command_output(_v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor";
                load="0",
                extra_env=Dict("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "3",
                               "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "30"),
            ))
            @test accepted.ok
            plist = read(_v22_plist(home, "live-monitor"), String)
            @test occursin("<key>LIVE_FUTURE_CLOCK_TOLERANCE_MIN</key>\n    <string>3</string>",
                           plist)
            @test occursin("<key>LIVE_MAX_FUTURE_CLOCK_SKEW_MIN</key>\n    <string>30</string>",
                           plist)
        end
    end

    @testset "V2.1 defaults remain collector-free" begin
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor & # <state>")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)

            render_only = _v22_launchd_command(
                home, monitor, receipts, logs, tools; load="0",
            )
            @test _v22_command_succeeds(render_only)
            @test isfile(_v22_plist(home, "live-monitor"))
            @test isfile(_v22_plist(home, "dashboard"))
            @test isfile(_v22_plist(home, "watchdog"))
            @test occursin(
                _v22_xml_text(monitor),
                read(_v22_plist(home, "live-monitor"), String),
            )
            @test occursin(
                _v22_xml_text(monitor),
                read(_v22_plist(home, "dashboard"), String),
            )
            @test !ispath(_v22_plist(home, "v22-receipt-collector"))
            @test isdir(joinpath(monitor, "logs"))
            @test !ispath(receipts)
            @test !ispath(logs)
            @test isempty(_v22_launchctl_events(tools))

            default_load = _v22_launchd_command(
                home, monitor, receipts, logs, tools; load=nothing,
            )
            @test _v22_command_succeeds(default_load)
            events = _v22_launchctl_events(tools)
            @test length(events) == 12
            @test first.(split.(events)) == repeat(
                ["bootout", "bootstrap", "enable", "kickstart"], 3,
            )
            @test all(!occursin("v22-receipt-collector", event) for event in events)
            for suffix in ("live-monitor", "dashboard", "watchdog")
                @test count(event -> occursin(suffix, event), events) == 4
            end
        end
    end

    @testset "collector-only special paths render exactly" begin
        mktempdir() do root
            special = " & # <tag> > \"double\" 'single' \\path"
            home = joinpath(root, "home")
            clone = joinpath(root, "clone" * special)
            monitor = joinpath(root, "monitor" * special)
            receipts = joinpath(root, "receipts" * special)
            logs = joinpath(root, "logs" * special)
            mkpath(home)
            mkpath(joinpath(clone, "examples"))
            mkpath(joinpath(clone, "app"))
            write(joinpath(clone, "examples", "v2_2_l1_receipt_collector.jl"), "")
            tools = _v22_fake_launchd_tools(root)

            command = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "collector";
                clone, load=nothing,
            )
            @test _v22_command_succeeds(command)
            @test isempty(_v22_launchctl_events(tools))
            plist = _v22_plist(home, "v22-receipt-collector")
            @test isfile(plist)
            @test all(!ispath(_v22_plist(home, suffix)) for suffix in
                      ("live-monitor", "dashboard", "watchdog"))
            @test !ispath(monitor)
            @test isdir(receipts)
            @test isdir(logs)

            text = read(plist, String)
            expected_arguments = (
                tools.julia_bin,
                "--startup-file=no",
                "--project=$clone",
                joinpath(clone, "examples", "v2_2_l1_receipt_collector.jl"),
                "--root=$receipts",
                "--interval-sec=60",
            )
            for argument in expected_arguments
                @test occursin(
                    "<string>$(_v22_xml_text(argument))</string>", text,
                )
            end
            @test occursin(
                "<string>$(_v22_xml_text(clone))</string>", text,
            )
            @test occursin(
                "<string>$(_v22_xml_text(logs))/launchd.out</string>", text,
            )
            @test occursin(
                "<string>$(_v22_xml_text(logs))/launchd.err</string>", text,
            )
            @test all(occursin(entity, text) for entity in
                      ("&amp;", "&lt;", "&gt;", "&quot;", "&apos;"))
            @test !occursin(r"__[A-Z0-9_]+__", text)
            @test success(pipeline(`plutil -lint $plist`, stdout=devnull))
        end
    end

    @testset "leading collector token is render-only unless explicitly loaded" begin
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)

            render_only = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "collector";
                include_clone=false, load=nothing,
            )
            @test _v22_command_succeeds(render_only)
            @test isfile(_v22_plist(home, "v22-receipt-collector"))
            @test all(!ispath(_v22_plist(home, suffix)) for suffix in
                      ("live-monitor", "dashboard", "watchdog"))
            @test !ispath(monitor)
            @test isempty(_v22_launchctl_events(tools))

            explicit_load = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "collector";
                include_clone=false, load="1",
            )
            @test _v22_command_succeeds(explicit_load)
            events = _v22_launchctl_events(tools)
            @test length(events) == 4
            @test first.(split.(events)) ==
                  ["bootout", "bootstrap", "enable", "kickstart"]
            @test all(occursin("v22-receipt-collector", event) for event in events)
            @test !ispath(monitor)
        end
    end

    @testset "allow-listed config keys reach the services that read them" begin
        # A launchd job inherits nothing from an interactive shell, so a key that is not rendered
        # into the plist does not reach the daemon. Before this the installed services carried only
        # the monitor directory, the thread count and a hard-coded 8723, while README and
        # solarsindy.env presented the file as the configuration mechanism.
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            webhook = "https://hooks.example.test/services/T&<0>"
            deploy_dir = joinpath(root, "v2_4 & <bundle>")
            stack_path = joinpath(root, "stack.csv")

            command = _v22_launchd_command(
                home, monitor, receipts, logs, tools; load="0",
                extra_env=Dict(
                    "SWM_HOST" => "0.0.0.0",
                    "SWM_PORT" => "9137",
                    "SWM_WEBHOOK_URL" => webhook,
                    "LIVE_MONITOR_INTERVAL_SEC" => "1800",
                    "LIVE_MONITOR_MAX_LOG_ROWS" => "1234",
                    "SOLARSINDY_V2_4_DEPLOY_DIR" => deploy_dir,
                    "SOLARSINDY_V2_2_STACK" => stack_path,
                ),
            )
            @test _v22_command_succeeds(command)

            dash = read(_v22_plist(home, "dashboard"), String)
            mon = read(_v22_plist(home, "live-monitor"), String)
            wd = read(_v22_plist(home, "watchdog"), String)

            @test occursin("<key>SWM_HOST</key>\n    <string>0.0.0.0</string>", dash)
            @test occursin("<key>SWM_PORT</key>\n    <string>9137</string>", dash)
            @test occursin(
                "<key>SWM_WEBHOOK_URL</key>\n    <string>$(_v22_xml_text(webhook))</string>", dash)
            @test !occursin("8723", dash)
            @test occursin("<key>LIVE_MONITOR_INTERVAL_SEC</key>\n    <string>1800</string>", mon)
            @test occursin("<key>LIVE_MONITOR_MAX_LOG_ROWS</key>\n    <string>1234</string>", mon)
            @test occursin(
                "<key>SOLARSINDY_V2_4_DEPLOY_DIR</key>\n    <string>$(_v22_xml_text(deploy_dir))</string>",
                mon)
            @test occursin(
                "<key>SOLARSINDY_V2_2_STACK</key>\n    <string>$(_v22_xml_text(stack_path))</string>",
                mon)
            # 0.0.0.0 is a bind address, not a probe destination: the watchdog health check has to
            # go over loopback on the configured port or it reports a healthy dashboard as down.
            @test occursin(
                "<key>SOLARSINDY_WATCHDOG_DASH_URL</key>\n    <string>http://127.0.0.1:9137/api/health</string>",
                wd)
            @test occursin("<key>SWM_WEBHOOK_URL</key>", wd)
            # Each key reaches only the services that read it.
            @test !occursin("LIVE_MONITOR_INTERVAL_SEC", dash)
            @test !occursin("SOLARSINDY_V2_4_DEPLOY_DIR", dash)
            @test !occursin("SWM_WEBHOOK_URL", mon)
            @test !occursin("SOLARSINDY_V2_4_DEPLOY_DIR", wd)
            for text in (dash, mon, wd)
                @test !occursin("__EXTRA_ENV__", text)
            end
            for suffix in ("live-monitor", "dashboard", "watchdog")
                @test success(pipeline(`plutil -lint $(_v22_plist(home, suffix))`,
                                       stdout=devnull))
            end
        end

        # An unset key leaves the daemon on its own documented default rather than pinning an
        # empty string, and the defaults still render when nothing is configured.
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            @test _v22_command_succeeds(
                _v22_launchd_command(home, monitor, receipts, logs, tools; load="0"))
            dash = read(_v22_plist(home, "dashboard"), String)
            mon = read(_v22_plist(home, "live-monitor"), String)
            wd = read(_v22_plist(home, "watchdog"), String)
            @test occursin("<key>SWM_HOST</key>\n    <string>127.0.0.1</string>", dash)
            @test occursin("<key>SWM_PORT</key>\n    <string>8723</string>", dash)
            @test !occursin("<key>SWM_WEBHOOK_URL</key>", dash)
            @test !occursin("<key>LIVE_MONITOR_INTERVAL_SEC</key>", mon)
            @test occursin(
                "<key>SOLARSINDY_WATCHDOG_DASH_URL</key>\n    <string>http://127.0.0.1:8723/api/health</string>",
                wd)
            @test occursin("<key>SOLARSINDY_WATCHDOG_STALE_SEC</key>\n    <string>7200</string>", wd)
            for text in (dash, mon, wd)
                @test !occursin("__EXTRA_ENV__", text)
            end
        end

        # A key that reaches none of the selected services is named at install time instead of
        # being silently dropped, and an unusable port fails before anything is written.
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            reported = _v22_command_output(_v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor"; load="0",
                extra_env=Dict("SWM_WEBHOOK_URL" => "https://hooks.example.test/x",
                               "SOLARSINDY_JULIA_THREADS" => "8"),
            ))
            @test reported.ok
            @test occursin("SWM_WEBHOOK_URL", reported.text)
            @test occursin("SOLARSINDY_JULIA_THREADS", reported.text)
            @test occursin("not rendered into the selected service", reported.text)
            @test !occursin("SOLARSINDY_V2_4_DEPLOY_DIR", reported.text)

            rejected = _v22_command_output(_v22_launchd_command(
                home, monitor, receipts, logs, tools; load="0",
                extra_env=Dict("SWM_PORT" => "not-a-port"),
            ))
            @test !rejected.ok
            @test occursin("SWM_PORT", rejected.text)
            @test isempty(_v22_launchctl_events(tools))
        end
    end

    @testset "preflight failures have no partial effects" begin
        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            invalid = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor", "invalid";
                include_clone=false, load="1",
            )
            @test !_v22_command_succeeds(invalid)
            @test isempty(_v22_launchctl_events(tools))
            @test !ispath(joinpath(home, "Library", "LaunchAgents"))
            @test !ispath(monitor)
            @test !ispath(receipts)
            @test !ispath(logs)
        end

        mktempdir() do root
            home = joinpath(root, "home")
            monitor = joinpath(root, "monitor")
            receipts = joinpath(root, "receipts")
            logs = joinpath(root, "receipt-logs")
            mkpath(home)
            tools = _v22_fake_launchd_tools(root)
            failed_render = _v22_launchd_command(
                home, monitor, receipts, logs, tools, "monitor", "collector";
                include_clone=false,
                load="1",
                plutil_fail_match="v22-receipt-collector.plist",
            )
            @test !_v22_command_succeeds(failed_render)
            @test isfile(_v22_plist(home, "live-monitor"))
            @test !ispath(_v22_plist(home, "v22-receipt-collector"))
            @test isempty(_v22_launchctl_events(tools))
        end

        for bad_component in ("bad\nreceipts", "bad\treceipts")
            mktempdir() do root
                home = joinpath(root, "home")
                monitor = joinpath(root, "monitor")
                receipts = joinpath(root, bad_component)
                logs = joinpath(root, "receipt-logs")
                mkpath(home)
                tools = _v22_fake_launchd_tools(root)
                invalid = _v22_launchd_command(
                    home, monitor, receipts, logs, tools, "collector";
                    load="1",
                )
                @test !_v22_command_succeeds(invalid)
                @test isempty(_v22_launchctl_events(tools))
                @test !ispath(joinpath(home, "Library", "LaunchAgents"))
                @test !ispath(monitor)
                @test !ispath(receipts)
            end
        end
    end
  end
end
