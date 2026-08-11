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

function _v22_launchd_command(
        home, monitor, receipts, logs, tools, services...;
        clone=_V22_LAUNCHD_PACKAGE_ROOT,
        include_clone=true,
        load="0",
        plutil_fail_match=nothing)
    arguments = String[]
    include_clone && push!(arguments, String(clone))
    append!(arguments, String.(services))
    command = `bash $_V22_LAUNCHD_INSTALLER $arguments`
    return addenv(
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
end

function _v22_command_succeeds(command)
    process = run(pipeline(command; stdout=devnull, stderr=devnull); wait=false)
    wait(process)
    return success(process)
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

@testset verbose=true "Explicit-only V2.2 collector launchd rendering" begin
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
                plutil_fail_match="v22-receipt-collector.plist.tmp",
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
