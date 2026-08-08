#!/usr/bin/env julia

# One-time, fail-closed migration of the accumulated Operational V2.0 hot log.
# Rows are copied byte-for-byte into the tracked historical tree before the
# current hot log is replaced by an empty schema for V2.1 accrual.

using CSV
using DataFrames
using SHA

include(joinpath(@__DIR__, "paths.jl"))
include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const V21_HOT_LOG = joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "live_forecast_log.csv")
const V20_ARCHIVE_DIR = joinpath(OPERATIONAL_PACKAGE_ROOT, "data", "historical", "v2_0")
const V20_ARCHIVE_LOG = joinpath(V20_ARCHIVE_DIR, "live_forecast_log.csv")
const V20_ARCHIVE_MANIFEST = joinpath(V20_ARCHIVE_DIR, "live_forecast_log_manifest.csv")

_sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

function _refresh_v2_1_monitor_artifacts!()
    state = _load_or_rebuild_live_state!(V21_HOT_LOG)
    Int(state["row_count"]) == 0 || error("V2.1 monitor state retained historical rows")
    !Bool(state["has_historical_model"]) || error(
        "V2.1 monitor state retained a historical-model flag",
    )
    write_live_comparison_report(
        V21_HOT_LOG,
        DEFAULT_REPORT_PATH;
        empty_identity=:v2_1,
    )
    return state
end

function _atomic_empty_schema!(path::AbstractString, schema::DataFrame)
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path))
    close(io)
    try
        CSV.write(tmp, first(schema, 0))
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return path
end

function migrate_v2_1_live_log!()
    isfile(V21_HOT_LOG) || error("current hot log is missing: $V21_HOT_LOG")
    hot = CSV.read(V21_HOT_LOG, DataFrame)
    "model_version" in names(hot) || error("hot log omits model_version")

    if nrow(hot) == 0
        isfile(V20_ARCHIVE_LOG) || error(
            "hot log is already empty but the historical V2.0 archive is missing",
        )
        _refresh_v2_1_monitor_artifacts!()
        println("V2.1 hot log is already empty; historical V2.0 archive is present")
        return V20_ARCHIVE_LOG
    end

    versions = sort(unique(String.(collect(skipmissing(hot.model_version)))))
    all(==("v2"), versions) || error(
        "refusing to archive a hot log containing nonhistorical versions: $(join(versions, ", "))",
    )
    if "sub_hourly_model_version" in names(hot)
        served = String.(collect(skipmissing(hot.sub_hourly_model_version)))
        any(startswith.(served, "v2.1")) && error(
            "refusing to archive a log that already contains V2.1 served rows",
        )
    end

    mkpath(V20_ARCHIVE_DIR)
    if isfile(V20_ARCHIVE_LOG)
        _sha256_file(V20_ARCHIVE_LOG) == _sha256_file(V21_HOT_LOG) || error(
            "existing historical log differs from the V2.0 hot log",
        )
    else
        cp(V21_HOT_LOG, V20_ARCHIVE_LOG; force=false)
    end
    archive_sha = _sha256_file(V20_ARCHIVE_LOG)
    archive_sha == _sha256_file(V21_HOT_LOG) || error(
        "historical copy failed byte-identity verification",
    )

    verified = "observation_dst_nt" in names(hot) ?
               count(!ismissing, hot.observation_dst_nt) : 0
    manifest = DataFrame(
        operational_version=["v2.0"],
        source_role=["historical_accumulated_live_log"],
        rows=[nrow(hot)],
        verified_rows=[verified],
        nonmissing_model_rows=[length(collect(skipmissing(hot.model_version)))],
        model_versions=[join(versions, ";")],
        sha256=[archive_sha],
        migration_target=["fresh_v2.1_hot_log_schema"],
    )
    CSV.write(V20_ARCHIVE_MANIFEST, manifest)
    _atomic_empty_schema!(V21_HOT_LOG, hot)

    empty = CSV.read(V21_HOT_LOG, DataFrame)
    nrow(empty) == 0 || error("V2.1 hot log reset did not produce zero rows")
    names(empty) == names(hot) || error("V2.1 hot log schema changed during reset")
    _sha256_file(V20_ARCHIVE_LOG) == archive_sha || error(
        "historical V2.0 archive changed after hot-log reset",
    )
    _refresh_v2_1_monitor_artifacts!()
    println("Archived $(nrow(hot)) historical V2.0 rows and initialized an empty V2.1 hot log")
    return V20_ARCHIVE_LOG
end

if abspath(PROGRAM_FILE) == @__FILE__
    migrate_v2_1_live_log!()
end
