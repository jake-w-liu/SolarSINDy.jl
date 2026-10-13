#!/usr/bin/env julia

# One-time, fail-closed migration of the accumulated Operational SINDy core hot log.
# Rows are copied byte-for-byte into the tracked historical tree before the
# current hot log is replaced by an empty schema for inertial-SINDy accrual.

using CSV
using DataFrames
using SHA

include(joinpath(@__DIR__, "paths.jl"))
isdefined(@__MODULE__, :LiveVerifyConfig) || include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const INERTIAL_SINDY_HOT_LOG = joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "live_forecast_log.csv")
const SindyCore_ARCHIVE_DIR = joinpath(OPERATIONAL_PACKAGE_ROOT, "data", "historical", "sindy_core")
const SindyCore_ARCHIVE_LOG = joinpath(SindyCore_ARCHIVE_DIR, "live_forecast_log.csv")
const SindyCore_ARCHIVE_MANIFEST = joinpath(SindyCore_ARCHIVE_DIR, "live_forecast_log_manifest.csv")

_sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

function _refresh_inertial_sindy_monitor_artifacts!()
    state = _load_or_rebuild_live_state!(INERTIAL_SINDY_HOT_LOG)
    Int(state["row_count"]) == 0 || error("inertial-SINDy monitor state retained historical rows")
    !Bool(state["has_historical_model"]) || error(
        "inertial-SINDy monitor state retained a historical-model flag",
    )
    write_live_comparison_report(
        INERTIAL_SINDY_HOT_LOG,
        DEFAULT_REPORT_PATH;
        empty_identity=:inertial_sindy,
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

function migrate_inertial_sindy_live_log!()
    isfile(INERTIAL_SINDY_HOT_LOG) || error("current hot log is missing: $INERTIAL_SINDY_HOT_LOG")
    hot = CSV.read(INERTIAL_SINDY_HOT_LOG, DataFrame)
    "model_version" in names(hot) || error("hot log omits model_version")

    if nrow(hot) == 0
        isfile(SindyCore_ARCHIVE_LOG) || error(
            "hot log is already empty but the historical SINDy core archive is missing",
        )
        _refresh_inertial_sindy_monitor_artifacts!()
        println("inertial-SINDy hot log is already empty; historical SINDy core archive is present")
        return SindyCore_ARCHIVE_LOG
    end

    versions = sort(unique(String.(collect(skipmissing(hot.model_version)))))
    all(==("inertial_sindy"), versions) || error(
        "refusing to archive a hot log containing nonhistorical versions: $(join(versions, ", "))",
    )
    if "sub_hourly_model_version" in names(hot)
        served = String.(collect(skipmissing(hot.sub_hourly_model_version)))
        any(startswith.(served, "inertial_sindy")) && error(
            "refusing to archive a log that already contains inertial-SINDy served rows",
        )
    end

    mkpath(SindyCore_ARCHIVE_DIR)
    if isfile(SindyCore_ARCHIVE_LOG)
        _sha256_file(SindyCore_ARCHIVE_LOG) == _sha256_file(INERTIAL_SINDY_HOT_LOG) || error(
            "existing historical log differs from the SINDy core hot log",
        )
    else
        cp(INERTIAL_SINDY_HOT_LOG, SindyCore_ARCHIVE_LOG; force=false)
    end
    archive_sha = _sha256_file(SindyCore_ARCHIVE_LOG)
    archive_sha == _sha256_file(INERTIAL_SINDY_HOT_LOG) || error(
        "historical copy failed byte-identity verification",
    )

    verified = "observation_dst_nt" in names(hot) ?
               count(!ismissing, hot.observation_dst_nt) : 0
    manifest = DataFrame(
        operational_version=["sindy_core"],
        source_role=["historical_accumulated_live_log"],
        rows=[nrow(hot)],
        verified_rows=[verified],
        nonmissing_model_rows=[length(collect(skipmissing(hot.model_version)))],
        model_versions=[join(versions, ";")],
        sha256=[archive_sha],
        migration_target=["fresh_inertial-sindy_hot_log_schema"],
    )
    CSV.write(SindyCore_ARCHIVE_MANIFEST, manifest)
    _atomic_empty_schema!(INERTIAL_SINDY_HOT_LOG, hot)

    empty = CSV.read(INERTIAL_SINDY_HOT_LOG, DataFrame)
    nrow(empty) == 0 || error("inertial-SINDy hot log reset did not produce zero rows")
    names(empty) == names(hot) || error("inertial-SINDy hot log schema changed during reset")
    _sha256_file(SindyCore_ARCHIVE_LOG) == archive_sha || error(
        "historical SINDy core archive changed after hot-log reset",
    )
    _refresh_inertial_sindy_monitor_artifacts!()
    println("Archived $(nrow(hot)) historical SINDy core rows and initialized an empty inertial-SINDy hot log")
    return SindyCore_ARCHIVE_LOG
end

if abspath(PROGRAM_FILE) == @__FILE__
    migrate_inertial_sindy_live_log!()
end
