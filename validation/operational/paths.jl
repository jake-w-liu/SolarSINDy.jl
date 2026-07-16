if !isdefined(@__MODULE__, :OPERATIONAL_PACKAGE_ROOT)
    const OPERATIONAL_PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    const OPERATIONAL_WORKSPACE_ROOT = normpath(joinpath(OPERATIONAL_PACKAGE_ROOT, ".."))
    const _OPERATIONAL_OUTPUT_OVERRIDE = strip(get(ENV, "SOLARSINDY_OPERATIONAL_OUTPUT_DIR", ""))
    const OPERATIONAL_OUTPUT_DIR = abspath(isempty(_OPERATIONAL_OUTPUT_OVERRIDE) ?
        joinpath(OPERATIONAL_PACKAGE_ROOT, "validation", "output", "operational") :
        _OPERATIONAL_OUTPUT_OVERRIDE)
    mkpath(OPERATIONAL_OUTPUT_DIR)

    function _operational_path(env_name::AbstractString, fallback::AbstractString,
                               candidates::AbstractString...; directory::Bool = false)
        override = strip(get(ENV, env_name, ""))
        !isempty(override) && return abspath(override)
        present = directory ? isdir : isfile
        for candidate in candidates
            present(candidate) && return abspath(candidate)
        end
        return abspath(fallback)
    end

    const _OPERATIONAL_PAPER_ROOT = joinpath(OPERATIONAL_WORKSPACE_ROOT, "paper_v2_monitor")
    const _OPERATIONAL_CACHE_FALLBACK = joinpath(OPERATIONAL_OUTPUT_DIR, "source_cache")
    const _OPERATIONAL_PAPER_EVIDENCE =
        joinpath(_OPERATIONAL_PAPER_ROOT, "data", "source", "operational")
    const _OPERATIONAL_PACKAGE_EVIDENCE =
        joinpath(OPERATIONAL_PACKAGE_ROOT, "data", "operational_validation")

    if !isdefined(@__MODULE__, :select_operational_evidence_dir)
        Base.include(@__MODULE__,
                     joinpath(OPERATIONAL_PACKAGE_ROOT, "app", "src", "operational_paths.jl"))
    end

    """Resolve a cohesive evidence directory containing every requested artifact.

    Explicit overrides are returned unchanged so a bad override fails closed at the consumer.
    Otherwise newly generated validation output takes precedence, followed by the preserved
    paper evidence and the small package snapshot. Merely creating the output directory is not
    enough to select it: all requested artifacts must be present.
    """
    function operational_evidence_dir(required_files::AbstractString...;
                                      output_dir::AbstractString=OPERATIONAL_OUTPUT_DIR,
                                      paper_dir::AbstractString=_OPERATIONAL_PAPER_EVIDENCE,
                                      package_dir::AbstractString=_OPERATIONAL_PACKAGE_EVIDENCE)
        return select_operational_evidence_dir(
            required_files...; output_dir=output_dir, paper_dir=paper_dir,
            package_dir=package_dir,
        )
    end

    const OPERATIONAL_OMNI = _operational_path(
        "SOLARSINDY_OMNI_EXTRACTED",
        joinpath(_OPERATIONAL_CACHE_FALLBACK, "omni_extracted.csv"),
        joinpath(_OPERATIONAL_PAPER_ROOT, "data", "omni_extracted.csv"),
    )
    const OPERATIONAL_STORM_CATALOG = joinpath(OPERATIONAL_PACKAGE_ROOT, "data", "storm_catalog.csv")
    const OPERATIONAL_GFZ_KP_SOURCE = _operational_path(
        "SOLARSINDY_GFZ_KP_SOURCE",
        joinpath(_OPERATIONAL_CACHE_FALLBACK, "Kp_ap_Ap_SN_F107_since_1932.txt"),
        joinpath(_OPERATIONAL_PAPER_ROOT, "data", "source_cache",
                 "Kp_ap_Ap_SN_F107_since_1932.txt"),
    )
    const OPERATIONAL_NOAA_KP_CACHE = _operational_path(
        "SOLARSINDY_NOAA_KP_CACHE",
        joinpath(_OPERATIONAL_CACHE_FALLBACK, "noaa_3day_forecast"),
        joinpath(_OPERATIONAL_PAPER_ROOT, "data", "source_cache", "noaa_3day_forecast");
        directory = true,
    )
    const OPERATIONAL_TEMERIN_CACHE = _operational_path(
        "SOLARSINDY_TEMERIN_CACHE",
        joinpath(_OPERATIONAL_CACHE_FALLBACK, "temerin_li_dst"),
        joinpath(_OPERATIONAL_PAPER_ROOT, "data", "source_cache", "temerin_li_dst");
        directory = true,
    )
    const OPERATIONAL_HRO_CACHE = _operational_path(
        "SOLARSINDY_HRO_CACHE",
        joinpath(_OPERATIONAL_CACHE_FALLBACK, "omni_hro");
        directory = true,
    )
    # `v2_replay_scored.csv` is the first core V2 replay artifact. Once regenerated, the audit
    # deliberately reads the output tree and fails on any still-missing companion evidence until
    # the complete operational workflow has run.
    const OPERATIONAL_EVIDENCE_DIR = operational_evidence_dir("v2_replay_scored.csv")
end
