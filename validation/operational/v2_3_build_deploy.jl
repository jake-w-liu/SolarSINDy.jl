#!/usr/bin/env julia

# v2_3_build_deploy.jl — assemble `deploy/v2_3_shadow/` from a scored V2.3 run tree.
#
# The confirmatory runner persists the fitted objects a live engine needs (the error layers, the
# analog standardisation) but not the deployment package: the analog frame, the analog origin
# identities, the correction calibration, the blend weights and a digest manifest live in three
# different places. This script collects them once, recomputes the pieces that must be derived
# rather than copied, and then verifies the result by loading it through the same
# `load_v23_serving_artifacts` path the live engine uses. A package that cannot be loaded and
# re-verified is not written out as usable.
#
# Two inputs are deliberately derived rather than copied:
#
#   * the analog frame is a slice of the causal hourly frame the scoring run read, so the served
#     retrieval searches the same driver archive rather than a re-cleaned copy;
#   * the analog origin identities are recomputed as the run's own archive rule — the run's DEV
#     anchors whose issue-time features are complete and whose seven continuation records exist,
#     inside the origin window the run recorded. Archive membership is not a function of the hourly
#     frame alone (an origin must also have been a V2.1 calibration anchor, which needs a
#     quality-flagged, non-gap-filled driver record at t-1), so the identities are shipped and the
#     loader re-checks every one of them against the frame.
#
# Usage (from the package root):
#   julia --startup-file=no --project=. validation/operational/v2_3_build_deploy.jl [options]
#
#   --from=<dir>        scored run tree (default validation/output/operational/v2_3_test)
#   --dev=<dir>         development tree holding the selection (default …/v2_3_dev)
#   --out=<dir>         deployment directory to write (default deploy/v2_3_shadow)
#   --from-test         build from the confirmatory TEST tree and its development tree (the default
#                       source; the flag is accepted so the intent can be written down explicitly)
#   --from-smoke        build from the pseudo-TEST smoke tree and its smoke development tree;
#                       the smoke tree has no decision, so the decision gate is waived
#   --force             overwrite a non-empty output directory
#
# The confirmatory decision for this candidate is NO_GO, so the package this script writes is a
# SHADOW deployment: the live engine computes and logs the V2.3 center from it, but never serves it
# and never uses it for severity. The decision gate below is therefore a completeness check (the
# scoring finished and recorded a verdict), not a GO gate.

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

"Deployment file names, in manifest order."
const V23_DEPLOY_FRAME = "analog_frame_2010_2019.csv"
const V23_DEPLOY_ORIGINS = "analog_origins.csv"
const V23_DEPLOY_STATS = "analog_feature_stats.csv"
const V23_DEPLOY_CALIBRATION = "t1r_calibration.csv"
const V23_DEPLOY_LAT = "lat_weights.csv"
const V23_DEPLOY_E_LAYERS = "e_layers.json"
const V23_DEPLOY_SELECTION = "selected_configuration.json"
const V23_DEPLOY_MANIFEST = "manifest.csv"

"""
    V23_DEPLOY_ARTIFACT_DIR

Subdirectory of a scored run tree holding the fitted objects the deployment needs. It mirrors
`V23_CONFIRM_ARTIFACT_DIR` in the confirmatory runner, which this script does not include: loading
the runner would pull in the whole scoring path to read one directory name.
"""
const V23_DEPLOY_ARTIFACT_DIR = "artifacts"

"First and last hour of the shipped analog frame slice."
const V23_DEPLOY_FRAME_FIRST = DateTime(2010, 1, 1, 0)
const V23_DEPLOY_FRAME_LAST = DateTime(2019, 12, 31, 23)

_v23_deploy_log(msg) = (println(msg); flush(stdout))

"Parse the command line into a build request."
function v23_deploy_options(args)
    from_smoke = any(==("--from-smoke"), args)
    from_test = any(==("--from-test"), args)
    from_smoke && from_test && error("--from-smoke and --from-test are mutually exclusive")
    force = any(==("--force"), args)
    from = from_smoke ? V23_TEST_SMOKE_DIR : V23_TEST_DIR
    dev = from_smoke ? V23_DEV_SMOKE_DIR : V23_DEV_DIR
    out = joinpath(OPERATIONAL_PACKAGE_ROOT, "deploy", "v2_3_shadow")
    for arg in args
        if startswith(arg, "--from=")
            from = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--dev=")
            dev = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--out=")
            out = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif arg in ("--from-smoke", "--from-test", "--force")
            continue
        else
            error("unknown option $arg")
        end
    end
    return (from = abspath(from), dev = abspath(dev), out = abspath(out),
            from_smoke = from_smoke, force = force)
end

"Manifest row `(entry_type, name)` of a scored run tree."
function _v23_deploy_source_row(manifest::DataFrame, kind::AbstractString, name::AbstractString)
    hits = [r for r in eachrow(manifest)
            if String(r.entry_type) == String(kind) && String(r.name) == String(name)]
    length(hits) == 1 || error(
        "the source manifest holds $(length(hits)) ($kind, $name) rows; expected exactly one",
    )
    return hits[1]
end

"""
    v23_deploy_origins(archive_bounds; table_path, frame_path) -> NamedTuple

Recompute the analog archive of a scored run: the run's DEV anchors inside the recorded origin
window whose issue-time features are complete and whose seven continuation records exist. Returns
the origin times, the sliced frame and the recomputed standardisation.

The window bounds come from the run's own manifest, so a smoke run (whose fitting archive stops at
the pseudo-TEST embargo) and the confirmed run (whose archive is all of DEV) are both reproduced by
the same rule.
"""
function v23_deploy_origins(archive_bounds::NamedTuple;
                            table_path::AbstractString = V23_BASE_TABLE,
                            frame_path::AbstractString = V23_BASE_HOURLY_FRAME)
    isfile(frame_path) || error("V2.3 hourly frame is missing: $frame_path")
    isfile(table_path) || error("V2.3 base table is missing: $table_path")
    frame = CSV.read(frame_path, DataFrame; types = Dict("time_utc" => DateTime))
    slice = frame[V23_DEPLOY_FRAME_FIRST .<= frame.time_utc .<= V23_DEPLOY_FRAME_LAST, :]
    nrow(slice) > 0 || error("the hourly frame holds no rows inside the shipped analog window")
    lookup = v23_serving_frame_lookup(slice)
    _v23_deploy_log("  analog frame slice: $(nrow(slice)) rows " *
                    "$(first(slice.time_utc)) .. $(last(slice.time_utc))")

    anchors = CSV.read(table_path, DataFrame; select = ["issue_time_utc", "partition"],
                       types = Dict("issue_time_utc" => DateTime))
    dev = Set(anchors.issue_time_utc[anchors.partition .== "DEV"])
    candidates = sort([t for t in slice.time_utc
                       if archive_bounds.first <= t <= archive_bounds.last && t in dev])
    isempty(candidates) && error("the recorded origin window contains no DEV anchor")
    X, ok = v23_feature_matrix(slice, candidates)
    continuable = v23_analog_origin_ok(lookup, candidates)
    keep = [j for j in eachindex(candidates) if ok[j] && continuable[j]]
    origins = candidates[keep]
    length(origins) == archive_bounds.count || error(
        "the recomputed analog archive holds $(length(origins)) origins but the scored run " *
        "recorded $(archive_bounds.count); the deployment would not search the scored archive",
    )
    first(origins) == archive_bounds.first || error(
        "the recomputed archive starts at $(first(origins)); the run recorded $(archive_bounds.first)",
    )
    last(origins) == archive_bounds.last || error(
        "the recomputed archive ends at $(last(origins)); the run recorded $(archive_bounds.last)",
    )
    stats = v23_feature_stats(X[keep, :])
    return (origins = origins, frame = slice, stats = stats)
end

"Append one manifest row."
function _v23_deploy_push!(rows::Vector{NamedTuple}, entry_type, name, count, value)
    push!(rows, (entry_type = String(entry_type), name = String(name),
                 count = Float64(count), value = String(value)))
    return rows
end

"""
    build_v2_3_deploy(; from, dev, out, from_smoke, force) -> NamedTuple

Assemble and verify a V2.3 deployment directory. Returns the output path, the origin count and the
loaded artifacts, so a caller (or a test) can assert on the package it just produced.
"""
function build_v2_3_deploy(; from::AbstractString, dev::AbstractString, out::AbstractString,
                           from_smoke::Bool = false, force::Bool = false,
                           table_path::AbstractString = V23_BASE_TABLE,
                           frame_path::AbstractString = V23_BASE_HOURLY_FRAME)
    source = abspath(String(from))
    dev_dir = abspath(String(dev))
    outdir = abspath(String(out))
    isdir(source) || error("scored V2.3 run tree not found: $source")
    isdir(dev_dir) || error("V2.3 development tree not found: $dev_dir")
    decision_path = joinpath(source, "decision.csv")
    if !from_smoke && !isfile(decision_path)
        error(
            "refusing to build a deployment from $source: it holds no decision.csv, so the " *
            "confirmatory scoring has not completed. Pass --from-smoke to package the " *
            "pseudo-TEST smoke tree instead.",
        )
    end
    artifact_dir = joinpath(source, V23_DEPLOY_ARTIFACT_DIR)
    isdir(artifact_dir) || error("the run tree holds no $(V23_DEPLOY_ARTIFACT_DIR)/ directory")
    source_manifest = CSV.read(joinpath(source, "manifest.csv"), DataFrame;
                               types = Dict("entry_type" => String, "name" => String,
                                            "value" => String))

    if isdir(outdir) && !isempty(readdir(outdir)) && !force
        error("deployment directory $outdir is not empty; pass --force to overwrite it")
    end
    mkpath(outdir)

    selection_path = joinpath(dev_dir, "selected_configuration.json")
    isfile(selection_path) || error("development selection is missing: $selection_path")
    selection = JSON3.read(read(selection_path, String), Dict{String,Any})
    String(selection["family"]) == "T1r" || error(
        "only the T1r family is deployable; the selection records $(selection["family"])",
    )
    Bool(selection["safeguards"]) == false || error(
        "the deployable T1r center takes no V2.1 safeguards; the selection records safeguards = true",
    )
    calibration_name = String(selection["t1r_calibration_csv"])
    calibration_path = joinpath(dev_dir, calibration_name)
    isfile(calibration_path) || error("T1r calibration is missing: $calibration_path")
    lat_path = joinpath(dev_dir, "lat_weights.csv")
    isfile(lat_path) || error("lead-aware blend weights are missing: $lat_path")

    e_layers_path = joinpath(artifact_dir, V23_DEPLOY_E_LAYERS)
    isfile(e_layers_path) || error("error-layer record is missing: $e_layers_path")
    e_layers = JSON3.read(read(e_layers_path, String))
    String(e_layers["selected_config"]) == String(selection["selected_config"]) || error(
        "the run's error layers were fitted for $(e_layers["selected_config"]) but the selection " *
        "names $(selection["selected_config"])",
    )
    Float64.(collect(e_layers["lat_weights"])) ==
        Float64.(collect(selection["lat_weights"])) || error(
        "the run's error-layer record and the selection disagree on the blend weights",
    )

    archive_row = _v23_deploy_source_row(source_manifest, "analog_archive", "origins")
    pairs = Dict{String,String}()
    for piece in split(String(archive_row.value), ';')
        parts = split(piece, '=', limit = 2)
        length(parts) == 2 && (pairs[strip(String(parts[1]))] = strip(String(parts[2])))
    end
    bounds = (count = Int(round(Float64(archive_row.count))),
              first = DateTime(pairs["first"]), last = DateTime(pairs["last"]))
    _v23_deploy_log("  scored archive: $(bounds.count) origins " *
                    "$(bounds.first) .. $(bounds.last)")

    rebuilt = v23_deploy_origins(bounds; table_path = table_path, frame_path = frame_path)

    # ---- write the package ----
    CSV.write(joinpath(outdir, V23_DEPLOY_FRAME), rebuilt.frame)
    CSV.write(joinpath(outdir, V23_DEPLOY_ORIGINS),
              DataFrame(origin_time_utc = rebuilt.origins))
    shipped_stats = joinpath(artifact_dir, V23_DEPLOY_STATS)
    isfile(shipped_stats) || error("analog standardisation is missing: $shipped_stats")
    cp(shipped_stats, joinpath(outdir, V23_DEPLOY_STATS); force = true)
    cp(calibration_path, joinpath(outdir, V23_DEPLOY_CALIBRATION); force = true)
    cp(lat_path, joinpath(outdir, V23_DEPLOY_LAT); force = true)
    cp(e_layers_path, joinpath(outdir, V23_DEPLOY_E_LAYERS); force = true)
    cp(selection_path, joinpath(outdir, V23_DEPLOY_SELECTION); force = true)
    layer_files = String[]
    for record in e_layers["steps"]
        artifact = record["artifact"]
        artifact === nothing && continue
        name = String(artifact)
        path = joinpath(artifact_dir, name)
        isfile(path) || error("error-layer artifact is missing: $path")
        cp(path, joinpath(outdir, name); force = true)
        push!(layer_files, name)
    end
    isempty(layer_files) && _v23_deploy_log(
        "  ! every model step keeps the identity error layer; no layer artifact is shipped",
    )

    # A shipped standardisation that disagrees with the recomputed one would make the loader's
    # archive check vacuous, so the disagreement is caught here, before the manifest is written.
    stats_table = CSV.read(joinpath(outdir, V23_DEPLOY_STATS), DataFrame)
    worst = max(maximum(abs.(rebuilt.stats.mean .- Float64.(stats_table.feature_mean))),
                maximum(abs.(rebuilt.stats.sd .- Float64.(stats_table.feature_sd))))
    worst <= V23_SERVING_STATS_ATOL || error(
        "the recomputed analog standardisation disagrees with the run's shipped table by $worst",
    )

    files = vcat(String[V23_DEPLOY_FRAME, V23_DEPLOY_ORIGINS, V23_DEPLOY_STATS,
                        V23_DEPLOY_CALIBRATION, V23_DEPLOY_LAT, V23_DEPLOY_E_LAYERS,
                        V23_DEPLOY_SELECTION], layer_files)
    rows = NamedTuple[]
    _v23_deploy_push!(rows, "build", "utc", NaN, string(now(UTC)))
    _v23_deploy_push!(rows, "build", "package_version", NaN,
                      string(pkgversion(SolarSINDy)))
    _v23_deploy_push!(rows, "build", "julia_version", NaN, string(VERSION))
    _v23_deploy_push!(rows, "build", "smoke", NaN, string(from_smoke))
    _v23_deploy_push!(rows, "source", "tree", NaN, source)
    _v23_deploy_push!(rows, "source", "tree_label", NaN, basename(source))
    _v23_deploy_push!(rows, "source", "development_dir", NaN, dev_dir)
    _v23_deploy_push!(rows, "source", "decision_csv", NaN,
                      isfile(decision_path) ? _v23_sha256_file(decision_path) : "absent")
    _v23_deploy_push!(rows, "source", "run_manifest_sha256", NaN,
                      _v23_sha256_file(joinpath(source, "manifest.csv")))
    _v23_deploy_push!(rows, "source", "hourly_frame_sha256", NaN, _v23_sha256_file(frame_path))
    _v23_deploy_push!(rows, "source", "base_table_sha256", NaN, _v23_sha256_file(table_path))
    _v23_deploy_push!(rows, "selection", "config", NaN, String(selection["selected_config"]))
    _v23_deploy_push!(rows, "selection", "family", NaN, String(selection["family"]))
    _v23_deploy_push!(rows, "selection", "k", Float64(selection["k"]), "")
    _v23_deploy_push!(rows, "selection", "weight_set", NaN,
                      String(selection["params"]["weight_set"]))
    _v23_deploy_push!(rows, "selection", "safeguards", NaN, string(selection["safeguards"]))
    _v23_deploy_push!(rows, "selection", "t1r_calibration_csv", NaN, calibration_name)
    _v23_deploy_push!(rows, "selection", "identity", NaN,
                      v23_serving_identity(Symbol(String(selection["params"]["weight_set"])),
                                           Int(selection["k"])))
    _v23_deploy_push!(rows, "analog_archive", "origins", Float64(length(rebuilt.origins)),
                      "first=$(first(rebuilt.origins));last=$(last(rebuilt.origins))")
    _v23_deploy_push!(rows, "analog_archive", "frame_window", Float64(nrow(rebuilt.frame)),
                      "first=$(first(rebuilt.frame.time_utc));last=$(last(rebuilt.frame.time_utc))")
    for name in files
        path = joinpath(outdir, name)
        _v23_deploy_push!(rows, "sha256", name, Float64(filesize(path)),
                          _v23_sha256_file(path))
    end
    CSV.write(joinpath(outdir, V23_DEPLOY_MANIFEST), DataFrame(rows))

    artifacts = load_v23_serving_artifacts(outdir)
    _v23_deploy_log("  wrote $outdir: $(length(files) + 1) files, " *
                    "$(length(artifacts.origins)) origins, identity $(artifacts.identity)")
    return (dir = outdir, origins = length(artifacts.origins), artifacts = artifacts,
            files = files, manifest = DataFrame(rows))
end

function main(args = ARGS)
    options = v23_deploy_options(args)
    _v23_deploy_log("V2.3 deployment build")
    _v23_deploy_log("  from $(options.from)")
    _v23_deploy_log("  dev  $(options.dev)")
    _v23_deploy_log("  out  $(options.out)")
    built = build_v2_3_deploy(; from = options.from, dev = options.dev, out = options.out,
                              from_smoke = options.from_smoke, force = options.force)
    _v23_deploy_log("V2.3 deployment build PASS")
    return built
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
