using SHA
using JSON3
using Dates
using CSV
using DataFrames

isdefined(@__MODULE__, :validation_run_mode) ||
    include(joinpath(@__DIR__, "output_paths.jl"))

const CANONICAL_OMNI_SHA256 =
    "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"
const PROVENANCE_SCHEMA_VERSION = 1
const STORM_CATALOG_SCHEMA_VERSION = 1
const STORM_CATALOG_COLUMNS = (
    "storm_id", "onset_time", "min_dst_star", "min_dst_star_time",
    "recovery_end_time", "duration_hr", "solar_cycle", "split",
    "onset_idx", "end_idx",
)

_provenance_mode(mode) = begin
    parsed = Symbol(lowercase(String(mode)))
    parsed in (:canonical, :noncanonical, :test) || throw(ArgumentError(
        "mode must be canonical, noncanonical, or test",
    ))
    parsed
end

function provenance_sha256(path::AbstractString)
    isfile(path) || throw(ArgumentError("file not found: $path"))
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""Verify the frozen OMNI input, enforcing the AISR hash in canonical mode."""
function verify_omni_input(path::AbstractString; mode=validation_run_mode())
    run_mode = _provenance_mode(mode)
    digest = provenance_sha256(path)
    run_mode == :canonical && digest != CANONICAL_OMNI_SHA256 && error(
        "canonical OMNI SHA-256 mismatch for $(abspath(path)): " *
        "expected $CANONICAL_OMNI_SHA256, got $digest",
    )
    return (; path=abspath(path), sha256=digest, bytes=filesize(path),
            canonical=run_mode == :canonical)
end

function _provenance_source_files(root::AbstractString)
    files = String[]
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(root, name)
        isfile(path) && push!(files, path)
    end
    for dir_name in ("src", "validation")
        dir = joinpath(root, dir_name)
        isdir(dir) || continue
        for (walk_root, _, names) in walkdir(dir)
            for name in names
                path = joinpath(walk_root, name)
                isfile(path) && push!(files, path)
            end
        end
    end
    sort!(files; by=path -> relpath(path, root))
    isempty(files) && throw(ArgumentError("no source files found under $root"))
    return files
end

function provenance_source_tree_sha256(root::AbstractString)
    absolute_root = abspath(root)
    records = String[]
    for path in _provenance_source_files(absolute_root)
        push!(records, relpath(path, absolute_root) * "\0" * provenance_sha256(path))
    end
    return bytes2hex(SHA.sha256(join(records, "\n")))
end

function _provenance_git_head(root::AbstractString)
    try
        return readchomp(pipeline(`git -C $root rev-parse HEAD`; stderr=devnull))
    catch error_value
        error_value isa InterruptException && rethrow()
        return "not-a-git-worktree"
    end
end

function provenance_source_identity(package_root::AbstractString)
    root = abspath(package_root)
    project = joinpath(root, "Project.toml")
    manifest = joinpath(root, "Manifest.toml")
    isfile(project) || throw(ArgumentError("Project.toml not found under $root"))
    isfile(manifest) || throw(ArgumentError("Manifest.toml not found under $root"))
    return (
        git_baseline = _provenance_git_head(root),
        source_tree_sha256 = provenance_source_tree_sha256(root),
        julia_version = string(VERSION),
        project = (path=project, sha256=provenance_sha256(project)),
        manifest = (path=manifest, sha256=provenance_sha256(manifest)),
    )
end

function _provenance_require_regular_target(path::AbstractString)
    islink(path) && throw(ArgumentError(
        "provenance output target must not be a symbolic link: $path",
    ))
    ispath(path) && !isfile(path) && throw(ArgumentError(
        "provenance output target exists but is not a regular file: $path",
    ))
    return path
end

function _provenance_atomic_replace(source::AbstractString, target::AbstractString)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "provenance replacement source must be a regular non-symlink file: $source",
    ))
    _provenance_require_regular_target(target)
    Base.Filesystem.rename(source, target)
    isfile(target) && !islink(target) || throw(ErrorException(
        "provenance replacement did not install a regular file: $target",
    ))
    return target
end

function _provenance_atomic_json(path::AbstractString, value)
    parent = dirname(path)
    mkpath(parent)
    _provenance_require_regular_target(path)
    temp_path, io = mktemp(parent; cleanup=false)
    try
        JSON3.write(io, value)
        flush(io)
        close(io)
        _provenance_atomic_replace(temp_path, path)
    finally
        isopen(io) && close(io)
        isfile(temp_path) && rm(temp_path; force=true)
    end
    return path
end

function _provenance_input_records(input_paths)
    pairs_list = collect(pairs(input_paths))
    names = String[string(pair.first) for pair in pairs_list]
    length(unique(names)) == length(names) ||
        throw(ArgumentError("input names must be unique"))
    order = sortperm(names)
    return [begin
        path = abspath(String(pairs_list[index].second))
        isfile(path) || throw(ArgumentError("input file not found: $path"))
        (name=names[index], path=path, sha256=provenance_sha256(path),
         bytes=filesize(path))
    end for index in order]
end

"""
    write_output_manifest(output_path; producer_script, input_paths,
                          selection_record, seed=nothing,
                          deterministic=false, ...)

Atomically write the provenance sidecar `<output_path>.manifest.json`. Exactly
one of a nonnegative `seed` or `deterministic=true` is required.
"""
function write_output_manifest(output_path::AbstractString;
                               producer_script::AbstractString,
                               input_paths,
                               selection_record,
                               seed::Union{Nothing,Integer}=nothing,
                               deterministic::Bool=false,
                               metadata=(;),
                               package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                               mode=validation_run_mode(),
                               manifest_path::AbstractString=String(output_path) * ".manifest.json")
    run_mode = _provenance_mode(mode)
    isfile(output_path) || throw(ArgumentError("output file not found: $output_path"))
    isfile(producer_script) ||
        throw(ArgumentError("producing script not found: $producer_script"))
    selection_record === nothing &&
        throw(ArgumentError("selection_record must be explicit"))
    (seed === nothing) == deterministic || throw(ArgumentError(
        "provide exactly one of seed or deterministic=true",
    ))
    seed isa Bool && throw(ArgumentError("seed must be an integer, not Bool"))
    seed !== nothing && !(0 <= seed <= typemax(Int)) && throw(ArgumentError(
        "seed must be representable as a nonnegative Int",
    ))
    inputs = _provenance_input_records(input_paths)
    if run_mode == :canonical
        for input in inputs
            occursin("omni", lowercase(input.name)) || continue
            input.sha256 == CANONICAL_OMNI_SHA256 || error(
                "canonical manifest input $(input.name) has unpinned OMNI SHA-256",
            )
        end
    end
    source = provenance_source_identity(package_root)
    output = (path=abspath(output_path), sha256=provenance_sha256(output_path),
              bytes=filesize(output_path))
    producer = (path=abspath(producer_script),
                sha256=provenance_sha256(producer_script))
    record = (
        schema_version = PROVENANCE_SCHEMA_VERSION,
        run_mode = String(run_mode),
        producer = producer,
        source_identity = source,
        inputs = inputs,
        selection_record = selection_record,
        randomness = seed === nothing ?
            (kind="deterministic", seed=nothing) :
            (kind="seeded", seed=Int(seed)),
        timestamp_utc = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z",
        output = output,
        metadata = metadata,
    )
    _provenance_atomic_json(manifest_path, record)
    return abspath(manifest_path)
end

function _manifest_require(object, key::AbstractString)
    haskey(object, key) || error("provenance manifest is missing '$key'")
    return object[key]
end

function _manifest_hash_matches(record, label::AbstractString)
    path = String(_manifest_require(record, "path"))
    isfile(path) && !islink(path) || error(
        "$label must be a regular non-symlink file: $path",
    )
    expected = String(_manifest_require(record, "sha256"))
    actual = provenance_sha256(path)
    actual == expected || error("$label SHA-256 mismatch for $path")
    haskey(record, "bytes") && filesize(path) != Int(record["bytes"]) &&
        error("$label size mismatch for $path")
    return path
end

"""Verify an output and every recorded input before downstream consumption."""
function verify_output_manifest(output_path::AbstractString;
                                manifest_path::AbstractString=String(output_path) * ".manifest.json",
                                package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                require_canonical::Bool=false,
                                verify_source::Bool=true)
    isfile(manifest_path) && !islink(manifest_path) || error(
        "provenance manifest must be a regular non-symlink file: $manifest_path",
    )
    record = try
        JSON3.read(read(manifest_path, String))
    catch error_value
        error_value isa InterruptException && rethrow()
        error("invalid provenance manifest $manifest_path: $(sprint(showerror, error_value))")
    end
    Int(_manifest_require(record, "schema_version")) == PROVENANCE_SCHEMA_VERSION ||
        error("unsupported provenance schema version")
    run_mode = String(_provenance_mode(String(_manifest_require(record, "run_mode"))))
    require_canonical && run_mode != "canonical" &&
        error("noncanonical output cannot be consumed by a canonical run")
    manifest_output = _manifest_require(record, "output")
    abspath(output_path) == String(_manifest_require(manifest_output, "path")) ||
        error("manifest output path does not match requested output")
    _manifest_hash_matches(manifest_output, "output")
    _manifest_hash_matches(_manifest_require(record, "producer"), "producer")
    _manifest_require(record, "selection_record") === nothing &&
        error("provenance manifest has no selection record")
    randomness = _manifest_require(record, "randomness")
    randomness_kind = String(_manifest_require(randomness, "kind"))
    randomness_kind in ("deterministic", "seeded") ||
        error("invalid provenance randomness marker")
    randomness_kind == "seeded" &&
        Int(_manifest_require(randomness, "seed")) < 0 && error("invalid provenance seed")
    randomness_kind == "deterministic" && randomness["seed"] !== nothing &&
        error("deterministic provenance record must not contain a seed")
    timestamp = String(_manifest_require(record, "timestamp_utc"))
    endswith(timestamp, "Z") || error("provenance timestamp is not UTC")
    try
        DateTime(chop(timestamp), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    catch error_value
        error_value isa InterruptException && rethrow()
        error("invalid provenance UTC timestamp")
    end
    inputs = _manifest_require(record, "inputs")
    input_names = String[String(_manifest_require(input, "name")) for input in inputs]
    length(unique(input_names)) == length(input_names) ||
        error("provenance manifest contains duplicate input names")
    for input in inputs
        _manifest_hash_matches(input, "input $(input["name"])")
        if run_mode == "canonical" && occursin("omni", lowercase(String(input["name"])))
            String(input["sha256"]) == CANONICAL_OMNI_SHA256 ||
                error("canonical manifest contains an unpinned OMNI input")
        end
    end
    if verify_source
        source = _manifest_require(record, "source_identity")
        current = provenance_source_identity(package_root)
        String(source["git_baseline"]) == current.git_baseline ||
            error("git baseline differs from the producing run")
        String(source["source_tree_sha256"]) == current.source_tree_sha256 ||
            error("source-tree SHA-256 differs from the producing run")
        String(source["julia_version"]) == current.julia_version ||
            error("Julia version differs from the producing run")
        String(source["project"]["sha256"]) == current.project.sha256 ||
            error("Project.toml SHA-256 differs from the producing run")
        String(source["manifest"]["sha256"]) == current.manifest.sha256 ||
            error("Manifest.toml SHA-256 differs from the producing run")
    end
    return record
end

verified_output_path(path::AbstractString; kwargs...) =
    (verify_output_manifest(path; kwargs...); abspath(path))

"""Write, manifest, and immediately verify one CSV while restoring any prior pair on failure."""
function write_manifested_csv(output_path::AbstractString, data;
                              producer_script::AbstractString,
                              input_paths,
                              selection_record,
                              seed::Union{Nothing,Integer}=nothing,
                              deterministic::Bool=seed === nothing,
                              metadata=(;),
                              package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                              mode=validation_run_mode(),
                              verify_source::Bool=true,
                              manifest_path::AbstractString=String(output_path) * ".manifest.json")
    run_mode = _provenance_mode(mode)
    abspath(output_path) != abspath(manifest_path) || throw(ArgumentError(
        "CSV output and provenance manifest paths must differ",
    ))
    frame = data isa DataFrame ? data : DataFrame(data)
    _provenance_require_regular_target(output_path)
    _provenance_require_regular_target(manifest_path)
    parent = dirname(output_path)
    mkpath(parent)
    staging = mktempdir(parent)
    staged_output = joinpath(staging, "output.new.csv")
    output_backup = joinpath(staging, "output.backup")
    manifest_backup = joinpath(staging, "manifest.backup")
    had_output = isfile(output_path)
    had_manifest = isfile(manifest_path)
    had_output && cp(output_path, output_backup)
    had_manifest && cp(manifest_path, manifest_backup)
    try
        CSV.write(staged_output, frame)
        _provenance_atomic_replace(staged_output, output_path)
        write_output_manifest(output_path;
            producer_script,
            input_paths,
            selection_record,
            seed,
            deterministic,
            metadata=merge((
                rows=nrow(frame), columns=Tuple(String.(names(frame))),
            ), metadata),
            package_root,
            mode=run_mode,
            manifest_path,
        )
        verify_output_manifest(output_path;
            manifest_path,
            package_root,
            require_canonical=run_mode == :canonical,
            verify_source,
        )
    catch
        if had_output
            _provenance_atomic_replace(output_backup, output_path)
        elseif isfile(output_path) && !islink(output_path)
            rm(output_path; force=true)
        end
        if had_manifest
            _provenance_atomic_replace(manifest_backup, manifest_path)
        elseif isfile(manifest_path) && !islink(manifest_path)
            rm(manifest_path; force=true)
        end
        rethrow()
    finally
        rm(staging; recursive=true, force=true)
    end
    return String(output_path)
end

"""
Render one PlotlySupply.jl or Diff3D.jl figure into a staged file, then install,
manifest, and immediately verify the figure and every manifested CSV input.
"""
function write_manifested_figure(output_path::AbstractString, render!::Function;
                                 producer_script::AbstractString,
                                 input_paths,
                                 selection_record,
                                 backend::AbstractString,
                                 seed::Union{Nothing,Integer}=nothing,
                                 deterministic::Bool=seed === nothing,
                                 metadata=(;),
                                 package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                 mode=validation_run_mode(),
                                 verify_source::Bool=true,
                                 manifest_path::AbstractString=String(output_path) * ".manifest.json")
    run_mode = _provenance_mode(mode)
    backend in ("PlotlySupply.jl", "Diff3D.jl") || throw(ArgumentError(
        "canonical figures require PlotlySupply.jl or Diff3D.jl",
    ))
    lowercase(splitext(output_path)[2]) in (".pdf", ".png", ".svg") ||
        throw(ArgumentError("figure output must be PDF, PNG, or SVG"))
    pairs_list = collect(pairs(input_paths))
    isempty(pairs_list) && throw(ArgumentError(
        "a data-driven figure requires at least one manifested CSV input",
    ))
    for pair in pairs_list
        input = String(pair.second)
        lowercase(splitext(input)[2]) == ".csv" || throw(ArgumentError(
            "figure input must be a manifested CSV: $input",
        ))
        verify_output_manifest(input;
            package_root,
            require_canonical=run_mode == :canonical,
            verify_source,
        )
    end

    abspath(output_path) != abspath(manifest_path) || throw(ArgumentError(
        "figure output and provenance manifest paths must differ",
    ))
    _provenance_require_regular_target(output_path)
    _provenance_require_regular_target(manifest_path)
    parent = dirname(output_path)
    mkpath(parent)
    staging = mktempdir(parent)
    staged_output = joinpath(staging, "figure.new" * splitext(output_path)[2])
    output_backup = joinpath(staging, "figure.backup")
    manifest_backup = joinpath(staging, "manifest.backup")
    had_output = isfile(output_path)
    had_manifest = isfile(manifest_path)
    had_output && cp(output_path, output_backup)
    had_manifest && cp(manifest_path, manifest_backup)
    try
        render!(staged_output)
        isfile(staged_output) && !islink(staged_output) && filesize(staged_output) > 0 ||
            error("figure renderer did not create a nonempty regular file")
        _provenance_atomic_replace(staged_output, output_path)
        write_output_manifest(output_path;
            producer_script,
            input_paths,
            selection_record,
            seed,
            deterministic,
            metadata=merge((
                artifact_kind="data_driven_figure",
                figure_backend=backend,
                input_count=length(pairs_list),
            ), metadata),
            package_root,
            mode=run_mode,
            manifest_path,
        )
        verify_output_manifest(output_path;
            manifest_path,
            package_root,
            require_canonical=run_mode == :canonical,
            verify_source,
        )
    catch
        if had_output
            _provenance_atomic_replace(output_backup, output_path)
        elseif isfile(output_path) && !islink(output_path)
            rm(output_path; force=true)
        end
        if had_manifest
            _provenance_atomic_replace(manifest_backup, manifest_path)
        elseif isfile(manifest_path) && !islink(manifest_path)
            rm(manifest_path; force=true)
        end
        rethrow()
    finally
        rm(staging; recursive=true, force=true)
    end
    return String(output_path)
end

"""Verify every top-level artifact and reject missing or orphan manifests."""
function verify_artifact_closure(directory::AbstractString;
                                 extensions=(".csv",),
                                 expected_artifacts::AbstractDict=Dict{String,String}(),
                                 package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                 require_canonical::Bool=false,
                                 verify_source::Bool=true)
    isdir(directory) || throw(ArgumentError("artifact directory not found: $directory"))
    normalized_extensions = Set(lowercase.(String.(extensions)))
    isempty(normalized_extensions) &&
        throw(ArgumentError("artifact extensions must not be empty"))
    files = sort(readdir(directory; join=true))
    artifacts = filter(files) do path
        isfile(path) && !endswith(path, ".manifest.json") &&
            lowercase(splitext(path)[2]) in normalized_extensions
    end
    isempty(artifacts) && error("artifact closure found no matching outputs in $directory")
    all(!islink, artifacts) || error("artifact closure rejects symbolic-link outputs")
    records = Dict{String,Any}()
    for artifact in artifacts
        record = verify_output_manifest(artifact;
            package_root,
            require_canonical,
            verify_source,
        )
        records[basename(artifact)] = record
    end
    for manifest in filter(path -> isfile(path) && endswith(path, ".manifest.json"),
                           files)
        artifact = chop(manifest; tail=length(".manifest.json"))
        lowercase(splitext(artifact)[2]) in normalized_extensions || continue
        isfile(artifact) || error("orphan artifact manifest: $manifest")
    end
    for (basename_value, expected_producer) in expected_artifacts
        name = String(basename_value)
        basename(name) == name || throw(ArgumentError(
            "expected artifact keys must be basenames: $name",
        ))
        lowercase(splitext(name)[2]) in normalized_extensions ||
            throw(ArgumentError("expected artifact extension was not requested: $name"))
        haskey(records, name) || error("missing required canonical artifact: $name")
        record = records[name]
        actual_producer = String(_manifest_require(
            _manifest_require(record, "producer"), "path",
        ))
        actual_producer == abspath(String(expected_producer)) || error(
            "canonical artifact $name has producer $actual_producer, expected " *
            abspath(String(expected_producer)),
        )
        if lowercase(splitext(name)[2]) != ".csv"
            metadata = _manifest_require(record, "metadata")
            String(_manifest_require(metadata, "artifact_kind")) ==
                "data_driven_figure" || error(
                    "canonical figure $name lacks data-driven figure provenance",
                )
            String(_manifest_require(metadata, "figure_backend")) in
                ("PlotlySupply.jl", "Diff3D.jl") || error(
                    "canonical figure $name has an unauthorized backend",
                )
        end
    end
    if !isempty(expected_artifacts)
        unexpected = sort!(collect(setdiff(
            Set(keys(records)), Set(String.(keys(expected_artifacts))),
        )))
        isempty(unexpected) || error(
            "unexpected top-level artifacts must be archived before closure: " *
            join(unexpected, ", "),
        )
    end
    return (; directory=abspath(directory), count=length(artifacts),
            artifacts=abspath.(artifacts), records)
end

function storm_catalog_parameters(; year_start::Int=1963, year_end::Int=2025,
                                  dst_thresh::Real=-50.0,
                                  window_pre::Int=24, window_post::Int=144,
                                  min_separation::Int=48)
    year_start <= year_end || throw(ArgumentError("year_start must not exceed year_end"))
    isfinite(dst_thresh) || throw(ArgumentError("dst_thresh must be finite"))
    all(>=(0), (window_pre, window_post, min_separation)) ||
        throw(ArgumentError("catalog window parameters must be nonnegative"))
    return (
        schema_version = STORM_CATALOG_SCHEMA_VERSION,
        year_start = year_start,
        year_end = year_end,
        dst_thresh = Float64(dst_thresh),
        window_pre = window_pre,
        window_post = window_post,
        min_separation = min_separation,
        minimum_sindy_usable_fraction = 0.60,
        onset_reference_nt = -20.0,
        recovery_threshold_nt = -20.0,
        recovery_buffer_hr = 24,
        cycle_boundaries = (
            "20:1964-10", "21:1976-03", "22:1986-09",
            "23:1996-08", "24:2008-12", "25:2019-12",
        ),
        split_policy = "SC20-SC23=train;SC24=val;SC25+=test;pre-SC20=exclude",
    )
end

function _validated_catalog_parameters(parameters::NamedTuple, mode)
    required = storm_catalog_parameters(
        year_start=parameters.year_start,
        year_end=parameters.year_end,
        dst_thresh=parameters.dst_thresh,
        window_pre=parameters.window_pre,
        window_post=parameters.window_post,
        min_separation=parameters.min_separation,
    )
    parameters == required || error(
        "storm-catalog policy fields do not match the implemented deterministic policy",
    )
    _provenance_mode(mode) == :canonical &&
        parameters != storm_catalog_parameters() && error(
            "canonical storm-catalog parameters differ from the frozen AISR protocol",
        )
    return parameters
end

function _catalog_frame_summary(frame::DataFrame)
    Tuple(names(frame)) == STORM_CATALOG_COLUMNS || error(
        "storm-catalog schema mismatch: expected $(collect(STORM_CATALOG_COLUMNS)), " *
        "got $(names(frame))",
    )
    nrow(frame) > 0 || error("storm catalog is empty")
    ids = Int.(frame.storm_id)
    ids == collect(1:nrow(frame)) || error("storm IDs must be contiguous from one")
    allowed_splits = Set(("exclude", "train", "val", "test"))
    all(split -> String(split) in allowed_splits, frame.split) ||
        error("storm catalog contains an invalid split")
    onsets = DateTime[value isa DateTime ? value : DateTime(value)
                      for value in frame.onset_time]
    all(index -> onsets[index] > onsets[index - 1], 2:length(onsets)) ||
        error("storm catalog must be strictly chronological")
    for row in eachrow(frame)
        onset = row.onset_time isa DateTime ? row.onset_time : DateTime(row.onset_time)
        minimum_time = row.min_dst_star_time isa DateTime ? row.min_dst_star_time :
            DateTime(row.min_dst_star_time)
        recovery_time = row.recovery_end_time isa DateTime ? row.recovery_end_time :
            DateTime(row.recovery_end_time)
        isfinite(Float64(row.min_dst_star)) ||
            error("storm $(row.storm_id) has a nonfinite Dst* minimum")
        isfinite(Float64(row.duration_hr)) && Float64(row.duration_hr) >= 0 ||
            error("storm $(row.storm_id) has an invalid duration")
        onset <= minimum_time <= recovery_time ||
            error("storm $(row.storm_id) has inconsistent event times")
        cycle = SolarSINDy._solar_cycle(onset)
        Int(row.solar_cycle) == cycle ||
            error("storm $(row.storm_id) has a stale solar-cycle assignment")
        String(row.split) == SolarSINDy._assign_split(cycle) ||
            error("storm $(row.storm_id) has a stale split assignment")
        Int(row.onset_idx) >= 1 && Int(row.end_idx) >= Int(row.onset_idx) ||
            error("storm $(row.storm_id) has invalid row bounds")
        Float64(row.duration_hr) == Int(row.end_idx) - Int(row.onset_idx) ||
            error("storm $(row.storm_id) duration does not match row bounds")
    end
    split_counts = [(split=split,
                     count=count(==(split), String.(frame.split)))
                    for split in ("exclude", "train", "val", "test")]
    cycles = sort(unique(Int.(frame.solar_cycle)))
    cycle_counts = [(solar_cycle=cycle,
                     count=count(==(cycle), Int.(frame.solar_cycle)))
                    for cycle in cycles]
    schema = [(name=name, logical_type=type) for (name, type) in zip(
        STORM_CATALOG_COLUMNS,
        ("Int", "DateTime", "Float64", "DateTime", "DateTime", "Float64",
         "Int", "String", "Int", "Int"),
    )]
    return (; schema, n_storms=nrow(frame), split_counts, cycle_counts)
end

function _provenance_atomic_catalog(catalog, path::AbstractString)
    parent = dirname(path)
    mkpath(parent)
    _provenance_require_regular_target(path)
    temp_path, io = mktemp(parent; cleanup=false)
    close(io)
    try
        frame = DataFrame(
            storm_id = [entry.storm_id for entry in catalog],
            onset_time = [entry.onset_time for entry in catalog],
            min_dst_star = [entry.min_dst_star for entry in catalog],
            min_dst_star_time = [entry.min_dst_star_time for entry in catalog],
            recovery_end_time = [entry.recovery_end_time for entry in catalog],
            duration_hr = [entry.duration_hr for entry in catalog],
            solar_cycle = [entry.solar_cycle for entry in catalog],
            split = [entry.split for entry in catalog],
            onset_idx = [entry.onset_idx for entry in catalog],
            end_idx = [entry.end_idx for entry in catalog],
        )
        _catalog_frame_summary(frame)
        CSV.write(temp_path, frame)
        _provenance_atomic_replace(temp_path, path)
    finally
        isfile(temp_path) && rm(temp_path; force=true)
    end
    return path
end

"""Atomically save a catalog and its standard provenance manifest."""
function write_verified_storm_catalog(catalog, catalog_path::AbstractString;
                                      omni_path::AbstractString,
                                      producer_script::AbstractString,
                                      parameters::NamedTuple=storm_catalog_parameters(),
                                      package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                      mode=validation_run_mode())
    _validated_catalog_parameters(parameters, mode)
    verify_omni_input(omni_path; mode=mode)
    manifest_path = String(catalog_path) * ".manifest.json"
    _provenance_require_regular_target(catalog_path)
    _provenance_require_regular_target(manifest_path)
    parent = dirname(catalog_path)
    mkpath(parent)
    staging = mktempdir(parent)
    catalog_backup = joinpath(staging, "catalog.backup")
    manifest_backup = joinpath(staging, "manifest.backup")
    staged_catalog = joinpath(staging, "catalog.new")
    had_catalog = isfile(catalog_path)
    had_manifest = isfile(manifest_path)
    had_catalog && cp(catalog_path, catalog_backup)
    had_manifest && cp(manifest_path, manifest_backup)
    try
        _provenance_atomic_catalog(catalog, staged_catalog)
        _provenance_atomic_replace(staged_catalog, catalog_path)
        frame = CSV.read(catalog_path, DataFrame)
        summary = _catalog_frame_summary(frame)
        catalog_metadata = merge(summary, (
            parameters = parameters,
            catalog_sha256 = provenance_sha256(catalog_path),
        ))
        write_output_manifest(catalog_path;
            producer_script,
            input_paths=(omni_extracted=omni_path,),
            selection_record=(kind="deterministic_storm_catalog", parameters=parameters),
            deterministic=true,
            metadata=(catalog=catalog_metadata,),
            package_root,
            mode,
        )
        verify_output_manifest(catalog_path;
            package_root,
            require_canonical=_provenance_mode(mode) == :canonical,
            verify_source=true,
        )
    catch
        if had_catalog
            _provenance_atomic_replace(catalog_backup, catalog_path)
        elseif isfile(catalog_path) && !islink(catalog_path)
            rm(catalog_path; force=true)
        end
        if had_manifest
            _provenance_atomic_replace(manifest_backup, manifest_path)
        elseif isfile(manifest_path) && !islink(manifest_path)
            rm(manifest_path; force=true)
        end
        rethrow()
    finally
        rm(staging; recursive=true, force=true)
    end
    return catalog_path
end


function _json_semantically_equal(left, right)
    if left isa AbstractDict || left isa JSON3.Object || left isa NamedTuple
        (right isa AbstractDict || right isa JSON3.Object || right isa NamedTuple) ||
            return false
        left_keys = sort!(String[string(key) for key in keys(left)])
        right_keys = sort!(String[string(key) for key in keys(right)])
        left_keys == right_keys || return false
        return all(key -> _json_semantically_equal(left[Symbol(key)], right[Symbol(key)]),
                   left_keys)
    elseif left isa AbstractVector || left isa Tuple || left isa JSON3.Array
        (right isa AbstractVector || right isa Tuple || right isa JSON3.Array) ||
            return false
        length(left) == length(right) || return false
        return all(_json_semantically_equal(a, b) for (a, b) in zip(left, right))
    elseif left isa Real && right isa Real
        return left == right
    end
    return left == right
end

function _json_get(value, key::AbstractString)
    haskey(value, key) && return value[key]
    symbol = Symbol(key)
    haskey(value, symbol) && return value[symbol]
    error("missing JSON key $key")
end

"""Fail closed unless the catalog, frozen input, parameters, and counts match."""
function verify_storm_catalog(catalog_path::AbstractString;
                              omni_path::AbstractString,
                              parameters::NamedTuple=storm_catalog_parameters(),
                              package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                              mode=validation_run_mode(),
                              verify_source::Bool=true)
    run_mode = _provenance_mode(mode)
    _validated_catalog_parameters(parameters, run_mode)
    verify_omni_input(omni_path; mode=run_mode)
    record = verify_output_manifest(catalog_path;
        package_root,
        require_canonical=run_mode == :canonical,
        verify_source,
    )
    catalog_metadata = _json_get(_json_get(record, "metadata"), "catalog")
    String(_json_get(catalog_metadata, "catalog_sha256")) ==
        provenance_sha256(catalog_path) || error("catalog metadata hash mismatch")
    _json_semantically_equal(_json_get(catalog_metadata, "parameters"), parameters) ||
        error("storm-catalog parameters differ from the requested canonical parameters")
    frame = CSV.read(catalog_path, DataFrame)
    summary = _catalog_frame_summary(frame)
    Int(_json_get(catalog_metadata, "n_storms")) == summary.n_storms ||
        error("storm-catalog count mismatch")
    _json_semantically_equal(_json_get(catalog_metadata, "schema"), summary.schema) ||
        error("storm-catalog schema metadata mismatch")
    _json_semantically_equal(_json_get(catalog_metadata, "split_counts"),
                             summary.split_counts) ||
        error("storm-catalog split counts mismatch")
    _json_semantically_equal(_json_get(catalog_metadata, "cycle_counts"),
                             summary.cycle_counts) ||
        error("storm-catalog cycle counts mismatch")
    input_record = only(filter(input -> String(input["name"]) == "omni_extracted",
                               record["inputs"]))
    abspath(omni_path) == String(input_record["path"]) ||
        error("catalog was generated from a different OMNI path")
    return record
end

function load_verified_storm_catalog(catalog_path::AbstractString; kwargs...)
    record = verify_storm_catalog(catalog_path; kwargs...)
    frame = CSV.read(catalog_path, DataFrame)
    catalog = [StormCatalogEntry(
        Int(row.storm_id),
        row.onset_time isa DateTime ? row.onset_time : DateTime(row.onset_time),
        Float64(row.min_dst_star),
        row.min_dst_star_time isa DateTime ? row.min_dst_star_time :
            DateTime(row.min_dst_star_time),
        row.recovery_end_time isa DateTime ? row.recovery_end_time :
            DateTime(row.recovery_end_time),
        Float64(row.duration_hr),
        Int(row.solar_cycle),
        String(row.split),
        Int(row.onset_idx),
        Int(row.end_idx),
    ) for row in eachrow(frame)]
    length(catalog) == Int(record["metadata"]["catalog"]["n_storms"]) ||
        error("loaded storm-catalog count mismatch")
    return catalog
end

"""Load legacy catalogs only outside canonical mode; manifested catalogs verify first."""
function load_validation_storm_catalog(catalog_path::AbstractString;
                                       omni_path::AbstractString,
                                       parameters::NamedTuple=storm_catalog_parameters(),
                                       package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                       mode=validation_run_mode(),
                                       verify_source::Bool=true)
    run_mode = _provenance_mode(mode)
    if isfile(String(catalog_path) * ".manifest.json")
        return load_verified_storm_catalog(catalog_path;
            omni_path, parameters, package_root, mode=run_mode, verify_source)
    end
    run_mode == :canonical && error(
        "canonical storm catalog manifest not found: $(catalog_path).manifest.json",
    )
    return load_storm_catalog(String(catalog_path))
end

"""Regenerate the catalog directly from one immutable extracted OMNI file."""
function regenerate_verified_storm_catalog(omni_path::AbstractString,
                                           catalog_path::AbstractString;
                                           producer_script::AbstractString,
                                           parameters::NamedTuple=storm_catalog_parameters(),
                                           package_root::AbstractString=normpath(joinpath(@__DIR__, "..")),
                                           mode=validation_run_mode())
    _validated_catalog_parameters(parameters, mode)
    verify_omni_input(omni_path; mode=mode)
    df = parse_omni2(String(omni_path);
                     year_start=parameters.year_start,
                     year_end=parameters.year_end)
    add_original_observation_flags!(df)
    clean_omni_data!(df)
    catalog = build_storm_catalog(df;
        dst_thresh=parameters.dst_thresh,
        window_pre=parameters.window_pre,
        window_post=parameters.window_post,
        min_separation=parameters.min_separation,
    )
    isempty(catalog) && error("deterministic catalog regeneration found no storms")
    write_verified_storm_catalog(catalog, catalog_path;
        omni_path, producer_script, parameters, package_root, mode)
    return catalog
end
