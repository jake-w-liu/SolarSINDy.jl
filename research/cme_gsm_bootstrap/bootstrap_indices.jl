module CMEGSMBootstrap

using Random, SHA, TOML, Dates

const INPUT_SHA = "14c7d83be58729868e8c86f949cbdc541f54f68f7891e07c3c569da91f9ff760"
const POPULATION_SHA = "b81228b4b0954dfec881d5a57ae9f7a76bb7e9a73aeee84bf17a0dcfdaf99fb9"
const PROVIDER_SHA = "4ab51b795b549e7a26c0a8f17206abcad0c493ddd7255f62549335d0300d0d24"
const PROTOCOL_SHA = "df44b33c835f1df7479d11e672676054db9c0925a829fee7c2059e481cd58276"
const FORMAT = "utf8-ascii-decimal-comma-noheader-LF-finalnewline-rowdraw-columnslot-index1-v1"
const PRIMARY_RUNTIME = v"1.12.6"
const CONTROL_RUNTIMES = (v"1.12.6", v"1.13.1")
const SEED = 170281
const DRAWS = 10000
const SOURCE_SHA = bytes2hex(sha256(read(@__FILE__)))
const DEFAULT_INPUT = joinpath(@__DIR__, "frozen_population_years.toml")

need(value, message) = value || throw(ArgumentError(message))
digest(bytes) = bytes2hex(sha256(bytes))
hex40(value) = value isa AbstractString && occursin(r"^[0-9a-f]{40}$", value)
hex64(value) = value isa AbstractString && occursin(r"^[0-9a-f]{64}$", value)

function source_guard()
    need(digest(read(@__FILE__)) == SOURCE_SHA, "generator loaded/disk source mismatch")
end

function bounded_bytes(path, maximum)
    need(isfile(path) && !islink(path), "ordinary bounded input required")
    before = stat(path)
    need(0 <= before.size <= maximum, "input byte bound")
    bytes = open(path, "r") do io
        read(io, maximum + 1)
    end
    after = stat(path)
    need(length(bytes) == before.size == after.size && length(bytes) <= maximum, "input grew or changed size")
    need(before.mtime == after.mtime && before.ctime == after.ctime, "input changed while reading")
    bytes
end

function population_values(data)
    need(Set(keys(data)) == Set(("schema", "frozen_population_sha256", "provider_sha256", "frozen_protocol_sha256",
        "event_count", "event_order", "year_universe", "event")), "population metadata keys")
    need(data["schema"] == "cme_gsm_frozen_population_years_v1", "population schema")
    need(data["frozen_population_sha256"] == POPULATION_SHA && data["provider_sha256"] == PROVIDER_SHA &&
        data["frozen_protocol_sha256"] == PROTOCOL_SHA, "frozen population provenance")
    need(data["event_count"] === 551 && data["event_order"] == "original_frozen_population_order", "full population required")
    rows = data["event"]
    need(rows isa AbstractVector && length(rows) == 551, "551 event lineage rows required")
    ids = String[]; years = Int[]
    for (index, row) in enumerate(rows)
        need(Set(keys(row)) == Set(("index", "id", "mo_start_utc", "utc_year")), "event metadata keys")
        need(row["index"] === index, "original event order changed")
        id = row["id"]; stamp = row["mo_start_utc"]; y = row["utc_year"]
        need(id isa String && occursin(r"^[A-Za-z0-9_.-]{1,128}$", id), "event ID")
        need(stamp isa String && endswith(stamp, "+00:00"), "strict UTC MO-start metadata")
        parsed = DateTime(chop(stamp; tail=6))
        need(y isa Integer && !(y isa Bool) && y == year(parsed), "MO-start calendar year differs")
        push!(ids, id); push!(years, Int(y))
    end
    need(allunique(ids), "duplicate event ID")
    universe = sort!(unique(years))
    need(data["year_universe"] == universe, "year universe must use all frozen starts in sorted order")
    (; ids=Tuple(ids), event_years=Tuple(years), years=Tuple(universe), K=length(universe))
end

function load_population(path=DEFAULT_INPUT)
    bytes = bounded_bytes(path, 262144)
    need(digest(bytes) == INPUT_SHA, "frozen population metadata byte hash differs")
    value = population_values(TOML.parse(String(copy(bytes))))
    (; value..., bytes, sha256=INPUT_SHA)
end

function exclusive_bytes(path, bytes)
    file = Base.Filesystem.open(path, Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT |
        Base.Filesystem.JL_O_EXCL, 0o600)
    try
        need(write(file, bytes) == length(bytes), "short exclusive write")
    finally
        close(file)
    end
    need(bounded_bytes(path, length(bytes)) == bytes, "exclusive write readback differs")
    path
end

function exclusive_toml(path, value)
    io = IOBuffer(); TOML.print(io, value; sorted=true)
    bytes = take!(io)
    isempty(bytes) || last(bytes) == 0x0a || push!(bytes, 0x0a)
    exclusive_bytes(path, bytes)
end

"Write one scalar choice per slot; the independent fake-draw test fixes loop/byte order."
function write_index_csv(io::IO, K::Integer, draws::Integer, pick)
    need(!(K isa Bool) && !(draws isa Bool) && 1 <= K <= 551 && 1 <= draws <= DRAWS, "matrix geometry bound")
    for draw in 1:draws
        for slot in 1:K
            value = pick(1:K)
            need(value isa Integer && !(value isa Bool) && 1 <= value <= K, "index outside frozen block domain")
            write(io, string(value))
            write(io, slot == K ? 0x0a : 0x2c)
        end
    end
    nothing
end

function csv_values(path, K::Integer, draws::Integer; expected_sha256::AbstractString)
    need(1 <= K <= 551 && 1 <= draws <= DRAWS && hex64(expected_sha256), "validator geometry/hash")
    bytes = bounded_bytes(path, draws * K * (ndigits(K) + 1))
    need(digest(bytes) == expected_sha256, "matrix byte hash differs")
    need(!isempty(bytes) && last(bytes) == 0x0a && all(b -> b in 0x30:0x39 || b == 0x2c || b == 0x0a, bytes), "canonical CSV bytes required")
    lines = split(String(copy(bytes)), '\n'; keepempty=true)
    need(length(lines) == draws + 1 && isempty(last(lines)), "matrix row geometry differs")
    values = Matrix{Int}(undef, draws, K)
    for draw in 1:draws
        fields = split(lines[draw], ','; keepempty=true)
        need(length(fields) == K, "matrix column geometry differs")
        for slot in 1:K
            need(occursin(r"^[1-9][0-9]*$", fields[slot]), "canonical decimal index required")
            value = parse(Int, fields[slot])
            need(1 <= value <= K, "index outside frozen block domain")
            values[draw, slot] = value
        end
    end
    values
end

function validate_artifact(directory; expected_source_head, expected_matrix_sha256, population=load_population())
    source_guard()
    need(hex40(expected_source_head) && hex64(expected_matrix_sha256), "caller-pinned artifact identities required")
    need(isdir(directory) && !islink(directory), "ordinary artifact directory required")
    need(sort(readdir(directory)) == ["BOOTSTRAP_INDICES.csv", "COMPLETED.toml", "INDEX_MANIFEST.toml", "POPULATION_YEARS.toml"], "artifact membership differs")
    need(bounded_bytes(joinpath(directory, "POPULATION_YEARS.toml"), 262144) == population.bytes, "artifact population lineage differs")
    manifest_bytes = bounded_bytes(joinpath(directory, "INDEX_MANIFEST.toml"), 65536)
    manifest = TOML.parse(String(copy(manifest_bytes)))
    marker = TOML.parse(String(bounded_bytes(joinpath(directory, "COMPLETED.toml"), 4096)))
    need(marker == Dict("schema" => "cme_gsm_bootstrap_completion_v1", "manifest_sha256" => digest(manifest_bytes),
        "matrix_sha256" => expected_matrix_sha256), "completion bindings differ")
    expected_keys = Set(("schema", "role", "runtime", "generator", "seed", "draws", "K", "year_universe", "event_count",
        "format", "matrix_sha256", "input_metadata_sha256", "population_sha256", "provider_sha256", "protocol_sha256",
        "claimed_source_head", "generator_source_sha256", "actual_GSM_values", "source_authority_granted", "scientific_qualification"))
    need(Set(keys(manifest)) == expected_keys, "manifest keys differ")
    need(manifest["schema"] == "cme_gsm_bootstrap_index_manifest_v1" && manifest["role"] in ("primary", "control"), "manifest schema/role")
    need(manifest["runtime"] in string.(CONTROL_RUNTIMES), "unsupported measured runtime")
    need(manifest["role"] != "primary" || manifest["runtime"] == string(PRIMARY_RUNTIME), "primary runtime must be Julia1.12.6")
    need(manifest["generator"] == "Random.MersenneTwister" && manifest["seed"] === SEED && manifest["draws"] === DRAWS, "RNG protocol differs")
    need(manifest["K"] == population.K && manifest["year_universe"] == collect(population.years) && manifest["event_count"] === 551, "frozen year universe/geometry differs")
    need(manifest["format"] == FORMAT && manifest["matrix_sha256"] == expected_matrix_sha256, "matrix format/hash differs")
    need(manifest["input_metadata_sha256"] == INPUT_SHA && manifest["population_sha256"] == POPULATION_SHA &&
        manifest["provider_sha256"] == PROVIDER_SHA && manifest["protocol_sha256"] == PROTOCOL_SHA, "artifact provenance differs")
    need(manifest["claimed_source_head"] == expected_source_head && manifest["generator_source_sha256"] == SOURCE_SHA, "source identity differs")
    need(manifest["actual_GSM_values"] === 0 && manifest["source_authority_granted"] === false &&
        manifest["scientific_qualification"] === false, "bootstrap is not source/scientific qualification")
    values = csv_values(joinpath(directory, "BOOTSTRAP_INDICES.csv"), population.K, DRAWS; expected_sha256=expected_matrix_sha256)
    source_guard()
    (; manifest, values, matrix_sha256=expected_matrix_sha256, manifest_sha256=digest(manifest_bytes))
end

function generate(directory; role::String, source_head::String, expected_runtime::VersionNumber, input=DEFAULT_INPUT)
    source_guard(); need(hex40(source_head), "claimed source head must be40hex")
    need(VERSION == expected_runtime && VERSION in CONTROL_RUNTIMES, "native runtime identity differs")
    need(role in ("primary", "control") && (role != "primary" || VERSION == PRIMARY_RUNTIME), "primary/control runtime contract")
    population = load_population(input)
    need(isabspath(directory) && isdir(dirname(directory)), "fresh absolute output under existing parent required")
    target = joinpath(realpath(dirname(directory)), basename(directory))
    need(!ispath(target) && !islink(target), "output already exists")
    mkdir(target)
    rng = MersenneTwister(SEED)
    buffer = IOBuffer()
    write_index_csv(buffer, population.K, DRAWS, range -> rand(rng, range))
    bytes = take!(buffer); matrix_sha = digest(bytes)
    exclusive_bytes(joinpath(target, "BOOTSTRAP_INDICES.csv"), bytes)
    exclusive_bytes(joinpath(target, "POPULATION_YEARS.toml"), population.bytes)
    manifest = Dict{String,Any}("schema" => "cme_gsm_bootstrap_index_manifest_v1", "role" => role, "runtime" => string(VERSION),
        "generator" => "Random.MersenneTwister", "seed" => SEED, "draws" => DRAWS, "K" => population.K,
        "year_universe" => collect(population.years), "event_count" => 551, "format" => FORMAT,
        "matrix_sha256" => matrix_sha, "input_metadata_sha256" => INPUT_SHA, "population_sha256" => POPULATION_SHA,
        "provider_sha256" => PROVIDER_SHA, "protocol_sha256" => PROTOCOL_SHA, "claimed_source_head" => source_head,
        "generator_source_sha256" => SOURCE_SHA, "actual_GSM_values" => 0, "source_authority_granted" => false, "scientific_qualification" => false)
    exclusive_toml(joinpath(target, "INDEX_MANIFEST.toml"), manifest)
    manifest_sha = digest(bounded_bytes(joinpath(target, "INDEX_MANIFEST.toml"), 65536))
    exclusive_toml(joinpath(target, "COMPLETED.toml"), Dict("schema" => "cme_gsm_bootstrap_completion_v1",
        "manifest_sha256" => manifest_sha, "matrix_sha256" => matrix_sha))
    validate_artifact(target; expected_source_head=source_head, expected_matrix_sha256=matrix_sha, population)
    source_guard()
    (; directory=target, matrix_sha256=matrix_sha, manifest_sha256=manifest_sha, K=population.K, draws=DRAWS)
end

end
