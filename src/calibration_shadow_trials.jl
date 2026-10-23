# Shared by issuance, auditing and the standalone dashboard environment.
module CalibrationShadowTrials
using Dates, SHA, TOML

struct ShadowTrialSpec
    trial_id::String
    identity::String
    config_sha256::String
    served_manifest_sha256::String
    registered_utc::Union{Nothing,DateTime}
    start_utc::DateTime
    legacy::Bool
    seal_sha256::Union{Nothing,String}
end

const LEGACY_A3 = ShadowTrialSpec(
    "legacy-a3", "superlearner_e-cal-shadow-A3+median24+width1h1.50+widthOther1.20+warm30",
    "31d67e5077ae6fe69cee133fa07dceec3aa639903a17861d179d3877ed0c21af",
    "057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede",
    nothing, DateTime(2026,8,24,23,15,51,419), true, nothing)

digest(path::AbstractString) = bytes2hex(open(SHA.sha256, path))
valid_digest(value) = value isa AbstractString && occursin(r"^[0-9a-f]{64}$",value)
function regular_file(path)
    isfile(path) && !islink(path) || error("trial evidence is missing or not a regular file: $path")
    path
end
function utc_time(value)
    value isa AbstractString && endswith(value,"Z") || throw(ArgumentError("trial time must be a UTC string ending Z"))
    DateTime(value[1:end-1])
end

"Load an explicitly pinned, immutable prospective spec; no live activation occurs."
function load_trial(path::AbstractString, expected_sha256::AbstractString; seal_sha256=nothing)
    seal_sha256===nothing || valid_digest(seal_sha256) || throw(ArgumentError("invalid expected legacy seal digest"))
    valid_digest(expected_sha256) || throw(ArgumentError("expected trial digest must be lowercase SHA256"))
    regular_file(path); filesize(path)<=65_536 || error("trial config exceeds size limit")
    bytes=read(path)
    actual=bytes2hex(sha256(bytes))
    actual==expected_sha256 || error("trial configuration digest mismatch")
    config=TOML.parse(String(bytes))
    required=Set(["schema","trial_id","shadow_identity","registered_utc","start_utc",
        "served_manifest_sha256","algorithm","history_policy"])
    Set(keys(config))==required || throw(ArgumentError("trial config has missing or unknown keys"))
    config["schema"] isa Integer && !(config["schema"] isa Bool) && config["schema"]==1 ||
        throw(ArgumentError("unsupported trial config schema"))
    all(config[key] isa AbstractString for key in setdiff(required,Set(["schema"]))) ||
        throw(ArgumentError("trial fields must be strings"))
    id=String(config["trial_id"])
    occursin(r"^[a-z0-9][a-z0-9-]{0,79}$",id) && id!="legacy-a3" ||
        throw(ArgumentError("invalid or reserved trial id"))
    identity="superlearner-cal-shadow-"*id*"+median24+width1h1.50+widthOther1.20+warm30"
    config["shadow_identity"]==identity || throw(ArgumentError("trial identity does not match trial id"))
    config["algorithm"]=="median24-width1h1.50-widthOther1.20-warm30" ||
        throw(ArgumentError("unsupported shadow algorithm"))
    config["history_policy"]=="cold-exact-trial-issued-since-start" ||
        throw(ArgumentError("unsupported shadow history policy"))
    manifest=String(config["served_manifest_sha256"])
    valid_digest(manifest) || throw(ArgumentError("invalid served-manifest digest"))
    registered=utc_time(config["registered_utc"]); start=utc_time(config["start_utc"])
    registered<=start || throw(ArgumentError("trial registration must not follow its start"))
    ShadowTrialSpec(id,identity,actual,manifest,registered,start,false,seal_sha256)
end

function configured_trial(; environment=ENV)
    path=strip(get(environment,"SOLARSINDY_SHADOW_TRIAL_CONFIG",""))
    pin=strip(get(environment,"SOLARSINDY_SHADOW_TRIAL_SHA256",""))
    seal=strip(get(environment,"SOLARSINDY_SHADOW_TRIAL_SEAL_SHA256",""))
    isempty(path) && isempty(pin) && isempty(seal) && return LEGACY_A3
    !isempty(path) && !isempty(pin) && !isempty(seal) ||
        error("trial config, expected config digest and external seal digest must be supplied together")
    load_trial(path,pin;seal_sha256=seal)
end

"Bind the independently frozen seal digest without changing the registered configuration."
function bind_legacy_seal(trial,expected_sha256::AbstractString)
    !trial.legacy && valid_digest(expected_sha256) || throw(ArgumentError("invalid prospective seal binding"))
    ShadowTrialSpec(trial.trial_id,trial.identity,trial.config_sha256,trial.served_manifest_sha256,
        trial.registered_utc,trial.start_utc,false,String(expected_sha256))
end

function require_started(trial,reference::DateTime=now(UTC))
    trial.legacy || reference>=trial.start_utc ||
        throw(ArgumentError("new calibration trial has not started; no acquisition or issue is allowed"))
    nothing
end

function trial_paths(log_path::AbstractString,trial)
    root=joinpath(dirname(abspath(log_path)),"calibration_shadow_trials",trial.trial_id)
    (;root,seal=joinpath(root,"legacy_seal"),status=joinpath(root,"status.json"),report=joinpath(root,"status.md"))
end

"Verify a pre-start legacy snapshot without modifying it or refreshing its date."
function verify_legacy_seal(log_path::AbstractString,trial)
    trial.legacy && throw(ArgumentError("legacy default does not have a new-trial seal"))
    valid_digest(trial.seal_sha256) || error("an externally pinned legacy seal digest is required")
    paths=trial_paths(log_path,trial)
    isdir(paths.seal) && !islink(paths.seal) || error("new trial requires a sealed legacy assessment")
    receipt_path=regular_file(joinpath(paths.seal,"seal.toml"))
    filesize(receipt_path)<=65_536 || error("legacy seal receipt exceeds size limit")
    digest(receipt_path)==trial.seal_sha256 || error("legacy seal digest does not match its external pin")
    receipt=TOML.parsefile(receipt_path)
    get(receipt,"schema",nothing)===1 || error("unknown legacy seal schema")
    get(receipt,"trial_id",nothing)==trial.trial_id &&
        get(receipt,"trial_config_sha256",nothing)==trial.config_sha256 &&
        get(receipt,"trial_start_utc",nothing)==string(trial.start_utc)*"Z" &&
        get(receipt,"legacy_identity",nothing)==LEGACY_A3.identity || error("legacy seal identity mismatch")
    sealed=utc_time(get(receipt,"sealed_utc",nothing))
    trial.registered_utc<=sealed<=trial.start_utc || error("legacy seal was not created before the trial start")
    files=(source=regular_file(joinpath(paths.seal,"source_rows.csv")),
        assessment=regular_file(joinpath(paths.seal,"assessment.json")),
        report=regular_file(joinpath(paths.seal,"assessment.md")))
    for (key,path) in pairs(files)
        expected=get(receipt,string(key)*"_sha256",nothing)
        valid_digest(expected) && digest(path)==expected || error("sealed legacy evidence digest mismatch: $key")
    end
    (;files...,receipt=receipt_path,receipt_sha256=digest(receipt_path),sealed_utc=string(sealed)*"Z")
end
end
