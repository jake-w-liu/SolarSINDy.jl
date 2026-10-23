# Synthetic configuration parsing only: no activation, sealing or observation acquisition.
isdefined(@__MODULE__, :CalibrationShadowTrials) ||
    include(joinpath(@__DIR__,"..","src","calibration_shadow_trials.jl"))

function run_versioned_shadow_trial_synthetic()
    config="""
    schema = 1
    trial_id = "synthetic-example-2027"
    shadow_identity = "superlearner-cal-shadow-synthetic-example-2027+median24+width1h1.50+widthOther1.20+warm30"
    registered_utc = "2026-12-31T00:00:00Z"
    start_utc = "2027-01-01T00:00:00Z"
    served_manifest_sha256 = "$(repeat("1",64))"
    algorithm = "median24-width1h1.50-widthOther1.20-warm30"
    history_policy = "cold-exact-trial-issued-since-start"
    """
    mktempdir() do directory
        path=joinpath(directory,"synthetic.toml")
        write(path,config)
        pin=CalibrationShadowTrials.digest(path)
        trial=CalibrationShadowTrials.load_trial(path,pin)
        @assert !trial.legacy && trial.trial_id=="synthetic-example-2027"
        @assert trial.config_sha256==pin
        paths=CalibrationShadowTrials.trial_paths(joinpath(directory,"log.csv"),trial)
        @assert !ispath(paths.root)
        @assert CalibrationShadowTrials.LEGACY_A3.identity!=trial.identity
        println("Synthetic versioned shadow spec checked; no trial or state was activated.")
    end
    true
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_versioned_shadow_trial_synthetic()
end
