using Documenter
using SolarSINDy

DocMeta.setdocmeta!(SolarSINDy, :DocTestSetup, :(using SolarSINDy); recursive=true)

makedocs(
    sitename = "SolarSINDy.jl",
    authors = "Jake W. Liu",
    modules = [SolarSINDy],
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(prettyurls=false),
    pages = [
        "Home" => "index.md",
        "API Reference" => [
            "Core API" => "api.md",
            "Interval-Mean Forecaster" => "interval-mean-forecaster.md",
            "Published Lund Dst Reference" => "lund-dst-reference.md",
            "Completed-Source Minute Replay" => "completed-minute-replay.md",
            "Forecasting And Alarms" => "forecast-api.md",
            "Operational API" => "operational-api.md",
            "Super-Learner" => "superlearner-api.md",
            "Static Stack Driver And Arrival Models" => "static-stack-research-api.md",
            "Static Stack Error And Shadow Models" => "static-stack-errors-api.md",
            "Analog Ensemble Study API" => "analog-api.md",
        ],
        "Examples" => "examples.md",
        "Initial GBM Reference Research Tools" => "initial-reference.md",
        "Live Verification" => "live-verification.md",
        "EKF Overlay Decision" => "ekf-overlay-decision.md",
    ],
)
