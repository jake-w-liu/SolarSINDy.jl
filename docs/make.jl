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
            "Forecasting And Alarms" => "forecast-api.md",
            "Operational API" => "operational-api.md",
            "V2.4e Super-Learner" => "operational-v24-api.md",
            "V2.2 Research Chain" => "operational-v22-research-api.md",
            "V2.3 Study API" => "operational-v23-api.md",
        ],
        "Examples" => "examples.md",
        "Live Verification" => "live-verification.md",
        "EKF V3 Decision" => "ekf-v3-decision.md",
    ],
)
