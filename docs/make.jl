using Documenter
using SolarSINDy

DocMeta.setdocmeta!(SolarSINDy, :DocTestSetup, :(using SolarSINDy); recursive=true)

makedocs(
    sitename = "SolarSINDy.jl",
    authors = "Jake W. Liu",
    modules = [SolarSINDy],
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls=false),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md",
    ],
)
