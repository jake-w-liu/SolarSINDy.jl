"""
    validation_run_mode()

Return `:canonical`, `:noncanonical`, or `:test`. Explicit output roots are
canonical by default; package-local validation remains noncanonical. Tests and
diagnostic runs must opt out explicitly with `SOLARSINDY_RUN_MODE`.
"""
function validation_run_mode()
    default = isempty(strip(get(ENV, "SOLARSINDY_OUTPUT_ROOT", ""))) ?
        "noncanonical" : "canonical"
    raw = lowercase(strip(get(ENV, "SOLARSINDY_RUN_MODE", default)))
    raw in ("canonical", "noncanonical", "test") || throw(ArgumentError(
        "SOLARSINDY_RUN_MODE must be canonical, noncanonical, or test",
    ))
    return Symbol(raw)
end

"""
    validation_output_paths()

Resolve validation outputs from `SOLARSINDY_OUTPUT_ROOT`. When unset, preserve
the package-local `data/` and `figs/` layout. A canonical external run writes
only below the explicit root. Its frozen OMNI input defaults to
`<root>/data/source/omni_extracted.csv`. Input-path overrides are accepted only
for explicitly noncanonical or test runs.
"""
function validation_output_paths()
    package_root = normpath(joinpath(@__DIR__, ".."))
    explicit_root = strip(get(ENV, "SOLARSINDY_OUTPUT_ROOT", ""))
    root = isempty(explicit_root) ? package_root : abspath(explicit_root)
    mode = validation_run_mode()
    data = joinpath(root, "data")
    figs = joinpath(root, "figs")
    mkpath(data)
    mkpath(figs)
    default_omni = isempty(explicit_root) ?
        joinpath(data, "omni_extracted.csv") :
        joinpath(data, "source", "omni_extracted.csv")
    omni_override = strip(get(ENV, "SOLARSINDY_OMNI_EXTRACTED", ""))
    mode == :canonical && !isempty(omni_override) && throw(ArgumentError(
        "canonical runs use <output-root>/data/source/omni_extracted.csv; " *
        "set SOLARSINDY_RUN_MODE=noncanonical or test to override it",
    ))
    omni = abspath(isempty(omni_override) ? default_omni : omni_override)
    return (; root, data, figs, omni, explicit=!isempty(explicit_root),
            mode, canonical=mode == :canonical)
end
