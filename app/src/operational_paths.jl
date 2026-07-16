"""Choose the first evidence directory containing every requested artifact.

An explicit `SOLARSINDY_OPERATIONAL_EVIDENCE_DIR` override is returned unchanged so a bad
override fails closed at the consumer. Generated output otherwise takes precedence over the
preserved paper evidence and the bundled package snapshot.
"""
function select_operational_evidence_dir(required_files::AbstractString...;
                                         output_dir::AbstractString,
                                         paper_dir::AbstractString,
                                         package_dir::AbstractString)
    override = strip(get(ENV, "SOLARSINDY_OPERATIONAL_EVIDENCE_DIR", ""))
    !isempty(override) && return abspath(override)
    isempty(required_files) && throw(ArgumentError("at least one evidence artifact is required"))
    for candidate in (output_dir, paper_dir, package_dir)
        isdir(candidate) || continue
        all(name -> isfile(joinpath(candidate, name)), required_files) &&
            return abspath(candidate)
    end
    # Use the writer's directory when no complete source exists. Consumers then report missing
    # artifacts instead of silently reading an unrelated partial snapshot.
    return abspath(output_dir)
end
