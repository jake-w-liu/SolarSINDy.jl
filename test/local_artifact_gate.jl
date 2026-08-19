"""
Ledger of oracles that depend on a locally generated artifact.

Several of this package's strongest oracles read artifacts that are produced by the offline
validation scripts and are deliberately not tracked: the OMNI archive, the 400 MB V2.3 base table and
hourly frame, and the three served-identity tables. A clean checkout does not have them, so those
oracles cannot run there — and an oracle that silently disappears is worse than one that is absent,
because the suite still reports success.

Every such oracle registers itself here before it decides whether to run. `runtests.jl` then prints
the ledger and asserts its shape, so a green suite states exactly which real-data oracles it did and
did not exercise. Setting `SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS=1` turns any absence into a failure,
which is how a full-evidence environment demands the complete set.

The module lives in `Main` so that every test file — each of which is its own module — shares one
ledger.
"""
module LocalArtifactGate

export local_artifact_available

const LedgerEntry = NamedTuple{(:oracle, :artifact, :present),Tuple{String,String,Bool}}
const LEDGER = LedgerEntry[]

"""
    local_artifact_available(oracle, artifact) -> Bool

Record `oracle` as depending on `artifact` and report whether that artifact is present. Registration
is idempotent per oracle name, so a gate inside a loop counts once.
"""
function local_artifact_available(oracle::AbstractString, artifact::AbstractString)
    present = ispath(artifact)
    entry = LedgerEntry((String(oracle), String(artifact), present))
    index = findfirst(e -> e.oracle == entry.oracle, LEDGER)
    index === nothing ? push!(LEDGER, entry) : (LEDGER[index] = entry)
    return present
end

"Oracles that could not run because their artifact is absent."
skipped_oracles() = [entry.oracle for entry in LEDGER if !entry.present]

"Oracles that ran against a present artifact."
exercised_oracles() = [entry.oracle for entry in LEDGER if entry.present]

"Whether the environment demands the complete real-data evidence set."
require_local_artifacts() = strip(get(ENV, "SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS", "")) in
                            ("1", "true", "TRUE", "yes", "YES")

"""
    report(io = stdout)

Print one line per registered oracle, then the exercised/skipped counts.
"""
function report(io::IO = stdout)
    println(io, "Local-artifact oracle ledger ($(length(LEDGER)) registered):")
    for entry in LEDGER
        println(io, "  ", entry.present ? "exercised" : "SKIPPED  ", "  ", entry.oracle,
                "  <- ", entry.artifact)
    end
    println(io, "  exercised: ", length(exercised_oracles()),
            "; skipped: ", length(skipped_oracles()))
    return nothing
end

end # module LocalArtifactGate
