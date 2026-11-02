# Unexecuted package adapter for unchanged standalone bootstrap controls.
module CMEGSMBootstrapPackageControls
using Test,SHA,TOML,SolarSINDy
const DIRECTORY=realpath(joinpath(pkgdir(SolarSINDy),"research","cme_gsm_bootstrap"))
const CORE=joinpath(DIRECTORY,"bootstrap_indices.jl")
const CORE_SHA="ffdebdfafe250328add95f0adc9fabff8dac70cf21cea5f5a04630fa74b2da9e"
@assert bytes2hex(sha256(read(CORE)))==CORE_SHA
# Untouched standalone tests refer explicitly to Main.CMEGSMBootstrap.
# Define that fixed module once; never replace an existing different module.
if !isdefined(Main,:CMEGSMBootstrap)
    Base.include(Main,CORE)
end
const BOOTSTRAP=getfield(Main,:CMEGSMBootstrap)
@assert BOOTSTRAP isa Module && BOOTSTRAP.SOURCE_SHA==CORE_SHA
@assert realpath(BOOTSTRAP.DEFAULT_INPUT)==realpath(joinpath(DIRECTORY,"frozen_population_years.toml"))
BOOTSTRAP.source_guard()
include(joinpath(DIRECTORY,"test_bootstrap_indices.jl"))
include(joinpath(DIRECTORY,"test_native_generated_artifact.jl"))

function run_package_native_fixture()
    @assert VERSION in BOOTSTRAP.CONTROL_RUNTIMES
    role=VERSION==BOOTSTRAP.PRIMARY_RUNTIME ? "primary" : "control"
    # This fixed tag is synthetic fixture metadata, never current-source proof.
    source_head=repeat("f",40)
    mktempdir() do temporary
        root=realpath(temporary);output=joinpath(root,"generated-native-fixture")
        result=BOOTSTRAP.generate(output;role,source_head,expected_runtime=VERSION)
        test_native_generated_artifact(result;role,source_head)
        manifest=TOML.parsefile(joinpath(output,"INDEX_MANIFEST.toml"))
        @test manifest["claimed_source_head"]==source_head
        @test manifest["actual_GSM_values"]==0 && manifest["source_authority_granted"]===false && manifest["scientific_qualification"]===false
    end
    println("BOOTSTRAP_PACKAGE_FIXTURE_ONLY;SYNTHETIC_HEAD_TAG=true;FOUR_JOB_COMPARISONS_NOT_SUBSTITUTED")
    nothing
end
end
