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

# Additive package regression; the standalone bootstrap numerical fixture above
# retains its original complete bytes and does not become study admission.
module CMEGSMBootstrapOwnerPinPackageControls
using Test,SHA,JSON3,SolarSINDy
include(joinpath(@__DIR__,"completed_refit_artifacts","run_ci_native_chain.jl"))
const DRIVER=CompletedRefitNativeDriver
const FIXED=joinpath(pkgdir(SolarSINDy),"test","cme_gsm_bootstrap","run_controls.py")
const FIXED_SHA="8d91348fa546967e23be5aab2a48e1e25fc81cc5b9772cd02569263a9be4b212"

function run_owner_pin_controls()
    @test bytes2hex(sha256(read(FIXED)))==FIXED_SHA
    python=DRIVER.choose_python()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    parent=realpath(mktempdir(base;prefix="solarsindy-bootstrap-pin-controls-",cleanup=false))
    output=joinpath(parent,"proof")
    command=addenv(Cmd([python,"-B","-X","utf8",FIXED,"--output",output]),
        "PYTHONDONTWRITEBYTECODE"=>"1","TMP"=>parent,"TEMP"=>parent,"TMPDIR"=>parent)
    DRIVER._run_support_command(command,parent)
    # The owned Python receipt includes measured floating wall time; it is a
    # bounded native-control JSON record, not the integer-only artifact codec.
    terminal=joinpath(output,"TERMINAL.json")
    isfile(terminal)&&!islink(terminal) || error("ordinary control receipt required")
    before=stat(terminal)
    0<before.size<=131072 || error("control receipt byte bound")
    bytes=open(io->read(io,131073),terminal)
    after=stat(terminal)
    length(bytes)==before.size==after.size && length(bytes)<=131072 &&
        before.inode==after.inode && before.mtime==after.mtime && before.ctime==after.ctime ||
        error("control receipt changed or exceeded its bound")
    receipt=JSON3.read(bytes,Dict{String,Any})
    @test receipt["complete"]===true && receipt["methods"]==9
    @test receipt["native"]["native_exit"] isa Integer && !(receipt["native"]["native_exit"] isa Bool)
    @test receipt["native"]["native_exit"]==0 && receipt["native"]["timed_out"]===false
    @test isempty(receipt["changed_controls"])
    @test receipt["controls_before_sha256"]==receipt["controls_after_sha256"]
    @test open(io->bytes2hex(sha256(io)),joinpath(output,"native.log"))==receipt["raw_sha256"]
    @test receipt["Julia_executed"]===false && receipt["actual_GSM_values"]==0
    @test receipt["source_authority_granted"]===false && receipt["scientific_qualification"]===false
    println("BOOTSTRAP_PIN_METADATA_CONTROLS_ONLY;RECEIPTS=",output)
    nothing
end

@testset "Bootstrap transitive helper pins and real clean-fixture admission" begin
    run_owner_pin_controls()
end
end
