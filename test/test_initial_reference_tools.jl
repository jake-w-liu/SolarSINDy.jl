module InitialReferenceToolPackageTests

using Test, SHA, Dates
const ROOT=normpath(joinpath(@__DIR__,".."))
const COMPONENT=joinpath(ROOT,"tools","initial_reference")
const FROZEN=joinpath(@__DIR__,"initial_reference","frozen")
const MEMBERS=[joinpath(COMPONENT,name) for name in
    ("DstInitialGBMReference.jl","InitialGBMReplay.jl","BoundInitialReferenceInputs.jl","AllGridInitialGBMReference.jl")]
push!(MEMBERS,joinpath(ROOT,"tools","initial_reference_audit","InitialReferenceAudit.jl"))
const BEFORE=Dict(path=>bytes2hex(sha256(read(path))) for path in MEMBERS)

module SourceAdmission
const ARGS=[joinpath(realpath(mktempdir(;prefix="solarsindy-reference-admission-",cleanup=false)),"evidence")]
end
Base.include(SourceAdmission,joinpath(FROZEN,"test_loader.jl"))

module RuntimeBinding
const ARGS=[normpath(joinpath(@__DIR__,"..","tools","initial_reference"))]
end
Base.include(RuntimeBinding,joinpath(FROZEN,"test_runtime_binding_v1.jl"))

module InputSchemas
const ARGS=[normpath(joinpath(@__DIR__,"..","tools","initial_reference"))]
end
Base.include(InputSchemas,joinpath(FROZEN,"test_input_metadata_schema_v1.jl"))

module IndependentCohorts end
Base.include(IndependentCohorts,joinpath(ROOT,"tools","initial_reference_audit","InitialReferenceAudit.jl"))
Base.include(IndependentCohorts,joinpath(FROZEN,"independent_reader_hand_oracles_v1.jl"))

module PublicEntryMismatch
using Test, Dates
const COMPONENT=normpath(joinpath(@__DIR__,"..","tools","initial_reference"))
for name in ("DstInitialGBMReference.jl","InitialGBMReplay.jl","BoundInitialReferenceInputs.jl","AllGridInitialGBMReference.jl")
    include(joinpath(COMPONENT,name))
end
include(joinpath(@__DIR__,"initial_reference","frozen","independent_fixture.jl"))
const B=BoundInitialReferenceInputs
const G=AllGridInitialGBMReference
@testset "Public reconstruction rejects a different admitted executable before source access" begin
    # Inert synthetic file/receipt objects cannot authorize executing the current
    # binary. The source/module/input/native objects below intentionally provide
    # no usable fields, so touching them before rejection fails this test.
    work=realpath(mktempdir(;prefix="solarsindy-reference-runtime-",cleanup=false))
    f=fixture(joinpath(work,"input"))
    admitted=B.admit(f.path,f.sha256;roots=f.roots)
    config=(validation_start=DateTime(2019)+Hour(500),test_start=DateTime(2019)+Hour(900))
    out=joinpath(work,"must_not_exist")
    exception=try
        G.reconstruct_reference(nothing,nothing,nothing,out;config,admitted)
        nothing
    catch error
        error
    end
    @test exception isa ArgumentError
    @test occursin("actual Julia executable",sprint(showerror,exception))
    @test !ispath(out)
    @test B.recheck(admitted)
end
end # PublicEntryMismatch

@testset "Reference component source immutability" begin
    @test Dict(path=>bytes2hex(sha256(read(path))) for path in MEMBERS)==BEFORE
    @test bytes2hex(sha256(read(joinpath(COMPONENT,"DstInitialGBMReference.jl"))))==
        "906e1a28bb7f56d6e607ae765ff2bb14197a3c2da2f2d1a7741501ea8cacc448"
    @test bytes2hex(sha256(read(joinpath(COMPONENT,"BoundInitialReferenceInputs.jl"))))==
        "6f4f1d105cbe925cfa1599d19904f945f5b563dbb76cb96753cac46bca258be0"
    @test bytes2hex(sha256(read(joinpath(COMPONENT,"InitialGBMReplay.jl"))))==
        "8e7c89d92f31095a890892fb6e33f9fdafddcd714644309623741a39ebf8cb05"
end

end # module
