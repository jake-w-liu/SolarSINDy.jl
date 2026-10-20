module ActualAChainPackageTests
using Test, LinearAlgebra

# A long, entirely generated component test exercises the same source adapter,
# bounded stream, complete disk design and saved certificate checker. The full
# audited-reference 18-model stage is exercised by the native CI Python driver.
const ROOT=realpath(mktempdir(;prefix="solarsindy-a-long-",cleanup=false))
const BLAS_THREADS_BEFORE=BLAS.get_num_threads()
module LongPrefix
const ARGS=[joinpath(parentmodule(@__MODULE__).ROOT,"output")]
end
const PASSED=Ref(false)
try
    Base.include(LongPrefix,joinpath(@__DIR__,"actual_a_chain","long_prefix_cases.jl"))
    Base.include(LongPrefix,joinpath(@__DIR__,"actual_a_chain","permit_scope_cases.jl"))
    # Julia 1.12+ resolves bindings introduced by include in their new world.
    checked=Base.invokelatest() do
        (exported=isdefined(LongPrefix.C,:verify_initial_l0),
         bytes=LongPrefix.X.stage_inventory(joinpath(ROOT,"output")).bytes)
    end
    @test checked.exported
    @test checked.bytes>0
    PASSED[]=true
finally
    BLAS.set_num_threads(BLAS_THREADS_BEFORE)
    if PASSED[]
        @test realpath(dirname(ROOT))==realpath(tempdir()) && startswith(basename(ROOT),"solarsindy-a-long-")
        @test !islink(ROOT) && !Base.invokelatest(()->LongPrefix.C.ReferenceImport.B.reparse(ROOT))
        # Inventory every descendant before recursively removing this fresh,
        # ordinary, exclusively generated test directory.
        Base.invokelatest(()->LongPrefix.X.stage_inventory(ROOT))
        rm(ROOT;recursive=true)
    else
        println(stderr,"FAILED_A_LONG_COMPONENT_EVIDENCE_ROOT=",ROOT)
    end
end
@test PASSED[] && !ispath(ROOT)
@test BLAS.get_num_threads()==BLAS_THREADS_BEFORE
end
