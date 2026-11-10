module MatchedInitial342Tests
using Test,Dates,SHA,JSON3,SolarSINDy
using LinearAlgebra: BLAS
include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","BMatchedReference342.jl"))
const Entry=BMatchedReference342
const FC=Entry.BMatched342Core
const I=Entry.BI
const P=Entry.P
const M=Entry.M
include("fixture.jl")
include("source_cell_tests.jl")
include("native_cell_tests.jl")
include("generator_tests.jl")
include("producer_audit_tests.jl")
include("full32_driver.jl")
include("profile_contract_tests.jl")

function run_tests()
    @testset "Initial matched342 source and per-lead independent contracts" begin
        parent=realpath(get(ENV,"RUNNER_TEMP",tempdir()))
        owned_case(source_and_cell_checks,parent,"source-cells")
        owned_case(training_only_encoder_checks,parent,"training-only")
        owned_case(generator_checks,parent,"generator")
        ProfileContractOracles342.run_tests(parent)
    end
    @testset "Initial matched342 nonlinear native, immutable snapshot and saved replay" begin
        @test Threads.nthreads()==1
        previous_blas=BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            withenv("OMP_NUM_THREADS"=>"1") do
                runtime=FC.runtime_identity()
                @test runtime.runtime.threads==runtime.runtime.blas_threads==runtime.runtime.omp_threads==1
                owned_case(saved_state_cell_checks,realpath(get(ENV,"RUNNER_TEMP",tempdir())),"saved-states")
                run_owned_full32_tests()
            end
        finally
            BLAS.set_num_threads(previous_blas)
        end
    end
end
end
MatchedInitial342Tests.run_tests()
