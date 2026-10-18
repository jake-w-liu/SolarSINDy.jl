using Test, Dates, SHA, TOML, LinearAlgebra
include(ARGS[1])
const A=IntervalMeanForecaster.DstIntervalAdapter
const OUTPUT=ARGS[2]
const OBSERVATIONS=Dict{String,Any}[]
function problem(n,kind;scale=1.)
    m=2n;Z=Matrix{Float64}(undef,m,2);b=Vector{Float64}(undef,m);labels=similar(b)
    patterns=((1.,0.),(0.,1.),(1.,1.))
    for i in 1:n
        fallback=i%10==0
        for slot in 1:2
            z=if fallback
                (0.,0.)
            elseif kind==:interior
                patterns[mod1(i+slot-1,3)]
            elseif kind==:rank_one
                (1.,2.)
            else
                slot==1 ? (1.,0.) : (0.,1.)
            end
            residual=if fallback
                5.
            elseif kind==:interior
                2z[1]+3z[2]
            elseif kind==:rank_one
                5.
            else
                slot==1 ? -2. : 3.
            end
            row=2i-2+slot
            Z[row,1]=scale*z[1];Z[row,2]=scale*z[2];b[row]=scale*residual
            labels[row]=fallback ? -50. : (slot==1 ? -60. : -40.)
        end
    end
    f=fld(n,10);storm=n+f
    weights=[1/m+(x<=-50. ? .25/storm : 0.) for x in labels]
    irreducible=2f*25*(1/m+.25/storm)
    kind==:axis && (irreducible+=(n-f)*4*(1/m+.25/storm))
    (;Z,b,weights,literal_loss=scale^2*irreducible)
end
failure=nothing
try
    @testset "independent large positive/rank/axis NNLS oracles" begin
        for n in (200,10_000,430_248,500_000), kind in (:interior,:rank_one,:axis)
            p=problem(n,kind)
            result=A.nnls2(p.Z,p.b;weights=p.weights)
            expected=kind==:interior ? (2.,3.) : kind==:rank_one ? (1.,2.) : (0.,3.)
            observation=Dict("origins"=>n,"kind"=>String(kind),"gains"=>collect(result.gains),
                "expected_gains"=>collect(expected),"gain_absolute_error"=>abs.(collect(result.gains).-collect(expected)),
                "weighted_sse"=>result.weighted_sse,"literal_sse"=>p.literal_loss,
                "rank"=>result.rank,"rank_cutoff"=>result.rank_cutoff,"selected_candidate"=>result.selected_candidate,
                "candidates"=>[Dict("gains"=>collect(c.gains),"gradient"=>collect(c.gradient),
                     "kkt_bound"=>c.kkt_bound,"kkt"=>c.kkt,"feasible"=>c.feasible) for c in result.candidates])
            push!(OBSERVATIONS,observation)
            @testset "$n-$kind" begin
                @test all(isapprox.(result.gains,expected;atol=1e-12,rtol=0))
                @test isapprox(result.weighted_sse,p.literal_loss;atol=1e-12,rtol=0)
                @test result.rank==(kind==:rank_one ? 1 : 2)
                @test length(result.candidates)==4
                @test result.candidates[result.selected_candidate].kkt
                @test result.candidates[result.selected_candidate].feasible
            end
            GC.gc()
        end
        for exponent in (-400,400)
            p=problem(200,:interior;scale=ldexp(1.,exponent))
            result=setprecision(BigFloat,64) do
                A.nnls2(p.Z,p.b;weights=p.weights)
            end
            @test all(isapprox.(result.gains,(2.,3.);atol=1e-12,rtol=0))
            @test isapprox(result.weighted_sse/p.literal_loss,1.;atol=1e-12,rtol=0)
        end
        zero=A.nnls2(zeros(400,2),ones(400))
        @test zero.gains==(0.,0.)
        @test zero.rank==0
        @test zero.weighted_sse==400.
    end
catch error
    global failure=error
finally
    open(OUTPUT,"w") do io
        TOML.print(io,Dict("synthetic_only"=>true,"actual_archive_read"=>false,
            "assertions_passed"=>failure===nothing,"source_hashes"=>Dict(A.verify_source_identity()),
            "observations"=>OBSERVATIONS);sorted=true)
    end
end
failure===nothing || throw(failure)
