"""Standalone cold public tuple-path regression; run in a fresh bounded Julia process."""
module DurationColdPublicAPIRegression
using SolarSINDy,Test,Dates,JSON3,SHA
const D=SolarSINDy.DurationConditionedForcing
const PATH=joinpath(@__DIR__,"mean_precode_oracles_v1.json")
@test bytes2hex(sha256(read(PATH)))=="ab5726e0bc4df1843e14f0a58bb272096ea71c61aec2f6e324f26544908886ca"
const O=JSON3.read(read(PATH))
fraction(x)=begin a,b=split(String(x),'/');parse(BigInt,a)//parse(BigInt,b) end
function model()
    cells(x)=ntuple(i->ntuple(j->Int(x[i][j]),5),49)
    D._model(5,DateTime(2024,2),Tuple(Float64.(fraction.(O.edges))),cells(O.risks),cells(O.flips),
      Tuple(Int.(O.global_risks)),Tuple(Int.(O.global_flips)),Tuple(Float64.(fraction.(O.marks))),
      Tuple(Int.(O.mark_counts)),repeat("a",64),repeat("b",64))
end
@testset "Cold ordinary public affine/evaluate h1:6,N/S" begin
    m=model()
    for state in (:N,:S)
        oracle=only(c for c in O.cases if c.beta==5&&Symbol(c.state)==state&&c.age==2)
        data=(;status=:available,reason=:available,issue=DateTime(2024,3),state,age=2,q=Float64(fraction(oracle.q)),
          input_prefix_sha256=repeat("c",64),availability_kind=:assumed_completed_hour)
        current=merge(data,(state_id=D._hash(data),))
        cp=D.R.Checkpoint(current.issue,[-2.,3.],zeros(2,2))
        for h in 1:6
            println("Cold public tuple path ",state," h",h);flush(stdout)
            tree=D.affine_rollout(m,current,cp,[1. -2.;3. 4.];lambdas=(0.,0.),sigmas=(1.,1.),
              pressure=0.,target_observation_variance=0.,horizon=h)
            # Ordinary public output goes directly to the public evaluator.
            result=D.evaluate_rollout(tree,(2.,3.))
            @test result.status==:available && result.paths isa Tuple && result.components isa Tuple
            @test length(result.paths)==Int(oracle.hours[h].leaves)
            branches=collect(result.paths) # Independent exact oracle only.
            @test sum(p.probability for p in branches)==1
            for t in 1:h
                @test isapprox(result.summaries[t].mean,Float64(fraction(oracle.hours[t].mean));atol=5e-11,rtol=5e-14)
                @test all(isapprox(result.summaries[t].coefficients[k],Float64(fraction(oracle.hours[t].coefficients[k]));atol=5e-11,rtol=5e-14) for k in 1:2)
            end
        end
    end
end
end
