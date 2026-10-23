module DurationMeanCompatibilityTests
using SolarSINDy,Test,Dates,JSON3,SHA
const D=SolarSINDy.DurationConditionedForcing
const FROZEN=joinpath(@__DIR__,"duration_forcing","frozen","evaluator_before_barrier_v1.jl")
@test bytes2hex(sha256(read(FROZEN)))=="f2dfca48267b5feaeb15174f6d1921fb1580d91a9ecc3d2d09896441fbecb558"
module FrozenReferenceHost
const IntervalMeanForecaster=Main.SolarSINDy.IntervalMeanForecaster
end
Base.include(FrozenReferenceHost,FROZEN)
const OLD=FrozenReferenceHost.DurationConditionedForcing
const O=JSON3.read(read(joinpath(@__DIR__,"duration_forcing","mean_precode_oracles_v1.json")))
fraction(x)=begin a,b=split(String(x),'/');parse(BigInt,a)//parse(BigInt,b) end
function model(mod)
    cells(x)=ntuple(i->ntuple(j->Int(x[i][j]),5),49)
    mod._model(5,DateTime(2024,2),Tuple(Float64.(fraction.(O.edges))),cells(O.risks),cells(O.flips),
      Tuple(Int.(O.global_risks)),Tuple(Int.(O.global_flips)),Tuple(Float64.(fraction.(O.marks))),
      Tuple(Int.(O.mark_counts)),repeat("a",64),repeat("b",64))
end
@noinline Base.@nospecializeinfer function same_values(@nospecialize(left),@nospecialize(right))
    pending=Tuple{Any,Any}[(left,right)]
    while !isempty(pending)
        a,b=pop!(pending)
        if a isa NamedTuple && b isa NamedTuple
            keys(a)==keys(b) || return false
            for name in keys(a);push!(pending,(getproperty(a,name),getproperty(b,name)));end
        elseif (a isa Tuple || a isa AbstractVector) && (b isa Tuple || b isa AbstractVector)
            length(a)==length(b) || return false
            for i in 1:length(a);push!(pending,(a[i],b[i]));end
        elseif hasproperty(a,:injections)&&hasproperty(b,:injections)&&hasproperty(a,:probability)&&hasproperty(b,:probability)
            propertynames(a)==propertynames(b) || return false
            for name in propertynames(a);push!(pending,(getproperty(a,name),getproperty(b,name)));end
        else
            isequal(a,b) || return false
        end
    end
    true
end
@testset "Exact frozen evaluator values, tuple order and metadata semantics" begin
    current_data=(;status=:available,reason=:available,issue=DateTime(2024,3),state=:N,age=2,q=0.,
      input_prefix_sha256=repeat("c",64),availability_kind=:assumed_completed_hour)
    current=merge(current_data,(state_id=D._hash(current_data),))
    cp=D.R.Checkpoint(current.issue,[-75.,12.],[9. 2.;2. 4.]);basis=[.25 -.5;.75 1.25]
    oldmodel=model(OLD);newmodel=model(D)
    @test oldmodel.model_id!=newmodel.model_id # Deliberate loaded-source identity change.
    for h in (2,4,6)
        args=(;lambdas=(.5,1/12),sigmas=(3.,3.),pressure=4.,target_observation_variance=1.,horizon=h)
        oldtree=OLD.affine_rollout(oldmodel,current,cp,basis;args...)
        tree=D.affine_rollout(newmodel,current,cp,basis;args...)
        result=D.evaluate_rollout(tree,(2.,3.))
        # Independent labelled reference uses every original branch in original order.
        reference=OLD.evaluate_rollout(merge(oldtree,(paths=collect(oldtree.paths),)),(2.,3.))
        @test keys(result)==keys(reference)
        @test result.paths isa Tuple && result.components isa Tuple
        for name in keys(result)
            if name==:model_id
                @test result.model_id==newmodel.model_id
            elseif name==:paths
                @test same_values(result.paths,oldtree.paths)
            else
                @test same_values(getproperty(result,name),getproperty(reference,name))
            end
        end
    end
    tree=D.affine_rollout(newmodel,current,cp,basis;lambdas=(0.,0.),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.,horizon=2)
    annotated=merge(tree,(caller_note="preserve caller metadata",))
    first=D.evaluate_rollout(annotated,(2.,3.))
    @test first.caller_note==annotated.caller_note
    @test keys(first)==(keys(annotated)...,:gains,:summaries,:components)
    again=D.evaluate_rollout(first,(3.,4.));fresh=D.evaluate_rollout(annotated,(3.,4.))
    @test same_values(again,fresh) && keys(again)==keys(fresh)
    # Same-input assembly must retain identities as well as numeric values.
    attached=D._attach_evaluation(annotated,first.gains,first.summaries,first.components)
    expected=merge(annotated,(gains=first.gains,summaries=first.summaries,components=first.components))
    @test keys(attached)==keys(expected)
    @test all(getproperty(attached,name)===getproperty(expected,name) for name in keys(expected))
    unavailable=(;status=:unavailable,reason=:synthetic,paths=(),claim_ready=false)
    @test D.evaluate_rollout(unavailable,(-1.,NaN))===unavailable
end
end
