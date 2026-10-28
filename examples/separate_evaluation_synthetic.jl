"""Generated frozen-model replay on a distinct future source bundle."""
module SeparateEvaluationSynthetic
using Dates,SHA,SolarSINDy
include("source_bound_static_synthetic.jl")
const T=SourceBoundStaticSynthetic
const S=T.S
module FutureFixture
using Dates,SHA,SolarSINDy
const START=DateTime(2021)
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","test","archive_source_bound_fit","case_fixture.jl"))
end
function run_example()
    mktempdir() do parent
        for role in ("primary","independent"),layout in ("ground-only","mixed")
            root=joinpath(parent,role*"-"*layout);mkdir(root)
            training=joinpath(root,"training");mkdir(training)
            value=T.make_cases(realpath(training),role,layout)
            recipe=T.C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
            first=T.START+Minute(40);last=T.START+Minute(90);cutoff=T.START+Minute(211)
            pe=T.F.prepare_expected(value,1,recipe;start=first,stop=last,asof=cutoff)
            point=T.F.fit_generated(value,1,recipe,pe;start=first,stop=last,asof=cutoff).capsule
            first=T.START+Minute(90);last=T.START+Minute(135);first_issue=T.START+Minute(150)
            ce=S.K.prepare_expected(value,1,point;start=first,stop=last,asof=cutoff,first_issue)
            calibration=S.K.calibrate_generated(value,1,point,ce;start=first,stop=last,asof=cutoff,first_issue)
            static=S.B.calibrate_generated(value,1,point,calibration,ce)
            before=T.F.model_bytes(point.model)
            future=joinpath(root,"future");mkdir(future)
            evaluation=FutureFixture.make_cases(realpath(future),role,layout)
            start=FutureFixture.START+Minute(150);stop=FutureFixture.START+Minute(166);asof=FutureFixture.START+Minute(211)
            binding=S.prepare_evaluation_binding(evaluation,1,point,calibration,ce;start,stop,asof)
            subscribers=(T.E.SubscriberSpec(:static,:static;nominal=.1),T.E.SubscriberSpec(:phase,:phase;nominal=.1,gamma=.01),
                T.E.SubscriberSpec(:fifo,:fifo;nominal=.1,gamma=.01/31),T.E.SubscriberSpec(:early,:early;nominal=.1,gamma=.01/31),
                T.E.SubscriberSpec(:maturity,:maturity;nominal=.1,gamma=.01/31))
            result=S.replay_generated(evaluation,1,point,calibration,ce;start,stop,asof,subscribers,nominal=.1,
                static_calibration=static,evaluation_binding=binding.expected)
            @assert result.audit.explicit_evaluation_binding_matched && result.evaluation_binding_sha256==binding.expected.context_sha256
            @assert result.audit.scheduled==result.audit.features_accepted==16 && result.audit.features_rejected==0
            @assert result.shared.source.completed_targets==1 && result.shared.source.censored_targets==15
            @assert result.static_bounds.methods==S.B.METHODS && all(c->c.issued==16,values(result.static_bounds.counts))
            @assert result.publisher_upper.attempted==result.publisher_absolute.attempted==16
            @assert only(result.publisher_upper.updates).target==10. && all(==(log(6.)),result.centers)
            @assert T.F.model_bytes(point.model)==before && !result.audit.actual_source_authority_granted && result.audit.actual_fits==0
        end
    end
    println("Generated separate-source future replay passed;contexts=4;fixed point models;actual source access=false;actual fits=0.")
    nothing
end
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    SeparateEvaluationSynthetic.run_example()
end
