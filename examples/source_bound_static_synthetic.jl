module SourceBoundStaticSynthetic
using Dates,SHA,SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
const S=SourceBoundSharedReplay
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const E=SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","test","archive_source_bound_fit","case_fixture.jl"))
"""Exercise the four static and common delayed consumers on generated catalogs."""
function run_example()
    mktempdir() do parent
        for role in ("primary","independent"),layout in ("ground-only","mixed")
            path=joinpath(parent,role*"-"*layout);mkdir(path)
            value=make_cases(realpath(path),role,layout)
            recipe=C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
            train_start=START+Minute(40);train_stop=START+Minute(90);asof=START+Minute(211)
            expected=F.prepare_expected(value,1,recipe;start=train_start,stop=train_stop,asof)
            point=F.fit_generated(value,1,recipe,expected;start=train_start,stop=train_stop,asof).capsule
            cal_start=START+Minute(90);cal_stop=START+Minute(135);start=START+Minute(150)
            ce=S.K.prepare_expected(value,1,point;start=cal_start,stop=cal_stop,asof,first_issue=start)
            calibration=S.K.calibrate_generated(value,1,point,ce;start=cal_start,stop=cal_stop,asof,first_issue=start)
            static=S.B.calibrate_generated(value,1,point,calibration,ce)
            subscribers=(E.SubscriberSpec(:static,:static;nominal=.1),E.SubscriberSpec(:phase,:phase;nominal=.1,gamma=.01),
                E.SubscriberSpec(:fifo,:fifo;nominal=.1,gamma=.01/31),E.SubscriberSpec(:early,:early;nominal=.1,gamma=.01/31),
                E.SubscriberSpec(:maturity,:maturity;nominal=.1,gamma=.01/31))
            result=S.replay_generated(value,1,point,calibration,ce;start,stop=START+Minute(166),asof,
                subscribers,nominal=.1,static_calibration=static)
            @assert result.audit.mandatory_static_inventory_complete && result.static_bounds.methods==S.B.METHODS
            @assert all(c->c.issued==16,values(result.static_bounds.counts))
            @assert result.shared.source.completed_targets==1 && result.shared.source.censored_targets==15
            @assert result.publisher_upper.attempted==result.publisher_absolute.attempted==16
            @assert result.audit.actual_fits==0 && !result.audit.actual_source_authority_granted
        end
    end
    println("Generated source-bound static/shared/publisher experiment passed;methods=4;actual source access=false;actual fits=0.")
    nothing
end
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    SourceBoundStaticSynthetic.run_example()
end
