"""Pure synthetic duration-model fit, immutable publication, reload and A+B forecast."""
module DurationForcingSyntheticExample
using SolarSINDy, Dates, SHA
const D=SolarSINDy.DurationConditionedForcing
const A=SolarSINDy.IntervalMeanForecaster
function run_example()
    states=vcat(fill(:N,49),[:S,:N,:S,:N,:S,:N,:S,:N])
    q=vcat(zeros(49),[0.,0.,0.,0.,2.,0.,2.,0.])
    function rows(start)
        [D.WindHour(start+Hour(i-1),start+Hour(i),start+Hour(i),states[i]==:S ? -1. : 1.,q[i]) for i in eachindex(q)]
    end
    training=rows(DateTime(2024));drivers=rows(DateTime(2024)+Day(40))
    model=D.fit_duration(training,DateTime(2024,2);beta=5,
        source_sha256=bytes2hex(sha256("synthetic duration forcing example v1")))
    current=D.current_state(drivers,last(drivers).completed)
    checkpoint=A.DstIntervalAdapter.Replay.Checkpoint(current.issue,[0.,0.],zeros(2,2))
    function forecast(m)
        tree=D.affine_rollout(m,current,checkpoint,zeros(2,2);lambdas=(0.,0.),sigmas=(1.,0.),
            pressure=0.,target_observation_variance=0.,horizon=2)
        D.evaluate_rollout(tree,(1.,0.))
    end
    before=forecast(model)
    @assert before.status==:available && length(before.paths)==4
    @assert before.summaries[2].mean==-12.
    @assert isapprox(before.summaries[2].total_variance,7/3;atol=5e-11,rtol=5e-14)
    mktempdir() do temporary
        saved=D.save_model(joinpath(temporary,"synthetic-model"),model)
        loaded=D.load_model(saved.directory;expected_manifest_sha256=saved.manifest_sha256)
        @assert forecast(loaded).summaries==before.summaries
    end
    println("Duration forcing synthetic fit/save/reload/A-interval mixture PASS; no actual fit, outcomes or skill verdict")
    (;complete=true,synthetic_only=true,paths=4,mean_h2=before.summaries[2].mean,
      variance_h2=before.summaries[2].total_variance,actual_fits=0,scientific_verdict=false)
end
run_example()
end
