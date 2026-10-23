"""Synthetic fit/publication/reload/as-of point design and independent gain fit."""
module DurationMeanSyntheticExample
using SolarSINDy, Dates, SHA
const D=SolarSINDy.DurationConditionedForcing
const A=SolarSINDy.IntervalMeanForecaster
const R=A.DstIntervalAdapter.Replay
function run_example()
    states=vcat(fill(:N,49),[:S,:N,:S,:N,:S,:N,:S,:N])
    q=vcat(zeros(49),[0.,0.,0.,0.,2.,0.,2.,0.])
    rows(start)=[D.WindHour(start+Hour(i-1),start+Hour(i),start+Hour(i),states[i]==:S ? -1. : 1.,q[i]) for i in eachindex(q)]
    training=rows(DateTime(2024));drivers=rows(DateTime(2024)+Day(40))
    m=D.fit_duration(training,DateTime(2024,2);beta=5,source_sha256=bytes2hex(sha256("synthetic mean-only fixture v1")))
    issue=last(drivers).completed;c=D.current_state(drivers,issue)
    cp=R.Checkpoint(issue,[-2.,3.],zeros(2,2))
    function design(model,input)
        state=D.current_state(input,issue)
        D.mean_affine_rollout(model,state,cp,[1. -2.;3. 4.];lambdas=(0.,0.),sigmas=(1.,0.),
          pressure=0.,target_observation_variance=0.,horizon=2)
    end
    result=design(m,drivers);pred=D.evaluate_mean_rollout(result,(2.,3.))
    # Independent Brownian hand design: E[q1]=1,E[q2]=0, initial column sums4/2.
    expected=[3.5 1.5;3. 1.];targets=[1.5,-1.]
    @assert all(result.coefficients[h][k]==expected[h,k] for h in 1:2,k in 1:2)
    @assert [s.mean for s in pred.summaries]==targets
    computed_design=[result.coefficients[h][k] for h in 1:2,k in 1:2]
    fit=A.nnls2(computed_design,targets .+ 10.)
    @assert all(isapprox(fit.gains[k],(2.,3.)[k];atol=5e-11,rtol=5e-14) for k in 1:2)
    future=D.WindHour(issue,issue+Hour(1),issue+Hour(20),-1.,100.)
    @assert design(m,vcat(drivers,[future]))==result
    mktempdir() do tmp
        saved=D.save_model(joinpath(tmp,"synthetic-model"),m)
        loaded=D.load_model(saved.directory;expected_manifest_sha256=saved.manifest_sha256)
        @assert design(loaded,drivers)==result
        @assert D.evaluate_mean_rollout(design(loaded,drivers),fit.gains).computation_scope==:point_mean_only
    end
    println("Synthetic mean-only fit/save/reload/as-of and independent NNLS PASS; no actual data or skill verdict")
    (;complete=true,synthetic_only=true,actual_fits=0,scientific_verdict=false)
end
run_example()
end
