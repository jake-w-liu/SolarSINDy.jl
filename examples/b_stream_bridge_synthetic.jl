"""Generated archival bridge example; no models, fits, datasets or serving."""
module BStreamBridgeSynthetic
using SolarSINDy, Dates, SHA
const A=SolarSINDy.IntervalMeanForecaster
const D=SolarSINDy.DurationConditionedForcing
const C=SolarSINDy.DurationCurrentStateStream
const n=180
const tags=[DateTime(2000)+Hour(i-1) for i in 1:n]
const raw=ones(17,n)
raw[8,:].=1000.;raw[6,:].=-1.49;raw[12,:].=0.;raw[15,:].=-11.
const source=bytes2hex(sha256("generated-only-bridge-example-v1"))
const input=A.HourlyInputs(tags,raw;source_sha256=source)
const wind=[D.WindHour(tag,tag+Hour(1),tag+Hour(1),-1.49,1.) for tag in tags]
for delay in (0,1)
    count=Ref(0);available=Ref(0)
    result=C.with_current_state_tracker(wind;source_sha256=source) do (advance,_)
        A.stream_basis(input;delay_hours=delay,taus=(2.,12.),sigma=1.,issue_sink=p->begin
            current=advance(p.issue);count[]+=1
            if p.a_model_available && current.status==:available
                @assert p.issue_checkpoint.boundary==current.issue
                available[]+=1
            end
        end)
    end
    @assert count[]==n && available[]==13
    @assert result.checkpoint.consumed_count==n
    println("GENERATED_BRIDGE_EXAMPLE delay=",delay," hours=",count[]," admitted_issue_states=",available[])
end
end # module
