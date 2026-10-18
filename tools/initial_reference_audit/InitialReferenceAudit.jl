module InitialReferenceAudit

# Independent arithmetic and raw cohort reconstruction. This module never fits
# a model, loads author code, or treats a replay as scientific qualification.
using Dates

const GRID = Tuple((max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255))
const METRIC_TOLERANCE = 5e-12
need(ok,message) = ok || throw(ArgumentError(message))
isint(x) = x isa Integer && !(x isa Bool)
finite_number(x) = x isa Real && !(x isa Bool) && isfinite(x)

"""Corrected Float64 sample deviation about a separately verified stored mean.

The mean is model state. Recomputing another rounded mean can turn a tiny
positive deviation into exact zero, whose legacy unit replacement is discontinuous.
Compensated accumulation limits reduction error; exactly zero maps to one.
"""
function sample_scale_about(values::AbstractVector,center)
    need(2<=length(values)<=1_000_000,"sample scale count outside 2:1000000")
    need(finite_number(center) && all(finite_number,values),"finite non-Boolean sample values/mean required")
    c=Float64(center)
    need(isfinite(c),"sample mean Float64 overflow")
    total=0.;correction=0.
    for value in values
        x=Float64(value)
        need(isfinite(x),"sample value Float64 overflow")
        term=abs2(x-c)
        need(isfinite(term),"sample variance overflow")
        adjusted=term-correction
        updated=total+adjusted
        correction=(updated-total)-adjusted
        total=updated
    end
    need(isfinite(total) && total>=0.,"sample variance sum overflow")
    scale=sqrt(total/(length(values)-1))
    need(isfinite(scale),"sample scale overflow")
    iszero(scale) ? 1. : scale
end

function field(record,key)
    if record isa AbstractDict
        need(haskey(record,String(key)),"missing reference field: $key")
        return record[String(key)]
    end
    need(hasproperty(record,key),"missing reference field: $key")
    getproperty(record,key)
end

function shift_hour(t::DateTime,hours::Integer)
    need(isint(hours),"Boolean hour shift")
    milliseconds = try
        Base.Checked.checked_add(Dates.value(t),Base.Checked.checked_mul(Int64(hours),Int64(3_600_000)))
    catch
        throw(ArgumentError("reference clock arithmetic overflow"))
    end
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(milliseconds)))
end

function exact_hour(t::DateTime)
    need(floor(t,Hour)==t,"reference clock must lie on an exact hour")
    t
end

function forecast_clock(issue::DateTime,horizon::Integer)
    need(isint(horizon) && horizon in (4,6),"initial reference requires L0 h4 or h6")
    exact_hour(issue)
    tag=shift_hour(issue,-1)
    complete=shift_hour(issue,horizon)
    (issue=issue,origin_tag=tag,driver_tag=tag,anchor_tag=tag,
     anchor_complete=issue,anchor_receipt=issue,target_tag=shift_hour(complete,-1),
     target_complete=complete,target_receipt=complete)
end

function eligible(issue::DateTime,complete::DateTime,receipt::DateTime,boundary::DateTime)
    exact_hour(boundary)
    need(issue<complete<=receipt,"invalid reference label chronology")
    cutoff=shift_hour(boundary,-168)
    issue<boundary && complete<=cutoff && receipt<=cutoff
end

"""Reconstruct separate raw per-lead cohorts, without an A joint-fit mask.

`available` restricts GBM training only. Validation and the A-window join retain
every finite raw anchor/target with both mature clocks, including fallbacks.
This performs no source admission, fitting, model selection, or data I/O.
"""
function expected_cohorts(times::AbstractVector{DateTime},dst::AbstractVector,
        available::AbstractVector{Bool},horizon::Integer;initial_boundary::DateTime,
        selection_boundary::DateTime,a_start::DateTime=initial_boundary,
        a_end::DateTime=selection_boundary)
    need(isint(horizon) && horizon in (4,6),"initial reference requires L0 h4 or h6")
    count=length(times)
    need(1<=count<=1_000_000 && length(dst)==length(available)==count,"reference raw dimensions or hour budget")
    need(all(t->floor(t,Hour)==t,times),"raw tags must lie on exact hours")
    need(all(i->times[i]==shift_hour(times[i-1],1),2:count),"raw tags must be contiguous and ordered")
    need(all(x->x isa Real && !(x isa Bool) && (isfinite(x)||isnan(x)),dst),"raw labels must be finite or NaN")
    foreach(exact_hour,(initial_boundary,selection_boundary,a_start,a_end))
    need(initial_boundary<selection_boundary && initial_boundary<=a_start<a_end<=selection_boundary,
         "invalid initial or A selection boundaries")
    fit=Int[];validation=Int[];a_join=Int[]
    for i in 1:max(0,count-horizon)
        isfinite(dst[i]) && isfinite(dst[i+horizon]) || continue
        clock=forecast_clock(shift_hour(times[i],1),horizon)
        if available[i] && eligible(clock.issue,clock.target_complete,clock.target_receipt,initial_boundary)
            push!(fit,i)
        end
        if initial_boundary<=clock.issue<selection_boundary &&
                eligible(clock.issue,clock.target_complete,clock.target_receipt,selection_boundary)
            push!(validation,i)
        end
        if a_start<=clock.issue<a_end && eligible(clock.issue,clock.target_complete,clock.target_receipt,a_end)
            push!(a_join,i)
        end
    end
    (fit=fit,validation=validation,a_join=a_join)
end

function metrics(prediction::AbstractVector,observed::AbstractVector)
    need(length(prediction)==length(observed)>0,"empty or inconsistent reference metric cohort")
    need(all(finite_number,prediction) && all(finite_number,observed),"nonfinite reference metric values")
    errors=Float64.(prediction).-Float64.(observed)
    count=length(errors)
    result=(n=count,rmse_nt=sqrt(sum(abs2,errors)/count),mae_nt=sum(abs,errors)/count,bias_nt=sum(errors)/count)
    need(all(isfinite,(result.rmse_nt,result.mae_nt,result.bias_nt)),"reference metric overflow")
    result
end

function verify_metric(actual,expected)
    a_count,e_count=field(actual,:n),field(expected,:n)
    need(isint(a_count) && isint(e_count) && a_count>0 && a_count==e_count,"reference metric count mismatch")
    for key in (:rmse_nt,:mae_nt,:bias_nt)
        a,e=field(actual,key),field(expected,key)
        need(finite_number(a) && finite_number(e) && abs(a-e)<=METRIC_TOLERANCE*max(1.,abs(e)),
             "reference metric mismatch: $key")
    end
    true
end

"""Verify the original eight-member grid and recover its strict first minimum.

The metric replay tolerance is never used to choose a winner. Every candidate
must describe the same complete per-lead validation cohort.
"""
function first_grid_choice(candidates)
    need(length(candidates)==8,"all original eight grid members required")
    selected=1;minimum=Inf;count=nothing
    for (index,candidate) in enumerate(candidates)
        configuration=field(candidate,:configuration)
        values=Tuple(field(configuration,key) for key in (:max_depth,:nrounds,:nbins))
        need(all(isint,values) && values==Tuple(GRID[index]),"original grid order/configuration differs")
        score=field(candidate,:metric)
        rows=field(score,:n)
        need(isint(rows) && rows>0,"invalid original grid metric count")
        count===nothing && (count=rows)
        need(rows==count,"original grid validation cohorts differ")
        rmse,mae,bias=(field(score,key) for key in (:rmse_nt,:mae_nt,:bias_nt))
        need(all(finite_number,(rmse,mae,bias)) && rmse>=0 && mae>=0,"invalid original grid metrics")
        if rmse<minimum
            minimum=rmse
            selected=index
        end
    end
    selected
end

end # module
