"""
    PreparedIssue

Feature-only admission record retaining issue, policy, layout, source timing,
rejection reason and optional ground11/mixed8 vectors. Label completeness never
determines feature eligibility. Construct records with [`prepare_issue`](@ref).
"""
struct PreparedIssue
    issue::DateTime
    policy::CMR.ReplayPolicy
    mode::Symbol
    timing::NamedTuple
    accepted::Bool
    reason::Symbol
    source_row::Union{Nothing,NamedTuple}
    ground11::Union{Nothing,Vector{Float64}}
    features::Union{Nothing,Vector{Float64}}
end

"""Prepare only issue features; ground11 never requires or consults wind."""
function prepare_issue(mag::AbstractDict,issue::DateTime;
                       policy=CMR.ReplayPolicy(),mode=:ground11,omni=nothing)
    check_mode(mode)
    mode===:mixed8 && !(omni isa AbstractDict) && throw(ArgumentError("mixed8 requires a wind dictionary"))
    timing=CMR.issue_timing(issue,policy)
    admitted=mode===:ground11 ? CMR.admit_ground_features(mag,issue;policy) :
        CMR.admit_features(mag,omni,issue;policy)
    admitted.accepted || return PreparedIssue(issue,policy,mode,timing,false,admitted.reason,nothing,nothing,nothing)
    row=admitted.value.row
    ground=vec(CF.feature_matrix([issue],[row.dbdt_now],[row.dbdt_mean30],[row.dbdt_max30],[row.dbdt_std30]))
    features=mode===:ground11 ? ground : [row.dbdt_now,row.dbdt_mean30,row.dbdt_max30,row.dbdt_std30,row.V,row.Bz,row.Bs,row.VBs]
    PreparedIssue(issue,policy,mode,timing,true,:complete,row,ground,features)
end

"""Audit every scheduled issue before any target-completeness decision."""
function prepare_schedule(mag::AbstractDict,times;kwargs...)
    issues=collect(times)
    all(time->time isa DateTime && whole(time),issues) || throw(ArgumentError("issues must be whole UTC minutes"))
    all(issues[i]<issues[i+1] for i in 1:length(issues)-1) || throw(ArgumentError("issues must increase strictly"))
    [prepare_issue(mag,time;kwargs...) for time in issues]
end

"""
    label_audit(prepared, mag, asof)

Read a target only after its declared label readiness. Rejected feature rows
return `:not_issued`; immature rows access no target source values.
"""
function label_audit(prepared::PreparedIssue,mag::AbstractDict,asof::DateTime)
    prepared.accepted || return (;accepted=false,reason=:not_issued,value=nothing)
    CMR.admit_target(mag,prepared.issue,asof;policy=prepared.policy)
end

"""Keep all scheduled rows; label rejection never changes feature eligibility."""
function case_audit(prepared,mag::AbstractDict;asof::DateTime)
    [begin
        label=label_audit(row,mag,asof)
        (;issue=row.issue,mode=row.mode,feature_accepted=row.accepted,feature_reason=row.reason,
          ground_anchor=row.timing.latest_ground_center,
          omni_anchor=row.mode===:mixed8 ? row.timing.latest_omni_arrival_start : nothing,
          label_ready=row.timing.label_assumed_ready,first_feedback=row.timing.first_scheduled_feedback,
          label_accepted=label.accepted,label_reason=label.reason,
          target=label.accepted ? label.value.target_max_dbdt : nothing,
          availability_kind=AVAILABILITY,measured_receipts_available=false)
    end for row in prepared]
end

"""
    split_eligible(row, next_split_issue)

Require feature eligibility and label readiness strictly before the next split.
Equality is excluded to preserve the supervised split boundary.
"""
split_eligible(row::PreparedIssue,next_split_issue::DateTime)=
    row.accepted && CMR.training_eligible(row.timing.label_assumed_ready,next_split_issue)

"""Bind point predictions from feature-eligible rows without examining labels."""
function bind_predictions(prepared,predictor)
    indices=findall(row->row.accepted,prepared)
    centers=fill(NaN,length(prepared));scales=fill(NaN,length(prepared))
    if !isempty(indices)
        X=reduce(vcat,permutedims(prepared[i].features) for i in indices)
        values=predictor(X)
        length(values)==length(indices) && all(valid_number,values) || throw(ArgumentError("invalid point predictions"))
        for (j,i) in enumerate(indices)
            centers[i]=Float64(values[j])
            scales[i]=1+prepared[i].ground11[3]
        end
    end
    (;indices,centers,scales)
end

"""
    CompletedCalibration(scores, label_ready; model_schema, point_model_sha256,
                         purpose, policy=CMR.ReplayPolicy(), mode=:ground11,
                         calibration_schema="dbdt_completed_source_calibration_v1")

Store chronological and sorted signed scores with readiness, layout and policy.
The point schema is `"dbdt_completed_source_point_v1"`; purpose is `:synthetic`
or `:refitted`. These declarations do not prove actual source-bound refitting
or artifact provenance. All labels must be strictly ready before replay starts.
"""
struct CompletedCalibration
    chronological_scores::Vector{Float64}
    sorted_scores::Vector{Float64}
    label_ready::Vector{DateTime}
    policy::CMR.ReplayPolicy
    mode::Symbol
    point_model_sha256::String
    purpose::Symbol
    function CompletedCalibration(scores,label_ready;policy=CMR.ReplayPolicy(),mode=:ground11,
                                  model_schema,calibration_schema=CALIBRATION_SCHEMA,
                                  point_model_sha256,purpose)
        check_mode(mode)
        model_schema==POINT_SCHEMA && calibration_schema==CALIBRATION_SCHEMA ||
            throw(ArgumentError("legacy or incompatible model/calibration clock"))
        purpose in (:synthetic,:refitted) || throw(ArgumentError("declare synthetic or refitted calibration purpose"))
        occursin(r"^[0-9a-f]{64}$",point_model_sha256) || throw(ArgumentError("point model SHA256 required"))
        length(scores)==length(label_ready)>0 && all(valid_number,scores) &&
            all(time->time isa DateTime,label_ready) && issorted(label_ready) ||
            throw(ArgumentError("invalid chronological calibration"))
        values=Float64.(scores)
        new(values,sort(values),collect(label_ready),policy,mode,String(point_model_sha256),purpose)
    end
end

function check_calibration(cal::CompletedCalibration,first_issue,policy,mode)
    policy_equal(cal.policy,policy) && cal.mode===mode || throw(ArgumentError("calibration clock/layout differs"))
    all(time->CMR.training_eligible(time,first_issue),cal.label_ready) ||
        throw(ArgumentError("calibration label not strictly mature before stream"))
end

function checked_stream(prepared,centers,scales,cal,asof)
    length(prepared)==length(centers)==length(scales) || throw(DimensionMismatch("issue/prediction lengths differ"))
    !isempty(prepared) && all(prepared[i].issue<prepared[i+1].issue for i in 1:length(prepared)-1) ||
        throw(ArgumentError("nonempty increasing schedule required"))
    indices=findall(row->row.accepted,prepared)
    !isempty(indices) || throw(ArgumentError("no feature-eligible issues"))
    policy=prepared[first(indices)].policy;mode=prepared[first(indices)].mode
    all(row->policy_equal(row.policy,policy) && row.mode===mode,prepared) || throw(ArgumentError("mixed stream clocks/layouts"))
    all(i->valid_number(centers[i]) && positive_number(scales[i]),indices) ||
        throw(ArgumentError("invalid eligible point/scale"))
    asof>=last(prepared).issue || throw(ArgumentError("asof precedes scheduled issues"))
    check_calibration(cal,prepared[first(indices)].issue,policy,mode)
    (;indices,policy,mode)
end
