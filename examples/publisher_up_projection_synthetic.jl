"""Generated completion-clock example for the publisher's numerical rule.

This example makes no archival-source or scientific-coverage claim.
"""
module PublisherUPProjectionSynthetic
using Dates,SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","publisher_up_projection.jl"))
const U=PublisherUPProjection
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
function run_example()
    start=DateTime(2020);asof=start+Minute(96)
    ground=Dict(start+Minute(k)=>(3.0k,4.0k) for k in -40:100)
    calibration=D.CompletedCalibration(zeros(9),fill(start-Millisecond(1),9);
        model_schema=D.POINT_SCHEMA,point_model_sha256="1"^64,purpose=:synthetic)
    prepared=D.prepare_schedule(ground,start:Minute(1):start+Minute(65))
    centers=fill(log1p(5.)-1,66);scales=ones(66)
    receipts=D.materialize_receipts(ground,start+Minute(1):Minute(1):start+Minute(95);
        asof,scheduled=false).batches
    shared=E.replay_shared(prepared,centers,scales,receipts,calibration;
        subscribers=(E.SubscriberSpec(:static,:static;nominal=.5),),
        stream_id="generated-publisher-up",protocol_id="generated-up31-v1",asof)
    result=U.project_shared(shared,calibration,();start,asof,nominal=.5)
    @assert result.phase_period_minutes==31
    @assert result.attempted==result.issued_bounds==result.binary_resolved==66
    @assert result.unavailable==result.binary_unknown==result.nonfinite_bounds==0
    @assert all(row->row.completed_at==row.issue+Minute(30)+Second(45) && row.signed_score==1.,result.updates)
    @assert result.own_numeric_study_pass && result.publisher_UP_numeric_implemented
    @assert !result.source_bound_provenance_verified && !result.scientific_coverage_qualified
    @assert !result.actual_source_authority_granted && result.actual_fits==0
    println("Generated publisher UP completion replay: ",result.binary_resolved,
        " resolved bounds, ",result.phase_period_minutes," phases; no archival or coverage qualification.")
    true
end
end
