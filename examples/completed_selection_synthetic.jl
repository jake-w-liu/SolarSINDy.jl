# Generated metadata only; no observations, fits, replay or source authority.
module CompletedSelectionSynthetic
using Dates, SolarSINDy
const S=SolarSINDy.CompletedSelectionPrimitives

function run_example()
    expected=S.GammaExpectation(:phase;cohorts=("generated-CMO","generated-FRD"),
        counts=(1,9),normalizations=(1.0,1.0),phase_period_minutes=31)
    rows=S.GammaMetrics[]
    for gamma in S.GAMMAS, (index,station) in enumerate(S.STATIONS)
        loss=gamma==0.001 ? (index==1 ? 0.0 : 1.0) : 1.0
        push!(rows,S.GammaMetrics(;station,schedule=:phase,grid="phase_declared",gamma,
            cohort=expected.cohorts[index],n=expected.counts[index],normalization=1.0,
            normalized_pinball=loss,full_count=0,empty_count=0,positive_inf_count=0,
            negative_inf_count=0,finite_negative_count=0,nan_count=0,phase_period_minutes=31))
    end
    gamma=S.choose_gamma(rows,expected)
    @assert gamma.gamma==0.001 && gamma.objective==0.5
    @assert !gamma.actual_source_access && gamma.actual_fits==0 && !gamma.source_bound_provenance_verified
    point_rows=[S.PointMetrics(;station="CMO",candidate_id=spec.id,family=spec.family,
        cohort="generated-ground-CMO",n=3,selection_log_mse=1.0,phase_period_minutes=31) for spec in S.POINT_GRID]
    point_expected=S.PointExpectation(;station="CMO",cohort="generated-ground-CMO",n=3,phase_period_minutes=31)
    point=S.choose_R2(point_rows,point_expected)
    @assert Tuple(row.candidate_id for row in point.family_winners)==("ridge-0.01","boost-2-50","forest-6-64")
    @assert point.primary_family===:ridge && !point.actual_coordinator_implemented && !point.scientific_qualification
    @assert S.split_clock_eligible(DateTime(2017,6,30,23,29),DateTime(2017,7,1))
    @assert !S.split_clock_eligible(DateTime(2017),DateTime(2017)+Minute(30)+Second(45))
    println("Generated pure-selection metadata decisions checked; actual fits=0; no scientific qualification.")
    nothing
end
end
