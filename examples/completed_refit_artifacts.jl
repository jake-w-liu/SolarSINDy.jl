module CompletedRefitArtifactsExample
using Dates, SolarSINDy
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
include(joinpath(@__DIR__,"..","test","completed_refit_artifacts","run_ci_native_chain.jl"))

function run_example(;fixture_root=nothing)
    fixture_root=fixture_root===nothing ? CompletedRefitNativeDriver.ensure_root() : CompletedRefitNativeDriver.verify_root(fixture_root)
    results=Dict{Symbol,Any}()
    for layout in (:ground11,:mixed8)
        directory=joinpath(fixture_root,String(layout))
        native=joinpath(directory,"native")
        terminal=R.decode_canonical(R.bounded_read(joinpath(native,"TERMINAL.json")))
        entries=R.decode_canonical(R.bounded_read(joinpath(native,"034-publish-cal-load","pins.json")))
        length(entries)<=R.MAX_GRAPH_FILES || error("fixture resolver exceeds bound")
        pins=Dict{String,String}(entry[1]=>entry[2] for entry in entries)
        length(pins)==length(entries) || error("duplicate fixture resolver pin")
        for path in values(pins)
            isabspath(path)&&!islink(path)&&(isfile(path)||isdir(path)) || error("fixture resolution must be ordinary")
        end
        index=Dict("point_pin"=>terminal["point_manifest_sha256"],"calibration_pin"=>terminal["calibration_manifest_sha256"])
        protocol=R._protocol_for(pins[index["point_pin"]],pins)
        point=R.load_point(pins[index["point_pin"]];pins,expected_role=layout===:ground11 ? :candidate : :fixed_mixed8)
        first_issue=DateTime(2017,11,1)
        cal=R.load_calibration(pins[index["calibration_pin"]];point,pins,first_issue)
        # Fixed research fixture plumbing attaches the already committed schedule.
        # It adds no public constructor or arbitrary feature/center callback.
        schedule_pin=R._context(point)["schedule_manifest_sha256_or_null"]
        source_pin=R._context(point)["source_manifest_sha256_or_null"]
        source=R.open_audited_sources(pins[source_pin];protocol,pins)
        schedule=R._load_schedule(pins[schedule_pin];protocol,source,pins)
        predictions=R.predict_log(point,schedule;protocol)
        bridge=R.completed_calibration(cal;first_issue)
        @assert length(predictions)==20 && all(row->row.accepted,predictions)
        @assert length(bridge.calibration.chronological_scores)==9
        @assert bridge.calibration.purpose===:synthetic
        @assert !bridge.source_bound_provenance_verified
        @assert bridge.artifact_pin==index["calibration_pin"]
        @assert bridge.point_manifest_sha256==index["point_pin"]
        results[layout]=(;point,cal,schedule,predictions,bridge,index,pins,protocol)
    end
    results
end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    CompletedRefitArtifactsExample.run_example()
    println("Synthetic ridge artifacts and complete signed calibration verified; actual-source provenance remains false.")
end
