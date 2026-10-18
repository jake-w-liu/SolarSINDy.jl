module AllGridInitialGBMReference

using Dates, SHA, JSON3, CSV, DataFrames, LinearAlgebra
using ..DstInitialGBMReference
using ..InitialGBMReplay
using ..BoundInitialReferenceInputs
const R=DstInitialGBMReference
const E=InitialGBMReplay
const B=BoundInitialReferenceInputs
const LOADED_SHA256=bytes2hex(sha256(read(@__FILE__)))
const NEXT_CONTRACT_SHA256="db833f075e82ad5a15e10587e538c9688958a0fac180ca5713405da43d145ff7"
export reconstruct_reference, verify_source_identity

function verify_source_identity()
    B.same(bytes2hex(sha256(read(@__FILE__))),LOADED_SHA256,"all-grid loaded/disk source mismatch")
    B.own_guard()
    (E.verify_source_identity()...,"BoundInitialReferenceInputs.jl"=>B.LOADED_SHA256,
        "AllGridInitialGBMReference.jl"=>LOADED_SHA256)
end
plain(x)=JSON3.read(JSON3.write(x),Dict{String,Any})
function bound_json(admitted,name)
    hash=B.getv(admitted.plan["files"]["artifacts"],name)
    path=B.resolved(admitted.roots,"artifacts",name)
    B.json_file(path,hash,admitted.limits)
end
function input_identity(input)
    context=SHA.SHA2_256_CTX()
    for t in input.time;SHA.update!(context,reinterpret(UInt8,[Int64(Dates.value(t))]));end
    (rows=length(input.time),first=first(input.time),last=last(input.time),source=input.source_sha256,
        times_sha256=bytes2hex(SHA.digest!(context)),
        matrix_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(input.values)))))
end
function input_metadata_identity(metadata,mode,expected_source)
    B.need(B.hex(expected_source),"invalid admitted raw source digest")
    if mode=="exposed_development"
        B.need(Set(keys(metadata))==Set(("rows","channels","start","stop","encoding","sha256")),
            "original raw metadata field set differs")
        B.same(metadata["encoding"],"little-endian Float64 hour-major; provider raw values",
            "original raw metadata encoding differs")
        channels=metadata["channels"]
        B.need(channels isa AbstractVector && length(channels)==17 &&
            all(x->x isa AbstractString,channels) && allunique(channels),"original raw channel schema differs")
        source=expected_source
        matrix=B.getv(metadata,"sha256")
    elseif mode=="synthetic_fixture"
        B.need(Set(keys(metadata))==Set(("start","rows","source_sha256","raw_matrix_sha256","synthetic_only")),
            "synthetic raw metadata field set differs")
        B.need(metadata["synthetic_only"]===true,"synthetic raw metadata evidence mode differs")
        source=B.getv(metadata,"source_sha256")
        matrix=B.getv(metadata,"raw_matrix_sha256")
    else
        throw(ArgumentError("unknown raw metadata evidence mode"))
    end
    B.need(B.integer(metadata["rows"],1) && metadata["rows"]<=1_000_000,"raw metadata row budget")
    B.need(B.hex(source) && B.hex(matrix),"invalid raw metadata digest")
    B.same(source,expected_source,"raw metadata source differs from admitted source")
    (source_sha256=String(source),raw_matrix_sha256=String(matrix))
end
function initial_model_names(admitted,M)
    metadata=bound_json(admitted,"initial_imputation.json")
    models=B.getv(metadata,"models")
    targets=Int[]
    for item in models
        target=B.getv(item,"target");features=B.getv(item,"features")
        B.need(B.integer(target,1) && target in M.Imputation.TARGETS &&
            features isa AbstractVector && all(x->B.integer(x,1),features),"initial imputer target/feature grammar")
        push!(targets,Int(target))
        name=B.getv(item,"file");parts=B.safe_components(name)
        B.need(length(parts)==1,"initial imputer filename must be one component")
        B.same(B.getv(item,"sha256"),B.getv(admitted.plan["files"]["artifacts"],name),"initial imputer metadata/actual map binding")
        B.resolved(admitted.roots,"artifacts",name)
    end
    B.need(allunique(targets) && issorted(targets),"initial imputer target set/order")
    metadata
end
function execution_runtime(admitted)
    support=B.getv(admitted.plan,"support")
    record=B.getv(support,"julia")
    expected=B.getv(record,"sha256")
    B.need(B.hex(expected),"invalid admitted Julia executable digest")
    admitted_path=B.resolved(admitted.roots,B.getv(record,"root"),B.getv(record,"path"))
    executing=realpath(joinpath(Sys.BINDIR,Base.julia_exename()))
    B.same(B.canonical(executing),B.canonical(admitted_path),
        "actual Julia executable differs from admitted runtime location")
    actual=B.digest_file(executing,B.Limits().file_bytes).sha256
    B.same(actual,expected,"actual Julia executable differs from admitted runtime digest")
    (path=executing,sha256=actual,version=string(VERSION),threads=Threads.nthreads())
end
function execution_permission(admitted,permit,own)
    admitted.mode=="synthetic_fixture" && return (kind="synthetic_only",controls=())
    B.need(permit isa NamedTuple && Set(keys(permit))==Set((:qualification,:harness,:clearance)),
        "actual fit requires separately pinned qualification/full-harness/root-clearance controls")
    controls=Pair{String,String}[];objects=Dict{Symbol,Any}()
    for role in (:qualification,:harness,:clearance)
        record=getproperty(permit,role)
        B.need(record isa NamedTuple && Set(keys(record))==Set((:path,:sha256)) && B.hex(record.sha256),"strict execution control record")
        parent=B.root_path(dirname(abspath(record.path)))
        path=B.resolved(Dict("control"=>parent),"control",basename(record.path))
        B.same(B.digest_file(path,64*1024^2).sha256,record.sha256,"pinned execution control bytes")
        push!(controls,path=>record.sha256)
        role==:clearance || (objects[role]=B.json_file(path,record.sha256,B.Limits()))
    end
    q=objects[:qualification];h=objects[:harness]
    B.need(q["schema"]=="all_grid_initial_reference_qualification_v1" && q["complete"]===true &&
        q["all_native_gates_passed"]===true && q["real_input_read"]===false,"complete isolated reference qualification")
    for (name,digest) in own
        B.same(B.getv(q["source_sha256"],name),digest,"qualified executed component identity")
    end
    B.same(q["original_source_sha256"],admitted.plan["files"]["source"],"qualified original source13")
    B.same(q["package_files"],admitted.plan["files"]["package"],"qualified original comparator runtime388")
    B.need(h["success"]===true && B.zero(h["native_exit"]) &&
        h["full_package_test_executed"]===true && h["experiments_executed"]===true &&
        isempty(h["changed_package_files"]) && isempty(h["changed_component_files"]) && isempty(h["changed_support_files"]),
        "complete unshortened package/examples/harness and unchanged component")
    B.same(h["component_source_sha256"],q["source_sha256"],"full harness binds complete qualified component group")
    B.same(h["qualification_sha256"],permit.qualification.sha256,"full harness binds actual qualification receipt")
    (kind="root_cleared_exposed_development",controls=Tuple(controls),
        qualification_sha256=permit.qualification.sha256,harness_sha256=permit.harness.sha256,
        root_clearance_sha256=permit.clearance.sha256)
end
function recheck_permission(permission)
    for (path,digest) in permission.controls
        parent=B.root_path(dirname(path));actual=B.resolved(Dict("control"=>parent),"control",basename(path))
        B.same(B.digest_file(actual,64*1024^2).sha256,digest,"execution qualification/clearance changed")
    end
end
function panel(refs)
    frame=E._frame_rows(refs,nothing)
    frame.driver_tag_utc=getproperty.(refs,:driver_tag)
    frame
end
function output_inventory(out;max_files=512,max_bytes=2*1024^3)
    result=Dict{String,String}();count=0;total=0
    for (directory,dirs,names) in walkdir(out)
        for dir in dirs;B.need(!islink(joinpath(directory,dir)) && !B.reparse(joinpath(directory,dir)),"unexpected output directory link");end
        for name in names
            count+=1;B.need(count<=max_files,"reference output file-count budget")
            path=joinpath(directory,name);size=filesize(path)
            B.need(size<=512*1024^2 && size<=max_bytes-total,"reference output byte budget")
            total+=size
            result[replace(relpath(path,out),'\\'=>'/')]=B.digest_file(path,512*1024^2).sha256
        end
    end
    (sha256=result,total_bytes=total,file_count=count)
end

"""Reconstruct all original INITIAL grid members behind a complete admitted
source proof. Actual execution additionally requires the separate root-cleared
full package/experiment/harness gate and a fresh process; this function neither
grants that clearance nor runs Candidate A. Synthetic and exposed modes keep
their exact distinct evidence labels.
"""
function reconstruct_reference(M,full_input,lib,out;config,admitted,
        selection_start=config.validation_start,selection_end=config.test_start,execution_permit=nothing)
    own=verify_source_identity();B.recheck(admitted)
    runtime=execution_runtime(admitted)
    permission=execution_permission(admitted,execution_permit,own)
    # The unchanged author module derives ROOT through ../.. and Julia retains
    # its final directory separator. Compare that benign spelling explicitly;
    # the admitted roots and all source members still pass the strict path guards.
    source_location=rstrip(B.canonical(M.ROOT),'/')
    package_location=rstrip(B.canonical(M.PACKAGE_ROOT),'/')
    B.same(source_location,rstrip(B.canonical(admitted.roots["source"]),'/'),"loaded original source location")
    B.same(package_location,rstrip(B.canonical(admitted.roots["package"]),'/'),"loaded original runtime location")
    B.same(plain(M.configuration(config)),admitted.plan["expected"]["configuration"],"original complete configuration")
    B.same(String(config.mode),admitted.mode,"source mode/admission mode")
    B.need(lib.version==(3,2,0) && lib.live_objects==0,"exclusive official3.2.0 native library required")
    native_record=admitted.plan["support"]["native"]
    B.same(lib.sha256,native_record["sha256"],"actual loaded native library identity")
    B.same(B.canonical(lib.path),B.canonical(B.resolved(admitted.roots,native_record["root"],native_record["path"])),
        "actual loaded native library location")
    B.need(lib.handle!=C_NULL,"actual native library is closed")
    B.need(selection_start>=config.validation_start && selection_end<=config.test_start && selection_start<selection_end,
        "A selection window exceeds original validation")
    identity=input_identity(full_input);meta=bound_json(admitted,"input_hourly_channels.json")
    metadata_identity=input_metadata_identity(meta,admitted.mode,admitted.plan["expected"]["input_sha256"])
    B.need(identity.rows==meta["rows"] && size(full_input.values)==(17,identity.rows) &&
        identity.first==DateTime(meta["start"]) &&
        all(i->full_input.time[i]==identity.first+Hour(i-1),eachindex(full_input.time)),"raw parsed clock/schema differs")
    B.same(identity.source,admitted.plan["expected"]["input_sha256"],"raw source identity")
    B.same(identity.source,metadata_identity.source_sha256,"raw matrix source metadata")
    B.same(identity.matrix_sha256,admitted.plan["expected"]["raw_hourly_matrix_sha256"],"actual parsed full raw matrix")
    B.same(identity.matrix_sha256,metadata_identity.raw_matrix_sha256,"raw matrix metadata digest")
    if admitted.mode=="exposed_development"
        B.same(meta["channels"],M.Features.RAW_NAMES,"original ordered raw channels differ")
        B.same(DateTime(meta["stop"]),identity.last,"original raw metadata final clock differs")
        B.need((identity.rows,identity.first,identity.last)==(430248,DateTime(1976,11,1),DateTime(2025,11,30,23)),
            "actual source coverage differs from locked archive")
    end
    initial_model_names(admitted,M)
    out=abspath(out);B.root_path(dirname(out));B.safe_components(basename(out))
    B.need(!ispath(out),"new reference directory required")
    for root in values(admitted.roots)
        B.need(B.canonical(out)!=B.canonical(root) && !startswith(B.canonical(out),B.canonical(root)*"/"),
            "new reference output overlaps an admitted input root")
    end
    mkdir(out);marker=joinpath(out,"completion.json");marker_bytes=nothing
    try
        BLAS.set_num_threads(1)
        final=findlast(tag->M._shift(tag,1)<=config.test_start,full_input.time)
        B.need(final!==nothing && final>=2,"no initial/validation source prefix")
        prefix=M.Features.HourlyInputs(full_input.time[1:final],full_input.values[:,1:final];source_sha256=full_input.source_sha256)
        source_dir=admitted.roots["artifacts"]
        imputed=E._reconstruct_imputation(M,lib,prefix,source_dir,out,config)
        E._write_new(joinpath(out,"prefix_raw17.f64"),Vector{UInt8}(reinterpret(UInt8,vec(prefix.values))))
        E._write_new(joinpath(out,"prefix_imputed17.f64"),Vector{UInt8}(reinterpret(UInt8,vec(imputed))))
        E._json_new(joinpath(out,"prefix.json"),(rows=final,start=first(prefix.time),last=last(prefix.time),
            source_sha256=prefix.source_sha256,raw_matrix_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(prefix.values)))),
            imputed_matrix_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(imputed))))))
        design=M.gbm29_design(prefix,imputed)
        B.need(length(design.names)==29 && allunique(design.names),"GBM29 ordered feature schema")
        raw=R.RawLabels(prefix.time,M.Features.censor_fills(prefix.values)[15,:];source_sha256=prefix.source_sha256)
        results=NamedTuple[]
        for horizon in (4,6)
            record=bound_json(admitted,"gbm29_selection_h$horizon.json")
            chosen=R.verify_choice(record,raw,design.available;horizon,
                validation_start=config.validation_start,validation_end=config.test_start)
            t=M.target_cohort(prefix,horizon,config)
            fitframe=DataFrame(row=chosen.fit_rows,issue_time_utc=t.issues[chosen.fit_rows],
                target_hour_complete_utc=t.complete[chosen.fit_rows],assumed_label_receipt_utc=t.receipts[chosen.fit_rows])
            E._csv_new(joinpath(out,"initial_fit_rows_h$horizon.csv"),fitframe)
            E._write_new(joinpath(out,"fit_h$horizon.txt"),Vector{UInt8}(codeunits(join(chosen.fit_rows,"\n")*"\n")))
            grid=NamedTuple[];fresh_candidates=NamedTuple[];selected_refs=nothing;selected_metric=nothing
            for (index,configuration) in enumerate(R.GRID)
                # Every fixed grid member is reconstructed; none is tuned from
                # new validation/test values or omitted after a favorable fit.
                model=M._gbm_fit(design,t,chosen.fit_rows,configuration)
                predictions=M._gbm_predictions(model,design,t,chosen.validation_rows)
                score=R.metric(predictions,t.outcomes[chosen.validation_rows])
                R.verify_metric(score,record["candidates"][index]["score"])
                tag="h$(horizon)_g$(lpad(index,2,'0'))"
                model_path=joinpath(out,"initial_gbm29_$tag.bson")
                E._save_generated(p->M.SolarSINDy.gbm_save(model.booster,p),model_path)
                restored=(booster=M.SolarSINDy.gbm_load(model_path),scaling=model.scaling,names=model.names)
                B.same(predictions,M._gbm_predictions(restored,design,t,chosen.validation_rows),"all-grid saved/reloaded predictions")
                scaling_path=joinpath(out,"initial_scaling_$tag.csv")
                E._csv_new(scaling_path,DataFrame(feature=design.names,center=model.scaling.center,scale=model.scaling.scale))
                scaling=CSV.read(scaling_path,DataFrame)
                B.need(scaling.feature==design.names && scaling.center==model.scaling.center && scaling.scale==model.scaling.scale,
                    "all-grid sample scaling exact roundtrip")
                refs=R.validation_rows(raw,horizon,chosen.validation_rows,predictions,design.available)
                prediction_path=joinpath(out,"initial_validation_$tag.csv");E._csv_new(prediction_path,panel(refs))
                push!(fresh_candidates,(candidate=configuration,score=score))
                push!(grid,(index=index,configuration=configuration,metric=score,
                    model=basename(model_path),model_sha256=E._sha(model_path),
                    scaling=basename(scaling_path),scaling_sha256=E._sha(scaling_path),
                    predictions=basename(prediction_path),predictions_sha256=E._sha(prediction_path),
                    roundtrip_predictions_exact=true))
                if index==chosen.selected_index
                    selected_refs=refs;selected_metric=score
                    E._csv_new(joinpath(out,"initial_validation_h$horizon.csv"),panel(refs))
                end
            end
            reconstructed=R.first_grid_choice(fresh_candidates,record["selected"];expected_count=length(chosen.validation_rows))
            B.same(reconstructed.selected_index,chosen.selected_index,"reconstructed strict first-grid choice")
            a_frame=R.cohorts(raw,horizon;validation_start=selection_start,validation_end=selection_end)
            a_rows=R.validation_rows(raw,horizon,a_frame.validation,raw.dst[a_frame.validation],falses(length(raw.time)))
            joined=R.join_for_a(selected_refs,a_rows,raw;horizon,validation_start=selection_start,validation_end=selection_end)
            E._csv_new(joinpath(out,"a_join_h$horizon.csv"),DataFrame(joined.rows))
            push!(results,(horizon=horizon,choice=chosen,grid=grid,ordered_names=design.names,
                original_selected_grid_metric=selected_metric,a_join_metric=joined.metric,
                a_join_keys_sha256=joined.row_keys_sha256,max_a_selection_target_complete=joined.max_target_complete,
                max_a_selection_label_receipt=joined.max_label_receipt))
            B.need(lib.live_objects==0,"native objects leaked after per-lead reconstruction")
        end
        B.same(input_identity(full_input),identity,"raw input changed during all-grid reconstruction")
        B.recheck(admitted);recheck_permission(permission);B.same(verify_source_identity(),own,"all-grid source changed")
        B.same(execution_runtime(admitted),runtime,"actual Julia runtime changed during reconstruction")
        inventory=output_inventory(out)
        result=(schema="all_grid_initial_validation_reference_v1",complete=true,
            evidence_mode=admitted.mode=="synthetic_fixture" ? "synthetic_only" : "exposed_development_reconstruction",
            contract_sha256=NEXT_CONTRACT_SHA256,original_reference_contract_sha256=R.REFERENCE_CONTRACT_SHA256,
            own_sources=own,admitted_metadata_fingerprint=admitted.sealed_sha256,
            execution_permission=permission,execution_runtime=runtime,
            actual_input_sha256=identity.source,actual_full_raw_matrix_sha256=identity.matrix_sha256,
            original_completion_sha256=admitted.plan["receipts"]["completion"]["sha256"],
            original_terminal_sha256=admitted.plan["receipts"]["terminal"]["sha256"],
            enhanced_closure_sha256=admitted.plan["receipts"]["closure"]["sha256"],
            initial_boundary=config.validation_start,original_selection_boundary=config.test_start,
            a_selection_start=selection_start,a_selection_end=selection_end,extra_dst_delay_hours=0,
            prefix_rows=final,grid_member_count=16,results=results,output_sha256=inventory.sha256,
            output_file_count=inventory.file_count,output_total_bytes=inventory.total_bytes,
            native_live_objects=lib.live_objects,scientific_verdict=false,candidate_a_fitting=false,
            original_model_byte_restoration_claim=false,source_run_modified=false)
        marker_bytes=Vector{UInt8}(codeunits(JSON3.write(result)));E._write_new(marker,marker_bytes)
        B.recheck(admitted);recheck_permission(permission);B.same(verify_source_identity(),own,"source changed at completion")
        B.same(execution_runtime(admitted),runtime,"actual Julia runtime changed at completion")
        (directory=out,completion_sha256=E._sha(marker),results=results)
    catch error
        if marker_bytes!==nothing && isfile(marker) && read(marker)==marker_bytes
            mv(marker,marker*".incomplete";force=false)
        end
        E._json_new(joinpath(out,"failed.json"),(complete=false,error=sprint(showerror,error),failed_utc=string(now(UTC))))
        rethrow()
    end
end

end # module
