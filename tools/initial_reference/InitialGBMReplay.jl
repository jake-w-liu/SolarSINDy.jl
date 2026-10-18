module InitialGBMReplay

using Dates, SHA, JSON3, CSV, DataFrames, LinearAlgebra
using ..DstInitialGBMReference
const R=DstInitialGBMReference
const LOADED_SHA256=bytes2hex(sha256(read(@__FILE__)))
export replay_initial

_require(x,message)=x || throw(ArgumentError(message))
_sha(path)=bytes2hex(sha256(read(path)))
_readjson(path)=JSON3.read(read(path),Dict{String,Any})
_plain(x)=JSON3.read(JSON3.write(x),Dict{String,Any})
function verify_source_identity()
    deps=R.verify_source_identity()
    _require(_sha(@__FILE__)==LOADED_SHA256,"loadedsource/disk mismatch: InitialGBMReplay.jl")
    (deps...,"InitialGBMReplay.jl"=>LOADED_SHA256)
end
function _write_new(path,bytes)
    file=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try;write(file,bytes)==length(bytes) || error("short reference artifact write")
    finally;close(file);end
    _require(read(path)==bytes,"reference artifact readback differs")
    path
end
_json_new(path,value)=_write_new(path,Vector{UInt8}(codeunits(JSON3.write(value))))
function _csv_new(path,frame)
    io=IOBuffer();CSV.write(io,frame);_write_new(path,take!(io))
end
function _save_generated(save_function,path)
    # Native/BSON writers accept filenames rather than exclusive IO streams.
    # Generate in a newly claimed private subdirectory, then publish its bytes
    # through O_EXCL. Existing public artifacts are never opened for overwrite.
    work=mktempdir(dirname(path);prefix="generated-model-",cleanup=false)
    intermediate=joinpath(work,basename(path));save_function(intermediate)
    _require(isfile(intermediate) && !islink(intermediate),"model writer did not produce a regular file")
    _write_new(path,read(intermediate))
end
function _bound_sources(binding)
    _require(length(binding.source_files)==13 && length(binding.package_files)==388 && length(binding.receipt_files)>=4,
        "incomplete frozen source/package/receipt binding")
    for group in (binding.source_files,binding.package_files,binding.receipt_files)
        for (path,digest) in group
            _require(isfile(path) && !islink(path) && _sha(path)==digest,"bound source/run artifact changed: $path")
        end
    end
    verify_source_identity()
end

function _load_original_imputers(M,lib,prefix,metadata)
    I=M.Imputation;N=M.Native
    config=metadata["configuration"]
    columns=I.ColumnImputer[]
    set=I.ImputerSet(columns,Int.(config["fit_rows"]),DateTime(config["training_cutoff_utc"]),
        config["training_sha256"],config["source_sha256"],Int(config["minimum_feature_count"]),Int(config["rounds"]),false)
    try
        for item in metadata["models"]
            target=Int(item["target"]);features=Int.(item["features"])
            _require(target in I.TARGETS && features==[j for j in I.TARGETS if j!=target],"saved initial imputer channel order differs")
            model=N.load_model(lib,item["resolved_file"];names=M.Features.RAW_NAMES[features],expected_sha256=item["sha256"])
            push!(columns,I.ColumnImputer(target,features,model,Int.(item["fit_rows"])))
        end
    catch
        close(set);rethrow()
    end
    set
end

function _reconstruct_imputation(M,lib,prefix,source_dir,out,config)
    I=M.Imputation;N=M.Native
    metadata=_readjson(joinpath(source_dir,"initial_imputation.json"))
    counts=_readjson(joinpath(source_dir,"initial_imputed_counts.json"))
    for item in metadata["models"]
        name=item["file"]
        _require(name isa String && basename(name)==name && !occursin('/',name) && !occursin('\\',name),"unsafe source imputer artifact name")
        item["resolved_file"]=joinpath(source_dir,name)
    end
    cutoff=M._shift(config.validation_start,-168)
    rows=findall(tag->M._shift(tag,1)<=cutoff,prefix.time)
    _require(metadata["configuration"]["fit_rows"]==rows,"initial imputer fit mask differs")
    set=nothing;original=nothing
    try
        set=I.fit_imputers(lib,prefix,rows;training_cutoff_utc=cutoff)
        _require(_plain(I.imputer_config(set))==metadata["configuration"],"initial imputer configuration/fingerprint differs")
        _require(set.training_sha256==counts["training_sha256"],"source initial imputation fingerprint differs")
        original=_load_original_imputers(M,lib,prefix,metadata)
        _require(length(set.columns)==length(original.columns),"initial imputer fitted-target count differs")
        clean=M.Features.censor_fills(prefix.values)
        records=NamedTuple[]
        for (fresh,old,item) in zip(set.columns,original.columns,metadata["models"])
            _require((fresh.target,fresh.features,fresh.fitted_rows)==(old.target,old.features,old.fitted_rows),"initial column training lineage differs")
            path=joinpath(out,"initial_imputer_$(fresh.target).json")
            _save_generated(p->N.save_model(fresh.model,p),path)
            roundtrip=N.load_model(lib,path;names=M.Features.RAW_NAMES[fresh.features],expected_sha256=_sha(path))
            probe_hash=SHA.SHA2_256_CTX();probe_count=0
            try
                for first in 1:4096:length(prefix.time)
                    probe=first:min(first+4095,length(prefix.time))
                    data=N.DenseData(lib,permutedims(clean[fresh.features,probe]);names=M.Features.RAW_NAMES[fresh.features])
                    try
                        a=N.predict(fresh.model,data);b=N.predict(old.model,data);c=N.predict(roundtrip,data)
                        _require(a==b==c,"saved/refitted/roundtrip imputer predictions differ")
                        SHA.update!(probe_hash,reinterpret(UInt8,a));probe_count+=length(a)
                    finally;close(data);end
                end
            finally;close(roundtrip);end
            push!(records,(target=fresh.target,features=fresh.features,fit_rows=fresh.fitted_rows,
                source_model_sha256=item["sha256"],new_model=basename(path),new_model_sha256=_sha(path),
                compared_prefix_rows=probe_count,prediction_sha256=bytes2hex(SHA.digest!(probe_hash))))
        end
        a=I.impute(set,prefix);b=I.impute(original,prefix)
        _require(isequal(a.values,b.values) && a.imputed_counts==b.imputed_counts,"initial imputed prefix differs from saved-source-model replay")
        _json_new(joinpath(out,"initial_imputation_replay.json"),(configuration=I.imputer_config(set),
            models=records,prefix_imputed_counts=a.imputed_counts,source_full_counts_not_substituted=counts["counts"],
            prefix_rows=length(prefix.time),prefix_matrix_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(a.values))))))
        a.values
    finally
        original===nothing || close(original)
        set===nothing || close(set)
    end
end

function _frame_rows(reference_rows,raw)
    DataFrame(row=getproperty.(reference_rows,:row),source_sha256=getproperty.(reference_rows,:source_sha256),
        origin_tag_utc=getproperty.(reference_rows,:origin_tag),issue_time_utc=getproperty.(reference_rows,:issue),
        anchor_tag_utc=getproperty.(reference_rows,:anchor_tag),anchor_hour_complete_utc=getproperty.(reference_rows,:anchor_complete),
        assumed_anchor_receipt_utc=getproperty.(reference_rows,:anchor_receipt),
        target_record_tag_utc=getproperty.(reference_rows,:target_tag),target_hour_complete_utc=getproperty.(reference_rows,:target_complete),
        assumed_target_receipt_utc=getproperty.(reference_rows,:target_receipt),completed_target_lead_hours=getproperty.(reference_rows,:horizon),
        extra_dst_delay_hours=zeros(Int,length(reference_rows)),anchor_dst_nt=getproperty.(reference_rows,:anchor),
        observation_dst_nt=getproperty.(reference_rows,:target),prediction_dst_nt=getproperty.(reference_rows,:prediction),
        status=getproperty.(reference_rows,:status))
end

"""Replay only the two selected INITIAL GBM29 models after external source-run
receipt/filesystem gates. The caller must use the fresh-process bound loader.
No final refit model or test prediction is read as a fitting/selection input.
"""
function replay_initial(M,full_input,lib,source_dir,out;
        config,binding,selection_start=config.validation_start,selection_end=config.test_start)
    own=verify_source_identity();_bound_sources(binding)
    _require(realpath(M.ROOT)==realpath(binding.source_root) && realpath(M.PACKAGE_ROOT)==realpath(binding.package_root),
        "loaded source/package location differs from bound immutable copy")
    _require(lib.version==(3,2,0) && lib.live_objects==0,"fresh exclusive official native library required")
    _require(config.mode==:synthetic_fixture,"this development implementation is currently synthetic-only")
    _require(selection_start>=config.validation_start && selection_end<=config.test_start && selection_start<selection_end,
        "A reference selection window exceeds original validation")
    _require(!ispath(out) && isdir(dirname(abspath(out))),"reference output must be a new directory under an existing parent")
    mkdir(out)
    completion_path=joinpath(out,"completion.json");completion_bytes=nothing
    try
        BLAS.set_num_threads(1)
        final_index=findlast(tag->M._shift(tag,1)<=config.test_start,full_input.time)
        _require(final_index!==nothing,"no validation-only source prefix")
        prefix=M.Features.HourlyInputs(full_input.time[1:final_index],full_input.values[:,1:final_index];source_sha256=full_input.source_sha256)
        imputed=_reconstruct_imputation(M,lib,prefix,source_dir,out,config)
        design=M.gbm29_design(prefix,imputed)
        _require(length(design.names)==29 && allunique(design.names),"GBM29 feature schema differs")
        raw=R.RawLabels(prefix.time,M.Features.censor_fills(prefix.values)[15,:];source_sha256=prefix.source_sha256)
        results=NamedTuple[]
        for horizon in (4,6)
            record=_readjson(joinpath(source_dir,"gbm29_selection_h$horizon.json"))
            chosen=R.verify_choice(record,raw,design.available;horizon,
                validation_start=config.validation_start,validation_end=config.test_start)
            t=M.target_cohort(prefix,horizon,config)
            model=M._gbm_fit(design,t,chosen.fit_rows,chosen.configuration)
            prediction=M._gbm_predictions(model,design,t,chosen.validation_rows)
            original_metric=R.metric(prediction,t.outcomes[chosen.validation_rows])
            R.verify_metric(original_metric,chosen.expected_metric)
            path=joinpath(out,"initial_gbm29_h$horizon.bson")
            _save_generated(p->M.SolarSINDy.gbm_save(model.booster,p),path)
            loaded=M.SolarSINDy.gbm_load(path)
            restored=(booster=loaded,scaling=model.scaling,names=model.names)
            _require(prediction==M._gbm_predictions(restored,design,t,chosen.validation_rows),"initial GBM save/reload prediction differs")
            scaling_path=joinpath(out,"initial_gbm29_scaling_h$horizon.csv")
            _csv_new(scaling_path,DataFrame(feature=design.names,center=model.scaling.center,scale=model.scaling.scale))
            scaling=CSV.read(scaling_path,DataFrame)
            _require(scaling.feature==design.names && scaling.center==model.scaling.center && scaling.scale==model.scaling.scale,
                "initial scaling roundtrip differs")
            refs=R.validation_rows(raw,horizon,chosen.validation_rows,prediction,design.available)
            reference_path=joinpath(out,"initial_validation_h$horizon.csv");_csv_new(reference_path,_frame_rows(refs,raw))
            a_frame=R.cohorts(raw,horizon;validation_start=selection_start,validation_end=selection_end)
            a_rows=R.validation_rows(raw,horizon,a_frame.validation,raw.dst[a_frame.validation],falses(length(raw.time)))
            joined=R.join_for_a(refs,a_rows,raw;horizon,validation_start=selection_start,validation_end=selection_end)
            _csv_new(joinpath(out,"a_join_h$horizon.csv"),DataFrame(joined.rows))
            mask_path=joinpath(out,"initial_fit_rows_h$horizon.csv");_csv_new(mask_path,DataFrame(row=chosen.fit_rows,
                issue_time_utc=t.issues[chosen.fit_rows],target_hour_complete_utc=t.complete[chosen.fit_rows],assumed_label_receipt_utc=t.receipts[chosen.fit_rows]))
            push!(results,(horizon=horizon,choice=chosen,original_grid_metric=original_metric,a_join_metric=joined.metric,
                a_join_keys_sha256=joined.row_keys_sha256,max_a_selection_target_complete=joined.max_target_complete,
                max_a_selection_label_receipt=joined.max_label_receipt,model=basename(path),model_sha256=_sha(path),
                scaling=basename(scaling_path),scaling_sha256=_sha(scaling_path),fit_mask_sha256=_sha(mask_path),
                ordered_names=design.names,model_identity="new deterministic initial-model reconstruction; original transient model absent"))
        end
        _require(lib.live_objects==0,"native objects leaked during initial reference replay")
        _bound_sources(binding);_require(verify_source_identity()==own,"reference implementation changed during execution")
        outputs=Dict{String,String}()
        for (directory,_,files) in walkdir(out),name in files
            path=joinpath(directory,name);_require(!islink(path),"unexpected output link")
            outputs[replace(relpath(path,out),'\\'=>'/')]=_sha(path)
        end
        receipt=(schema="candidate_a_initial_validation_reference_v1",evidence_mode="synthetic_only_unqualified_for_real_application",
            reference_contract_sha256=R.REFERENCE_CONTRACT_SHA256,own_sources=own,source_bindings=binding,
            source_sha256=prefix.source_sha256,prefix_rows=final_index,
            prefix_raw_matrix_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(prefix.values)))),
            initial_boundary=config.validation_start,original_selection_boundary=config.test_start,
            a_selection_start=selection_start,a_selection_end=selection_end,extra_dst_delay_hours=0,
            results=results,output_sha256=outputs,native_live_objects=lib.live_objects,complete=true)
        completion_bytes=Vector{UInt8}(codeunits(JSON3.write(receipt)));_write_new(completion_path,completion_bytes)
        _bound_sources(binding);_require(verify_source_identity()==own,"reference implementation changed at completion")
        (directory=abspath(out),completion_sha256=_sha(completion_path),results=results)
    catch error
        if completion_bytes!==nothing && isfile(completion_path) && read(completion_path)==completion_bytes
            mv(completion_path,completion_path*".incomplete";force=false)
        end
        _json_new(joinpath(out,"failed.json"),(complete=false,error=sprint(showerror,error),failed_utc=string(now(UTC))))
        rethrow()
    end
end

end # module
