module DstIntervalExperiment

using Dates, SHA, TOML
using ..DstIntervalAdapter
using ..DstIntervalStream
using ..DstIntervalTraining
using ..DstIntervalArtifacts
const A=DstIntervalAdapter
const S=DstIntervalStream
const T=DstIntervalTraining
const B=DstIntervalArtifacts
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
B.verify_source_identity()

export run_synthetic_experiment

function verify_source_identity()
    dependencies=B.verify_source_identity()
    bytes2hex(sha256(read(@__FILE__)))==LOADED_SOURCE_SHA256 ||
        throw(ArgumentError("loadedsource/disk mismatch: DstIntervalExperiment.jl"))
    (dependencies...,"DstIntervalExperiment.jl"=>LOADED_SOURCE_SHA256)
end

function _input_digest(input)
    io=IOBuffer();write(io,"candidate-A-synthetic-input-v1|",input.source_sha256,"|",String(input.availability_kind))
    for index in eachindex(input.time)
        write(io,htol(Int64(Dates.value(input.time[index]))))
        for channel in 1:17
            write(io,htol(reinterpret(UInt64,input.values[channel,index])))
        end
    end
    bytes2hex(sha256(take!(io)))
end

function _guard(input,digest,hashes)
    verify_source_identity()==hashes || throw(ArgumentError("experiment loaded source identity changed"))
    _input_digest(input)==digest || throw(ArgumentError("experiment input changed during execution"))
end

function _selection_document(selection)
    Dict("status"=>String(selection.status),"selected_index"=>selection.selected_index===nothing ? 0 : selection.selected_index,
        "cohort_sha256"=>collect(selection.cohort_sha256),
        "panels"=>[Dict("index"=>p.index,"qualifies_pooled"=>p.qualifies_pooled,
            "pooled_count"=>[m.n for m in p.pooled],"pooled_rmse"=>[string(m.rmse) for m in p.pooled],
            "storm_count"=>[m.n for m in p.storm],"storm_rmse"=>[string(m.rmse) for m in p.storm],
            "normalized_storm_rmse"=>string(p.normalized_storm_rmse)) for p in selection.panels])
end

"""Exercise the complete A-only experiment control flow on explicitly declared
synthetic inputs. This bounded prototype caps inputs at20,000 hours and retains
one basis panel plus18 validation panels at a time. It does not run a real raw
archive, qualify reference GBM training, alter serving state, or make a skill
verdict. fit_impl/publish_impl are injection points for independent call-count
oracles; returned models and bundles are still validated against loaded code.
"""
function run_synthetic_experiment(input::S.HourlyInputs,directory;
        initial_boundary::DateTime,validation_end::DateTime,final_boundary::DateTime,
        references_by_delay::AbstractDict,provenance,
        fit_impl=T.fit_gains,publish_impl=B.save_bundle,chunk_hours=24)
    hashes=verify_source_identity()
    B._require(provenance isa NamedTuple && hasproperty(provenance,:synthetic_fixture) &&
        provenance.synthetic_fixture===true && hasproperty(provenance,:input_sha256) &&
        provenance.input_sha256==input.source_sha256,"explicit synthetic input provenance required")
    B._require(1<=length(input.time)<=20_000,"synthetic prototype input bound exceeded")
    B._require(first(input.time)<initial_boundary<validation_end==final_boundary<=A._shift(last(input.time),1),
        "synthetic split boundaries differ from initial/validation/final contract")
    B._require(all(t->floor(t,Hour)==t,(initial_boundary,validation_end,final_boundary)),"split boundaries must be exact hours")
    B._require(Set(keys(references_by_delay))==Set((0,1)),"both latency reference slots are required")
    digest=_input_digest(input)
    destination=B._directory(directory);mkdir(destination)
    initial_calls=0;final_calls=0;publish_calls=0
    results=Dict{Int,Any}();records=Dict{String,Any}[];completion_bytes=nothing
    try
        for delay in (0,1)
            panels=Dict{Int,Any}();initial_models=String[]
            for setting in A.settings()[1:2:end]
                _guard(input,digest,hashes)
                rows=NamedTuple[]
                S.stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,
                    chunk_hours=chunk_hours,sink=x->append!(rows,x))
                training_rows=[r for r in rows if r.issue<initial_boundary]
                validation_rows=[r for r in rows if initial_boundary<=r.issue<validation_end]
                for eta_index in (setting.index,setting.index+1)
                    recipe=A.settings()[eta_index]
                    initial_calls+=1
                    model=fit_impl(training_rows,initial_boundary,recipe;delay_hours=delay,source_sha256=input.source_sha256)
                    _guard(input,digest,hashes);B._validate_model(model)
                    B._require(model.boundary==initial_boundary && model.setting_index==eta_index && model.delay_hours==delay,
                        "initial fit returned a different configuration or cutoff")
                    panels[eta_index]=T.predict_rows(model,validation_rows)
                    push!(initial_models,model.model_id)
                end
            end
            selection=T.select_setting(panels,references_by_delay[delay];validation_start=initial_boundary,
                validation_end=validation_end,delay_hours=delay)
            _guard(input,digest,hashes)
            selection_path=joinpath(destination,"selection-L$delay.toml")
            B._write_new(selection_path,B._toml_bytes(_selection_document(selection)))
            final_model=nothing;bundle=nothing
            if selection.status==:selected
                setting=A.settings()[selection.selected_index]
                rows=NamedTuple[]
                streamed=S.stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,
                    chunk_hours=chunk_hours,sink=x->append!(rows,x))
                final_calls+=1
                final_model=fit_impl(rows,final_boundary,setting;delay_hours=delay,source_sha256=input.source_sha256)
                _guard(input,digest,hashes);B._validate_model(final_model)
                B._require(final_model.boundary==final_boundary && final_model.setting_index==selection.selected_index &&
                    final_model.delay_hours==delay,"final fit returned different configuration or cutoff")
                publish_calls+=1
                bundle=publish_impl(joinpath(destination,"final-L$delay"),final_model,streamed.checkpoint;
                    provenance=(synthetic_fixture=true,input_sha256=input.source_sha256,input_values_sha256=digest,
                        initial_boundary=initial_boundary,validation_end=validation_end,final_boundary=final_boundary,
                        selection_status=:selected,selected_index=selection.selected_index,
                        execution_sources=hashes,caller_provenance=provenance))
                _guard(input,digest,hashes)
                loaded=B.load_bundle(bundle.directory;expected_manifest_sha256=bundle.manifest_sha256,
                    expected_sources=B.verify_source_identity())
                B._require(isequal(loaded.model,final_model),"published model differs from actual final fit")
                # Predictions through the reloaded model prove the saved object
                # is usable; this does not score the synthetic test partition.
                T.predict_rows(loaded.model,rows)
            else
                B._require(selection.selected_index===nothing,"unselected panel contains a selected index")
            end
            results[delay]=(selection=selection,initial_model_ids=Tuple(initial_models),final_model=final_model,bundle=bundle)
            push!(records,Dict("delay_hours"=>delay,"status"=>String(selection.status),
                "selected_index"=>selection.selected_index===nothing ? 0 : selection.selected_index,
                "initial_model_ids"=>initial_models,"final_model_id"=>final_model===nothing ? "" : final_model.model_id,
                "bundle_manifest_sha256"=>bundle===nothing ? "" : bundle.manifest_sha256,
                "selection_sha256"=>bytes2hex(sha256(read(selection_path)))))
        end
        _guard(input,digest,hashes)
        B._require(initial_calls==36 && 0<=final_calls==publish_calls<=2,"experiment call-count invariant failed")
        completion=Dict("schema"=>"candidate_a_synthetic_experiment_v1","contract_sha256"=>T.CONTRACT_SHA256,
            "evidence_mode"=>"synthetic_only_no_skill_verdict","input_sha256"=>input.source_sha256,
            "input_values_sha256"=>digest,"source_hashes"=>Dict(hashes),
            "initial_boundary"=>string(initial_boundary),"validation_end"=>string(validation_end),
            "final_boundary"=>string(final_boundary),"initial_fit_calls"=>initial_calls,
            "fit_implementation"=>fit_impl===T.fit_gains ? "loaded_default" : "injected_synthetic_oracle",
            "publish_implementation"=>publish_impl===B.save_bundle ? "loaded_default" : "injected_synthetic_oracle",
            "final_fit_calls"=>final_calls,"publish_calls"=>publish_calls,"results"=>records)
        completion_bytes=B._toml_bytes(completion);B._write_new(joinpath(destination,"completion.toml"),completion_bytes)
        _guard(input,digest,hashes)
        (directory=destination,completion_sha256=bytes2hex(sha256(completion_bytes)),by_delay=results,
            initial_fit_calls=initial_calls,final_fit_calls=final_calls,publish_calls=publish_calls,
            source_hashes=hashes,input_values_sha256=digest)
    catch error
        # No completion marker is created until all successful control paths
        # finish. Partial products remain available for diagnosis.
        B._quarantine_completion(joinpath(destination,"completion.toml"),completion_bytes)
        path=joinpath(destination,"failure.txt")
        ispath(path) || B._write_new(path,Vector{UInt8}(codeunits(sprint(showerror,error))))
        rethrow()
    end
end

end # module
