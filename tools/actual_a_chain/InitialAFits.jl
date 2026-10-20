module InitialAFits

using Dates, SHA, TOML, LinearAlgebra, SolarSINDy
using ..InitialReferenceImport, ..InitialAInputs
const I=InitialReferenceImport
const P=InitialAInputs
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
const S=F.DstIntervalStream
const T=F.DstIntervalTraining
const B=F.DstIntervalArtifacts
const Q=I.B
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
const MAX_FILE_BYTES=512*1024^2
const MAX_STAGE_BYTES=4*1024^3
const ROW_FIELDS=(:issue,:driver_tag,:driver_receipt,:anchor_tag,:anchor_complete,:anchor_receipt,
    :anchor,:target_tag,:target_complete,:target_receipt,:target,:scoreable,:model_available,:reason,
    :offset,:coefficients,:variance,:consecutive_valid,:reset_tag,:completed_target_leads,
    :dst_delay_hours,:taus,:sigma,:source_sha256,:latest_q,:latest_pressure)
export InitialExecutionPermit, run_initial_l0, verify_initial_l0, code_identity, read_basis

need(ok,message)=ok || throw(ArgumentError(message))
function code_identity()
    own=P.code_identity()
    need(I.sha(@__FILE__)==LOADED_SOURCE_SHA256,"loaded initial-A executor changed")
    own["tools/actual_a_chain/InitialAFits.jl"]=LOADED_SOURCE_SHA256
    for name in ("Project.toml","Manifest.toml")
        own[name]=I.sha(joinpath(I.PROJECT,name))
    end
    own
end

"Caller-pinned, separately qualified permission for exactly eighteen actual L0 fits."
struct InitialExecutionPermit
    path::String
    sha256::String
end

"Validate the fixed stage grammar; this alone grants no execution permission."
function validate_permit_scope(plan)
    fields=Set(("schema","complete","stage","implementation_sources","reference_completion_sha256",
        "independent_audit_completion_sha256","source_sha256","admitted_input_seal","initial_fit_count",
        "delay_hours","final_fit_count","test_scoring","scientific_verdict","serving_promotion",
        "qualification_file","qualification_sha256","julia_version","julia_binary_sha256",
        "training_boundary","validation_start","validation_end"))
    need(plan isa AbstractDict && Set(keys(plan))==fields,"exact initial-L0 permit fields required")
    need(plan["schema"]=="qualified_actual_initial_a_l0_permit_v1" && plan["complete"]===true &&
         plan["stage"]=="initial_l0","initial-L0 execution permit required")
    need(Q.integer(plan["initial_fit_count"]) && plan["initial_fit_count"]==18 &&
         Q.zero(plan["delay_hours"]) && Q.zero(plan["final_fit_count"]),"actual fit count or latency differs")
    for name in ("test_scoring","scientific_verdict","serving_promotion")
        need(plan[name]===false,"actual permit exceeds the initial-fit stage")
    end
    for (name,value) in (("training_boundary","2020-01-01T00:00:00"),
            ("validation_start","2020-01-01T00:00:00"),("validation_end","2023-01-01T00:00:00"))
        need(plan[name]==value,"actual permit changes locked boundaries")
    end
    for name in ("reference_completion_sha256","independent_audit_completion_sha256","source_sha256",
            "admitted_input_seal","qualification_sha256","julia_binary_sha256")
        need(Q.hex(plan[name]),"actual permit digest grammar differs")
    end
    need(plan["source_sha256"]==P.ACTUAL_SOURCE,"actual permit uses a different raw source")
    I.hash_map(plan["implementation_sources"])
    need(plan["qualification_file"] isa AbstractString && isabspath(plan["qualification_file"]) &&
         plan["julia_version"] isa AbstractString && !isempty(plan["julia_version"]),"actual permit path/runtime grammar differs")
    true
end

function checked_permit(reference,permit,own)
    proof=I.verified_metadata(reference)
    if proof.meta["evidence_mode"]=="synthetic_only"
        need(permit===nothing,"generated execution cannot consume an actual-stage permit")
        return nothing
    end
    need(permit isa InitialExecutionPermit,"actual fitting requires a separately qualified stage permit")
    path=I.absolute_file(permit.path)
    plan=Q.json_file(path,permit.sha256,I.LIMITS)
    validate_permit_scope(plan)
    need(plan["implementation_sources"]==own,"actual permit uses different execution sources")
    need(plan["reference_completion_sha256"]==reference.completion_sha256 &&
         plan["independent_audit_completion_sha256"]==reference.audit_completion_sha256 &&
         plan["source_sha256"]==P.ACTUAL_SOURCE==proof.meta["source_sha256"] &&
         plan["admitted_input_seal"]==reference.admitted.sealed_sha256,"actual permit input/reference bindings differ")
    for name in ("training_boundary","validation_start","validation_end")
        need(plan[name]==proof.meta[name],"actual permit/reference boundaries differ")
    end
    release_path=I.absolute_file(plan["qualification_file"])
    release=Q.json_file(release_path,plan["qualification_sha256"],I.LIMITS)
    need(release["complete"]===true && Q.zero(release["harness"]["counts"]["FAIL"]) &&
         release["harness"]["all17_independent_workers_complete"]===true,"complete executor software qualification required")
    need(length(release["native_jobs"])==6 && length(release["docs_and_containers"])==2 &&
         all(x->x["conclusion"]=="success",vcat(release["native_jobs"],release["docs_and_containers"])),
         "all native, documentation and image gates required")
    need(Set(x["name"] for x in release["native_jobs"])==Set("Julia $version / $os" for
            version in ("1.12.6","1.13.1") for os in ("ubuntu-latest","windows-latest","macos-latest")) &&
         Set(x["name"] for x in release["docs_and_containers"])==
            Set(("Documentation build","Julia 1.12 deployment images")),"exact native, documentation and image cells required")
    need(all(get(release["source_sha256"],name,nothing)==value for (name,value) in own),
         "qualified release does not contain this exact executor")
    need(plan["julia_version"]==string(VERSION) && plan["julia_binary_sha256"]==
         I.sha(joinpath(Sys.BINDIR,Base.julia_exename())),"actual Julia runtime differs from permit")
    need(I.sha(path)==permit.sha256 && I.sha(release_path)==plan["qualification_sha256"],"actual permit changed")
    (path=path,sha256=permit.sha256,qualification_path=release_path,qualification_sha256=String(plan["qualification_sha256"]))
end

function guard(frame,own,permission)
    need(code_identity()==own,"initial-A execution source changed")
    P.recheck(frame)
    if permission!==nothing
        need(I.sha(permission.path)==permission.sha256 && I.sha(permission.qualification_path)==permission.qualification_sha256,
             "actual stage permission/qualification changed")
    end
    true
end

nullable(x::Float64)=isnan(x) ? nothing : (need(isfinite(x),"infinite basis value");x)
function nullable_row(row)
    need(keys(row)==ROW_FIELDS,"stream basis field set differs")
    merge(row,(target=Tuple(nullable(x) for x in row.target),variance=Tuple(nullable(x) for x in row.variance),
        latest_q=nullable(row.latest_q),latest_pressure=nullable(row.latest_pressure)))
end
function restore_row(value)
    need(value isa NamedTuple && keys(value)==ROW_FIELDS,"typed basis row schema differs")
    restore(x)=x===nothing ? NaN : (need(x isa Float64 && isfinite(x),"invalid nullable basis value");x)
    need(value.model_available isa Bool && value.consecutive_valid isa Int && value.consecutive_valid>=0 &&
         value.reason in ("model","anchor_missing","known_driver_missing","warmup") &&
         value.completed_target_leads==(4,6),"basis availability metadata differs")
    merge(value,(target=Tuple(restore(x) for x in value.target),variance=Tuple(restore(x) for x in value.variance),
        latest_q=restore(value.latest_q),latest_pressure=restore(value.latest_pressure)))
end

function write_basis_row(io,row)
    bytes=B._typed_payload_bytes(nullable_row(row))
    need(1<=length(bytes)<=65536,"bounded typed basis record required")
    write(io,htol(UInt32(length(bytes))))
    need(write(io,bytes)==length(bytes),"short basis record write")
    4+length(bytes)
end

function read_basis(path,digest,reference_rows,setting,source)
    path=I.absolute_file(path)
    need(Q.hex(digest) && filesize(path)<=MAX_FILE_BYTES && I.sha(path)==digest,"pinned bounded basis file required")
    rows=NamedTuple[]
    open(path,"r") do io
        for expected in reference_rows
            need(!eof(io),"basis omitted a required reference-union row")
            size=Int(ltoh(read(io,UInt32)))
            need(1<=size<=65536,"basis record byte budget")
            bytes=read(io,size);need(length(bytes)==size,"truncated basis record")
            object=TOML.parse(String(bytes))
            need(Set(keys(object))==Set(("value",)),"typed basis envelope differs")
            row=restore_row(B._decode(object["value"]))
            T._check_rows([row],0,source;setting)
            need(isequal(T._identity(row),T._identity(expected)),"basis/raw-reference identity differs")
            push!(rows,row)
        end
        need(eof(io),"basis contains extra union rows")
    end
    need(I.sha(path)==digest,"basis changed during replay")
    rows
end

function point_file(path,rows)
    io=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try
        buffer=IOBuffer()
        for (index,row) in enumerate(rows)
            for point in row.prediction
                need(isfinite(point),"required-union candidate point is nonfinite")
                write(buffer,htol(reinterpret(UInt64,point)))
            end
            if index%4096==0;write(io,take!(buffer));end
        end
        position(buffer)>0 && write(io,take!(buffer))
    finally;close(io);end
    need(filesize(path)==16length(rows),"point file size differs")
    I.sha(path)
end

function stage_inventory(root)
    result=Dict{String,String}();total=0;directories=1
    for (directory,dirs,files) in walkdir(root;follow_symlinks=false)
        for name in dirs
            path=joinpath(directory,name)
            need(!islink(path) && !Q.reparse(path),"linked stage directory")
            directories+=1;need(directories<=128,"stage directory budget")
        end
        for name in files
            relative=replace(relpath(joinpath(directory,name),root),'\\'=>'/')
            path=Q.resolved(Dict("stage"=>root),"stage",relative)
            need(length(result)<512 && filesize(path)<=MAX_FILE_BYTES && filesize(path)<=MAX_STAGE_BYTES-total,
                 "stage file/total byte budget")
            total+=filesize(path);result[relative]=I.sha(path)
        end
    end
    need(allunique(lowercase.(collect(keys(result)))),"case-aliased stage files")
    (sha256=result,bytes=total,directories=directories)
end

"""Fit all frozen initial L0 settings from nine bounded source streams.

Actual inputs require a separately pinned software-qualified execution permit.
No final fit, test-period score, calibration or serving publication occurs here.
"""
function run_initial_l0(reference::I.ArtifactReference,directory;execution_permit=nothing,chunk_hours=24)
    own=code_identity();permission=checked_permit(reference,execution_permit,own)
    need(Q.integer(chunk_hours,1) && chunk_hours<=4096,"bounded chunk hours required")
    need(Threads.nthreads()==1 && BLAS.get_num_threads()==1,"initial-fit stage requires one Julia and BLAS thread")
    frame=P.load_prefix(reference)
    proof=I.materialize(reference);base=proof.reference.rows;meta=proof.proof.meta
    initial=I.datehour(meta["training_boundary"]);validation_start=I.datehour(meta["validation_start"]);validation_end=I.datehour(meta["validation_end"])
    need(validation_start<=validation_end && initial<=validation_start && validation_end==A._shift(Base.last(frame.input.time),1),"complete pre-selection source prefix required")
    destination=abspath(directory);Q.root_path(dirname(destination));Q.safe_components(basename(destination))
    need(!ispath(destination),"new initial-stage destination required")
    input_roots=vcat([I.PROJECT,reference.directory],collect(values(proof.proof.roots)))
    reference.admitted===nothing || append!(input_roots,collect(values(reference.admitted.roots)))
    for root in input_roots
        a=Q.canonical(destination);b=Q.canonical(root)
        need(a!=b && !startswith(a,b*"/") && !startswith(b,a*"/"),"initial-stage output overlaps its inputs/code")
    end
    guard(frame,own,permission)
    mkdir(destination);marker=joinpath(destination,"completion.json");marker_bytes=nothing
    fit_calls=0;schedules=0;models=Any[];streams=Any[];panels=Dict{Int,Any}()
    locations=Dict(row.issue=>index for (index,row) in enumerate(base))
    try
        for setting in A.settings()[1:2:end]
            guard(frame,own,permission)
            basis_name="basis-s"*lpad(string(setting.index),2,'0')*".bin"
            basis_path=joinpath(destination,basis_name)
            basis_io=Base.Filesystem.open(basis_path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
            second_export=Ref{Any}();streamed=Ref{Any}();matched=0;basis_bytes=0
            fit_names=("fit-s"*lpad(string(setting.index),2,'0'),"fit-s"*lpad(string(setting.index+1),2,'0'))
            exported=try
                T.export_fit_input(joinpath(destination,fit_names[1]),initial,setting;
                        delay_hours=0,source_sha256=frame.input.source_sha256,max_source_rows=length(frame.input.time),max_chunk_rows=chunk_hours) do first_sink
                    second_export[]=T.export_fit_input(joinpath(destination,fit_names[2]),initial,A.settings()[setting.index+1];
                            delay_hours=0,source_sha256=frame.input.source_sha256,max_source_rows=length(frame.input.time),max_chunk_rows=chunk_hours) do second_sink
                        streamed[]=S.stream_basis(frame.input;delay_hours=0,taus=setting.taus,sigma=setting.sigma,
                            chunk_hours=chunk_hours,sink=rows->begin
                                first_sink(rows);second_sink(rows)
                                for row in rows
                                    if haskey(locations,row.issue)
                                        expected=base[locations[row.issue]]
                                        need(isequal(T._identity(row),T._identity(expected)),"stream/raw-reference row differs")
                                        basis_bytes+=write_basis_row(basis_io,row);matched+=1
                                        need(basis_bytes<=MAX_FILE_BYTES,"basis file byte budget")
                                    end
                                end
                            end)
                    end
                end
            finally;close(basis_io);end
            schedules+=1
            need(matched==length(base) && streamed[].row_count==length(frame.input.time) &&
                 streamed[].max_buffered_rows<=chunk_hours,"full stream or reference-union coverage differs")
            guard(frame,own,permission)
            basis_sha=I.sha(basis_path)
            basis=read_basis(basis_path,basis_sha,base,setting,frame.input.source_sha256)
            inputs=(exported,second_export[])
            for slot in 1:2
                index=setting.index+slot-1
                guard(frame,own,permission)
                model=T.fit_gains_from_input(inputs[slot].directory;expected_manifest_sha256=inputs[slot].manifest_sha256)
                fit_calls+=1;B._validate_model(model)
                need(model.boundary==initial && model.setting_index==index && model.delay_hours==0,"initial fit configuration differs")
                before=T.predict_rows(model,basis)
                bundle_name="initial-model-s"*lpad(string(index),2,'0')
                provenance=(stage=:initial_l0,evidence_mode=frame.evidence_mode,source_sha256=frame.input.source_sha256,
                    input_prefix_sha256=frame.input_sha256,reference_completion_sha256=reference.completion_sha256,
                    independent_audit_completion_sha256=reference.audit_completion_sha256,
                    source_admission_seal=reference.admitted===nothing ? nothing : reference.admitted.sealed_sha256,
                    training_boundary=initial,validation_start=validation_start,validation_end=validation_end,
                    basis_sha256=basis_sha,fit_input_manifest_sha256=inputs[slot].manifest_sha256,
                    final_refit=false,test_scoring=false,scientific_verdict=false,serving_promotion=false)
                bundle=B.save_bundle(joinpath(destination,bundle_name),model,streamed[].checkpoint;provenance)
                loaded=B.load_bundle(bundle.directory;expected_manifest_sha256=bundle.manifest_sha256,expected_sources=B.verify_source_identity())
                after=T.predict_rows(loaded.model,basis)
                need(isequal(loaded.model,model) && isequal(before,after),"saved initial-model point roundtrip differs")
                panels[index]=after
                points_name="initial-points-s"*lpad(string(index),2,'0')*".f64"
                point_sha=point_file(joinpath(destination,points_name),after)
                push!(models,Dict("index"=>index,"model_id"=>model.model_id,"bundle"=>bundle_name,
                    "bundle_manifest_sha256"=>bundle.manifest_sha256,"fit_input"=>fit_names[slot],
                    "fit_input_manifest_sha256"=>inputs[slot].manifest_sha256,"basis"=>basis_name,"basis_sha256"=>basis_sha,
                    "points"=>points_name,"points_sha256"=>point_sha,"fit_count"=>model.fit_count,
                    "informative_count"=>model.informative_count,"storm_count"=>model.storm_count,
                    "gains"=>collect(model.gains),"fit_mask_sha256"=>model.fit_mask_sha256,"fit_data_sha256"=>model.fit_data_sha256))
                guard(frame,own,permission)
            end
            push!(streams,Dict("setting_pair"=>[setting.index,setting.index+1],"row_count"=>streamed[].row_count,
                "max_buffered_rows"=>streamed[].max_buffered_rows,"fallback_counts"=>streamed[].fallback_counts,
                "observations_used"=>streamed[].observations_used,"negative_pressure_hours"=>streamed[].negative_pressure_hours,
                "reference_union_rows"=>matched,"basis_sha256"=>basis_sha))
            stage_inventory(destination)
        end
        need(fit_calls==18 && schedules==9 && sort!(collect(keys(panels)))==collect(1:18),"complete initial grid required")
        selection=T.select_setting(panels,reference;validation_start=validation_start,validation_end=validation_end,delay_hours=0)
        guard(frame,own,permission)
        document=F.DstIntervalExperiment._selection_document(selection)
        B._write_new(joinpath(destination,"selection.toml"),B._toml_bytes(document))
        inventory=stage_inventory(destination)
        result=Dict("schema"=>"source_bound_a_initial_l0_stage_v1","complete"=>true,"evidence_mode"=>frame.evidence_mode,
            "source_sha256"=>frame.input.source_sha256,"reference_completion_sha256"=>reference.completion_sha256,
            "independent_audit_completion_sha256"=>reference.audit_completion_sha256,"input_prefix_sha256"=>frame.input_sha256,
            "parsed_source_rows"=>frame.parser_rows,"retained_prefix_rows"=>length(frame.input.time),
            "initial_boundary"=>string(initial),"validation_start"=>string(validation_start),"validation_end"=>string(validation_end),
            "initial_fit_count"=>fit_calls,"state_schedule_count"=>schedules,"final_fit_count"=>0,"delay_hours"=>0,
            "selection_status"=>String(selection.status),"selected_index"=>selection.selected_index===nothing ? 0 : selection.selected_index,
            "models"=>models,"streams"=>streams,"implementation_sources"=>own,"output_sha256"=>inventory.sha256,
            "output_bytes"=>inventory.bytes,"test_scoring"=>false,"scientific_verdict"=>false,"serving_promotion"=>false,
            "actual_execution_permit_sha256"=>permission===nothing ? nothing : permission.sha256)
        marker_bytes=Vector{UInt8}(codeunits(I.JSON3.write(result)))
        B._write_new(marker,marker_bytes)
        guard(frame,own,permission)
        need(stage_inventory(destination).sha256==merge(inventory.sha256,Dict("completion.json"=>I.sha(marker))),
             "initial-stage publication inventory changed")
        (directory=destination,completion_sha256=I.sha(marker),selection=selection,initial_fit_count=fit_calls,state_schedule_count=schedules)
    catch error
        if marker_bytes!==nothing && isfile(marker) && read(marker)==marker_bytes
            mv(marker,marker*".incomplete";force=false)
        end
        try
            B._write_new(joinpath(destination,"failed.toml"),B._toml_bytes(Dict("complete"=>false,"error"=>sprint(showerror,error),
                "completed_initial_fits"=>fit_calls,"completed_state_schedules"=>schedules)))
        catch
        end
        rethrow()
    end
end

"Validate the saved full-matrix certificate without optimizing or fitting gains."
function verify_fit_certificate(model,manifest,path)
    fit_rows=2model.fit_count
    bytes=T._disk_read_bound(path,manifest["data_sha256"],32fit_rows)
    need(length(bytes)==32fit_rows,"saved fit scalar byte fit_rows differs")
    words=reinterpret(UInt64,bytes)
    a=Matrix{Float64}(undef,fit_rows,2);rhs=Vector{Float64}(undef,fit_rows)
    for i in 1:fit_rows
        values=Tuple(reinterpret(Float64,ltoh(words[4(i-1)+j])) for j in 1:4)
        need(all(isfinite,values),"saved fit contains a nonfinite scalar")
        c1,c2,residual,label=values
        weight=1/fit_rows+(model.storm_count>0 && label<=-50 ? model.eta/model.storm_count : 0.)
        root=sqrt(weight);a[i,1]=c1*root;a[i,2]=c2*root;rhs[i]=residual*root
    end
    scale=max(maximum(abs,a),maximum(abs,rhs))
    if scale!=0
        a./=scale;rhs./=scale
    end
    factor=svd(a;full=false);largest=maximum(factor.S)
    cutoff=max(fit_rows,2)*eps(Float64)*largest
    solver=model.solver
    need(keys(solver)==(:gains,:weighted_sse,:normalized_sse,:normalization_scale,:rank,:rank_cutoff,
         :selected_candidate,:candidates),"saved solver certificate fields differ")
    need(solver.normalization_scale===scale && solver.rank_cutoff===cutoff &&
         solver.rank==Base.count(>(cutoff),factor.S),"saved matrix normalization/rank certificate differs")
    need(solver.candidates isa Tuple && length(solver.candidates)==4 &&
         Q.integer(solver.selected_candidate,1) && solver.selected_candidate<=4,"four-candidate solver certificate required")
    setprecision(BigFloat,256) do
        need(rounding(BigFloat)===RoundNearest,"readback requires nearest MPFR rounding")
        checked=NamedTuple[]
        for candidate in solver.candidates
            need(keys(candidate)==(:gains,:feasible,:kkt,:normalized_sse,:gradient,:kkt_bound) &&
                 candidate.gains isa NTuple{2,Float64} && all(isfinite,candidate.gains),"invalid saved candidate certificate")
            gains=BigFloat.(candidate.gains);loss=zero(BigFloat);gradient=zeros(BigFloat,2)
            for i in 1:fit_rows
                x,y=BigFloat(a[i,1]),BigFloat(a[i,2])
                residual=x*gains[1]+y*gains[2]-BigFloat(rhs[i])
                loss+=residual^2;gradient[1]+=x*residual;gradient[2]+=y*residual
            end
            bound=128*BigFloat(eps(Float64))*max(fit_rows,2)*BigFloat(largest)*
                (BigFloat(largest)*sqrt(sum(abs2,gains))+BigFloat(norm(rhs)))
            feasible=all(>=(0),candidate.gains)
            kkt=feasible && all(candidate.gains[j]==0 ? gradient[j]>=-bound : abs(gradient[j])<=bound for j in 1:2)
            need(candidate.feasible===feasible && candidate.kkt===kkt &&
                 candidate.normalized_sse==string(loss) && candidate.gradient==Tuple(string.(gradient)) &&
                 candidate.kkt_bound==string(bound),"saved residual/gradient/KKT certificate differs")
            push!(checked,(gains=candidate.gains,loss=loss,kkt=kkt,squared_gain_norm=sum(abs2,gains)))
        end
        need(first(checked).gains==(0.,0.) && checked[2].gains[2]==0 && checked[3].gains[1]==0,
             "saved zero/axis candidates differ")
        valid=findall(x->x.kkt,checked);need(!isempty(valid),"saved model has no feasible KKT certificate")
        best=first(valid);baseline=first(checked).loss
        for index in valid[2:end]
            current,previous=checked[index],checked[best]
            tolerance=64*BigFloat(eps(Float64))*max(baseline,current.loss,previous.loss)
            if current.loss<previous.loss-tolerance || (abs(current.loss-previous.loss)<=tolerance &&
                    (current.squared_gain_norm,current.gains)<(previous.squared_gain_norm,previous.gains))
                best=index
            end
        end
        winner=checked[best]
        need(best==solver.selected_candidate && winner.gains==model.gains &&
             Float64(winner.loss)===solver.normalized_sse && Float64(winner.loss*BigFloat(scale)^2)===solver.weighted_sse,
             "saved loss/winner policy differs")
    end
    true
end

"""Read back a pinned initial stage, replaying raw design and selected points.

This performs nine bounded state replays and certificate checks, with zero gain
fits, final refits, test-period scores or serving writes. It shares the already
independently tested state kernel; it is an artifact closure, not new independent
physics validation or an independent implementation of the optimizer.
"""
function verify_initial_l0(directory,completion_sha256,reference::I.ArtifactReference;execution_permit=nothing,chunk_hours=7)
    own=code_identity();permission=checked_permit(reference,execution_permit,own)
    need(Q.integer(chunk_hours,1) && chunk_hours<=4096 && Threads.nthreads()==1 && BLAS.get_num_threads()==1,
         "bounded serial readback required")
    root=Q.root_path(directory);marker=Q.resolved(Dict("stage"=>root),"stage","completion.json")
    need(filesize(marker)<=1_048_576,"bounded initial-stage completion required")
    meta=Q.json_file(marker,completion_sha256,I.LIMITS)
    fields=Set(("schema","complete","evidence_mode","source_sha256","reference_completion_sha256",
        "independent_audit_completion_sha256","input_prefix_sha256","parsed_source_rows","retained_prefix_rows",
        "initial_boundary","validation_start","validation_end","initial_fit_count","state_schedule_count",
        "final_fit_count","delay_hours","selection_status","selected_index","models","streams",
        "implementation_sources","output_sha256","output_bytes","test_scoring","scientific_verdict",
        "serving_promotion","actual_execution_permit_sha256"))
    need(Set(keys(meta))==fields && meta["schema"]=="source_bound_a_initial_l0_stage_v1" && meta["complete"]===true,
         "complete initial-stage schema required")
    need(meta["implementation_sources"]==own && meta["reference_completion_sha256"]==reference.completion_sha256 &&
         meta["independent_audit_completion_sha256"]==reference.audit_completion_sha256,"saved stage source/reference differs")
    for (name,value) in (("initial_fit_count",18),("state_schedule_count",9),("final_fit_count",0),("delay_hours",0))
        need(Q.integer(meta[name]) && meta[name]==value,"saved fit/stage count differs")
    end
    for name in ("test_scoring","scientific_verdict","serving_promotion")
        need(meta[name]===false,"saved initial stage exceeds its scope")
    end
    need(isequal(meta["actual_execution_permit_sha256"],permission===nothing ? nothing : permission.sha256),
         "saved actual-stage permission differs")
    expected=I.hash_map(meta["output_sha256"])
    allowed=Set(["selection.toml"])
    for index in 1:18
        suffix=lpad(string(index),2,'0')
        push!(allowed,"initial-points-s"*suffix*".f64")
        for name in ("manifest.toml","joint-fit.f64");push!(allowed,"fit-s"*suffix*"/"*name);end
        for name in ("model.toml","checkpoint.toml","provenance.toml","manifest.toml")
            push!(allowed,"initial-model-s"*suffix*"/"*name)
        end
        isodd(index) && push!(allowed,"basis-s"*suffix*".bin")
    end
    need(Set(keys(expected))==allowed,"saved initial-stage member set differs")
    need(stage_inventory(root).sha256==merge(expected,Dict("completion.json"=>completion_sha256)),"saved stage inventory differs")
    need(Q.integer(meta["output_bytes"]) && sum(filesize(joinpath(root,n)) for n in keys(expected))==meta["output_bytes"],
         "saved stage byte count differs")
    frame=P.load_prefix(reference);proof=I.materialize(reference);base=proof.reference.rows
    need(meta["evidence_mode"]==frame.evidence_mode && meta["source_sha256"]==frame.input.source_sha256 &&
         meta["input_prefix_sha256"]==frame.input_sha256 && Q.integer(meta["parsed_source_rows"]) &&
         meta["parsed_source_rows"]==frame.parser_rows && Q.integer(meta["retained_prefix_rows"]) &&
         meta["retained_prefix_rows"]==length(frame.input.time),"saved raw prefix provenance differs")
    initial=I.datehour(proof.proof.meta["training_boundary"])
    need(meta["initial_boundary"]==string(initial) && meta["validation_start"]==proof.proof.meta["validation_start"] &&
         meta["validation_end"]==proof.proof.meta["validation_end"],"saved selection/training clocks differ")
    need(meta["models"] isa AbstractVector && length(meta["models"])==18 && meta["streams"] isa AbstractVector &&
         length(meta["streams"])==9,"complete saved initial grid required")
    panels=Dict{Int,Any}();locations=Dict(row.issue=>i for (i,row) in enumerate(base))
    for (pair,setting) in enumerate(A.settings()[1:2:end])
        guard(frame,own,permission)
        records=meta["models"][[setting.index,setting.index+1]]
        basis_name="basis-s"*lpad(string(setting.index),2,'0')*".bin"
        basis_path=Q.resolved(Dict("stage"=>root),"stage",basis_name)
        basis=read_basis(basis_path,expected[basis_name],base,setting,frame.input.source_sha256)
        matched=0;joint=0;informative=0;storms=0;last_joint=nothing
        mask=T._hash(("A-joint-fit-mask-v1",initial,0));data=T._hash(("A-joint-fit-data-v1",initial,0))
        binary=SHA.SHA2_256_CTX()
        streamed=S.stream_basis(frame.input;delay_hours=0,taus=setting.taus,sigma=setting.sigma,chunk_hours,
            sink=rows->begin
                buffer=IOBuffer()
                for row in rows
                    if haskey(locations,row.issue)
                        matched+=1;need(isequal(row,basis[matched]),"saved basis differs from raw source replay")
                    end
                    all(row.scoreable) && all(A.fit_eligible(row.issue,row.target_complete[j],row.target_receipt[j],initial) for j in 1:2) || continue
                    joint+=1;informative+=row.model_available && any(!iszero,Iterators.flatten(row.coefficients));last_joint=row
                    mask=T._hash((mask,(row.issue,row.anchor_tag,row.target_tag,row.target_complete,row.target_receipt)))
                    data=T._hash((data,(row.issue,row.anchor,row.offset,row.coefficients,row.target)))
                    for slot in 1:2
                        target=row.target[slot];residual=target-row.offset[slot];storms+=target<=-50
                        for value in (row.coefficients[slot]...,residual,target)
                            write(buffer,htol(reinterpret(UInt64,value)))
                        end
                    end
                end
                SHA.update!(binary,take!(buffer))
            end)
        scalar_sha=bytes2hex(SHA.digest!(binary))
        need(matched==length(base) && last_joint!==nothing,"saved raw replay omitted required origins")
        stream_chunk=0
        for (slot,record) in enumerate(records)
            index=setting.index+slot-1;recipe=A.settings()[index]
            need(Set(keys(record))==Set(("index","model_id","bundle","bundle_manifest_sha256","fit_input",
                 "fit_input_manifest_sha256","basis","basis_sha256","points","points_sha256","fit_count",
                 "informative_count","storm_count","gains","fit_mask_sha256","fit_data_sha256")),"saved model record fields differ")
            need(Q.integer(record["index"]) && record["index"]==index,"saved initial model order differs")
            for (key,name) in (("bundle","initial-model-s"*lpad(string(index),2,'0')),
                    ("fit_input","fit-s"*lpad(string(index),2,'0')),("points","initial-points-s"*lpad(string(index),2,'0')*".f64"),
                    ("basis",basis_name))
                need(record[key]==name,"saved stage member name differs")
            end
            need(record["basis_sha256"]==expected[basis_name],"saved model basis pin differs")
            fit_root=joinpath(root,record["fit_input"]);fit_name=record["fit_input"]*"/manifest.toml"
            need(record["fit_input_manifest_sha256"]==expected[fit_name],"saved fit manifest pin differs")
            manifest=TOML.parse(String(T._disk_read_bound(joinpath(fit_root,"manifest.toml"),expected[fit_name],1_048_576)))
            need(Set(keys(manifest))==Set(("schema","complete","evidence_mode","contract_sha256","format","boundary_ms",
                 "delay_hours","source_sha256","source_hashes","setting_index","max_source_rows","max_chunk_rows",
                 "total_source_rows","joint_origins","informative_origins","fit_rows","storm_rows","data_bytes",
                 "data_sha256","fit_mask_sha256","fit_data_sha256","max_target_complete_ms","max_label_receipt_ms","max_fit_issue_ms")),
                 "saved fit input fields differ")
            need(manifest["schema"]=="disk_affine_gain_input_v1" && manifest["complete"]===true &&
                 manifest["evidence_mode"]=="unqualified_research_fit_input" && manifest["contract_sha256"]==T.CONTRACT_SHA256 &&
                 manifest["format"]=="little_endian_float64_c1_c2_residual_label_h4_then_h6_v1" &&
                 manifest["source_hashes"]==Dict(T.verify_source_identity()) && manifest["source_sha256"]==frame.input.source_sha256 &&
                 Q.integer(manifest["setting_index"]) && manifest["setting_index"]==index && Q.zero(manifest["delay_hours"]) &&
                 manifest["boundary_ms"]==string(Dates.value(initial)),"saved fit configuration differs")
            need(Q.integer(manifest["max_chunk_rows"],1) && manifest["max_chunk_rows"]<=4096,"saved fit chunk bound differs")
            slot==1 ? (stream_chunk=manifest["max_chunk_rows"]) : need(stream_chunk==manifest["max_chunk_rows"],"paired chunk limits differ")
            for (name,value) in (("max_fit_issue_ms",last_joint.issue),("max_target_complete_ms",maximum(last_joint.target_complete)),
                    ("max_label_receipt_ms",maximum(last_joint.target_receipt)))
                need(manifest[name]==string(Dates.value(value)),"saved fit maturity clocks differ")
            end
            for (name,value) in (("joint_origins",joint),("informative_origins",informative),("storm_rows",storms),
                    ("fit_rows",2joint),("data_bytes",64joint),("total_source_rows",length(frame.input.time)),("max_source_rows",length(frame.input.time)))
                need(Q.integer(manifest[name]) && manifest[name]==value,"saved full joint fit count differs")
            end
            need(manifest["fit_mask_sha256"]==mask && manifest["fit_data_sha256"]==data &&
                 manifest["data_sha256"]==scalar_sha==expected[record["fit_input"]*"/joint-fit.f64"],"saved joint records/masks differ from raw replay")
            loaded=B.load_bundle(joinpath(root,record["bundle"]);expected_manifest_sha256=record["bundle_manifest_sha256"],expected_sources=B.verify_source_identity())
            model=loaded.model
            provenance=(stage=:initial_l0,evidence_mode=frame.evidence_mode,source_sha256=frame.input.source_sha256,
                input_prefix_sha256=frame.input_sha256,reference_completion_sha256=reference.completion_sha256,
                independent_audit_completion_sha256=reference.audit_completion_sha256,
                source_admission_seal=reference.admitted===nothing ? nothing : reference.admitted.sealed_sha256,
                training_boundary=initial,validation_start=I.datehour(meta["validation_start"]),validation_end=I.datehour(meta["validation_end"]),
                basis_sha256=expected[basis_name],fit_input_manifest_sha256=expected[fit_name],
                final_refit=false,test_scoring=false,scientific_verdict=false,serving_promotion=false)
            need(isequal(loaded.provenance,provenance),"saved initial-model provenance differs")
            need(model.setting_index==index && model.boundary==initial && model.delay_hours==0 &&
                 model.fit_count==joint && model.informative_count==informative && model.storm_count==storms &&
                 model.fit_mask_sha256==mask && model.fit_data_sha256==data && model.source_sha256==frame.input.source_sha256 &&
                 model.max_fit_issue==last_joint.issue && model.max_target_complete==maximum(last_joint.target_complete) &&
                 model.max_label_receipt==maximum(last_joint.target_receipt),"saved model fit/source/clocks differ")
            need(record["model_id"]==model.model_id && record["gains"]==collect(model.gains) &&
                 record["fit_mask_sha256"]==mask && record["fit_data_sha256"]==data,"saved model summary differs")
            for (key,value) in (("fit_count",joint),("informative_count",informative),("storm_count",storms))
                need(Q.integer(record[key]) && record[key]==value,"saved model summary counts differ")
            end
            need(isequal(B._checkpoint_data(loaded.checkpoint),B._checkpoint_data(streamed.checkpoint)),"saved checkpoint differs from raw replay")
            verify_fit_certificate(model,manifest,joinpath(fit_root,"joint-fit.f64"))
            points=T.predict_rows(model,basis);buffer=IOBuffer()
            for row in points,point in row.prediction
                write(buffer,htol(reinterpret(UInt64,point)))
            end
            point_bytes=take!(buffer);point_name=record["points"]
            need(record["points_sha256"]==expected[point_name] && read(joinpath(root,point_name))==point_bytes,
                 "saved points differ from saved model/raw basis")
            panels[index]=points
        end
        saved=meta["streams"][pair]
        need(Set(keys(saved))==Set(("setting_pair","row_count","max_buffered_rows","fallback_counts","observations_used",
             "negative_pressure_hours","reference_union_rows","basis_sha256")),"saved stream record fields differ")
        need(saved["setting_pair"]==[setting.index,setting.index+1] && saved["basis_sha256"]==expected[basis_name] &&
             saved["fallback_counts"]==streamed.fallback_counts,"saved state schedule identity/fallback counts differ")
        for (key,value) in (("row_count",streamed.row_count),("observations_used",streamed.observations_used),
                ("negative_pressure_hours",streamed.negative_pressure_hours),("reference_union_rows",length(base)))
            need(Q.integer(saved[key]) && saved[key]==value,"saved state schedule count differs")
        end
        need(Q.integer(saved["max_buffered_rows"],1) && saved["max_buffered_rows"]==min(stream_chunk,length(frame.input.time)),
             "saved stream buffer bound differs")
    end
    selection=T.select_setting(panels,reference;validation_start=I.datehour(meta["validation_start"]),
        validation_end=I.datehour(meta["validation_end"]),delay_hours=0)
    need(meta["selection_status"]==String(selection.status) && Q.integer(meta["selected_index"]) &&
         meta["selected_index"]==(selection.selected_index===nothing ? 0 : selection.selected_index) &&
         TOML.parsefile(joinpath(root,"selection.toml"))==F.DstIntervalExperiment._selection_document(selection),"saved selection differs")
    guard(frame,own,permission)
    need(stage_inventory(root).sha256==merge(expected,Dict("completion.json"=>completion_sha256)),"saved stage changed during readback")
    (complete=true,models_checked=18,state_replays=9,new_gain_fits=0,final_refits=0,selection=selection,
     completion_sha256=completion_sha256,scientific_verdict=false)
end

end # module
