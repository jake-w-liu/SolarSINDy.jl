module DstIntervalTraining

using Dates, SHA, TOML
using ..DstIntervalAdapter
using ..DstIntervalStream
const A=DstIntervalAdapter
const S=DstIntervalStream
const K=A.Kernel
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
const CONTRACT_SHA256="18bdd26ae44267f7992378b82663558af3a5af3b64d45800366c107af6201658"
S.verify_source_identity()

export fit_gains, predict_rows, select_setting, export_fit_input, fit_gains_from_input

_hash(x)=bytes2hex(sha256(repr(x)))
_valid_hash(x)=x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)

function verify_source_identity()
    dependencies=S.verify_source_identity()
    bytes2hex(sha256(read(@__FILE__)))==LOADED_SOURCE_SHA256 ||
        throw(ArgumentError("loadedsource/disk mismatch: DstIntervalTraining.jl"))
    (dependencies...,"DstIntervalTraining.jl"=>LOADED_SOURCE_SHA256)
end

function _check_setting(setting)
    # Grid membership is by value; model and bundle metadata use grid types.
    for canonical in A.settings()
        setting==canonical && return canonical
    end
    throw(ArgumentError("setting differs from frozen18grid"))
end

function _check_rows(rows,delay,source;setting=nothing)
    isempty(rows) && throw(ArgumentError("emptybasis panel"))
    issorted(getproperty.(rows,:issue)) && allunique(getproperty.(rows,:issue)) || throw(ArgumentError("issues must beordered andunique"))
    _valid_hash(source) || throw(ArgumentError("invalidsourceidentity"))
    for row in rows
        row.dst_delay_hours==delay && row.source_sha256==source || throw(ArgumentError("mixedsource/latency panel"))
        for (slot,h) in enumerate((4,6))
            clock=A.forecast_clock(row.issue,h,delay)
            (row.driver_tag,row.driver_receipt,row.anchor_tag,row.anchor_complete,row.anchor_receipt,
             row.target_tag[slot],row.target_complete[slot],row.target_receipt[slot])==
            (clock.driver_tag,row.issue,clock.anchor_tag,clock.anchor_complete,clock.anchor_receipt,
             clock.target_tag,clock.target_complete,clock.target_receipt) || throw(ArgumentError("rowclock contract mismatch"))
            (isfinite(row.target[slot]) || isnan(row.target[slot])) || throw(ArgumentError("invalidtargetvalue"))
            row.scoreable[slot]==(isfinite(row.anchor)&&isfinite(row.target[slot])) || throw(ArgumentError("scoreablecohort flag mismatch"))
        end
        if setting!==nothing
            row.taus==setting.taus && row.sigma==setting.sigma || throw(ArgumentError("mixedA dynamics panel"))
        end
        if setting!==nothing && row.model_available
            isfinite(row.anchor) && all(isfinite,row.offset) && all(isfinite,Iterators.flatten(row.coefficients)) ||
                throw(ArgumentError("nonfiniteavailableAdesign"))
        elseif setting!==nothing && isfinite(row.anchor)
            row.offset==(row.anchor,row.anchor) && row.coefficients==S._zero_basis() ||
                throw(ArgumentError("fallbackmust beexactcommonpersistence"))
        end
    end
    nothing
end

function _check_model_boundary(boundary::DateTime)
    floor(boundary,Hour)==boundary ||
        throw(ArgumentError("canonical training boundary must lie on an exact hour"))
    boundary
end

function _fit_model(solved,selected,boundary;delay_hours,source_sha256,hashes,
        fit_count,informative_count,storm_count,fit_data_sha256,fit_mask_sha256,
        max_target_complete,max_label_receipt,max_fit_issue)
    verify_source_identity()==hashes || throw(ArgumentError("sourceschanged duringmodelassembly"))
    _check_model_boundary(boundary)
    base=(schema="candidate_a_intervalmean_research_v1",evidence_mode="synthetic_or_unqualified_research",contract_sha256=CONTRACT_SHA256,
        gains=solved.gains,setting_index=selected.index,taus=selected.taus,sigma=selected.sigma,eta=selected.eta,
        delay_hours=Int(delay_hours),boundary=boundary,source_sha256=String(source_sha256),source_hashes=hashes,
        fit_count=fit_count,informative_count=informative_count,storm_count=storm_count,
        fit_data_sha256=fit_data_sha256,fit_mask_sha256=fit_mask_sha256,
        max_target_complete=max_target_complete,max_label_receipt=max_label_receipt,
        max_fit_issue=max_fit_issue,solver=solved)
    merge(base,(model_id=_hash(base),))
end

"""Fit two gains on a supplied synthetic/research affine panel at an exact-hour
boundary, as required by the canonical model/bundle contract. This performs
no raw-data I/O and does not qualify a real experiment or serving model.
"""
function fit_gains(rows,boundary::DateTime,setting;delay_hours::Integer,source_sha256::AbstractString)
    hashes=verify_source_identity()
    _check_model_boundary(boundary)
    selected=_check_setting(setting)
    delay_hours in (0,1) && !(delay_hours isa Bool) || throw(ArgumentError("unknownlatency"))
    _check_rows(rows,delay_hours,source_sha256;setting=selected)
    admitted=Int[]
    for (index,row) in enumerate(rows)
        mature=all(A.fit_eligible(row.issue,row.target_complete[j],row.target_receipt[j],boundary) for j in 1:2)
        mature && all(row.scoreable) && push!(admitted,index)
    end
    informative=count(i->rows[i].model_available && any(!iszero,Iterators.flatten(rows[i].coefficients)),admitted)
    informative>=100 || throw(ArgumentError("fewerthan100informative jointorigins"))
    design=Matrix{Float64}(undef,2length(admitted),2)
    residual=Vector{Float64}(undef,2length(admitted))
    labels=similar(residual)
    mask_hash=_hash(("A-joint-fit-mask-v1",boundary,delay_hours))
    data_hash=_hash(("A-joint-fit-data-v1",boundary,delay_hours))
    cursor=0
    for index in admitted
        row=rows[index]
        mask_hash=_hash((mask_hash,(row.issue,row.anchor_tag,row.target_tag,row.target_complete,row.target_receipt)))
        data_hash=_hash((data_hash,(row.issue,row.anchor,row.offset,row.coefficients,row.target)))
        for slot in 1:2
            cursor+=1
            design[cursor,:].=row.coefficients[slot]
            labels[cursor]=row.target[slot]
            residual[cursor]=K._number(row.target[slot]-row.offset[slot],"trainingresidual")
        end
    end
    count_all=length(residual);count_storm=count(<=( -50.),labels)
    weights=[1/count_all+(count_storm>0 && labels[i]<=-50 ? selected.eta/count_storm : 0.) for i in eachindex(labels)]
    solved=A.nnls2(design,residual;weights)
    maximum_complete=maximum(maximum(rows[i].target_complete) for i in admitted)
    maximum_receipt=maximum(maximum(rows[i].target_receipt) for i in admitted)
    verify_source_identity()==hashes || throw(ArgumentError("sourceschanged duringgainfit"))
    _fit_model(solved,selected,boundary;delay_hours,source_sha256,hashes,
        fit_count=length(admitted),informative_count=informative,storm_count=count_storm,
        fit_data_sha256=data_hash,fit_mask_sha256=mask_hash,
        max_target_complete=maximum_complete,max_label_receipt=maximum_receipt,
        max_fit_issue=maximum(rows[i].issue for i in admitted))
end

function predict_rows(model,rows)
    model.source_hashes==verify_source_identity() || throw(ArgumentError("model/loadedsource mismatch"))
    setting=A.settings()[model.setting_index]
    (model.taus,model.sigma,model.eta)==(setting.taus,setting.sigma,setting.eta) || throw(ArgumentError("modelsettings mismatch"))
    all(isfinite,model.gains) && all(>=(0),model.gains) || throw(ArgumentError("invalidmodelgains"))
    _check_rows(rows,model.delay_hours,model.source_sha256;setting)
    result=[begin
        prediction=Tuple(isfinite(row.anchor) ? (row.model_available ?
            row.offset[j]+sum(row.coefficients[j][k]*model.gains[k] for k in 1:2) : row.anchor) : NaN for j in 1:2)
        isfinite(row.anchor) && !all(isfinite,prediction) && throw(ArgumentError("nonfinitepointprediction"))
        merge(row,(prediction=prediction,setting_index=model.setting_index,eta=model.eta,model_id=model.model_id))
    end for row in rows]
    verify_source_identity()
    result
end

function _metrics(rows,indices,slot)
    isempty(indices) && return (n=0,rmse=NaN)
    value=K._precise() do
        total=zero(BigFloat)
        for index in indices
            value=rows[index].prediction[slot]
            isfinite(value) || throw(ArgumentError("predictionmissing onscoreable row"))
            total+=(BigFloat(value)-BigFloat(rows[index].target[slot]))^2
        end
        sqrt(total/length(indices))
    end
    (n=length(indices),rmse=A._nonzero_result(value,"panelRMSE"))
end

_identity(row)=(row.issue,row.driver_tag,row.anchor_tag,row.anchor,row.target_tag,row.target_complete,
    row.target_receipt,row.target,row.scoreable,row.source_sha256,row.dst_delay_hours)

function _selection_result(result,hashes)
    verify_source_identity()==hashes || throw(ArgumentError("sourceschanged duringselection"))
    result
end

"""Initial-reference-constrained A selection on supplied rows. Synthetic
reference metadata exercises this gate; realGBM lineage still requires its
separate source/model/prediction replay qualification before actual use.
"""
function select_setting(panels::AbstractDict,reference;validation_start::DateTime,
        validation_end::DateTime,delay_hours::Integer)
    reference===nothing || reference.evidence_kind==:synthetic_initial_reference ||
        throw(ArgumentError("unqualifiedreferencekind"))
    reference===nothing || reference.training_boundary==validation_start ||
        throw(ArgumentError("referenceisnotinitialprevalidationmodel"))
    _select_initial_reference(panels,reference;validation_start,validation_end,delay_hours)
end

# Shared numerical selection only. The public NamedTuple route above retains
# its synthetic guard. The separate artifact importer must verify serialized
# source/model/prediction proof before and after using this internal helper.
function _select_initial_reference(panels::AbstractDict,reference;validation_start::DateTime,
        validation_end::DateTime,delay_hours::Integer)
    hashes=verify_source_identity()
    reference===nothing && return _selection_result((status=:blocked_reference_absent,selected_index=nothing,panels=(),reference=nothing,cohort_sha256=()),hashes)
    validation_start<validation_end || throw(ArgumentError("unorderedvalidationwindow"))
    sort(collect(keys(panels)))==collect(1:18) || throw(ArgumentError("all18candidatepanels required"))
    delay_hours in (0,1) && !(delay_hours isa Bool) || throw(ArgumentError("unknownlatency"))
    reference.training_boundary<=validation_start && reference.delay_hours==delay_hours || throw(ArgumentError("referenceisnotinitialprevalidationmodel"))
    # An independently audited artifact may expose a later selection subwindow.
    # Fit-label maturity remains anchored to its original training boundary.
    cutoff=A._shift(reference.training_boundary,-168)
    reference.max_fit_target_complete isa NTuple{2,DateTime} && reference.max_fit_label_receipt isa NTuple{2,DateTime} ||
        throw(ArgumentError("referencefitclocks require exactlytwo DateTime values"))
    all(j->reference.max_fit_target_complete[j]<=reference.max_fit_label_receipt[j]<=cutoff &&
        reference.max_fit_label_receipt[j]==A._shift(reference.max_fit_target_complete[j],delay_hours),1:2) ||
        throw(ArgumentError("referencefitlabelmaturity violation"))
    _valid_hash(reference.source_sha256) || throw(ArgumentError("invalidreferencesourceidentity"))
    length(reference.model_ids)==2 && all(_valid_hash,reference.model_ids) || throw(ArgumentError("missingreferencemodelidentities"))
    baseline=reference.rows
    _check_rows(baseline,delay_hours,reference.source_sha256)
    selection_rows=[i for i in eachindex(baseline) if validation_start<=baseline[i].issue<validation_end]
    isempty(selection_rows) && throw(ArgumentError("emptyvalidationreference"))
    reference_keys=getproperty.(baseline[selection_rows],:issue)
    admitted=Tuple([i for i in selection_rows if baseline[i].scoreable[j] &&
        A.fit_eligible(baseline[i].issue,baseline[i].target_complete[j],baseline[i].target_receipt[j],validation_end)] for j in 1:2)
    all(!isempty,admitted) || throw(ArgumentError("emptyperleadvalidationcohort"))
    storm=Tuple([i for i in admitted[j] if baseline[i].target[j]<=-50] for j in 1:2)
    base_metrics=Tuple((pooled=_metrics(baseline,admitted[j],j),storm=_metrics(baseline,storm[j],j)) for j in 1:2)
    cohort=Tuple(begin
        digest=_hash(("A-validation-cohort-v1",j,delay_hours,validation_start,validation_end))
        for i in admitted[j]
            digest=_hash((digest,(baseline[i].issue,baseline[i].target_tag[j])))
        end
        digest
    end for j in 1:2)
    results=NamedTuple[]
    for index in 1:18
        candidate=panels[index];setting=A.settings()[index]
        _check_rows(candidate,delay_hours,reference.source_sha256;setting)
        all(x->x.setting_index==index && x.eta==setting.eta,candidate) || throw(ArgumentError("candidateindex/eta mismatch"))
        locations=Dict(row.issue=>i for (i,row) in enumerate(candidate))
        keys=[row.issue for row in candidate if validation_start<=row.issue<validation_end]
        keys==reference_keys || throw(ArgumentError("candidatevalidationkeyset differs"))
        for i in selection_rows
            isequal(_identity(candidate[locations[baseline[i].issue]]),_identity(baseline[i])) || throw(ArgumentError("candidate/reference rowidentity differs"))
        end
        pooled=Tuple(_metrics(candidate,[locations[baseline[i].issue] for i in admitted[j]],j) for j in 1:2)
        activity=Tuple(_metrics(candidate,[locations[baseline[i].issue] for i in storm[j]],j) for j in 1:2)
        qualifies=all(pooled[j].rmse<=base_metrics[j].pooled.rmse for j in 1:2)
        normalized=any(isempty,storm) ? NaN : sum(base_metrics[j].storm.rmse==0 ?
            (activity[j].rmse==0 ? 1. : Inf) : activity[j].rmse/base_metrics[j].storm.rmse for j in 1:2)/2
        push!(results,(index=index,pooled=pooled,storm=activity,qualifies_pooled=qualifies,normalized_storm_rmse=normalized))
    end
    any(isempty,storm) && return _selection_result((status=:blocked_storm_reference,selected_index=nothing,
        panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
    qualifiers=[x for x in results if x.qualifies_pooled && isfinite(x.normalized_storm_rmse)]
    isempty(qualifiers) && return _selection_result((status=:none,selected_index=nothing,panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
    best=first(qualifiers)
    for row in qualifiers[2:end]
        row.normalized_storm_rmse<best.normalized_storm_rmse && (best=row)
    end
    _selection_result((status=:selected,selected_index=best.index,panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
end


# The disk-backed path is part of this source-bound training implementation.
# Export buffering is bounded; the existing solver still materializes all rows.
_disk_need(ok,message)=ok || throw(ArgumentError(message))
_disk_integer(x,lo,hi)=x isa Integer && !(x isa Bool) && lo<=x<=hi
_disk_hex(x)=x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)
_disk_digest(bytes)=bytes2hex(sha256(bytes))
_disk_milliseconds(t::DateTime)=string(Dates.value(t))
_disk_datetime(s)=DateTime(Dates.UTInstant{Millisecond}(Millisecond(parse(Int64,s))))

function _disk_new_file(path,bytes)
    io=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try
        _disk_need(write(io,bytes)==length(bytes),"short owned artifact write")
    finally
        close(io)
    end
end
function _disk_toml_bytes(data)
    io=IOBuffer();TOML.print(io,data;sorted=true);take!(io)
end
function _disk_file_hash(path,limit)
    _disk_need(isfile(path) && !islink(path),"ordinary input file required")
    context=SHA.SHA2_256_CTX();count=0
    open(path,"r") do io
        while !eof(io)
            bytes=read(io,min(65536,limit-count+1));count+=length(bytes)
            _disk_need(count<=limit,"file byte budget exceeded")
            SHA.update!(context,bytes)
        end
    end
    (sha256=bytes2hex(SHA.digest!(context)),bytes=count)
end
function _disk_read_bound(path,expected,limit)
    _disk_need(_disk_hex(expected) && isfile(path) && !islink(path),"ordinary pinned input required")
    bytes=open(io->read(io,limit+1),path,"r")
    _disk_need(length(bytes)<=limit && _disk_digest(bytes)==expected,"pinned input byte/hash mismatch")
    bytes
end

"""Export joint fit records from bounded affine chunks into a new owned directory.
`produce(sink)` supplies chunks; caller owns paths throughout I/O. This research
input does not grant actual-data, selection, serving or skill qualification.
"""
function export_fit_input(produce::Function,directory::AbstractString,boundary::DateTime,setting;
        delay_hours::Integer,source_sha256::AbstractString,max_source_rows::Integer=500_000,
        max_chunk_rows::Integer=4096)
    sources=verify_source_identity()
    selected=_check_setting(setting)
    _disk_need(_disk_integer(delay_hours,0,1) && _disk_hex(source_sha256),"latency/source grammar")
    _disk_need(_disk_integer(max_source_rows,1,500_000) && _disk_integer(max_chunk_rows,1,4096),"source/chunk bounds")
    destination=abspath(directory)
    _disk_need(isdir(dirname(destination)) && !islink(dirname(destination)),"existing ordinary parent required")
    mkdir(destination) # No overwrite, including existing incomplete/foreign paths.
    path=joinpath(destination,"joint-fit.f64")
    io=nothing;completion_bytes=nothing
    total=0;joint=0;informative=0;storm=0;last_issue=nothing
    maximum_complete=nothing;maximum_receipt=nothing;maximum_issue=nothing
    mask_hash=_hash(("A-joint-fit-mask-v1",boundary,Int(delay_hours)))
    data_hash=_hash(("A-joint-fit-data-v1",boundary,Int(delay_hours)))
    try
        io=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
        function sink(rows)
            _disk_need(rows isa AbstractVector && 1<=length(rows)<=max_chunk_rows,"bounded nonempty chunk required")
            _disk_need(length(rows)<=max_source_rows-total,"source row budget exceeded")
            for row in rows
                _disk_need(all(length(getproperty(row,key))==2 for key in
                    (:target_tag,:target_complete,:target_receipt,:target,:scoreable,:offset,:coefficients)),
                    "exactly two horizon entries required")
                _disk_need(all(length(coefficients)==2 for coefficients in row.coefficients),
                    "exactly two affine coefficients per horizon required")
            end
            _check_rows(rows,delay_hours,source_sha256;setting=selected)
            _disk_need(last_issue===nothing || first(rows).issue>last_issue,"cross-chunk issue ordering")
            total+=length(rows);last_issue=last(rows).issue
            buffer=IOBuffer()
            for row in rows
                mature=all(A.fit_eligible(row.issue,row.target_complete[j],row.target_receipt[j],boundary) for j in 1:2)
                mature && all(row.scoreable) || continue
                joint+=1
                informative+=row.model_available && any(!iszero,Iterators.flatten(row.coefficients))
                mask_hash=_hash((mask_hash,(row.issue,row.anchor_tag,row.target_tag,row.target_complete,row.target_receipt)))
                data_hash=_hash((data_hash,(row.issue,row.anchor,row.offset,row.coefficients,row.target)))
                complete=maximum(row.target_complete);received=maximum(row.target_receipt)
                maximum_complete=maximum_complete===nothing ? complete : max(maximum_complete,complete)
                maximum_receipt=maximum_receipt===nothing ? received : max(maximum_receipt,received)
                maximum_issue=maximum_issue===nothing ? row.issue : max(maximum_issue,row.issue)
                for slot in 1:2
                    target=K._number(row.target[slot],"disk raw label")
                    residual=K._number(row.target[slot]-row.offset[slot],"disk training residual")
                    storm+=target<=-50.
                    values=(K._number(row.coefficients[slot][1],"disk design"),
                            K._number(row.coefficients[slot][2],"disk design"),residual,target)
                    for value in values
                        write(buffer,htol(reinterpret(UInt64,value)))
                    end
                end
            end
            bytes=take!(buffer)
            _disk_need(length(bytes)<=64max_chunk_rows,"chunk serialization byte bound")
            _disk_need(write(io,bytes)==length(bytes),"short fit-record write")
            nothing
        end
        produce(sink)
        close(io);io=nothing
        _disk_need(informative>=100 && 100<=joint<=total,"fewer than100 informative joint origins")
        checked=_disk_file_hash(path,64max_source_rows)
        _disk_need(checked.bytes==64joint,"serialized fit record size mismatch")
        _disk_need(verify_source_identity()==sources,"source drift during fit export")
        manifest=Dict("schema"=>"disk_affine_gain_input_v1","complete"=>true,
            "evidence_mode"=>"unqualified_research_fit_input","contract_sha256"=>CONTRACT_SHA256,
            "format"=>"little_endian_float64_c1_c2_residual_label_h4_then_h6_v1",
            "boundary_ms"=>_disk_milliseconds(boundary),"delay_hours"=>Int(delay_hours),
            "source_sha256"=>String(source_sha256),"source_hashes"=>Dict(sources),
            "setting_index"=>selected.index,"max_source_rows"=>Int(max_source_rows),
            "max_chunk_rows"=>Int(max_chunk_rows),"total_source_rows"=>total,
            "joint_origins"=>joint,"informative_origins"=>informative,"fit_rows"=>2joint,
            "storm_rows"=>storm,"data_bytes"=>checked.bytes,"data_sha256"=>checked.sha256,
            "fit_mask_sha256"=>mask_hash,"fit_data_sha256"=>data_hash,
            "max_target_complete_ms"=>_disk_milliseconds(maximum_complete),
            "max_label_receipt_ms"=>_disk_milliseconds(maximum_receipt),"max_fit_issue_ms"=>_disk_milliseconds(maximum_issue))
        payload=_disk_toml_bytes(manifest)
        completion_bytes=payload
        _disk_new_file(joinpath(destination,"manifest.toml"),payload)
        _disk_need(verify_source_identity()==sources,"source drift at export completion")
        (directory=destination,manifest_sha256=_disk_digest(payload),joint_origins=joint,informative_origins=informative)
    catch error
        io===nothing || close(io)
        # This component never reuses or removes a caller's existing directory.
        completion=joinpath(destination,"manifest.toml")
        if completion_bytes!==nothing && isfile(completion) && read(completion)==completion_bytes
            mv(completion,completion*".incomplete";force=false)
        end
        try
            _disk_new_file(joinpath(destination,"failed.toml"),_disk_toml_bytes(Dict("complete"=>false,"error"=>sprint(showerror,error))))
        catch
            # Preserve the original export failure if its diagnostic cannot be written.
        end
        rethrow()
    end
end

"""Fit the unchanged full-matrix solver from a caller-pinned research input.
Numeric buffers are explicitly bounded by max_source_rows; no constant-memory
SVD, actual archive admission or forecasting superiority is claimed.
"""
function _solve_fit_input(directory::AbstractString;expected_manifest_sha256::AbstractString,
        require_canonical_boundary::Bool=false)
    sources=verify_source_identity();root=abspath(directory)
    _disk_need(isdir(root) && !islink(root),"ordinary fit directory required")
    manifest_bytes=_disk_read_bound(joinpath(root,"manifest.toml"),expected_manifest_sha256,1_048_576)
    m=TOML.parse(String(copy(manifest_bytes)))
    expected_keys=Set(("schema","complete","evidence_mode","contract_sha256","format","boundary_ms",
        "delay_hours","source_sha256","source_hashes","setting_index","max_source_rows","max_chunk_rows",
        "total_source_rows","joint_origins","informative_origins","fit_rows","storm_rows","data_bytes",
        "data_sha256","fit_mask_sha256","fit_data_sha256","max_target_complete_ms","max_label_receipt_ms","max_fit_issue_ms"))
    _disk_need(Set(keys(m))==expected_keys,"fit manifest field set mismatch")
    _disk_need(m["schema"]=="disk_affine_gain_input_v1" && m["complete"]===true &&
         m["evidence_mode"]=="unqualified_research_fit_input" && m["contract_sha256"]==CONTRACT_SHA256,"fit schema/evidence mismatch")
    _disk_need(m["format"]=="little_endian_float64_c1_c2_residual_label_h4_then_h6_v1","fit scalar format mismatch")
    _disk_need(Dict(sources)==m["source_hashes"] && _disk_hex(m["source_sha256"]),"fit source identity mismatch")
    _disk_need(_disk_integer(m["delay_hours"],0,1) && _disk_integer(m["setting_index"],1,18),"fit latency/recipe")
    _disk_need(_disk_integer(m["max_source_rows"],100,500_000) && _disk_integer(m["max_chunk_rows"],1,4096),"fit declared limits")
    _disk_need(_disk_integer(m["total_source_rows"],100,m["max_source_rows"]) &&
         _disk_integer(m["joint_origins"],100,m["total_source_rows"]) &&
         _disk_integer(m["informative_origins"],100,m["joint_origins"]),"fit origin counts")
    count=m["fit_rows"]
    _disk_need(_disk_integer(count,200,1_000_000) && count==2*m["joint_origins"] &&
         m["data_bytes"]==32*count && _disk_integer(m["storm_rows"],0,count),"fit scalar counts/bytes")
    _disk_need(all(_disk_hex(m[key]) for key in ("data_sha256","fit_mask_sha256","fit_data_sha256")),"fit _disk_digest grammar")
    boundary=_disk_datetime(m["boundary_ms"])
    require_canonical_boundary && _check_model_boundary(boundary)
    cutoff=A._shift(boundary,-168)
    maximum_issue=_disk_datetime(m["max_fit_issue_ms"])
    maximum_complete=_disk_datetime(m["max_target_complete_ms"])
    maximum_receipt=_disk_datetime(m["max_label_receipt_ms"])
    _disk_need(maximum_complete==A._shift(maximum_issue,6) &&
         maximum_receipt==A._shift(maximum_complete,m["delay_hours"]) &&
         maximum_complete<=maximum_receipt<=cutoff && maximum_issue<boundary,"fit maturity bound")
    bytes=_disk_read_bound(joinpath(root,"joint-fit.f64"),m["data_sha256"],32count)
    _disk_need(length(bytes)==32count,"exact fit file length required")
    words=reinterpret(UInt64,bytes)
    design=Matrix{Float64}(undef,count,2);rhs=Vector{Float64}(undef,count)
    weights=Vector{Float64}(undef,count);actual_storm=0;actual_informative=0
    selected=A.settings()[m["setting_index"]]
    for row in 1:count
        offset=4(row-1)
        a,b,residual,label=Tuple(reinterpret(Float64,ltoh(words[offset+j])) for j in 1:4)
        _disk_need(all(isfinite,(a,b,residual,label)),"nonfinite fit record")
        design[row,1]=a;design[row,2]=b;rhs[row]=residual
        actual_storm+=label<=-50.
        weights[row]=1/count+(m["storm_rows"]>0 && label<=-50. ? selected.eta/m["storm_rows"] : 0.)
        if iseven(row)
            actual_informative+=any(!iszero,view(design,row-1:row,:))
        end
    end
    _disk_need(actual_storm==m["storm_rows"] && actual_informative==m["informative_origins"],"fit label/informative recount differs")
    solved=A.nnls2(design,rhs;weights)
    checked=_disk_file_hash(joinpath(root,"joint-fit.f64"),32count)
    _disk_need(checked.sha256==m["data_sha256"] && checked.bytes==32count,"fit file changed during solve")
    _disk_need(_disk_read_bound(joinpath(root,"manifest.toml"),expected_manifest_sha256,1_048_576)==manifest_bytes,"fit manifest changed during solve")
    _disk_need(verify_source_identity()==sources,"fit source changed during solve")
    (schema="disk_affine_gain_research_result_v1",evidence_mode="unqualified_research_fit",
     source_sha256=m["source_sha256"],setting_index=m["setting_index"],delay_hours=m["delay_hours"],
     boundary=boundary,gains=solved.gains,solver=solved,fit_count=m["joint_origins"],
     informative_count=actual_informative,storm_count=actual_storm,
     fit_mask_sha256=m["fit_mask_sha256"],fit_data_sha256=m["fit_data_sha256"],
     input_manifest_sha256=String(expected_manifest_sha256),source_hashes=sources,
     max_target_complete=_disk_datetime(m["max_target_complete_ms"]),max_label_receipt=_disk_datetime(m["max_label_receipt_ms"]),
     max_fit_issue=_disk_datetime(m["max_fit_issue_ms"]))
end


"""Fit the canonical research model from an explicitly pinned, disk-backed
affine input with an exact-hour training boundary. The full-matrix solver,
joint maturity mask and storm weights are
the same as `fit_gains`. This does not qualify actual-data skill or serving.
"""
function fit_gains_from_input(directory::AbstractString;expected_manifest_sha256::AbstractString)
    result=_solve_fit_input(directory;expected_manifest_sha256,require_canonical_boundary=true)
    selected=A.settings()[result.setting_index]
    _fit_model(result.solver,selected,result.boundary;
        delay_hours=result.delay_hours,source_sha256=result.source_sha256,
        hashes=result.source_hashes,fit_count=result.fit_count,
        informative_count=result.informative_count,storm_count=result.storm_count,
        fit_data_sha256=result.fit_data_sha256,fit_mask_sha256=result.fit_mask_sha256,
        max_target_complete=result.max_target_complete,max_label_receipt=result.max_label_receipt,
        max_fit_issue=result.max_fit_issue)
end

end # module
