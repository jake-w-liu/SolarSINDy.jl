const BASIS_MANIFEST_FIELDS=(:schema,:complete,:role,:contract_sha256,:evidence_mode,:source_sha256,:input_view_id,:input_prefix_sha256,:source_hashes,:fold_id,:boundary,:delay_hours,:validation_start,:validation_end,:training_mask_sha256,:forecast_union_sha256,:members)
const BASIS_MEMBER_FIELDS=(:name,:physical_schedule,:beta,:bytes,:sha256,:training_origins,:forecast_origins)
function _export_basis(v::T.PointInputView,laws,directory;delay_hours,chunk_hours,fold_id="initial",validation_start=laws[1].boundary,validation_end=T.Adapter._shift(last(v.input.time),1),probe=T._NO_PROBE)
    sources=source_identity();T._recheck(v);delay=T._int(delay_hours,0,1);chunk=T._int(chunk_hours,1,T.MAX_CHUNK)
    masks=T._masks(v,laws[1].boundary,validation_start,validation_end,delay)
    root=_root_new(directory);ios=Base.Filesystem.File[];completion=nothing;errors=Any[];result=nothing
    counts=zeros(Int,18)
    try
        for index in 1:18
            io=Base.Filesystem.open(_member(root,"basis-$index.bin"),Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
            push!(ios,io);_header(io,"SSBPBAS1",248)
        end
        result=T._coordinate_issue_streams(v,laws,(physical,bi,rows)->begin
            index=2*(physical-1)+bi
            for row in rows;write(ios[index],_basis_bytes(row));counts[index]+=1;end
        end;delay_hours=delay,chunk_hours=chunk,training_mask=masks.training,forecast_union_mask=masks.forecast,fold_id,validation_start,validation_end,probe)
        foreach(close,ios);empty!(ios)
        _need(all(==(count(masks.training)+count(masks.forecast)),counts),"B basis row count differs")
        members=[begin
            name="basis-$i.bin";hash,bytes=_file_sha(_member(root,name))
            _need(bytes==16+248*counts[i],"B basis byte geometry differs")
            (;name,physical_schedule=(i-1)÷2+1,beta=(5,20)[(i-1)%2+1],bytes,sha256=hash,training_origins=count(masks.training),forecast_origins=count(masks.forecast))
        end for i in 1:18]
        manifest=(;schema="b_point_basis_panels_v1",complete=true,role=:basis_only_disjoint_training_and_forecast,contract_sha256=T.CONTRACT,evidence_mode=v.evidence_mode,source_sha256=v.source_sha256,input_view_id=v.view_id,input_prefix_sha256=v.input_prefix_sha256,source_hashes=_sources_data(sources),fold_id,boundary=laws[1].boundary,delay_hours=delay,validation_start,validation_end,training_mask_sha256=result.training_bindings[1].training_mask_sha256,forecast_union_sha256=_sha(reinterpret(UInt8,collect(masks.forecast))),members)
        completion=_bytes(manifest);_exclusive(_member(root,"manifest.json"),completion)
        T._recheck(v);_need(source_identity()==sources,"B basis sources changed")
    catch e;push!(errors,e)
    finally
        for io in ios;try close(io) catch e;push!(errors,e);end;end
        try T._recheck(v);_need(source_identity()==sources,"B basis exit source differs") catch e;push!(errors,e);end
        if !isempty(errors);try _invalidate(root,"manifest.json",completion) catch e;push!(errors,e);end;end
    end
    T._raise(errors)
    (;directory=root,manifest_sha256=_sha(completion),result,masks)
end
function _a_data(cp::Stream.StreamCheckpoint)
    pending=cp.pending===nothing ? nothing : (;index=cp.pending.index,start=cp.pending.start,stop=cp.pending.stop,q=cp.pending.q,pressure=cp.pending.pressure)
    (;last_index=cp.last_index,last_issue=cp.last_issue,source_sha256=cp.source_sha256,prefix_sha256=cp.prefix_sha256,delay_hours=cp.delay_hours,taus=cp.taus,sigma=cp.sigma,sealed=(;boundary=cp.sealed.boundary,mean=cp.sealed.mean,covariance=cp.sealed.covariance),basis=cp.basis,pending,active=cp.active,consecutive_valid=cp.consecutive_valid,reset_tag=cp.reset_tag)
end
struct BVerifiedBasisSet
    directory::String
    manifest_sha256::String
    view_id::String
    source_hashes::T.HashMap
    laws::NTuple{2,D.DurationModel}
    bindings::NTuple{2,T.PointTrainingBinding}
    a_checkpoints::Vector{Stream.StreamCheckpoint}
    wind_checkpoint::C.CurrentStateCheckpoint
    training_mask::BitVector
    forecast_mask::BitVector
    data_sha256::Vector{String}
    certificates::Dict{Int,NamedTuple} # At most36 fixed settings; never an origin cache.
    verification_id::String
end
function _verified_data(p::BVerifiedBasisSet)
    (;schema="b_verified_raw_source_replay_v1",manifest_sha256=p.manifest_sha256,input_view_id=p.view_id,source_hashes=_sources_data(p.source_hashes),law_ids=(p.laws[1].model_id,p.laws[2].model_id),context_ids=(p.bindings[1].context_id,p.bindings[2].context_id),a_checkpoints=[_a_data(x) for x in p.a_checkpoints],wind_checkpoint=merge(C._checkpoint_data(p.wind_checkpoint),(checkpoint_id=p.wind_checkpoint.checkpoint_id,)),training_mask_sha256=_sha(reinterpret(UInt8,collect(p.training_mask))),forecast_mask_sha256=_sha(reinterpret(UInt8,collect(p.forecast_mask))),data_sha256=p.data_sha256)
end
function _check_verified(p::BVerifiedBasisSet,v::T.PointInputView)
    T._recheck(v);_need(p.view_id==v.view_id&&p.source_hashes==T._hashmap(source_identity())&&p.verification_id==_id(_verified_data(p)),"B raw-source replay proof differs")
    _need(_sha(_bound(_member(p.directory,"manifest.json")))==p.manifest_sha256,"B basis completion changed")
    true
end
function _verify_basis(v::T.PointInputView,laws,directory;expected_manifest_sha256,expected_sources,chunk_hours=24,probe=T._NO_PROBE)
    sources=_source(expected_sources);T._recheck(v);root=_ordinary(directory)
    _need(isdir(root)&&sort(readdir(root))==sort(vcat(["basis-$i.bin" for i in 1:18],["manifest.json"])),"B basis membership differs")
    manifest,body=_json(_member(root,"manifest.json");fields=BASIS_MANIFEST_FIELDS)
    _need(_sha(body)==expected_manifest_sha256&&manifest.schema=="b_point_basis_panels_v1"&&manifest.complete===true&&manifest.role=="basis_only_disjoint_training_and_forecast"&&manifest.contract_sha256==T.CONTRACT,"B basis completion/schema differs")
    _need(manifest.input_view_id==v.view_id&&manifest.source_sha256==v.source_sha256&&manifest.input_prefix_sha256==v.input_prefix_sha256&&_sources(manifest.source_hashes)==sources,"B basis source/view differs")
    boundary=T._clock(manifest.boundary);delay=T._int(manifest.delay_hours,0,1);start=T._clock(manifest.validation_start);stop=T._clock(manifest.validation_end)
    _need(boundary==laws[1].boundary==laws[2].boundary&&length(manifest.members)==18,"B basis law/membership differs")
    masks=T._masks(v,boundary,start,stop,delay);_need(_sha(reinterpret(UInt8,collect(masks.forecast)))==manifest.forecast_union_sha256,"B basis forecast union differs")
    ios=IOStream[];errors=Any[];result=nothing;hashes=String[]
    try
        for i in 1:18
            member=manifest.members[i];_fields(member,BASIS_MEMBER_FIELDS)
            _need(member.name=="basis-$i.bin"&&member.physical_schedule==(i-1)÷2+1&&member.beta==(5,20)[(i-1)%2+1]&&member.training_origins==count(masks.training)&&member.forecast_origins==count(masks.forecast),"B basis setting/count differs")
            hash,bytes=_file_sha(_member(root,String(member.name)))
            _need(hash==member.sha256&&bytes==member.bytes==16+248*(count(masks.training)+count(masks.forecast)),"B basis data bytes differ")
            push!(hashes,hash);io=open(_member(root,String(member.name)),"r");push!(ios,io);_check_header(io,"SSBPBAS1",248)
        end
        result=T._coordinate_issue_streams(v,laws,(physical,bi,rows)->begin
            index=2*(physical-1)+bi
            for row in rows
                expected=_basis_bytes(row);observed=read(ios[index],248)
                _need(observed==expected,"B basis differs from independent raw-source replay")
            end
        end;delay_hours=delay,chunk_hours,training_mask=masks.training,forecast_union_mask=masks.forecast,fold_id=String(manifest.fold_id),validation_start=start,validation_end=stop,probe)
        _need(all(eof,ios)&&manifest.training_mask_sha256==result.training_bindings[1].training_mask_sha256,"B replay EOF/mask differs")
    catch e;push!(errors,e)
    finally
        for io in ios;try close(io) catch e;push!(errors,e);end;end
        try T._recheck(v);_need(source_identity()==Tuple(sources)||T._hashmap(source_identity())==sources,"B replay exit sources differ") catch e;push!(errors,e);end
    end
    T._raise(errors)
    a_checkpoints=[x.checkpoint for x in result.value.streams]
    fields=(root,String(expected_manifest_sha256),v.view_id,copy(sources),laws,result.training_bindings,a_checkpoints,result.wind_checkpoint,copy(masks.training),copy(masks.forecast),hashes,Dict{Int,NamedTuple}())
    provisional=BVerifiedBasisSet(fields...,repeat("0",64));verified=BVerifiedBasisSet(fields...,_id(_verified_data(provisional)))
    _check_verified(verified,v);verified
end
struct BVerifiedBasisPanel
    replay::BVerifiedBasisSet
    physical_schedule::Int
    beta_index::Int
    training::T.BTrainingPanel
    forecast::T.BForecastPanel
    basis_sha256::String
end
function _read_verified_panel(proof::BVerifiedBasisSet,v::T.PointInputView,physical,bi)
    _check_verified(proof,v);physical=T._int(physical,1,9);bi=T._int(bi,1,2);index=2*(physical-1)+bi
    path=_member(proof.directory,"basis-$index.bin");hash,bytes=_file_sha(path);_need(hash==proof.data_sha256[index],"B verified basis data changed")
    training=T.BTrainingRow[];forecast=T.BForecastRow[];binding=proof.bindings[bi];law=proof.laws[bi]
    open(path,"r") do io
        _check_header(io,"SSBPBAS1",248)
        for i in 1:v.source_hours
            (proof.training_mask[i]||proof.forecast_mask[i])||continue
            role=proof.training_mask[i] ? :training : :forecast
            row=_read_basis_row(io,binding,law,physical,role,i)
            _need(row.design.values.issue==T.Adapter._shift(v.input.time[i],1),"B decoded issue/source index differs")
            row isa T.BTrainingRow ? push!(training,row) : push!(forecast,row)
        end
        _need(eof(io),"extra B basis row bytes")
    end
    panel=T.BTrainingPanel(training,binding,law,physical,hash,v.evidence_mode)
    result=BVerifiedBasisPanel(proof,physical,bi,panel,T.BForecastPanel(forecast,binding,law,physical),hash)
    _check_verified(proof,v);_need(_file_sha(path)==(hash,bytes),"B basis changed during decode");result
end
