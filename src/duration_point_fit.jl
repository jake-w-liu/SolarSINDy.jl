_informative(row::BTrainingRow)=row.design.values.model_available&&any(!=(0.),Iterators.flatten(row.design.values.coefficients))
function _fit_mask_bytes(panel::BTrainingPanel)
    b=panel.binding;io=IOBuffer()
    write(io,"B-point-joint-mask-v1|$(b.fold_id)|$(Dates.value(b.boundary))|$(b.delay_hours)\n")
    for row in panel.rows
        issue=row.design.values.issue;a=Adapter.forecast_clock(issue,4,b.delay_hours);c=Adapter.forecast_clock(issue,6,b.delay_hours)
        write(io,join(Dates.value.((issue,a.anchor_tag,a.target_tag,a.target_complete,a.target_receipt,c.target_tag,c.target_complete,c.target_receipt)),"|")*"\n")
    end
    take!(io)
end
function _fit_data_bytes(panel::BTrainingPanel)
    io=IOBuffer();write(io,"SSBPFIT1");write(io,htol(UInt32(1)));write(io,htol(UInt32(64)))
    for row in panel.rows,k in 1:2
        d=row.design.values
        for x in (d.coefficients[k]...,row.target[k]-d.offsets[k],row.target[k])
            _need(isfinite(x),"nonfinite B fit scalar")
            write(io,htol(reinterpret(UInt64,x)))
        end
    end
    take!(io)
end
function _fit_problem(panel::BTrainingPanel,s::BSetting;minimum=100)
    _setting(s);b=panel.binding;n=length(panel.rows)
    _need(1<=n<=MAX_ROWS&&panel.physical_schedule==s.physical_schedule&&panel.law.beta==s.beta&&panel.law.model_id==b.hazard_model_id,"B fitting panel/setting/law differs")
    _need(b.context_id==_id(_binding_data(b))&&b.contract_sha256==CONTRACT&&b.source_hashes==_hashmap(source_identity()),"B fit binding identity differs")
    _need(panel.evidence_mode in (:synthetic_only,:archival_assumed_completion_research),"B fit evidence differs")
    issues=DateTime[];informative=0;storm=0;Z=Matrix{Float64}(undef,2n,2);rhs=Vector{Float64}(undef,2n);labels=similar(rhs)
    for (i,row) in enumerate(panel.rows)
        d=row.design.values
        _need(d.context_id==b.context_id&&d.law_id==panel.law.model_id&&isfinite(d.anchor)&&all(isfinite,row.target),"B fit row context/labels differ")
        _need(isempty(issues)||last(issues)<d.issue,"B fit rows must be strictly ordered")
        for k in 1:2
            clock=Adapter.forecast_clock(d.issue,(4,6)[k],b.delay_hours)
            _need(Adapter.fit_eligible(d.issue,clock.target_complete,clock.target_receipt,b.boundary),"B target immature at cutoff")
            index=2i-2+k
            for j in 1:2;Z[index,j]=d.coefficients[k][j];end
            rhs[index]=row.target[k]-d.offsets[k];labels[index]=row.target[k]
            _need(all(isfinite,(Z[index,1],Z[index,2],rhs[index],labels[index])),"B fit scalar range differs")
            storm+=labels[index]<=-50
        end
        push!(issues,d.issue);informative+=_informative(row)
    end
    _need(_sha(_fit_mask_bytes(panel))==b.training_mask_sha256,"B fitted mask is not the complete bound joint mask")
    _need(informative>=minimum,"B gain fit requires100 informative joint origins")
    weights=[1/(2n)+(labels[i]<=-50&&storm>0 ? s.eta/storm : 0.) for i in 1:2n]
    (;Z,rhs,weights,labels,informative,storm,issues)
end

# Independent readback of the complete serialized certificate. No solve occurs.
function _verify_solver(result,Z,rhs,weights)
    expected=(:gains,:weighted_sse,:normalized_sse,:normalization_scale,:rank,:rank_cutoff,:selected_candidate,:candidates)
    _need(keys(result)==expected&&length(result.candidates)==4,"B NNLS certificate shape differs")
    a=Matrix{Float64}(undef,size(Z));b=similar(rhs)
    for i in eachindex(rhs)
        root=sqrt(weights[i]);b[i]=rhs[i]*root
        for j in 1:2;a[i,j]=Z[i,j]*root;end
    end
    scale=max(maximum(abs,a),maximum(abs,b));_need(isequal(scale,result.normalization_scale),"B NNLS scale differs")
    if scale!=0;a ./=scale;b ./=scale;end
    singular=svd(a;full=false).S;largest=maximum(singular);cutoff=max(size(a,1),2)*eps(Float64)*largest
    _need(count(>(cutoff),singular)==result.rank&&isequal(cutoff,result.rank_cutoff),"B NNLS rank certificate differs")
    _need(result.candidates[1].gains==(0.,0.)&&result.candidates[2].gains[2]==0&&result.candidates[3].gains[1]==0,"B NNLS active-face certificate differs")
    K._precise() do
        losses=BigFloat[];norms=BigFloat[];valid=Int[]
        for (ci,candidate) in enumerate(result.candidates)
            _need(keys(candidate)==(:gains,:feasible,:kkt,:normalized_sse,:gradient,:kkt_bound),"B NNLS candidate fields differ")
            gain=BigFloat.(candidate.gains);loss=BigFloat(0);gradient=[BigFloat(0),BigFloat(0)]
            for i in eachindex(b)
                z1,z2=BigFloat(a[i,1]),BigFloat(a[i,2]);res=z1*gain[1]+z2*gain[2]-BigFloat(b[i])
                loss+=res*res;gradient[1]+=z1*res;gradient[2]+=z2*res
            end
            bound=128*BigFloat(eps(Float64))*max(size(a,1),2)*BigFloat(largest)*(BigFloat(largest)*sqrt(sum(abs2,gain))+BigFloat(norm(b)))
            feasible=all(>=(0),candidate.gains)
            kkt=feasible&&all(candidate.gains[j]==0 ? gradient[j]>=-bound : abs(gradient[j])<=bound for j in 1:2)
            _need(parse(BigFloat,candidate.normalized_sse)==loss&&Tuple(parse.(BigFloat,candidate.gradient))==Tuple(gradient)&&parse(BigFloat,candidate.kkt_bound)==bound,"B NNLS objective/gradient/bound differs")
            _need(candidate.feasible==feasible&&candidate.kkt==kkt,"B NNLS feasible/KKT flag differs")
            push!(losses,loss);push!(norms,sum(abs2,gain));kkt&&push!(valid,ci)
        end
        _need(!isempty(valid),"B NNLS certificate has no KKT candidate")
        best=first(valid)
        for ci in valid[2:end]
            tolerance=64*BigFloat(eps(Float64))*max(losses[1],losses[ci],losses[best])
            if losses[ci]<losses[best]-tolerance||(abs(losses[ci]-losses[best])<=tolerance&&(norms[ci],result.candidates[ci].gains)<(norms[best],result.candidates[best].gains));best=ci;end
        end
        _need(result.selected_candidate==best&&result.gains==result.candidates[best].gains,"B NNLS selected candidate differs")
        _need(isequal(result.normalized_sse,Float64(losses[best]))&&isequal(result.weighted_sse,Float64(losses[best]*BigFloat(scale)^2)),"B NNLS final losses differ")
    end
    true
end
const MODEL_FIELDS=(:schema,:evidence_mode,:training_role,:contract_sha256,:fold_id,:setting_index,:a_setting_index,:beta,:taus,:sigma,:eta,:delay_hours,:boundary,:cutoff,:gains,:hazard_model_id,:hazard_training_prefix_sha256,:source_sha256,:input_prefix_sha256,:input_view_id,:input_admission_sha256,:context_id,:source_hashes,:fit_count,:informative_count,:storm_scalar_count,:fit_mask_sha256,:fit_data_sha256,:fit_input_manifest_sha256,:max_fit_issue,:max_target_complete,:max_label_receipt,:solver,:model_id)
function _model_data(m::BPointModel;identity=false)
    names=identity ? MODEL_FIELDS : MODEL_FIELDS[1:end-1]
    NamedTuple{names}(Tuple(k==:source_hashes ? _hashdata(m.source_hashes) : getproperty(m,k) for k in names))
end
function _validate_model(m::BPointModel)
    _need(m.schema=="b_duration_point_model_v1"&&m.training_role==:in_sample_two_stage_training&&m.contract_sha256==CONTRACT,"B point model schema/role differs")
    _need(m.source_hashes==_hashmap(source_identity())&&m.model_id==_id(_model_data(m)),"B point model source/identity differs")
    s=settings()[_int(m.setting_index,1,36)]
    _need((m.a_setting_index,m.beta,m.taus,m.sigma,m.eta)==(s.a_index,s.beta,s.taus,s.sigma,s.eta)&&m.delay_hours in (0,1),"B point model setting differs")
    _need(m.cutoff==Adapter._shift(m.boundary,-168)&&m.law.boundary==m.boundary&&m.law.model_id==m.hazard_model_id&&m.law.training_prefix_sha256==m.hazard_training_prefix_sha256&&m.law.source_sha256==m.source_sha256,"B point model actual law differs")
    D._validate(m.law)
    _need(all(isfinite,m.gains)&&all(>=(0),m.gains)&&m.fit_count>=m.informative_count>=100&&0<=m.storm_scalar_count<=2m.fit_count,"B point model fit count/gain differs")
    _need(m.max_fit_issue<m.boundary&&m.max_target_complete<=m.cutoff&&m.max_label_receipt<=m.cutoff,"B point model maturity differs")
    _need(all(_hex,(m.source_sha256,m.input_prefix_sha256,m.input_view_id,m.input_admission_sha256,m.context_id,m.fit_mask_sha256,m.fit_data_sha256,m.fit_input_manifest_sha256,m.model_id)),"B point model digest differs")
    true
end
function _fit_point_gains(panel::BTrainingPanel,s::BSetting;fit_input_manifest_sha256=nothing)
    hashes=source_identity();D._validate(panel.law);problem=_fit_problem(panel,s)
    data=_fit_data_bytes(panel);mask=_fit_mask_bytes(panel);b=panel.binding
    manifest=fit_input_manifest_sha256===nothing ? _id((;schema="b_point_in_memory_fit_v1",context_id=b.context_id,setting_index=s.index,fit_data_sha256=_sha(data))) : String(fit_input_manifest_sha256)
    _need(_hex(manifest),"B fit manifest digest differs")
    solver=Adapter.nnls2(problem.Z,problem.rhs;weights=problem.weights)
    _verify_solver(solver,problem.Z,problem.rhs,problem.weights)
    lastissue=last(problem.issues);clock=Adapter.forecast_clock(lastissue,6,b.delay_hours)
    fields=("b_duration_point_model_v1",panel.evidence_mode,:in_sample_two_stage_training,CONTRACT,b.fold_id,s.index,s.a_index,s.beta,s.taus,s.sigma,s.eta,b.delay_hours,b.boundary,b.cutoff,solver.gains,panel.law.model_id,panel.law.training_prefix_sha256,b.source_sha256,b.input_prefix_sha256,b.input_view_id,b.input_admission_sha256,b.context_id,copy(b.source_hashes),length(panel.rows),problem.informative,problem.storm,_sha(mask),_sha(data),manifest,lastissue,clock.target_complete,clock.target_receipt,solver)
    provisional=BPointModel(fields...,repeat("0",64),panel.law)
    model=BPointModel(fields...,_id(_model_data(provisional)),panel.law)
    _validate_model(model);_need(source_identity()==hashes,"B sources changed during gain fit");model
end
fit_point_gains(panel::BTrainingPanel,s::BSetting)=_fit_point_gains(panel,s)
function predict_point_rows(model::BPointModel,panel::BForecastPanel)
    _validate_model(model);b=panel.binding
    _need(panel.law.model_id==model.hazard_model_id&&b.source_sha256==model.source_sha256&&b.input_view_id==model.input_view_id&&b.context_id==model.context_id&&panel.physical_schedule==settings()[model.setting_index].physical_schedule,"B prediction panel/source/law differs")
    result=BPointPredictionRow[];previous=nothing
    for row in panel.rows
        d=row.design.values
        _need(d.context_id==model.context_id&&d.law_id==model.hazard_model_id&&d.issue>=model.boundary&&d.issue>=panel.law.boundary&&(previous===nothing||previous<d.issue),"B forecast role/clock/order differs")
        means=ntuple(k->d.offsets[k]+sum(d.coefficients[k][j]*model.gains[j] for j in 1:2),2)
        _need(all(isfinite,means),"nonfinite B point prediction")
        push!(result,BPointPredictionRow(d.issue,means,d.anchor,row.target,d.model_available,d.model_available ? :available : d.a_reason!=:model ? d.a_reason : d.b_reason,b.source_sha256,b.delay_hours));previous=d.issue
    end
    _validate_model(model);result
end
