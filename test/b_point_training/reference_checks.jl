const GRID=((4,200,64),(4,200,255),(4,400,64),(4,400,255),(6,200,64),(6,200,255),(6,400,64),(6,400,255))
const BASELINE_SOURCES=["src/synthetic_matched_baseline.jl"=>T._sha("generated component source"),"contracts/features.json"=>T._sha("generated feature contract"),"contracts/loss.json"=>T._sha("generated maintained metric contract")]
function view_fixture(zero=false)
    t0=DateTime(1900);n=1024;raw=ones(17,n)
    for i in 1:n;raw[6,i]=isodd(i) ? -2. : 1.;raw[8,i]=400.;raw[15,i]=zero ? 0. : -60. + i%7;end
    source=T._sha(reinterpret(UInt8,vec(raw)))
    input=T.A.HourlyInputs([t0+Hour(i) for i in 0:n-1],raw;source_sha256=source)
    wind=[T.D.WindHour(t,t+Hour(1),t+Hour(1),raw[6,i],max(400. * max(-raw[6,i],0.)/1000-.49,0.)) for (i,t) in enumerate(input.time)]
    T._point_input_view(input,wind,(;evidence_mode=:synthetic_only,input_admission_sha256=T._sha("generated independent admission"),contract_sha256=T.CONTRACT))
end
hash(path)=P._file_sha(path)[1]
function identified(data,key)
    merge(data,NamedTuple{(key,)}((T._id(data),)))
end
function write_reference(root,v;delay=0,variant=:good,kind=:matched_information_gbm_initial_v1)
    mkdir(root);matched=kind==:matched_information_gbm_initial_v1
    matched&&(mkpath(joinpath(root,"candidates/h4"));mkpath(joinpath(root,"candidates/h6")))
    start=first(v.input.time);boundary=start+Hour(512);stop=start+Hour(1024)
    # Independent integer-clock membership, including the terminal two h4-only origins.
    train=ntuple(k->[i for i in 1:1024 if i-delay>=1&&i+(4,6)[k]+delay<=512-168],2)
    leads=ntuple(k->[i for i in 512:1024 if i+(4,6)[k]+delay<=1024-168],2)
    union_indices=sort(union(leads...));@test length(union_indices)==341-delay
    issue(i)=start+Hour(i)
    mask(indices,h,role)=T._id((;schema="b_reference_per_lead_mask_v1",source_sha256=v.source_sha256,input_view_id=v.view_id,role,horizon=h,delay_hours=delay,training_boundary=boundary,validation_start=boundary,validation_end=stop,clocks=[(;issue=issue(i),target_complete=issue(i)+Hour(h),target_receipt=issue(i)+Hour(h+delay)) for i in indices]))
    feature=Dict(BASELINE_SOURCES)["contracts/features.json"];loss=Dict(BASELINE_SOURCES)["contracts/loss.json"]
    qualified_head=repeat("a",40)
    qualification=(;schema="b_reference_component_software_qualification_v1",evidence_mode=:synthetic_only,head=qualified_head,complete=true,harness=(;counts=(;PASS=438,WARN=3,FAIL=0),all17_independent_workers_complete=true,unfiltered_pkg_test_complete=true,experiments_entry_complete=true),native_jobs=[(;name="Julia $v / $os",conclusion="success",receipt_sha256=T._sha("synthetic gate receipt $v/$os"),log_sha256=T._sha("synthetic gate log $v/$os")) for v in ("1.12.6","1.13.1") for os in ("ubuntu-latest","windows-latest","macos-latest")],docs_and_containers=[(;name,conclusion="success",receipt_sha256=T._sha("synthetic gate receipt $name"),log_sha256=T._sha("synthetic gate log $name")) for name in ("Documentation build","Julia 1.12 deployment images")],source_sha256=Dict(BASELINE_SOURCES),root_qualification_sha256=T._sha("synthetic root gate verification"),source_proof_sha256=T._sha("synthetic exact source proof"),canonical_harness_sha256=T._sha("synthetic harness raw receipt"),software_qualification_only=true,actual_panel_fit_or_outcome_qualified=false,serving_promotion=false)
    if variant==:qualification_head;qualification=merge(qualification,(;head=repeat("b",40)));end
    if variant==:qualification_float_FAIL;qualification=merge(qualification,(;harness=merge(qualification.harness,(;counts=(;PASS=438,WARN=3,FAIL=0.)))));end
    if variant==:qualification_bool_FAIL;qualification=merge(qualification,(;harness=merge(qualification.harness,(;counts=(;PASS=438,WARN=3,FAIL=false)))));end
    P._exclusive(joinpath(root,"qualification.json"),collect(codeunits(JSON3.write(qualification)*"\n")));qpin=hash(joinpath(root,"qualification.json"))
    common=(;complete=true,evidence_mode=:synthetic_only,reference_kind=kind,source_sha256=v.source_sha256,input_prefix_sha256=v.input_prefix_sha256,input_view_id=v.view_id,delay_hours=delay,training_boundary=boundary,validation_start=boundary,validation_end=stop,training_history_start=start,feature_contract_sha256=feature,tuning_loss_contract_sha256=loss,baseline_qualification_sha256=qpin,baseline_sources=T._hashdata(T._hashmap(BASELINE_SOURCES)))
    errors=([3.,2.,2.,4.,5.,6.,7.,8.],[4.,3.,1.,1.,6.,7.,8.,9.])
    selected=[2,3]
    if variant in (:near_tie_stored,:exact_tie_wrong)
        errors=([nextfloat(1.),1.,3.,4.,5.,6.,7.,8.],[4.,3.,1.,1.,6.,7.,8.,9.]);selected=[2,3]
    end
    if variant==:exact_tie_wrong;errors=([1.,1.,3.,4.,5.,6.,7.,8.],errors[2]);end
    chosen=ntuple(k->[v.input.values[15,i+(4,6)[k]]+errors[k][selected[k]] for i in union_indices],2)
    rows=[T.BPointPredictionRow(issue(i),(chosen[1][j],chosen[2][j]),v.input.values[15,i-delay],ntuple(k->v.input.values[15,i+(4,6)[k]],2),true,:admitted_reference_prediction,v.source_sha256,delay) for (j,i) in enumerate(union_indices)]
    io=IOBuffer();P._header(io,"SSBPPTS1",16);for row in rows,x in row.means;P._put_float(io,x);end
    P._exclusive(joinpath(root,"predictions.bin"),take!(io));ppin=hash(joinpath(root,"predictions.bin"))
    modelids=[T._sha("synthetic model $h/$(selected[k])") for (k,h) in enumerate((4,6))]
    gridpin=T._sha("separate original diagnostic grid");tuningpin=T._sha("separate original diagnostic tuning")
    if matched
        settings=[(;index=i,max_depth=g[1],nrounds=g[2],nbins=g[3]) for (i,g) in enumerate(GRID)]
        if variant==:wrong_grid;settings[1]=merge(settings[1],(;max_depth=999));end
        grid=identified((;schema="fixed_eight_gbm_settings_v2",complete=true,evidence_mode=:synthetic_only,feature_contract_sha256=feature,settings),:grid_id)
        P._exclusive(joinpath(root,"grid.json"),T._bytes(grid));gridpin=hash(joinpath(root,"grid.json"))
        leadsdata=NamedTuple[]
        for k in 1:2
            h=(4,6)[k];candidates=NamedTuple[]
            for s in 1:8
                name="candidates/h$h/setting-$s.bin";io=IOBuffer();P._header(io,"SSBRFPR1",8)
                for i in union_indices;P._put_float(io,v.input.values[15,i+h]+errors[k][s]);end
                P._exclusive(joinpath(root,name),take!(io));g=GRID[s];e=errors[k][s]
                stored_rmse=variant==:near_tie_stored&&k==1&&s==1 ? 1. : abs(e)
                push!(candidates,(;index=s,max_depth=g[1],nrounds=g[2],nbins=g[3],model_id=T._sha("synthetic model $h/$s"),fit_input_sha256=T._sha("synthetic fixed training matrix $h"),prediction_member=name,prediction_bytes=16+8length(union_indices),prediction_sha256=hash(joinpath(root,name)),n=length(leads[k]),rmse_nt=stored_rmse,mae_nt=abs(e),bias_nt=e))
            end
            push!(leadsdata,(;horizon=h,training_mask_sha256=mask(train[k],h,:training),validation_cohort_sha256=mask(leads[k],h,:validation),validation_count=length(leads[k]),max_fit_target_complete=issue(last(train[k]))+Hour(h),max_fit_label_receipt=issue(last(train[k]))+Hour(h+delay),max_selection_target_complete=issue(last(leads[k]))+Hour(h),max_selection_label_receipt=issue(last(leads[k]))+Hour(h+delay),candidates,selected_index=selected[k],selected_model_id=modelids[k]))
        end
        tuning=identified(merge((;schema="b_matched_reference_tuning_v2"),common,(;grid_sha256=gridpin,lead_order=[4,6],leads=leadsdata)),:tuning_id)
        P._exclusive(joinpath(root,"tuning.json"),T._bytes(tuning));tuningpin=hash(joinpath(root,"tuning.json"))
    end
    audit=identified(merge((;schema=matched ? "b_matched_reference_audit_v2" : "b_original_reference_audit_v1"),common,(;audit_contract_sha256=T._sha("synthetic independently pinned audit contract"),raw_row_identity_sha256=S._row_identity(rows),prediction_panel_sha256=ppin,grid_sha256=gridpin,tuning_sha256=tuningpin,all_eight_settings_checked=matched,full_per_lead_cohorts_checked=true,feature_receipt_history_checked=true,selected_prediction_columns_checked=true,initial_only=true)),:audit_id)
    P._exclusive(joinpath(root,"audit.json"),T._bytes(audit));apin=hash(joinpath(root,"audit.json"))
    names=matched ? vcat(["audit.json","grid.json","tuning.json","predictions.bin","qualification.json"],["candidates/h$h/setting-$s.bin" for h in (4,6) for s in 1:8]) : ["audit.json","predictions.bin","qualification.json"]
    members=[(;name,bytes=filesize(joinpath(root,name)),sha256=hash(joinpath(root,name))) for name in sort(names)]
    completion=identified(merge((;schema=matched ? "b_matched_reference_completion_v2" : "b_original_reference_completion_v1"),common,(;audit_contract_sha256=audit.audit_contract_sha256,model_ids=modelids,selected_setting_ids=selected,raw_row_identity_sha256=S._row_identity(rows),prediction_panel_sha256=ppin,grid_sha256=gridpin,tuning_sha256=tuningpin,audit_sha256=apin,members)),:completion_id)
    P._exclusive(joinpath(root,"completion.json"),T._bytes(completion))
    pins=I.BReferencePins(;completion_sha256=hash(joinpath(root,"completion.json")),audit_sha256=apin,grid_sha256=gridpin,tuning_sha256=tuningpin,prediction_sha256=ppin,feature_contract_sha256=feature,tuning_loss_contract_sha256=loss,baseline_qualification_sha256=qpin,baseline_sources=BASELINE_SOURCES,baseline_qualified_head=qualified_head,feature_contract_member="contracts/features.json",tuning_loss_contract_member="contracts/loss.json")
    (;root,pins,view=v,expected_sources=P.source_identity())
end
admit(f)=I.admit_reference(joinpath(f.root,"completion.json");trusted_pins=f.pins,roots=Dict("reference"=>f.root),view=f.view,expected_sources=f.expected_sources)

function reference_checks(OUT)
v=view_fixture();positive=write_reference(joinpath(OUT,"matched-good"),v)
ref=admit(positive);@test ref.metadata.selected_setting_ids==[2,3]
@test length(ref.rows)==341;@test S._check_reference(ref;matched=true)
other=admit(write_reference(joinpath(OUT,"matched-good-L1"),v;delay=1));@test length(other.rows)==340
@test S._check_reference(other;matched=true)
original=admit(write_reference(joinpath(OUT,"original-good"),v;kind=:original_reference_diagnostic_only));@test original.metadata.reference_kind==:original_reference_diagnostic_only
@test_throws ArgumentError S._check_reference(original;matched=true)
metric=I._reference_metric([-3.,1.,-2.,4.],zeros(4));@test metric.n==4&&metric.rmse_nt==sqrt(7.5)&&metric.mae_nt==2.5&&metric.bias_nt==0.
results=NamedTuple[]
for variant in (:wrong_grid,:near_tie_stored,:exact_tie_wrong,:qualification_head,:qualification_float_FAIL,:qualification_bool_FAIL)
    witness=write_reference(joinpath(OUT,String(variant)),variant in (:near_tie_stored,:exact_tie_wrong) ? view_fixture(true) : v;variant)
    rejected=false;error=""
    try admit(witness) catch e;rejected=e isa ArgumentError;error=sprint(showerror,e);end
    push!(results,(;variant,rejected,error));@test rejected
end
old=ref.rows[1];ref.rows[1]=T.BPointPredictionRow(old.issue,(nextfloat(old.means[1]),old.means[2]),old.anchor,old.target,old.model_available,old.reason,old.source_sha256,old.delay_hours)
@test_throws ArgumentError S._check_reference(ref;matched=true)
ref.rows[1]=old;@test S._check_reference(ref;matched=true)

true
end
