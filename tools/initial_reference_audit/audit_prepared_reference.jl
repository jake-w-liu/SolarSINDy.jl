using Dates, SHA, JSON3, CSV, DataFrames, Statistics
include("InitialReferenceAudit.jl")
include("IndependentScalarTreeReplay.jl")
module IndependentFeatureArithmetic
include("independent_gbm_math.jl")
end
const R=InitialReferenceAudit
const F=IndependentFeatureArithmetic
const T=IndependentScalarTreeReplay
const FILL_LIMITS=(999.,999.,999.,999.,999.,999.,999.,9999.,999.,9999999.,999.,99.99,999.,99.,99999.,9999.,9.)
const CHECKS=Ref(0)
function check(ok,message)
    CHECKS[]+=1
    ok || throw(ArgumentError(message))
end
digest(path)=open(path,"r") do io;bytes2hex(sha256(io));end
function readjson(path;maximum=64*1024^2)
    check(isfile(path) && !islink(path) && filesize(path)<=maximum,"bounded regular JSON input required")
    JSON3.read(read(path),Dict{String,Any})
end
function near(actual,expected,message)
    ok=actual isa Real && expected isa Real && !(actual isa Bool) && !(expected isa Bool) &&
          isfinite(actual) && isfinite(expected) && abs(actual-expected)<=5e-12*max(1.,abs(expected))
    check(ok,ok ? message : string(message,"; actual=",repr(actual)," expected=",repr(expected)))
end
function row_indices(actual,expected,message)
    check(length(actual)==length(expected) && all(x->x isa Integer && !(x isa Bool),actual) && collect(actual)==expected,message)
end
function datehour(value)
    parsed=DateTime(value)
    check(floor(parsed,Hour)==parsed,"exact-hour serialized clock required")
    parsed
end
function source_recheck(plan)
    for (path,expected) in plan["input_files_sha256"]
        check(isfile(path) && !islink(path) && digest(path)==expected,"bound input changed during independent replay")
    end
end
function fingerprint(raw,start,boundary)
    cutoff=boundary-Hour(168)
    rows=findall(i->start+Hour(i)<=cutoff,1:size(raw,2))
    context=SHA.SHA2_256_CTX()
    SHA.update!(context,codeunits("sandu-imputer-training-v1\n"))
    values=Vector{UInt64}(undef,18)
    for i in rows
        values[1]=reinterpret(UInt64,Int64(Dates.value(start+Hour(i-1))))
        for j in 1:17;values[j+1]=reinterpret(UInt64,raw[j,i]);end
        SHA.update!(context,reinterpret(UInt8,values))
    end
    (sha256=bytes2hex(SHA.digest!(context)),rows=collect(rows))
end
function check_clock(point,row,h,start,source,raw)
    clock=R.forecast_clock(start+Hour(row),h)
    fields=((:issue_time_utc,:issue),(:origin_tag_utc,:origin_tag),(:driver_tag_utc,:driver_tag),
        (:anchor_tag_utc,:anchor_tag),(:anchor_hour_complete_utc,:anchor_complete),
        (:assumed_anchor_receipt_utc,:anchor_receipt),(:target_record_tag_utc,:target_tag),
        (:target_hour_complete_utc,:target_complete),(:assumed_target_receipt_utc,:target_receipt))
    for (name,expected) in fields
        check(DateTime(getproperty(point,name))==getproperty(clock,expected),"independent prediction clock mismatch: $name")
    end
    check(point.source_sha256==source,"prediction raw source differs")
    check(point.completed_target_lead_hours isa Integer && !(point.completed_target_lead_hours isa Bool) &&
          point.completed_target_lead_hours==h,"prediction completed target lead differs")
    check(point.extra_dst_delay_hours isa Integer && !(point.extra_dst_delay_hours isa Bool) && point.extra_dst_delay_hours==0,
          "prediction latency differs")
    check(point.anchor_dst_nt==raw[15,row] && point.observation_dst_nt==raw[15,row+h],"prediction raw anchor/target differs")
end
function reference_key(source,i,h,start)
    clock=R.forecast_clock(start+Hour(i),h)
    (String(source),clock.issue,clock.origin_tag,clock.driver_tag,clock.anchor_tag,
        clock.anchor_complete,clock.anchor_receipt,clock.target_tag,clock.target_complete,clock.target_receipt,h,0)
end
function audit(plan_path,expected_plan_sha)
    CHECKS[]=0
    check(digest(plan_path)==expected_plan_sha,"pinned audit-plan digest differs")
    plan=readjson(plan_path;maximum=8*1024^2)
    check(plan["schema"]=="independent_initial_reference_audit_inputs_v1" && plan["preflight_complete"]===true,
          "complete independent preflight required")
    check(plan["author_or_ml_imported"]===false && plan["new_fitting"]===false && plan["scientific_verdict"]===false,
          "independent replay scope differs")
    output=dirname(abspath(plan_path));reference=plan["reference"];source=plan["source"]
    marker=joinpath(output,"numerical-audit.json")
    check(!ispath(marker),"exclusive new numerical-audit marker required")
    source_recheck(plan)
    complete=readjson(joinpath(reference,"completion.json"))
    original=readjson(joinpath(source,"completion.json"))
    check(digest(joinpath(reference,"completion.json"))==plan["reference_completion_sha256"] &&
          digest(joinpath(source,"completion.json"))==plan["source_completion_sha256"],"completion identity differs")
    check(complete["evidence_mode"]==plan["expected_mode"] && complete["complete"]===true,"reference evidence mode differs")
    plain_path=joinpath(output,"plain-trees.json")
    check(digest(plain_path)==plan["plain_models_sha256"],"plain saved-model digest differs")
    plain=readjson(plain_path)
    check(plain["schema"]=="independent_plain_initial_gbm16_v1" && plain["expected_mode"]==plan["expected_mode"] &&
          plain["source_completion_sha256"]==plan["reference_completion_sha256"],"plain-model binding differs")
    check(length(plain["models"])==16,"all sixteen models required")
    n=plan["prefix_rows"];full_n=plan["full_rows"]
    check(n isa Integer && !(n isa Bool) && 1<=n<=full_n<=1_000_000,"raw hour budget")
    check(ENDIAN_BOM==0x04030201,"little-endian numerical reader required")
    raw_path=joinpath(source,"input_hourly_channels.f64")
    check(filesize(raw_path)==17*full_n*8,"original raw matrix dimensions differ")
    bytes=open(raw_path,"r") do io;read(io,17*n*8);end
    check(length(bytes)==17*n*8,"truncated original raw prefix")
    raw=reshape(reinterpret(Float64,bytes),17,n) |> Matrix{Float64}
    check(bytes==read(joinpath(reference,"prefix_raw17.f64")),"reference raw prefix differs from original source")
    clean=copy(raw)
    for j in 1:17,i in 1:n
        isfinite(clean[j,i]) && clean[j,i]<FILL_LIMITS[j] || (clean[j,i]=NaN)
    end
    imputed_path=joinpath(reference,"prefix_imputed17.f64")
    check(filesize(imputed_path)==17*n*8,"imputed prefix dimensions differ")
    imputed=reshape(reinterpret(Float64,read(imputed_path)),17,n) |> Matrix{Float64}
    for j in 1:17,i in 1:n
        isfinite(clean[j,i]) && check(imputed[j,i]==clean[j,i],"observed raw value was imputed")
    end
    for j in 15:17;check(isequal(imputed[j,:],clean[j,:]),"excluded observation channel was imputed");end
    start=datehour(plan["start"]);initial=datehour(plan["initial_boundary"]);selection=datehour(plan["selection_boundary"])
    a_start=datehour(plan["a_start"]);a_end=datehour(plan["a_end"])
    check(start+Hour(n)==selection,"prefix extends beyond original initial-reference boundary")
    training=fingerprint(clean,start,initial)
    original_imputation=readjson(joinpath(source,"initial_imputation.json"))
    row_indices(original_imputation["configuration"]["fit_rows"],training.rows,"independent imputer fingerprint mask differs")
    check(training.sha256==original_imputation["configuration"]["training_sha256"]==plan["independent_imputation"]["training_sha256"],
          "independent imputer training fingerprint differs")
    X,available=F.features29(imputed)
    check(length(available)==n && size(X)==(n,29),"independent feature shape differs")
    times=[start+Hour(i-1) for i in 1:n]
    model_points=0;fallback_points=0;results=Any[]
    for (slot,h) in enumerate((4,6))
        item=complete["results"][slot]
        check(item["horizon"] isa Integer && !(item["horizon"] isa Bool) && item["horizon"]==h,"ordered integer leads required")
        cohort=R.expected_cohorts(times,vec(clean[15,:]),available,h;initial_boundary=initial,selection_boundary=selection,a_start,a_end)
        check(!isempty(cohort.fit) && !isempty(cohort.validation) && !isempty(cohort.a_join),"empty independent reference cohort")
        row_indices(item["choice"]["fit_rows"],cohort.fit,"reference complete per-lead fit pool differs")
        row_indices(item["choice"]["validation_rows"],cohort.validation,"reference full validation pool differs")
        fitpath=joinpath(reference,"fit_h$h.txt")
        row_indices(parse.(Int,readlines(fitpath)),cohort.fit,"independent saved fit mask differs")
        center,scale=F.fit_scaling(X,available,fitpath)
        old=readjson(joinpath(source,"gbm29_selection_h$h.json"))
        check(old["horizon"]==h,"original selected lead differs")
        row_indices(old["train_rows"],cohort.fit,"original full fit mask differs")
        row_indices(old["validation_common_rows"],cohort.validation,"original full selection mask differs")
        check(item["ordered_names"]==F.NAMES,"original feature order differs")
        for (field,rows) in (("max_fit_target_complete",cohort.fit),("max_fit_label_receipt",cohort.fit),
                             ("max_selection_target_complete",cohort.validation),("max_selection_label_receipt",cohort.validation))
            check(datehour(item["choice"][field])==start+Hour(maximum(rows)+h),"independent fit/selection clock maximum differs")
        end
        candidates=Any[];selected_points=nothing
        for (index,record) in enumerate(item["grid"])
            check(record["index"]==index && record["configuration"]==Dict(string(k)=>v for (k,v) in pairs(R.GRID[index])),
                  "complete original grid order differs")
            data=only(filter(x->x["horizon"]==h && x["index"]==index,plain["models"]))
            check(data["configuration"]==record["configuration"] && data["model_sha256"]==record["model_sha256"],"plain/model identity differs")
            trees=[(feature=x["feature"],condition=x["condition"],prediction=x["prediction"],split=x["split"]) for x in data["trees"]]
            config=record["configuration"]
            model=T.Model(data["edges"],trees;max_depth=config["max_depth"],nrounds=config["nrounds"])
            saved=CSV.read(joinpath(reference,record["scaling"]),DataFrame)
            check(names(saved)==["feature","center","scale"] && saved.feature==F.NAMES,"exact sample-scaling schema differs")
            for j in 1:29
                context=" h=$h grid=$index feature=$(F.NAMES[j])"
                near(saved.center[j],center[j],"independent sample center differs"*context)
                # Source variance is centered at the stored, independently
                # verified rounded mean. Preserve its zero-to-unit mapping.
                expected_scale=R.sample_scale_about(view(X,cohort.fit,j),saved.center[j])
                near(saved.scale[j],expected_scale,"independent sample scale differs"*context)
                check(saved.scale[j]>0,"saved scale must be positive")
            end
            points=CSV.read(joinpath(reference,record["predictions"]),DataFrame)
            expected_columns=Set(["row","source_sha256","origin_tag_utc","driver_tag_utc","issue_time_utc","anchor_tag_utc",
                "anchor_hour_complete_utc","assumed_anchor_receipt_utc","target_record_tag_utc","target_hour_complete_utc",
                "assumed_target_receipt_utc","completed_target_lead_hours","extra_dst_delay_hours","anchor_dst_nt",
                "observation_dst_nt","prediction_dst_nt","status"])
            check(Set(names(points))==expected_columns,"exact prediction schema differs")
            row_indices(points.row,cohort.validation,"independent prediction cohort differs")
            for point in eachrow(points)
                i=point.row;check_clock(point,i,h,start,original["input_sha256"],clean)
                if available[i]
                    value=clean[15,i]+T.predict_delta(model,collect((X[i,:].-saved.center)./saved.scale))
                    near(point.prediction_dst_nt,value,"independent saved-tree point differs")
                    check(point.status=="model","available model status differs");model_points+=1
                else
                    check(point.prediction_dst_nt==clean[15,i] && point.status=="persistence_feature_unavailable",
                          "exact persistence fallback differs");fallback_points+=1
                end
            end
            score=R.metrics(points.prediction_dst_nt,points.observation_dst_nt)
            R.verify_metric(score,record["metric"])
            old_member=old["candidates"][index]
            check(old_member["candidate"]==record["configuration"],"original grid configuration differs")
            R.verify_metric(score,old_member["score"])
            push!(candidates,(configuration=record["configuration"],metric=score))
            index==item["choice"]["selected_index"] && (selected_points=points)
        end
        chosen=R.first_grid_choice(candidates)
        old_candidates=[(configuration=x["candidate"],metric=x["score"]) for x in old["candidates"]]
        check(R.first_grid_choice(old_candidates)==chosen==item["choice"]["selected_index"],"strict original first-grid choice differs")
        check(item["choice"]["configuration"]==item["grid"][chosen]["configuration"],"selected configuration metadata differs")
        R.verify_metric(candidates[chosen].metric,item["choice"]["expected_metric"])
        check(old["selected"]==item["grid"][chosen]["configuration"],"original chosen configuration differs")
        check(selected_points!==nothing,"selected validation points missing")
        selected_file=CSV.read(joinpath(reference,"initial_validation_h$h.csv"),DataFrame)
        check(isequal(selected_file,selected_points),"selected vector differs from selected grid member")
        joined=CSV.read(joinpath(reference,"a_join_h$h.csv"),DataFrame)
        check(Set(names(joined))==Set(["key","row","issue","target","anchor","prediction","status"]),"exact A-join schema differs")
        row_indices(joined.row,cohort.a_join,"complete separate A reference join differs")
        lookup=Dict(row=>i for (i,row) in enumerate(selected_points.row))
        for point in eachrow(joined)
            expected=selected_points[lookup[point.row],:]
            check(DateTime(point.issue)==start+Hour(point.row) && point.anchor==expected.anchor_dst_nt &&
                  point.target==expected.observation_dst_nt && point.prediction==expected.prediction_dst_nt && point.status==expected.status,
                  "exact A-join raw/reference point differs")
            check(String(point.key)==repr(reference_key(original["input_sha256"],point.row,h,start)),"A-join serialized key differs")
        end
        keys=[reference_key(original["input_sha256"],i,h,start) for i in cohort.a_join]
        check(bytes2hex(sha256(repr(keys)))==item["a_join_keys_sha256"],"independent complete A-join keyset digest differs")
        joined_score=R.metrics(joined.prediction,joined.target)
        R.verify_metric(joined_score,item["a_join_metric"])
        R.verify_metric(candidates[chosen].metric,item["original_selected_grid_metric"])
        for field in ("max_a_selection_target_complete","max_a_selection_label_receipt")
            check(datehour(item[field])==start+Hour(maximum(cohort.a_join)+h),"A-join clock maximum differs")
        end
        push!(results,(horizon=h,fit_count=length(cohort.fit),validation_count=length(cohort.validation),
            a_join_count=length(cohort.a_join),selected_index=chosen,original_grid_metric=candidates[chosen].metric,
            a_join_metric=joined_score))
    end
    source_recheck(plan)
    check(digest(plan_path)==expected_plan_sha && digest(plain_path)==plan["plain_models_sha256"],"prepared control changed during replay")
    receipt=(schema="independent_initial_reference_numerical_audit_v1",complete=true,checks=CHECKS[],
        expected_mode=plan["expected_mode"],input_plan_sha256=expected_plan_sha,results=results,
        model_points=model_points,fallback_points=fallback_points,imputer_training_fingerprint_verified=true,
        author_or_ml_imported=false,new_fitting=false,scientific_verdict=false)
    bytes=Vector{UInt8}(codeunits(JSON3.write(receipt)))
    file=Base.Filesystem.open(marker,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try;check(write(file,bytes)==length(bytes),"short numerical-audit write");finally;close(file);end
    check(read(marker)==bytes,"numerical-audit readback differs")
    println("INDEPENDENT_INITIAL_REFERENCE_NUMERICAL_AUDIT_COMPLETE=true CHECKS=",receipt.checks)
    receipt
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    length(ARGS)==2 || error("AUDIT_PLAN EXPECTED_PLAN_SHA256")
    audit(abspath(ARGS[1]),ARGS[2])
end
