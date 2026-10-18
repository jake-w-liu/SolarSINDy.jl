using Test, SHA, JSON3
include("DstInitialGBMReference.jl")
include("BoundInitialReferenceInputs.jl")
include("independent_fixture.jl")
const B=BoundInitialReferenceInputs
const WORK=length(ARGS)==1 ? abspath(ARGS[1]) : error("new owned test output path required")
!ispath(WORK) || error("never overwrite a previous qualification")
mkdir(WORK)
snapshot(root)=Dict(replace(relpath(joinpath(d,n),root),'\\'=>'/')=>fixture_sha(joinpath(d,n)) for (d,_,ns) in walkdir(root) for n in ns)
f=fixture(joinpath(WORK,"literal_fixture"))
before=snapshot(f.base)
function manifest_variant(name,change)
    p=deepcopy(f.plan);change(p)
    path=joinpath(WORK,name*".json");digest=new_json(path,p)
    (path,digest)
end
function receipt_variant(role,name,change)
    original=getproperty(f,Symbol(role));modified=deepcopy(original);change(modified)
    namefile=name*".json";digest=new_json(joinpath(f.roots["closure"],namefile),modified)
    manifest_variant(name) do p
        p["receipts"][role]=record("closure",namefile,digest)
    end
end
# Julia do-blocks pass their callback first.
manifest_variant(change::Function,name)=manifest_variant(name,change)
receipt_variant(change::Function,role,name)=receipt_variant(role,name,change)
function rejected_variant(name,change)
    p,d=manifest_variant(name,change)
    @test_throws Exception B.admit(p,d;roots=f.roots)
end
rejected_variant(change::Function,name)=rejected_variant(name,change)

@testset "literal independent filesystem admission" begin
    token=B.admit(f.path,f.sha256;roots=f.roots)
    @test token.mode=="synthetic_fixture"
    @test token.new_model_fitting===false && token.scientific_verdict===false
    @test length(token.plan["files"]["source"])==13
    @test length(token.plan["files"]["package"])==388
    @test token.records["closure"]["checks"]==29
    @test token.records["semantics"]["gbm_predictions"]==28
    @test token.records["semantics"]["stages"]["initial"]["fit_count"]==Dict("4"=>24,"6"=>22)
    @test B.recheck(token)
    @test snapshot(f.base)==before
    expected_files=length(token.files)
    expected_bytes=sum(filesize(path) for (path,_) in values(token.files))
    @test expected_files==446
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(files=expected_files-1))
    @test B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(files=expected_files)).mode=="synthetic_fixture"
    excluding_manifest=expected_bytes-filesize(f.path)
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,
        limits=B.Limits(total_bytes=excluding_manifest,file_bytes=excluding_manifest,json_bytes=excluding_manifest))
    @test B.admit(f.path,f.sha256;roots=f.roots,
        limits=B.Limits(total_bytes=expected_bytes,file_bytes=expected_bytes,json_bytes=expected_bytes)).mode=="synthetic_fixture"
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,
        limits=B.Limits(total_bytes=expected_bytes-1,file_bytes=expected_bytes-1,json_bytes=expected_bytes-1))
    @test_throws ArgumentError B.admit(f.path,repeat("a",64);roots=f.roots)
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=merge(f.roots,Dict("extra"=>WORK)))
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=merge(f.roots,Dict("reader"=>f.roots["source"])))
    for mutate in (t->(t.records["terminal"]["success"]=false),t->(t.plan["mode"]="exposed_development"),
        t->delete!(t.files,first(keys(t.files))),t->(t.roots["source"]=WORK))
        changed=B.admit(f.path,f.sha256;roots=f.roots);mutate(changed)
        @test_throws ArgumentError B.recheck(changed)
    end
end

@testset "old audit and semantic contradiction controls" begin
    rejected_variant("missing_enhanced") do p;delete!(p["receipts"],"closure");end
    rejected_variant("legacy_disguised") do p;p["receipts"]["closure"]=p["receipts"]["legacy"];end
    rejected_variant("unsupported_L1") do p;p["expected"]["configuration"]["extra_dst_delay_hours"]=1;end
    rejected_variant("manifest_mode_promotion") do p;p["mode"]="exposed_development";end
    cases=[("closure","false_exit",x->(x["native_exit"]=false)),
        ("closure","nonzero_exit",x->(x["native_exit"]=1)),
        ("closure","wrong_terminal",x->(x["original_terminal_sha256"]=repeat("d",64))),
        ("closure","wrong_completion",x->(x["original_completion_sha256"]=repeat("d",64))),
        ("closure","wrong_source",x->(x["original_source_sha256"]["s2.txt"]=repeat("d",64))),
        ("closure","wrong_log",x->(x["log_sha256"]=repeat("d",64))),
        ("closure","wrong_plan",x->(x["verified_input_plan"]["terminal"]["sha256"]=repeat("d",64))),
        ("closure","promoted_verdict",x->(x["scientific_verdict"]=true)),
        ("semantics","semantic_fake_zero",x->(x["helper_runs"][1]["native_exit"]=false)),
        ("semantics","semantic_wrong_binary",x->(x["helper_runs"][2]["binary_sha256"]=repeat("d",64))),
        ("semantics","semantic_wrong_source",x->(x["source_sha256"]["bounded_bson.py"]=repeat("d",64))),
        ("semantics","semantic_wrong_stage",x->(x["stages"]["initial"]["fit_count"]["4"]=false)),
        ("semantics","semantic_no_points",x->(x["gbm_predictions"]=0)),
        ("qualification","qual_skipped_native",x->(x["unit_runs"][3]["native_exit"]=1)),
        ("harness","harness_shortened",x->pop!(x["package_files_before"])),
        ("runtime","runtime_shortened",x->pop!(x["runs"])),
        ("terminal","terminal_false_native",x->(x["runs"][1]["native_exit"]=false))]
    for (role,name,change) in cases
        p,d=receipt_variant(role,name,change)
        @test_throws Exception B.admit(p,d;roots=f.roots)
    end
    # Rebind semantics AND closure to rule out rejection merely by a stale digest.
    s=deepcopy(f.semantics);s["helper_runs"][1]["log_sha256"]=repeat("e",64)
    semantic_name="fully_rebound_semantics.json";sd=new_json(joinpath(f.roots["closure"],semantic_name),s)
    z=deepcopy(f.closure);z["additive_receipt_sha256"]=sd
    closure_name="fully_rebound_closure.json";zd=new_json(joinpath(f.roots["closure"],closure_name),z)
    p,d=manifest_variant("fully_rebound") do p
        p["receipts"]["semantics"]=record("closure",semantic_name,sd)
        p["receipts"]["closure"]=record("closure",closure_name,zd)
    end
    @test_throws ArgumentError B.admit(p,d;roots=f.roots)
    for (name,change,expected) in (
        ("rebound_helper_bool",x->(x["helper_runs"][1]["native_exit"]=false),"math native exit/binary"),
        ("rebound_helper_binary",x->(x["helper_runs"][2]["binary_sha256"]=repeat("f",64)),"math native exit/binary"),
        ("rebound_core_source",x->(x["source_sha256"]["bounded_bson.py"]=repeat("f",64)),"exact semantic core source group"),
        ("rebound_fit_bool",x->(x["stages"]["initial"]["fit_count"]["4"]=false),"positive full per-lead fit counts"),
        ("rebound_no_points",x->(x["gbm_predictions"]=0),"semantic positive integer counts"))
        semantic=deepcopy(f.semantics);change(semantic)
        sn=name*"_semantics.json";sh=new_json(joinpath(f.roots["closure"],sn),semantic)
        closed=deepcopy(f.closure);closed["additive_receipt_sha256"]=sh
        cn=name*"_closure.json";ch=new_json(joinpath(f.roots["closure"],cn),closed)
        p,d=manifest_variant(name) do p
            p["receipts"]["semantics"]=record("closure",sn,sh)
            p["receipts"]["closure"]=record("closure",cn,ch)
        end
        observed=try;B.admit(p,d;roots=f.roots);nothing;catch error;error;end
        @test observed isa ArgumentError
        @test occursin(expected,sprint(showerror,observed))
    end
end

@testset "path, duplicate-key and resource controls" begin
    for name in ("source","support","s1","a_9")
        @test B.role_name(name)
    end
    for name in ("support\n","support\r","support\t","A","1support","support-","süpport","support_x\n")
        @test !B.role_name(name)
    end
    @test B.hex(repeat("a",64))
    @test !B.hex(repeat("a",64)*"\n")
    @test !B.hex(repeat("a",64)*"\r")
    p,d=manifest_variant("root_role_newline") do p
        p["root_roles"]=[x=="support" ? "support\n" : x for x in p["root_roles"]]
        for item in values(p["support"]);item["root"]="support\n";end
    end
    renamed_roots=copy(f.roots);renamed_roots["support\n"]=pop!(renamed_roots,"support")
    @test_throws ArgumentError B.admit(p,d;roots=renamed_roots)
    for name in ("../escape","/absolute","s2.txt/../s3.txt","s2.txt/","s2.txt\\x","C:evil","NUL.txt","s2.txt.","s2.txt ","s2.txt//a")
        @test_throws ArgumentError B.safe_components(name)
    end
    rejected_variant("traversal_map") do p;p["files"]["source"]["../support/raw.txt"]=f.plan["support"]["raw"]["sha256"];end
    rejected_variant("case_alias") do p;p["files"]["source"]["S2.TXT"]=p["files"]["source"]["s2.txt"];end
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(files=400))
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(json_bytes=32,file_bytes=64,total_bytes=128))
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(nodes=50))
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(depth=2))
    @test_throws ArgumentError B.admit(f.path,f.sha256;roots=f.roots,limits=B.Limits(total_bytes=64*1024^2,file_bytes=64*1024^2,json_bytes=32*1024^2,files=1))
    for text in ("{\"a\":1,\"a\":2}","{\"a\":1,\"\\u0061\":2}","{\"x\":{\"a\":1,\"a\":2}}")
        @test_throws ArgumentError B.lexical_json(Vector{UInt8}(codeunits(text)),B.Limits())
    end
    @test B.lexical_json(Vector{UInt8}(codeunits("{\"a\":1,\"x\":{\"a\":2}}")),B.Limits())===nothing
    @test_throws ArgumentError B.lexical_json(Vector{UInt8}(codeunits("[[[[0]]]]")),B.Limits(depth=4))
    @test_throws ArgumentError B.lexical_json(Vector{UInt8}(codeunits("[0,0,0,0]")),B.Limits(nodes=4))
    @test B.lexical_json(Vector{UInt8}(codeunits("[0,0,0,0]")),B.Limits(nodes=5))===nothing
    huge=joinpath(WORK,"oversize.dat");new_file(huge,fill(UInt8(0x78),4096))
    @test_throws ArgumentError B.digest_file(huge,4095)
    @test B.digest_file(huge,4096).bytes==4096
    @test B.digest_file(huge,4096).sha256==fixture_sha(huge)
    # Pre-materialization guards: a megabyte of malformed bytes must report the
    # byte gate, with bounded allocation, before a parser/hash can inspect it.
    oversized_json=joinpath(WORK,"oversized-malformed.json")
    new_file(oversized_json,fill(UInt8(0x78),1024^2))
    tiny=B.Limits(json_bytes=32,file_bytes=1024^2,total_bytes=2*1024^2)
    oversized_sha=fixture_sha(oversized_json)
    capture(call)=try;call();nothing;catch error;error;end
    err=capture(()->B.json_file(oversized_json,oversized_sha,tiny))
    @test err isa ArgumentError && occursin("JSON byte budget before materialization",sprint(showerror,err))
    json_allocation=@allocated capture(()->B.json_file(oversized_json,oversized_sha,tiny))
    @test json_allocation<128*1024
    capture(()->B.digest_file(oversized_json,32))
    hash_allocation=@allocated capture(()->B.digest_file(oversized_json,32))
    @test hash_allocation<128*1024
    println("REJECT_JSON_BYTES=",filesize(oversized_json)," ALLOCATED=",json_allocation," HASH_REJECT_ALLOCATED=",hash_allocation)
    if Sys.iswindows()
        junction=joinpath(WORK,"owned-junction")
        quoted(x)="'"*replace(x,"'"=>"''")*"'"
        command="New-Item -ItemType Junction -Path "*quoted(junction)*" -Target "*quoted(f.roots["source"])*" -ErrorAction Stop | Out-Null"
        run(Cmd(Cmd(["powershell.exe","-NoProfile","-NonInteractive","-Command",command]);windows_hide=true))
        @test B.reparse(junction)
        @test_throws ArgumentError B.admit(f.path,f.sha256;roots=merge(f.roots,Dict("source"=>junction)))
        @test_throws ArgumentError B.root_path(joinpath(f.base,"SOURCE"))
        crowded=joinpath(WORK,"crowded");mkdir(crowded)
        for i in 1:10000;new_file(joinpath(crowded,"entry_$i.txt"),"x");end
        new_file(joinpath(crowded,"needle.txt"),"needle")
        narrow_roots=Dict("target"=>crowded)
        @test B.resolved(narrow_roots,"target","needle.txt")==joinpath(crowded,"needle.txt")
        path_allocation=@allocated B.resolved(narrow_roots,"target","needle.txt")
        @test path_allocation<200*1024
        println("CROWDED_DIRECTORY_ENTRIES=10001 PATH_ALLOCATED=",path_allocation)
    end
end

@testset "relocation and detected post-admission change" begin
    relocated=joinpath(WORK,"relocated");cp(f.base,relocated)
    roots=Dict(k=>joinpath(relocated,k) for k in keys(f.roots))
    t=B.admit(joinpath(relocated,"manifest.json"),f.sha256;roots)
    @test B.recheck(t)
    @test t.records==B.admit(f.path,f.sha256;roots=f.roots).records
    @test fixture_sha(joinpath(relocated,"manifest.json"))==f.sha256
    changed=joinpath(roots["source"],"s2.txt")
    open(changed,"a") do io;write(io,"CHANGED");end
    @test_throws ArgumentError B.recheck(t)
    @test fixture_sha(joinpath(f.roots["source"],"s2.txt"))==f.plan["files"]["source"]["s2.txt"]
    @test all(fixture_sha(joinpath(f.base,name))==digest for (name,digest) in before)
end

println("LOADER_ONLY_SYNTHETIC_QUALIFICATION=true")
println("NO_REAL_ARCHIVE_OR_FIT=true")
println("LOADER_SHA256=",B.LOADED_SHA256)
println("REFERENCE_SHA256=",DstInitialGBMReference.LOADED_SHA256)

# Entire original pure reference regression, unchanged inside a private namespace.
module OriginalReferencePureRegression
using Dates, Test, SHA
include("DstInitialGBMReference.jl")
using .DstInitialGBMReference
const R=DstInitialGBMReference
const source=bytes2hex(sha256("synthetic reference labels only"))

@testset "independentintegerclockandperleadjoin" begin
    boundary=DateTime(2020);cutoff=DateTime(2019,12,25)
    for (h,issue) in ((4,DateTime(2019,12,24,20)),(6,DateTime(2019,12,24,18)))
        c=clock(issue,h)
        @test c.target_complete==c.target_receipt==cutoff
        @test eligible(issue,c.target_complete,c.target_receipt,boundary)
        @test !eligible(issue,c.target_complete,c.target_receipt+Millisecond(1),boundary)
        @test !eligible(issue+Hour(1),c.target_complete+Hour(1),c.target_receipt+Hour(1),boundary)
        @test c.origin_tag==c.anchor_tag==issue-Hour(1)
    end
    @test_throws ArgumentError clock(DateTime(2020),4;delay_hours=1)
    @test_throws ArgumentError clock(DateTime(2020),true)
    @test_throws ArgumentError eligible(DateTime(2020),DateTime(2019),DateTime(2021),DateTime(2022))
    t=DateTime(2019,12,24,15);dst=Float64.(1:15);dst[7]=NaN
    raw=RawLabels([t+Hour(i-1) for i in 1:15],dst;source_sha256=source)
    start=DateTime(2019,12,24,16)
    four=cohorts(raw,4;validation_start=start,validation_end=boundary)
    six=cohorts(raw,6;validation_start=start,validation_end=boundary)
    @test four.validation==[1,2,4,5]
    @test six.validation==[2,3]
    @test intersect(four.validation,six.validation)==[2]
    for (h,idx,prediction,expected) in ((4,[1,2,4,5],[1.,7.,6.,11.],(n=4,rmse_nt=2.5,mae_nt=2.25,bias_nt=-.75)),
            (6,[2,3],[2.,10.],(n=2,rmse_nt=sqrt(18.5),mae_nt=3.5,bias_nt=-2.5)))
        available=trues(15);available[first(idx)]=false
        refs=validation_rows(raw,h,idx,prediction,available)
        a=[merge(r,(prediction=999.,status="irrelevantAstatus",)) for r in refs]
        joined=join_for_a(refs,a,raw;horizon=h,validation_start=start,validation_end=boundary)
        @test verify_metric(joined.metric,expected)
        @test length(joined.rows)==length(idx)
        @test joined.rows[1].status=="persistence_feature_unavailable"
        @test joined.max_target_complete==joined.max_label_receipt==cutoff
        @test_throws ArgumentError join_for_a(refs[2:end],a,raw;horizon=h,validation_start=start,validation_end=boundary)
        @test_throws ArgumentError join_for_a(vcat(refs,[refs[1]]),a,raw;horizon=h,validation_start=start,validation_end=boundary)
        @test_throws ArgumentError join_for_a(refs,a[2:end],raw;horizon=h,validation_start=start,validation_end=boundary)
        changed=copy(a);changed[1]=merge(changed[1],(target=-100.,))
        @test_throws ArgumentError join_for_a(refs,changed,raw;horizon=h,validation_start=start,validation_end=boundary)
        @test_throws ArgumentError validation_rows(raw,h,idx,prediction.+1,available)
    end
end

@testset "strictfirstgridtiesandfulllabelmasks" begin
    rmse=[3.,2.5,2.5,4.,5.,6.,7.,8.]
    candidates=[(candidate=c,score=(n=4,rmse_nt=rmse[i],mae_nt=i==2 ? 2.25 : 2.,bias_nt=i==2 ? -.75 : 0.)) for (i,c) in enumerate(R.GRID)]
    first=first_grid_choice(candidates,R.GRID[2];expected_count=4)
    @test first.selected_index==2 && first.configuration==(max_depth=4,nrounds=200,nbins=255)
    @test_throws ArgumentError first_grid_choice(candidates,R.GRID[3];expected_count=4)
    @test_throws ArgumentError first_grid_choice(reverse(candidates),R.GRID[2];expected_count=4)
    @test_throws ArgumentError first_grid_choice(candidates,R.GRID[2];expected_count=3)
    @test_throws ArgumentError first_grid_choice(candidates[1:7],R.GRID[2];expected_count=4)
    @test_throws ArgumentError verify_metric((n=4,rmse_nt=2.500000001,mae_nt=2.25,bias_nt=-.75),first.expected_metric)
    @test verify_metric((n=4,rmse_nt=2.5,mae_nt=2.25,bias_nt=-.75),first.expected_metric)
    time=[DateTime(2019,12,1)+Hour(i-1) for i in 1:1100]
    raw=RawLabels(time,[-70+sin(i/19) for i in 1:1100];source_sha256=source)
    start=DateTime(2020);ending=DateTime(2020,1,25)
    available=trues(1100);available[100]=false
    views=Dict{Int,Any}()
    for h in (4,6)
        frame=cohorts(raw,h;validation_start=start,validation_end=ending)
        fit=[i for i in frame.train if available[i]]
        records=[(candidate=c,score=(n=length(frame.validation),rmse_nt=rmse[i],mae_nt=2.,bias_nt=0.)) for (i,c) in enumerate(R.GRID)]
        record=(horizon=h,train_rows=fit,validation_common_rows=frame.validation,candidates=records,selected=R.GRID[2])
        checked=verify_choice(record,raw,available;horizon=h,validation_start=start,validation_end=ending)
        @test checked.max_fit_target_complete==checked.max_fit_label_receipt==DateTime(2019,12,25)
        @test checked.fit_rows==fit && checked.validation_rows==frame.validation
        @test_throws ArgumentError verify_choice(merge(record,(train_rows=fit[2:end],)),raw,available;horizon=h,validation_start=start,validation_end=ending)
        @test_throws ArgumentError verify_choice(merge(record,(validation_common_rows=frame.validation[2:end],)),raw,available;horizon=h,validation_start=start,validation_end=ending)
        views[h]=checked
    end
    @test length(views[4].fit_rows)==length(views[6].fit_rows)+2
    @test views[4].fit_rows!=intersect(views[4].fit_rows,views[6].fit_rows)
end

function receipt_fixture()
    hash=repeat("a",64)
    source_map=Dict("source$i"=>hash for i in 1:13)
    package=Dict("package$i"=>hash for i in 1:388)
    complete_sources=merge(copy(source_map),Dict("pkg$i"=>hash for i in 1:5))
    config=Dict("mode"=>"synthetic_fixture","horizons"=>[4,6],"embargo_hours"=>168,
        "extra_dst_delay_hours"=>0,"extra_driver_delay_hours"=>0,"target_imputation"=>false,
        "novel_or_operational_superiority_claim"=>false)
    expected=Dict{String,Any}("source_sha256"=>source_map,"package_files"=>package,"package_commit"=>repeat("a",40),
        "configuration"=>config,"completion_source_sha256"=>complete_sources)
    for key in ("driver_sha256","runtime_gate_sha256","synthetic_gate_sha256","input_sha256","format_sha256",
            "native_library_sha256","julia_binary_sha256","raw_hourly_matrix_sha256","completion_sha256","checker_sha256")
        expected[key]=hash
    end
    terminal=Dict{String,Any}(k=>deepcopy(v) for (k,v) in expected if !(k in ("configuration","completion_source_sha256","raw_hourly_matrix_sha256","completion_sha256","checker_sha256")))
    merge!(terminal,Dict("success"=>true,"all13_runner_files_unchanged"=>true,"all388_package_files_unchanged"=>true,
        "independent_confirmation"=>false,"novelty_or_operational_superiority"=>false,
        "runs"=>[Dict("name"=>name,"native_exit"=>0,"timed_out"=>false,"log_sha256"=>hash) for name in ("real_fit","independent_reader")]))
    outputs=Dict(n=>hash for n in ("started.json","initial_imputation.json","initial_imputed_counts.json",
        "gbm29_selection_h4.json","gbm29_selection_h6.json","input_hourly_channels.f64","input_hourly_channels.json"))
    completion=Dict{String,Any}("complete"=>true,"input_sha256"=>hash,"raw_hourly_matrix_sha256"=>hash,
        "source_sha256"=>complete_sources,"configuration"=>config,"no_original_model_parity"=>true,
        "no_live_serving_changes"=>true,"native"=>Dict("sha256"=>hash,"version"=>[3,2,0],"live_objects"=>0),"output_sha256"=>outputs)
    audit=Dict{String,Any}("complete"=>true,"completion_sha256"=>hash,"checker_sha256"=>hash,"source_mode"=>"synthetic_fixture",
        "horizons"=>[4,6],"checks"=>100,"new_model_fitting"=>false,"author_or_ml_library_imported"=>false,"scientific_verdict"=>false)
    terminal,completion,audit,expected
end
@testset "syntheticreceiptcontentsonlynotafilesystemqualification" begin
    t,c,a,e=receipt_fixture()
    @test validate_receipts(t,c,a,e)
    for field in ("success","all13_runner_files_unchanged","all388_package_files_unchanged")
        bad=deepcopy(t);bad[field]=false
        @test_throws ArgumentError validate_receipts(bad,c,a,e)
    end
    for mutation in (x->(x["runs"]=x["runs"][1:1]),x->reverse!(x["runs"]),
            x->(x["runs"][1]["native_exit"]=1),x->(x["runs"][1]["native_exit"]=false),
            x->(x["runs"][2]["timed_out"]=true),x->delete!(x,"input_sha256"),
            x->(x["source_sha256"]["source1"]=repeat("b",64)),x->delete!(x["package_files"],"package1"))
        bad=deepcopy(t);mutation(bad)
        @test_throws ArgumentError validate_receipts(bad,c,a,e)
    end
    for mutation in (x->(x["complete"]=false),x->(x["native"]["live_objects"]=1),
            x->(x["native"]["live_objects"]=false),x->(x["configuration"]["extra_dst_delay_hours"]=1),
            x->delete!(x["output_sha256"],"initial_imputation.json"),x->(x["raw_hourly_matrix_sha256"]=repeat("c",64)))
        bad=deepcopy(c);mutation(bad)
        @test_throws ArgumentError validate_receipts(t,bad,a,e)
    end
    for field in ("complete","new_model_fitting","author_or_ml_library_imported","scientific_verdict")
        bad=deepcopy(a);bad[field]=!bad[field]
        @test_throws ArgumentError validate_receipts(t,c,bad,e)
    end
    bad=deepcopy(a);bad["completion_sha256"]=repeat("d",64)
    @test_throws ArgumentError validate_receipts(t,c,bad,e)
end
println("REFERENCE_SOURCE_SHA256=",R.LOADED_SHA256)

end # OriginalReferencePureRegression

using Dates
@testset "independent exact reference digest grammar" begin
    valid=repeat("a",64)
    @test DstInitialGBMReference._hex(valid)
    raw=DstInitialGBMReference.RawLabels([DateTime(2019)],[1.];source_sha256=valid)
    @test raw.source_sha256==valid
    for suffix in ("\n","\r","\t","x"," ","\0")
        # Decode just these literal escape spellings independently of production.
        value=valid*unescape_string(suffix)
        @test !DstInitialGBMReference._hex(value)
        @test_throws ArgumentError DstInitialGBMReference.RawLabels([DateTime(2019)],[1.];source_sha256=value)
    end
    terminal,completion,audit,expected=OriginalReferencePureRegression.receipt_fixture()
    expected["package_commit"]*= "\n"
    terminal["package_commit"]=expected["package_commit"]
    @test_throws ArgumentError DstInitialGBMReference.validate_receipts(terminal,completion,audit,expected)
end
println("STRICT_REFERENCE_WITH_ORIGINAL_PURE_SUITE_COMPLETE=true")
