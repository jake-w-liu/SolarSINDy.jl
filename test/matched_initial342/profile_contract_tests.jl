module ProfileContractOracles342
using Test,SHA,JSON3

# These entry modules expose data-only predicates. No main, authority admission,
# native model construction, profile execution or qualification issuance is used.
const PACKAGE=dirname(dirname(@__DIR__))
include(joinpath(PACKAGE,"tools","actual_b_chain","issue_matched342_resource_qualification.jl"))
include(joinpath(PACKAGE,"tools","actual_b_chain","run_matched342_initial_actual.jl"))
const Issuer=Matched342ResourceIssuer
const Profile=Issuer.Profile
const Actual=Matched342InitialActual
const Parent=parentmodule(@__MODULE__)
plain(x)=getfield(Parent,:b2_plain)(x)
body(x)=getfield(Parent,:b2_bytes)(x)
digest(x)=bytes2hex(sha256(body(x)))
parsed(x)=JSON3.read(body(x))
freeze(x::AbstractDict)=(;(Symbol(k)=>freeze(v) for (k,v) in pairs(x))...)
freeze(x::AbstractVector)=map(freeze,x)
freeze(x)=x
const HEX=repeat("a",64)
const HEAD=repeat("b",40)
const FEATURE="2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683"
const SCHEMA="109464af1630d16ac766b4f8791ad9f065a528d0aabac201a11fb68a4156249d"
const EVO="f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f"
const LOSS="f24c0dd5b9ef4baf58f0595056d5b5a30ab21ff5c278126b5ebefaa9435bc73c"
const AUDIT="60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a"
const SPEC="tools/actual_b_chain/fixtures/full_shape342_spec_v1.json"
const EXTREMA=(("c04f800000000000","3ff0000000000000"),
    ("c04f400000000000","3ff8000000000000"),("c04f600000000000","3ff4000000000000"),
    ("c04f200000000000","3ffc000000000000"))
const CALENDAR=((0,4,1:378212,378384:404516,378384:404516,378212,26133,26133),
    (0,6,1:378210,378384:404514,378384:404516,378210,26131,26133),
    (1,4,2:378211,378384:404515,378384:404515,378210,26132,26132),
    (1,6,2:378209,378384:404513,378384:404515,378208,26130,26132))

function identified(x,key="request_id")
    d=plain(x);pop!(d,key,nothing);d[key]=digest(d);d
end
function changed(x,key,value;id="request_id")
    d=plain(x);d[key]=value;identified(d,id)
end
function dropped(x,key;id="request_id")
    d=plain(x);delete!(d,key);identified(d,id)
end
function member(root,name,data)
    path=joinpath(root,name);@assert !ispath(path)
    open(io->write(io,data),path,"w");path,bytes2hex(sha256(data))
end
ref(root,name)=(;path=joinpath(root,name),sha256=HEX)
budgets()=(;peak_rss_bytes=8*1024^3,julia_gc_allocated_bytes=123456789,descriptor_growth=2)
limits()=(;source_rows=500000,model_member_bytes=4194304,total_model_member_bytes=134217728,
    required_available_bytes=8589934592,required_disk_bytes=17179869184,stage_wall_seconds=120,fit_wall_seconds=60)
runtime()=(;julia_version="1.12.6",julia_binary_sha256=HEX,evotrees_version="0.18.7",
    evotrees_source_sha256=EVO,threads=1,blas_threads=1,omp_threads=1)
contracts()=(;protocol_sha256="7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404",
    feature_contract_sha256=FEATURE,feature_schema_sha256=SCHEMA,loss_contract_sha256=LOSS,
    audit_contract_sha256=AUDIT,evotrees_source_sha256=EVO)

function requests(root)
    # Grammar-only, explicitly unqualified metadata. None of the named files is
    # an authority, software receipt, observation or generated resource result.
    fixture=(;member=SPEC,path=joinpath(root,"spec.json"),sha256=HEX)
    sources=[(;name="test/member"*lpad(string(i),3,'0')*".jl",sha256=HEX) for i in 1:136]
    p=identified((;schema="matched342_full_shape_profile_request_v1",evidence_mode="synthetic_only",
        operation="initial32_full_shape_profile",qualified_head=HEAD,implementation_sources=sources,
        source_proof=ref(root,"proof.json"),component_qualification=ref(root,"component.json"),contracts=contracts(),
        runtime=runtime(),fixture_spec=fixture,limits=limits(),budgets=budgets(),
        output_directory=joinpath(root,"profile-output"),owner_directory=joinpath(root,"profile-owner")))
    i=identified((;schema="matched342_resource_issuance_request_v1",operation="issue_initial32_full_shape_resource_qualification",
        head=HEAD,source_proof=ref(root,"proof.json"),root_qualification=ref(root,"root.json"),
        component_qualification=ref(root,"component.json"),fixture_spec=fixture,profile_result=ref(root,"result.json"),
        native=ref(root,"native.json"),raw=ref(root,"raw.log"),prelaunch_admission=ref(root,"prelaunch.json"),
        measurement=ref(root,"measurement.json"),limits=limits(),budgets=budgets(),output_directory=joinpath(root,"issuance")))
    a=identified((;schema="matched342_initial_actual_request_v1",evidence_mode="archival_assumed_completion_research",
        operation="initial_matched342_32fits",execution_authority=ref(root,"permit.json"),
        sources=[(;delay_hours=L,source_proof=ref(root,"source$L.json"),roots=Dict("source"=>joinpath(root,"source$L"))) for L in (0,1)],
        source_proof=ref(root,"proof.json"),runtime=runtime(),output_directory=joinpath(root,"execution"),owner_directory=joinpath(root,"actual-owner")))
    p,i,a
end

function metadata_checks(root)
    p,i,a=requests(root)
    validators=(Profile.validate_profile_metadata,Issuer.validate_issuer_metadata,Actual.validate_actual_metadata)
    @test length(p)==15&&length(i)==16&&length(a)==10
    @test length(Profile.RESULT_FIELDS)==15
    for (d,validate) in zip((p,i,a),validators)
        @test validate(parsed(d)).request_id==d["request_id"]
        @test_throws ArgumentError validate(parsed(changed(d,"future_field",nothing)))
        @test_throws ArgumentError validate(parsed(dropped(d,"schema")))
        @test_throws ArgumentError validate(parsed(changed(d,"schema","future_schema")))
        @test_throws ArgumentError validate(parsed(changed(d,"operation","future_operation")))
        @test_throws ArgumentError validate(parsed(changed(d,"output_directory","relative")))
        @test_throws ArgumentError validate(parsed(changed(d,"source_proof",(;path=joinpath(root,"proof"),sha256=uppercase(HEX)))))
        stale=copy(d);stale["operation"]="future_operation"
        @test_throws ArgumentError validate(parsed(stale))
    end
    @test_throws ArgumentError Profile.validate_profile_metadata(parsed(changed(p,"evidence_mode","software_qualified")))
    @test_throws ArgumentError Actual.validate_actual_metadata(parsed(changed(a,"evidence_mode","synthetic_only")))
    for (d,validate) in zip((p,i),validators[1:2])
        for value in (nothing,merge(budgets(),(;descriptor_growth=true)),merge(budgets(),(;peak_rss_bytes=8589934593)))
            @test_throws ArgumentError validate(parsed(changed(d,"budgets",value)))
        end
        @test_throws ArgumentError validate(parsed(dropped(d,"budgets")))
        for bad in ((;source_rows=1024),(;required_available_bytes=8589934591),(;required_disk_bytes=17179869183),
                (;fit_wall_seconds=121),(;stage_wall_seconds=true))
            @test_throws ArgumentError validate(parsed(changed(d,"limits",merge(limits(),bad))))
        end
    end
    for bad in ((;threads=true),(;blas_threads=2),(;julia_binary_sha256="bad"))
        @test_throws ArgumentError Profile.validate_profile_metadata(parsed(changed(p,"runtime",merge(runtime(),bad))))
    end
    for L in 1:2
        s=deepcopy(a["sources"]);s[L]["delay_hours"]=1-(L-1)
        @test_throws ArgumentError Actual.validate_actual_metadata(parsed(changed(a,"sources",s)))
    end
    for label in ("", "outside/path", "with space")
        s=deepcopy(a["sources"]);s[1]["roots"]=Dict(label=>joinpath(root,"source0"))
        @test_throws ArgumentError Actual.validate_actual_metadata(parsed(changed(a,"sources",s)))
    end
    @test Profile.validate_budgets(budgets())==budgets()
    for bad in ((;peak_rss_bytes=0),(;julia_gc_allocated_bytes=0),(;descriptor_growth=-1),
            (;peak_rss_bytes=true),(;descriptor_growth=1.0),(;future=0))
        @test_throws ArgumentError Profile.validate_budgets(merge(budgets(),bad))
    end
    bridge=(;schema="matched342_native_on_launch_bridge_v1",request_sha256=HEX,prelaunch_admission=ref(root,"PRELAUNCH.json"))
    @test Profile.validate_bridge_metadata(parsed(bridge),HEX).path==joinpath(root,"PRELAUNCH.json")
    for bad in ((;request_sha256=repeat("c",64)),(;schema="future"),(;native=nothing))
        @test_throws ArgumentError Profile.validate_bridge_metadata(parsed(merge(bridge,bad)),HEX)
    end
    # Invalid request metadata must reject without opening the missing proof/Q.
    invalid=changed(p,"budgets",merge(budgets(),(;descriptor_growth=-1)))
    path,sha=member(root,"invalid-request.json",body(invalid))
    @test_throws ArgumentError Profile.read_profile_request(path;expected_sha256=sha)
    @test_throws ArgumentError Profile.read_profile_request(path;expected_sha256=repeat("c",64))
    @test_throws ArgumentError Profile.read_profile_request("relative.json";expected_sha256=sha)
    huge=joinpath(root,"oversize.json");open(io->truncate(io,2*1024^2+1),huge,"w")
    @test_throws ArgumentError Profile.read_profile_request(huge;expected_sha256=HEX)
    @test_throws ArgumentError Profile.fresh_directory(joinpath(root,"overlap"),[root])
end

function calendar_and_response_checks()
    cells=Issuer.derive_full_cells()
    @test cells isa Tuple&&length(cells)==4
    for (g,literal) in zip(cells,CALENDAR)
        @test Tuple(g)==literal
        @test first(g.validation_rows)==378384
        @test last(g.training_rows)+g.horizon+g.delay_hours+168==378384
        @test Issuer.check_full_geometry(g)==g
        @test_throws ArgumentError Issuer.check_full_geometry(merge(g,(;training_rows=(first(g.training_rows)+1):(last(g.training_rows)+1))))
        @test_throws ArgumentError Issuer.check_full_geometry(merge(g,(;training_rows=collect(g.training_rows))))
        @test_throws ArgumentError Issuer.check_full_geometry(merge(g,(;union_count=g.union_count-1)))
    end
    literals=((6,0,-4.),(6,15,-4.),(6,16,-3.),(6,128,4.),(6,143,4.),(6,144,-4.),
        (8,0,350.),(8,7,350.),(8,8,354.),(8,247,470.),(8,248,350.),
        (12,0,1.),(12,3,1.),(12,4,1.125),(12,63,2.875),(12,64,1.),
        (15,0,-80.),(15,1,-79.75),(15,255,-16.25),(15,256,-80.),(15,499999,-72.25),
        (1,0,1.75),(1,1,1.),(1,2,.25),(1,15,2.5),(1,16,1.75),(2,0,3.5),(2,1,2.75))
    for (j,t,value) in literals;@test Issuer.independent_value(j,t)===value;end
    for (j,t) in ((0,0),(18,0),(1,-1),(1,500000),(true,0),(1,true))
        @test_throws ArgumentError Issuer.independent_value(j,t)
    end
    # Literal quarter-cycle, independent of the issuer/generator formula.
    cycle=collect(-80.0:0.25:-16.25);dst=repeat(cycle,cld(500000,256))[1:500000]
    for (g,(lo,hi)) in zip(cells,EXTREMA)
        @test Tuple(Issuer.response_extrema(dst,g))==(lo,hi)
    end
    @test_throws ArgumentError Issuer.response_extrema(dst[1:1024],cells[1])
    @test_throws ArgumentError Issuer.response_extrema(fill(-40.,500000),cells[1])
    poisoned=copy(dst);poisoned[end]=NaN
    @test_throws ArgumentError Issuer.response_extrema(poisoned,cells[1])
    @test_throws ArgumentError Issuer.response_extrema(dst,merge(cells[1],(;training_rows=1:1024,training_count=1024)))
    cells
end

function encoder_checks(cells)
    g=first(cells)
    o=Dict{String,Any}("schema"=>"matched342_encoder_v1","feature_count"=>342,
        "feature_schema_sha256"=>SCHEMA,"feature_contract_sha256"=>FEATURE,
        "training_rows"=>collect(g.training_rows),"finite_counts"=>fill(g.training_count,165),
        "imputation_f64_bits"=>fill("0000000000000000",165),"center_f64_bits"=>fill("0000000000000000",342),
        "scale_f64_bits"=>fill("3ff0000000000000",342),"training_design_sha256"=>HEX)
    check(d;sha=digest(d),design=HEX)=Issuer.check_full_encoder(freeze(d),g;
        expected_preprocessing_sha256=sha,expected_training_design_sha256=design)
    @test check(o).training_rows==collect(g.training_rows)
    @test_throws ArgumentError check(o;sha=repeat("c",64))
    @test_throws ArgumentError check(o;design=repeat("c",64))
    for bad in (collect(1:1024),collect(2:378213),reverse(collect(g.training_rows)))
        d=copy(o);d["training_rows"]=bad;@test_throws ArgumentError check(d)
    end
    for (key,value) in (("finite_counts",fill(g.training_count+1,165)),("finite_counts",fill(true,165)),
            ("imputation_f64_bits",fill("0000000000000000",171)),("center_f64_bits",fill("7ff0000000000000",342)),
            ("scale_f64_bits",fill("0000000000000000",342)),("scale_f64_bits",fill("bff0000000000000",342)),
            ("scale_f64_bits",fill("7ff0000000000000",342)),("future",nothing))
        d=copy(o);d[key]=value;@test_throws ArgumentError check(d)
    end
end

function saved_model_checks(root)
    expected=(;schema="matched_gbm_scalar342_model_v1",feature_count=342,feature_schema_sha256=SCHEMA,
        evotrees_version="0.18.7",evotrees_source_sha256=EVO,cell_delay=0,cell_horizon=4,setting_index=1,
        max_depth=4,nrounds=200,nbins=64,seed=22022026,training_mask_sha256=HEX,
        preprocessing_sha256=HEX,input_admission_sha256=HEX,feature_contract_sha256=FEATURE,loss_contract_sha256=LOSS)
    tree(n)=Dict{String,Any}("feature"=>zeros(Int,n),"condition"=>zeros(Int,n),
        "prediction_f32_bits"=>fill("00000000",n),"split"=>fill(false,n))
    model=plain(expected);model["numeric_feature_types"]=fill(true,342)
    model["edges_f64_bits"]=[fill("0000000000000000",63) for _ in 1:342]
    model["trees"]=vcat([tree(1)],[tree(15) for _ in 1:200])
    stump=identified(model,"model_id")
    path,sha=member(root,"stump.json",body(stump))
    # A valid ordinary saved model is insufficient full-shape resource evidence.
    @test Issuer.Audit.decode_model(path,sha,expected,1)[2]==stump["model_id"]
    @test_throws ArgumentError Issuer.saved_model_facts(path,sha,expected,1)
    split=deepcopy(model);split["trees"][2]["feature"][1]=1;split["trees"][2]["condition"][1]=1;split["trees"][2]["split"][1]=true
    split=identified(split,"model_id");path,sha=member(root,"one-split.json",body(split))
    @test Issuer.saved_model_facts(path,sha,expected,1)==(;model_id=split["model_id"],nonbias_split_count=1)
    @test_throws ArgumentError Issuer.saved_model_facts(path,repeat("c",64),expected,1)
    @test_throws ArgumentError Issuer.saved_model_facts(path,sha,merge(expected,(;cell_delay=1)),1)
    @test_throws ArgumentError Issuer.saved_model_facts(path,sha,expected,2)
    unreachable=deepcopy(split);unreachable["trees"][2]["split"][4]=true
    unreachable["trees"][2]["feature"][4]=1;unreachable["trees"][2]["condition"][4]=1
    unreachable=identified(unreachable,"model_id");up,us=member(root,"unreachable.json",body(unreachable))
    @test_throws ArgumentError Issuer.saved_model_facts(up,us,expected,1)
end

function measurement_checks(root)
    d=identified((;schema="matched342_julia_process_measurement_v1",complete=true,evidence_mode="synthetic_only",
        head=HEAD,source_proof_sha256=HEX,fixture_spec_sha256=HEX,fixture_sha256=HEX,runtime_sha256=HEX,
        julia_pid=123,scope="initial32_producer_and_independent_saved_state_audit",observed=true,
        peak_rss_bytes=budgets().peak_rss_bytes,peak_rss_method="Julia Sys.maxrss bytes; OS process-lifetime maximum",
        julia_gc_allocated_bytes=budgets().julia_gc_allocated_bytes,allocation_method="Julia Base.@timed.bytes; GC_Diff.allocd managed cumulative bytes",
        elapsed_ns=119999999999,elapsed_method="Julia time_ns difference around complete timed body",
        descriptors=(;kind="posix_file_descriptors",method="Linux /proc/self/fd count on Julia process",before=10,after=12,supported=true,observed=true)),"measurement_id")
    check(x)=Issuer.check_measurement_metadata(parsed(x);head=HEAD,proof_sha=HEX,spec_sha=HEX,fixture_sha=HEX,
        runtime_sha=HEX,pid=123,budgets=budgets(),limits=limits())
    @test length(d)==19&&check(d).measurement_id==d["measurement_id"]
    for (key,value) in (("julia_pid",124),("julia_pid",true),("complete",false),("observed",false),
            ("evidence_mode","software_qualified"),("head",repeat("c",40)),("runtime_sha256",repeat("c",64)),
            ("peak_rss_bytes",8589934593),("peak_rss_bytes",true),("julia_gc_allocated_bytes",123456790),
            ("elapsed_ns",120000000000),("elapsed_ns",0),("peak_rss_method","current RSS samples"),
            ("allocation_method","native allocation total"),("elapsed_method","Python monotonic epoch"),("future",nothing))
        @test_throws ArgumentError check(changed(d,key,value;id="measurement_id"))
    end
    desc=d["descriptors"]
    for bad in ((;after=13),(;supported=false),(;observed=false),(;before=true),(;method="unobserved"),(;kind="unknown"))
        @test_throws ArgumentError check(changed(d,"descriptors",merge(freeze(desc),bad);id="measurement_id"))
    end
    for (kind,method) in (("windows_handles","Windows GetProcessHandleCount on Julia current process"),
            ("posix_file_descriptors","macOS /dev/fd count on Julia process"))
        @test check(changed(d,"descriptors",merge(freeze(desc),(;kind,method));id="measurement_id")).complete
    end
    @test check(changed(d,"descriptors",merge(freeze(desc),(;after=9));id="measurement_id")).complete
    summary=(;profile_request=ref(root,"original-profile.json"),julia_pid=123,profile_result_sha256=HEX,
        measurement_sha256=HEX,fixture_sha256=HEX,runtime_sha256=HEX,source_proof_sha256=HEX)
    raw,_=member(root,"summary.log",vcat(codeunits("ordinary output\nMATCHED342_PROFILE_COMPLETE "),codeunits(JSON3.write(summary)),codeunits("\n")))
    @test Issuer.completion_summary(raw).profile_request.sha256==HEX
    duplicate,_=member(root,"duplicate-summary.log",vcat(read(raw),read(raw)))
    @test_throws ArgumentError Issuer.completion_summary(duplicate)
    missing,_=member(root,"missing-summary.log",codeunits("ordinary output\n"))
    @test_throws ArgumentError Issuer.completion_summary(missing)
end

function run_tests(parent=realpath(tempdir()))
    @testset "Unqualified profile metadata and independent resource predicates" begin
        mktempdir(parent;prefix="matched342-profile-contract-") do root
            metadata_checks(root)
            cells=calendar_and_response_checks()
            encoder_checks(cells)
            saved_model_checks(root)
            measurement_checks(root)
        end
    end
end
end
