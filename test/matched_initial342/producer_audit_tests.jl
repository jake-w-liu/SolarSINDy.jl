const Producer=Entry.BMatchedInitialProducer
const Audit=Entry.MatchedReferenceAudit342
# Frozen independent oracle/interface V2, supplied before the auditor source.
# This static contract is distinct from every subsequently generated CA hash.
const AUDIT_CONTRACT_SHA="60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a"

function genuine_whole_source_proof(root)
    git(args...)=FC._git(args...)
    @assert isempty(git("status","--porcelain=v1","--untracked-files=all")) "Full generated audit requires a real clean committed source tree"
    head=strip(String(git("rev-parse","HEAD")))
    tree=strip(String(git("rev-parse","HEAD^{tree}")))
    hashes=Dict{String,Any}();objects=Dict{String,Any}()
    for row in split(String(git("ls-files","--stage","-z")),'\0';keepempty=false)
        header,name=split(row,'\t';limit=2)
        mode,oid,stage=split(header)
        @assert mode in ("100644","100755")&&stage=="0"&&!haskey(objects,name)
        path=joinpath(pkgdir(SolarSINDy),split(name,'/')...)
        hashes[name]=b2_file(path);objects[name]=String(oid)
    end
    proof=Dict{String,Any}("complete"=>true,"head"=>head,"tree"=>tree,"source_files"=>length(hashes),
        "source_sha256"=>hashes,"commit_object_ids"=>objects,"all_clean_and_checkout_filters_match"=>true)
    path=joinpath(root,"whole-source-proof.json");sha=b2_write(path,proof)
    # The production reader independently verifies every committed raw object,
    # clean filter and checkout filter before this proof can authorize auditing.
    FC.recheck_source_proof((;path,sha256=sha);expected_head=head,expected_sources=Entry.source_identity())
    (;head,path,sha256=sha)
end
function synthetic_parser_qualification(root,proof)
    # Test DATA for the synthetic-only qualification parser. These named cells
    # are generated fixtures, not receipts claiming that GitHub jobs executed.
    native=Any[];other=Any[]
    names=vcat(["Julia $v / $os" for v in ("1.12.6","1.13.1") for os in ("ubuntu-latest","windows-latest","macos-latest")],
        ["Documentation build","Julia 1.12 deployment images"])
    for (k,name) in enumerate(names)
        path=joinpath(root,"synthetic-parser-cell-$k.json")
        sha=b2_write(path,Dict("schema"=>"synthetic_parser_cell_fixture_v1","evidence_mode"=>"synthetic_only","name"=>name))
        row=Dict("name"=>name,"conclusion"=>"success","receipt_sha256"=>sha,"log_sha256"=>sha)
        push!(k<=6 ? native : other,row)
    end
    witness=joinpath(root,"synthetic-parser-root.json")
    witness_sha=b2_write(witness,Dict("schema"=>"synthetic_parser_qualification_witness_v1","evidence_mode"=>"synthetic_only"))
    data=Dict{String,Any}("schema"=>"b_reference_component_software_qualification_v1","complete"=>true,
        "evidence_mode"=>"synthetic_only","head"=>proof.head,"source_sha256"=>Dict(Entry.source_identity()),
        "harness"=>Dict("counts"=>Dict("PASS"=>1,"WARN"=>0,"FAIL"=>0),"all17_independent_workers_complete"=>true,
            "unfiltered_pkg_test_complete"=>true,"experiments_entry_complete"=>true),
        "native_jobs"=>native,"docs_and_containers"=>other,"root_qualification_sha256"=>witness_sha,
        "source_proof_sha256"=>proof.sha256,"canonical_harness_sha256"=>witness_sha,
        "software_qualification_only"=>true,"actual_panel_fit_or_outcome_qualified"=>false,"serving_promotion"=>false)
    path=joinpath(root,"synthetic-parser-qualification.json");sha=b2_write(path,data)
    (;path,sha256=sha)
end
function restore_bytes_after(f,path)
    original=read(path)
    try
        f(original)
    finally
        open(io->write(io,original),path,"w")
    end
end
function full_producer_audit_checks(root;flat)
    f=source_pair_fixture(root;flat,routing_stress=true)
    controls=mkdir(joinpath(root,"controls"));proof=genuine_whole_source_proof(controls)
    qualification=synthetic_parser_qualification(controls,proof)
    feature=(;path=joinpath(pkgdir(SolarSINDy),Entry.FEATURE_CONTRACT_MEMBER),sha256=Entry.FEATURE_CONTRACT_SHA256,member=Entry.FEATURE_CONTRACT_MEMBER)
    loss=(;path=joinpath(pkgdir(SolarSINDy),Entry.LOSS_CONTRACT_MEMBER),sha256=Entry.LOSS_CONTRACT_SHA256,member=Entry.LOSS_CONTRACT_MEMBER)
    draft=Producer.produce_initial_generated(f.admissions...,joinpath(root,"initial32");qualification,
        feature_contract=feature,loss_contract=loss,qualified_head=proof.head,expected_sources=Entry.source_identity())
    meta=Producer.draft_metadata(draft);component=b2_json(meta.component_completion_path)
    @test meta.evidence_mode=="synthetic_only"
    @test length(component["members"])==GEOMETRY["component_noncompletion_members"]
    @test component["execution_authority_sha256"]===nothing
    @test length(component["cells"])==4
    @test sum(length(c["models"]) for c in component["cells"])==32
    @test all(!ispath(joinpath(p,"completion.json"))&&!ispath(joinpath(p,"audit.json")) for p in meta.reference_roots)
    @test_throws MethodError Producer.publish_initial(draft,nothing;expected_audit_completion_sha256=repeat("0",64))
    reopened=Producer.read_draft(f.pair,meta.component_completion_path;expected_component_sha256=meta.component_completion_sha256,
        expected_sources=Entry.source_identity())
    @test Producer.draft_metadata(reopened)==meta
    for (k,cell) in enumerate(component["cells"])
        oracle=GEOMETRY["cells"][k]
        @test (cell["delay_hours"],cell["horizon"])==(oracle["delay"],oracle["horizon"])
        @test cell["training_count"]==oracle["train_count"]
        @test cell["validation_count"]==oracle["validation_count"]
        @test cell["union_count"]==oracle["union_count"]
        @test cell["validation_persistence_count"]==4
        @test length(unique(m["model_id"] for m in cell["models"]))==8
        @test all(m["native_increment_sha256"]==m["reloaded_increment_sha256"] for m in cell["models"])
    end
    for (k,d) in enumerate(component["drafts"])
        @test length(d["members"])==20
        @test d["persistence_count"]==4
        @test d["model_count"]==GEOMETRY["cells"][2k-1]["union_count"]-4
        flat&&(@test d["selected_setting_ids"]==[1,1])
        routepath=joinpath(meta.component_root,d["routing_member"]["name"])
        open(routepath,"r") do io
            @test String(read(io,8))=="SSBMRTE1"
            @test ltoh(read(io,UInt64))==1
            route=read(io)
            expected=zeros(UInt8,GEOMETRY["cells"][2k-1]["union_count"])
            for issue in (600,650,700,730);expected[issue-512+1]=1;end
            @test route==expected
            @test route[620-512+1]==0
        end
    end
    contracts=(;protocol_sha256="7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404",
        feature_contract_sha256=Entry.FEATURE_CONTRACT_SHA256,feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,audit_contract_sha256=AUDIT_CONTRACT_SHA,
        evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256)
    auditroot=joinpath(root,"independent-audit")
    request=Dict{String,Any}("schema"=>"matched342_initial_component_audit_request_v1","evidence_mode"=>"synthetic_only",
        "operation"=>"independent_saved_state_audit_initial32","component"=>Dict("path"=>meta.component_completion_path,"sha256"=>meta.component_completion_sha256),
        "component_qualification"=>qualification,"source_proof"=>(;path=proof.path,sha256=proof.sha256),
        "source_bindings"=>component["source_bindings"],"contracts"=>contracts,"runtime"=>component["runtime"],
        "execution_authority"=>nothing,"drafts"=>[Dict("delay_hours"=>k-1,"root"=>meta.reference_roots[k],"members"=>component["drafts"][k]["members"]) for k in 1:2],
        "output_directory"=>auditroot)
    request["request_id"]=b2_id(request)
    requestpath=joinpath(controls,"independent-audit-request.json");requestsha=b2_write(requestpath,request)
    # A changed saved state fails before an audit store/completion is written.
    modelpath=joinpath(meta.component_root,component["cells"][1]["models"][1]["model_member"]["name"])
    restore_bytes_after(modelpath) do body
        changed=copy(body);changed[end]=UInt8(' ');open(io->write(io,changed),modelpath,"w")
        @test_throws ArgumentError Producer.recheck_draft(draft)
        @test_throws ArgumentError Audit.audit_component(f.pair,requestpath;request_sha256=requestsha,
            expected_sources=Entry.source_identity(),expected_audit_contract_sha256=AUDIT_CONTRACT_SHA)
        @test !ispath(auditroot)
    end
    checked=Audit.audit_component(f.pair,requestpath;request_sha256=requestsha,
        expected_sources=Entry.source_identity(),expected_audit_contract_sha256=AUDIT_CONTRACT_SHA)
    auditmeta=Audit.component_audit_metadata(checked)
    capath=joinpath(auditroot,"audit-completion.json");casha=b2_file(capath);ca=b2_json(capath)
    @test length(readdir(auditroot))==4
    @test ca["source_proof"]["sha256"]==proof.sha256
    @test ca["execution_authority"]===nothing
    @test casha==auditmeta.audit_completion_sha256
    report=b2_json(joinpath(auditroot,"audit-report.json"))
    @test report["fitting_performed"]===false&&report["native_model_deserialization"]===false
    @test sum(length(c["models"]) for c in report["cells"])==32
    for c in report["cells"],m in c["models"]
        @test m["native_increment_sha256"]==m["reloaded_increment_sha256"]==m["audited_increment_sha256"]
    end
    restart=Audit.read_component_audit(f.pair,capath,casha;expected_component_sha256=meta.component_completion_sha256,
        expected_sources=Entry.source_identity(),expected_audit_contract_sha256=AUDIT_CONTRACT_SHA)
    @test Audit.component_audit_metadata(restart)==auditmeta
    restore_bytes_after(joinpath(auditroot,"audit-report.json")) do body
        changed=copy(body);changed[end]=UInt8(' ')
        open(io->write(io,changed),joinpath(auditroot,"audit-report.json"),"w")
        @test_throws ArgumentError Audit.recheck_component_audit(checked)
    end
    published=Producer.publish_initial(draft,restart;expected_audit_completion_sha256=casha)
    @test length(published.references)==2
    @test_throws ArgumentError Producer.publish_initial(draft,restart;expected_audit_completion_sha256=casha)
    for (k,ref) in enumerate(published.references)
        completion=b2_json(ref.completion_path)
        @test length(completion["members"])==GEOMETRY["reference_noncompletion_members"]
        rootpath=dirname(ref.completion_path)
        files=[joinpath(p,n) for (p,_,names) in walkdir(rootpath) for n in names]
        @test length(files)==GEOMETRY["reference_total_members"]
        @test completion["audit_contract_sha256"]==AUDIT_CONTRACT_SHA
        @test completion["audit_sha256"]==b2_file(joinpath(rootpath,"audit.json"))
        @test completion["selected_setting_ids"]==component["drafts"][k]["selected_setting_ids"]
        if flat
            @test completion["selected_setting_ids"]==[1,1]
            open(joinpath(rootpath,"predictions.bin"),"r") do io
                read(io,16)
                n=GEOMETRY["cells"][2k-1]["union_count"]
                @test filesize(joinpath(rootpath,"predictions.bin"))==16+16n
                for _ in 1:2n;@test reinterpret(Float64,ltoh(read(io,UInt64)))==-40.;end
                @test eof(io)
            end
        end
    end
    println("MATCHED342_FULL32_AUDIT_B22_DONE flat=",flat);flush(stdout)
end
