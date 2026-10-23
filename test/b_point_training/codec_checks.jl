plain(x::JSON3.Object)=Dict{String,Any}(String(k)=>plain(v) for (k,v) in pairs(x))
plain(x::JSON3.Array)=[plain(v) for v in x]
plain(x)=x
function codec_checks(root,result)
    a=result.admission;stage=result.stage.directory
    laws=ntuple(2) do i
        path=joinpath(stage,"laws","beta-$((5,20)[i])")
        T.D.load_model(path;expected_manifest_sha256=T._sha(read(joinpath(path,"MANIFEST.json"))))
    end
    basis=joinpath(stage,"basis");proof=P._verify_basis(a.view,laws,basis;expected_manifest_sha256=T._sha(read(joinpath(basis,"manifest.json"))),expected_sources=P.source_identity(),chunk_hours=7)
    panel=P._read_verified_panel(proof,a.view,1,1);context=B.BInitialPointFits._export_context(a,panel,nothing)
    bundle=joinpath(stage,"models","setting-1");pin=T._sha(read(joinpath(bundle,"manifest.json")))
    genuine=P.load_point_bundle(bundle;expected_manifest_sha256=pin,expected_sources=P.source_identity(),expected_binding=context)
    @test genuine.provenance.initial_hazard_fit_count==2&&genuine.provenance.initial_gain_fit_count==36
    @test private_context_checks(root,a,panel,context,genuine.model)
    fitonly=P.BFitExportContext(context.binding,context.view,context.basis)
    @test_throws ArgumentError P.load_point_bundle(bundle;expected_manifest_sha256=pin,expected_sources=P.source_identity(),expected_binding=fitonly)
    record(data)=NamedTuple{Tuple(Symbol.(collect(keys(data))))}(Tuple(values(data)))
    positive=joinpath(root,"positive-resaved-bundle")
    saved=P.save_point_bundle(positive,genuine.model,genuine.a_checkpoint,genuine.wind_checkpoint;provenance=(;data=record(plain(genuine.provenance)),verification=context))
    @test P.load_point_bundle(positive;expected_manifest_sha256=saved.manifest_sha256,expected_sources=P.source_identity(),expected_binding=context).model.model_id==genuine.model.model_id
    mutations=[
        :schema=>"wrong",:stage=>"calibration",:evidence_mode=>"archival_assumed_completion_research",
        :source_sha256=>repeat("0",64),:input_view_id=>repeat("0",64),:input_admission_sha256=>repeat("0",64),:input_prefix_sha256=>repeat("0",64),
        :reference_completion_sha256=>repeat("0",64),:independent_audit_completion_sha256=>repeat("0",64),
        :training_boundary=>string(Dates.value(a.metadata.training_boundary+Hour(1))),:validation_start=>string(Dates.value(a.metadata.validation_start-Hour(1))),:validation_end=>string(Dates.value(a.metadata.validation_end+Hour(1))),
        :basis_sha256=>repeat("0",64),:fit_input_manifest_sha256=>repeat("0",64),
        :initial_hazard_fit_count=>3,:initial_gain_fit_count=>35,:final_fit_count=>1,:calibration_fit_count=>1,
        :exposed_development_scoring=>true,:scientific_verdict=>true,:serving_promotion=>true,:actual_execution_permit_sha256=>repeat("d",64),
        :final_fit_count=>false,:calibration_fit_count=>false,:initial_hazard_fit_count=>true,:initial_gain_fit_count=>true]
    for (index,(field,value)) in enumerate(mutations)
        changed=plain(genuine.provenance);changed[String(field)]=value
        output=joinpath(root,"rejected-save-$index")
        @test_throws ArgumentError P.save_point_bundle(output,genuine.model,genuine.a_checkpoint,genuine.wind_checkpoint;provenance=(;data=record(changed),verification=context))
        @test !ispath(output)
        clone=joinpath(root,"rehashed-provenance-$index");cp(bundle,clone)
        body=T._bytes(changed);write(joinpath(clone,"provenance.json"),body)
        manifest=plain(JSON3.read(read(joinpath(clone,"manifest.json"))))
        for member in manifest["members"]
            if member["name"]=="provenance.json";member["bytes"]=length(body);member["sha256"]=T._sha(body);end
        end
        bytes=T._bytes(manifest);write(joinpath(clone,"manifest.json"),bytes)
        @test_throws ArgumentError P.load_point_bundle(clone;expected_manifest_sha256=T._sha(bytes),expected_sources=P.source_identity(),expected_binding=context)
    end
    @test length(mutations)==26
    fit=joinpath(stage,"fits","setting-1");clone=joinpath(root,"fit-Bool");cp(fit,clone)
    fields=plain(JSON3.read(read(joinpath(clone,"manifest.json"))));fields["a_setting_index"]=true
    bytes=T._bytes(fields);write(joinpath(clone,"manifest.json"),bytes)
    @test_throws ArgumentError P.fit_point_gains_from_input(clone;expected_manifest_sha256=T._sha(bytes),expected_sources=P.source_identity(),expected_context=context.binding,view=a.view,basis=panel)
    # Genuine D artifact equality is independent of B's embedded two-member law.
    genuine_law=joinpath(root,"genuine-duration-law");saved=T.D.save_model(genuine_law,laws[1]);@test P._preflight_duration_artifact(genuine_law)
    loaded=T.D.load_model(genuine_law;expected_manifest_sha256=saved.manifest_sha256)
    @test loaded.model_id==laws[1].model_id&&T.D._json(T.D._data(loaded))==T.D._json(T.D._data(laws[1]))
    @test_throws ArgumentError T.D.load_model(joinpath(bundle,"hazard");expected_manifest_sha256=T._sha(read(joinpath(bundle,"hazard","manifest.json"))))
    metadata=joinpath(root,"metadata-only");mkdir(metadata)
    for (label,start) in (("valid-Nov1",DateTime(1976,11,1)),("false-Jan1",DateTime(1976)))
        input=plain(JSON3.read(read(joinpath(dirname(a.envelope_path),"envelope.json"))))
        stop=DateTime(2023);n=Int((Dates.value(stop)-Dates.value(start))÷3_600_000)
        merge!(input,Dict("evidence_mode"=>"archival_assumed_completion_research","source_sha256"=>I.ACTUAL_SOURCE,"prefix_start_ms"=>string(Dates.value(start)),"prefix_stop_ms"=>string(Dates.value(stop)),"prefix_rows"=>n,"training_boundary_ms"=>string(Dates.value(DateTime(2020))),"validation_start_ms"=>string(Dates.value(DateTime(2020))),"validation_end_ms"=>string(Dates.value(stop)),"raw_prefix_bytes"=>136n,"raw_prefix_member"=>"metadata/no-actual-body.bin","members"=>Any[]))
        bytes=T._bytes(input);path=joinpath(metadata,label*".json");P._exclusive(path,bytes)
        error=try I.admit_prefix(path;expected_envelope_sha256=T._sha(bytes),roots=Dict("metadata"=>metadata),expected_sources=P.source_identity(),independent_admission=nothing);nothing catch e;e;end
        @test error isa ArgumentError
        @test occursin(label=="valid-Nov1" ? "member count" : "archival source/split",sprint(showerror,error))
    end
    @test all(endswith(path,".json") for path in readdir(metadata)) # zero raw members
    true
end
function completion_checks(root,result)
    a=result.admission;stage=result.stage.directory;m=a.metadata
    changes=["evidence_mode"=>"archival_assumed_completion_research","source_sha256"=>repeat("0",64),"training_semantics"=>"wrong","training_boundary"=>string(Dates.value(m.training_boundary+Hour(1))),"validation_start"=>string(Dates.value(m.validation_start-Hour(1))),"validation_end"=>string(Dates.value(m.validation_end+Hour(1))),"actual_execution_permit_sha256"=>repeat("d",64),"qualified_head"=>repeat("e",40),"source_rows"=>a.view.source_hours+1,"retained_prefix_rows"=>a.view.source_hours+1,"input_envelope_sha256"=>repeat("0",64),"reference_completion_sha256"=>repeat("0",64),"reference_audit_sha256"=>repeat("0",64),"input_prefix_sha256"=>repeat("0",64),"initial_gain_fit_count"=>35,"scientific_verdict"=>true,"input_view_id"=>repeat("0",64)]
    for (index,(field,value)) in enumerate(changes)
        clone=joinpath(root,"stage-metadata-$index");cp(stage,clone)
        data=plain(JSON3.read(read(joinpath(clone,"completion.json"))));data[field]=value
        bytes=T._bytes(data);write(joinpath(clone,"completion.json"),bytes)
        @test_throws ArgumentError B.BInitialPointFits.verify_initial_l0(clone,a,result.reference;expected_completion_sha256=T._sha(bytes),expected_sources=P.source_identity(),chunk_hours=2)
    end
    @test length(changes)==17
    true
end
