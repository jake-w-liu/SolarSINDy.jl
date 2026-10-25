# The expected field family and bounds precede the production correction.
const FIT_HEADER_INTEGER_ORACLES=(
    (:setting_index,1,36),(:a_setting_index,1,18),(:beta,5,20),
    (:delay_hours,0,1),(:max_source_rows,1,500000),(:max_chunk_rows,1,4096),
    (:total_source_rows,1,500000),(:joint_origins,1,500000),
    (:informative_origins,0,500000),(:fit_scalar_rows,0,1000000),
    (:storm_scalar_rows,0,1000000),(:data_bytes,16,536870912),(:mask_bytes,1,536870912))

function fit_header_grammar_checks(root,result)
    a=result.admission;stage=result.stage.directory
    laws=ntuple(2) do i
        path=joinpath(stage,"laws","beta-$((5,20)[i])")
        T.D.load_model(path;expected_manifest_sha256=T._sha(read(joinpath(path,"MANIFEST.json"))))
    end
    replay()=P._verify_basis(a.view,laws,joinpath(stage,"basis");expected_manifest_sha256=T._sha(read(joinpath(stage,"basis/manifest.json"))),expected_sources=P.source_identity(),chunk_hours=7)
    proof=replay();panel=P._read_verified_panel(proof,a.view,1,1)
    context=B.BInitialPointFits._export_context(a,panel,nothing)
    bundle=joinpath(stage,"models/setting-1")
    loaded=P.load_point_bundle(bundle;expected_manifest_sha256=T._sha(read(joinpath(bundle,"manifest.json"))),expected_sources=P.source_identity(),expected_binding=context)
    fit=joinpath(stage,"fits/setting-1")
    @test B.BInitialPointFits._verify_fit_input_readback(fit,loaded.model,panel,a.view,T.settings()[1])===nothing
    header=plain(JSON3.read(read(joinpath(fit,"manifest.json"))))
    @test length(header)==41&&length(FIT_HEADER_INTEGER_ORACLES)==13

    # Every modified header gets a matching new model identity and fit pin.
    # These negatives therefore cannot pass merely by tripping the old pin.
    for field in sort!(collect(keys(header)))
        clone=joinpath(root,"header-field-"*field);cp(fit,clone)
        data=deepcopy(header)
        data[field]=field in ("max_source_rows","max_chunk_rows") ? 0 : field=="complete" ? false : "invalid-field-value"
        body=T._bytes(data);write(joinpath(clone,"manifest.json"),body)
        modeldata=plain(JSON3.read(T._bytes(T._model_data(loaded.model;identity=true))))
        modeldata["fit_input_manifest_sha256"]=T._sha(body)
        modeldata["model_id"]=T._id(Dict(k=>v for (k,v) in modeldata if k!="model_id"))
        model=P._decode_model(JSON3.read(T._bytes(modeldata)),loaded.model.law)
        @test model.fit_input_manifest_sha256==T._sha(body)
        @test read(joinpath(clone,"joint-fit.bin"))==read(joinpath(fit,"joint-fit.bin"))
        @test read(joinpath(clone,"fit-mask.ascii"))==read(joinpath(fit,"fit-mask.ascii"))
        @test_throws ArgumentError B.BInitialPointFits._verify_fit_input_readback(clone,model,panel,a.view,T.settings()[1])
    end

    # Public fit-entry negatives use a fresh independent replay with an empty
    # certificate ledger. A stale or conflicting certificate cannot reject them.
    fresh=replay();freshpanel=P._read_verified_panel(fresh,a.view,1,1)
    @test isempty(fresh.certificates)
    for (field,lo,hi) in FIT_HEADER_INTEGER_ORACLES
        for (kind,value) in (("bool",false),("float",Float64(lo)),("string",string(lo)),("below",lo-1),("above",hi+1))
            clone=joinpath(root,"public-fit-"*String(field)*"-"*kind);cp(fit,clone)
            data=deepcopy(header);data[String(field)]=value;body=T._bytes(data);write(joinpath(clone,"manifest.json"),body)
            @test_throws ArgumentError P.fit_point_gains_from_input(clone;expected_manifest_sha256=T._sha(body),expected_sources=P.source_identity(),expected_context=freshpanel.training.binding,view=a.view,basis=freshpanel)
            @test isempty(fresh.certificates)
        end
    end

    envelope=plain(JSON3.read(read(a.envelope_path)))
    admission=plain(JSON3.read(read(joinpath(dirname(a.envelope_path),"admission.json"))))
    for field in ("prefix_rows","receipt_delay_hours")
        changed=deepcopy(admission);changed[field]=false;body=T._bytes(changed)
        path=joinpath(root,"proof-"*field*".json");write(path,body)
        @test_throws ArgumentError I._proof(path,T._sha(body),JSON3.read(T._bytes(envelope)),a.view.evidence_mode)
    end
    closure=JSON3.read(read(joinpath(bundle,"checkpoints.json")))
    @test P._decode_a(closure.a_checkpoint) isa T.Stream.StreamCheckpoint
    for value in (false,1,"invalid",Any[])
        changed=plain(closure.a_checkpoint);changed["pending"]=value
        @test_throws ArgumentError P._decode_a(JSON3.read(T._bytes(changed)))
    end
    @test result.verified.certificate_count==36&&result.verified.gain_fit_calls==0&&result.verified.hazard_fit_calls==0
    true
end
