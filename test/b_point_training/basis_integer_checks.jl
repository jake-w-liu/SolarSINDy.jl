function basis_integer_checks(root)
    # Zero masks deliberately exercise Boolean false versus integer0. The
    # physical schedule1 additionally exercises Boolean true versus integer1.
    fixture=protocol_view();v=fixture.view;boundary=first(v.input.time)+Hour(32);stop=first(v.input.time)+Hour(64)
    laws=ntuple(k->T.D.fit_duration(v.wind,boundary;beta=(5,20)[k],source_sha256=v.source_sha256),2)
    for delay in (0,1)
        directory=joinpath(root,"basis-int-L$delay")
        produced=P._export_basis(v,laws,directory;delay_hours=delay,chunk_hours=7,validation_start=boundary,validation_end=stop)
        positive=P._verify_basis(v,laws,directory;expected_manifest_sha256=produced.manifest_sha256,expected_sources=P.source_identity(),chunk_hours=2)
        @test positive.view_id==v.view_id&&!any(positive.training_mask)&&!any(positive.forecast_mask)
        genuine=read(joinpath(directory,"manifest.json"))
        for (field,value) in ((:physical_schedule,true),(:beta,true),(:bytes,true),(:training_origins,false),(:forecast_origins,false))
            data=plain(JSON3.read(genuine));data["members"][1][String(field)]=value
            body=T._bytes(data);write(joinpath(directory,"manifest.json"),body)
            @test_throws ArgumentError P._verify_basis(v,laws,directory;expected_manifest_sha256=T._sha(body),expected_sources=P.source_identity(),chunk_hours=2)
            @test read(joinpath(directory,"manifest.json"))==body
            write(joinpath(directory,"manifest.json"),genuine)
        end
        @test P._verify_basis(v,laws,directory;expected_manifest_sha256=T._sha(genuine),expected_sources=P.source_identity(),chunk_hours=2).verification_id==positive.verification_id
    end
    true
end
