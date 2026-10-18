# Frozen before implementation: generated fixtures, literal metadata type oracles.
using Test,Dates,SHA
const RF=DISK_FIT_NAMESPACE
const RM=DISK_METADATA
const RECIPE_OUTPUT=abspath(ARGS[2])
const RECIPE_SOURCE_BEFORE=RF.source_identity()

function equivalent_recipe_variants(setting)
    [merge(setting,(index=Int32(setting.index),)),
     merge(setting,(taus=Float32.(setting.taus),)),
     merge(setting,(sigma=Float32(setting.sigma),)),
     merge(setting,(eta=Float32(setting.eta),)),
     (index=Int32(setting.index),taus=Float32.(setting.taus),
      sigma=Float32(setting.sigma),eta=Float32(setting.eta))]
end

@testset "accepted recipe representations produce canonical model metadata" begin
    for delay in (0,1), index in (1,2)
        setting=RF.settings()[index]
        rows=RM.metadata_rows(setting,delay)
        normal=RF.fit_gains(rows,RM.MBOUNDARY,setting;
            delay_hours=delay,source_sha256=RM.MSOURCE)
        @test RF.DstIntervalArtifacts._validate_model(normal)===normal
        for (kind,alternate) in enumerate(equivalent_recipe_variants(setting))
            @test alternate==setting && alternate in RF.settings()
            input_before=bytes2hex(sha256(repr((rows,alternate))))
            model=RF.fit_gains(rows,RM.MBOUNDARY,alternate;
                delay_hours=delay,source_sha256=RM.MSOURCE)
            @test model.setting_index isa Int
            @test model.taus isa NTuple{2,Float64}
            @test model.sigma isa Float64
            @test model.eta isa Float64
            @test isequal(model,normal)
            @test model.model_id==normal.model_id
            @test model.gains[1]≈2. atol=1e-12 rtol=0
            @test model.gains[2]≈3. atol=1e-12 rtol=0
            validation_error=try
                RF.DstIntervalArtifacts._validate_model(model)
                nothing
            catch error
                error
            end
            @test validation_error===nothing
            @test isequal(RF.predict_rows(model,rows),RF.predict_rows(normal,rows))
            exported=RF.export_fit_input(joinpath(RECIPE_OUTPUT,"typed-L$delay-R$index-K$kind"),
                RM.MBOUNDARY,alternate;delay_hours=delay,source_sha256=RM.MSOURCE) do sink
                for i in 1:13:length(rows)
                    sink(rows[i:min(i+12,end)])
                end
            end
            disk=RF.fit_gains_from_input(exported.directory;
                expected_manifest_sha256=exported.manifest_sha256)
            @test isequal(disk,normal)
            @test disk.model_id==model.model_id
            @test bytes2hex(sha256(repr((rows,alternate))))==input_before
        end
        off_grid=merge(setting,(sigma=nextfloat(setting.sigma),))
        @test !(off_grid in RF.settings())
        @test_throws ArgumentError RF.fit_gains(rows,RM.MBOUNDARY,off_grid;
            delay_hours=delay,source_sha256=RM.MSOURCE)
        destination=joinpath(RECIPE_OUTPUT,"off-grid-L$delay-R$index")
        called=Ref(false)
        @test_throws ArgumentError RF.export_fit_input(sink->(called[]=true),
            destination,RM.MBOUNDARY,off_grid;delay_hours=delay,source_sha256=RM.MSOURCE)
        @test !called[] && !ispath(destination)
    end
end

@testset "equivalent recipe fit survives actual stream bundle roundtrip" begin
    for delay in (0,1)
        origin=DateTime(2018)
        values=ones(17,900)
        values[8,:].=400.;values[6,:].=-5.;values[12,:].=2.;values[15,:].=-30.
        digest=bytes2hex(sha256("generated canonical recipe representation stream"))
        setting=RF.settings()[9]
        input=RF.HourlyInputs([origin+Hour(i-1) for i in 1:900],values;source_sha256=digest)
        rows=NamedTuple[]
        streamed=RF.stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,
            chunk_hours=7,sink=part->append!(rows,part))
        normal=RF.fit_gains(rows,origin+Hour(700),setting;delay_hours=delay,source_sha256=digest)
        alternate=last(equivalent_recipe_variants(setting))
        model=RF.fit_gains(rows,origin+Hour(700),alternate;delay_hours=delay,source_sha256=digest)
        @test isequal(model,normal)
        validation_error=try
            RF.DstIntervalArtifacts._validate_model(model)
            nothing
        catch error
            error
        end
        @test validation_error===nothing
        # Let the type assertions above fail directly on the old source without
        # replacing their failure with an unrelated unsupported-checkpoint error.
        if validation_error===nothing
            saved=RF.save_bundle(joinpath(RECIPE_OUTPUT,"typed-bundle-L$delay"),model,
                streamed.checkpoint;provenance=(synthetic_fixture=true,input_sha256=digest))
            loaded=RF.load_bundle(saved.directory;expected_manifest_sha256=saved.manifest_sha256,
                expected_sources=RF.DstIntervalArtifacts.verify_source_identity())
            @test isequal(loaded.model,normal)
            @test isequal(loaded.checkpoint,streamed.checkpoint)
            @test isequal(RF.predict_rows(loaded.model,rows),RF.predict_rows(normal,rows))
        end
    end
end
@test RF.source_identity()==RECIPE_SOURCE_BEFORE
