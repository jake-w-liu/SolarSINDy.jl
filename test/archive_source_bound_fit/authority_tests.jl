function run_authority_tests(results)
    @testset "Actual case lineage and numerical fitting without execution certificates" begin
        entry=results[("primary","ground-only")];value=entry.value;root=entry.root
        reread=B.read_reverified(joinpath(root,"case-request.json"),joinpath(root,"cases");request_sha256=value.request_sha256,
            manifest_sha256=value.manifest_sha256,complete_sha256=value.complete_sha256)
        r=recipe();start=START+Minute(40);stop=START+Minute(120);asof=START+Minute(211)
        @test_throws ArgumentError F.fit_archival(reread,1,r,entry.fit.capsule.expected;start,stop,asof)
        @test_throws ArgumentError F._recipe_bytes(R.canonical_bytes(["foreign"];limit=16384))
        @test_throws ArgumentError F._fit_clock("2020-02-30T00:00:00.000Z")
        @test_throws ArgumentError F._fit_clock("2020-01-01T00:00:00Z")
        @test F._fit_clock("2020-01-01T01:02:45.001Z")==START+Minute(62)+Second(45)+Millisecond(1)
        mktempdir() do temporary
            directory=realpath(temporary);epoch=DateTime(2017)
            cases=make_cases(directory,"primary","ground-only";epoch,actual=true)
            actual=B.read_reverified(joinpath(directory,"case-request.json"),joinpath(directory,"cases");
                request_sha256=cases.request_sha256,manifest_sha256=cases.manifest_sha256,complete_sha256=cases.complete_sha256)
            @test actual.cases.actual_source_authority_granted && !actual.cases.actual_fit_performed
            start=epoch+Minute(40);stop=epoch+Minute(120);asof=epoch+Minute(211)
            expected=F.prepare_expected(actual,1,r;start,stop,asof)
            fitted=F.fit_archival(actual,1,r,expected;start,stop,asof)
            arrays=Q.split_arrays(actual.cases,1;start,stop,asof)
            @test fitted.training_rows==50 && arrays.targets==vcat(fill(5.,20),fill(10.,30))
            @test reinterpret(UInt64,F.predict_log(fitted.capsule,arrays.features))==reinterpret(UInt64,SolarSINDy.gbm_predict(fitted.native,arrays.features))
            pins=F.publish_fit(fitted.capsule,joinpath(directory,"point"))
            saved=F.read_published(joinpath(directory,"point");expected,model_sha256=pins.model_sha256,
                receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
            @test F.predict_log(saved,arrays.features)==F.predict_log(fitted.capsule,arrays.features)
            receipt=R.decode_canonical(F.receipt_bytes(saved));@test receipt["authorization"]["schema"]=="archive_point_fit_binding_v2"
            @test receipt["runtime_observations"]["elapsed_ns"] isa Integer
            @test !(receipt["runtime_observations"]["elapsed_ns"] isa Bool)
            @test receipt["runtime_observations"]["elapsed_ns"]>=0
            wrong=F.ExpectedFit("0"^64,expected.recipe_sha256,expected.source_sha256,expected.library_sha256)
            @test_throws ArgumentError F.fit_archival(actual,1,r,wrong;start,stop,asof)
            @test_throws ArgumentError F.prepare_expected(actual,1,r;start=DateTime(2016),stop,asof)
            @test_throws ArgumentError F.fit_archival(actual,1,r,expected;start,stop,asof,authority=Dict())
            bad=joinpath(directory,"cases","manifest.json");original=read(bad);write(bad,vcat(original,UInt8(' ')))
            @test_throws ArgumentError B.read_reverified(joinpath(directory,"case-request.json"),joinpath(directory,"cases");
                request_sha256=cases.request_sha256,manifest_sha256=cases.manifest_sha256,complete_sha256=cases.complete_sha256)
        end
    end
end
