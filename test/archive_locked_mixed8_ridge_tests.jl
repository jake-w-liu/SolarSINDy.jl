# Distinct locked mixed8 ridge: no actual observations or archival fits.
module ArchiveLockedMixed8RidgeTests
using Dates,SHA,LinearAlgebra,Statistics,Test,SolarSINDy
const L=SolarSINDy.ArchiveLockedMixed8Ridge
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const B=SolarSINDy.ArchiveCatalogCaseReader
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"archive_source_bound_fit","case_fixture.jl"))
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
const S=SourceBoundSharedReplay

function historical_reference(X,y)
    mu=vec(mean(X,dims=1));sigma=vec(std(X,dims=1)).+1e-9
    Z=hcat(ones(size(X,1)),(X.-mu')./sigma')
    A=Z'*Z+1.0I;b=Z'*y
    (;mu,sigma,Z,A,b,beta=A\b)
end

@testset "Locked convention is distinct and has independent literal expectations" begin
    recipe=L.Recipe()
    @test recipe.sha256!=C.Recipe(kind=:ridge).sha256
    @test F.family(recipe)=="locked_mixed8_ridge_log1p_v1"
    for value in (false,0.,2.,NaN,Inf)
        @test_throws ArgumentError L.Recipe(;penalty=value)
    end
    X=zeros(2,8);y=ones(2)
    fit=L.fit_numeric(X,y,recipe)
    @test fit.model.coefficients[1]==2/3
    @test fit.model.coefficients[2:end]==ntuple(_->0.0,8)
    @test fit.model.center isa NTuple{8,Float64}
    @test fit.model.scale==ntuple(_->1e-9,8)
    @test C.fit_numeric(X,y,C.Recipe(kind=:ridge)).model.coefficients[1]==1.
    @test L.predict(fit.model,X)==fill(2/3,2)
    @test L.predict(fit.model,zeros(0,8))==Float64[]
    before=L.encode_model(fit.model);fit.native.coefficients[1]+=1.
    @test L.encode_model(fit.model)==before
    X=zeros(3,8);X[:,1]=[-1.,0.,1.];y=[0.,1.,2.]
    fit=L.fit_numeric(X,y,recipe);reference=historical_reference(X,y)
    @test fit.model.center==Tuple(reference.mu) && fit.model.scale==Tuple(reference.sigma)
    @test isequal(fit.statistics.A,reference.A) && isequal(fit.statistics.b,reference.b)
    @test isequal(collect(fit.model.coefficients),reference.beta)
    @test fit.model.coefficients[1]==3/4
    # Exact rational elimination gives slope 2s/(2+s^2), s=1+10^-9.
    exact_scale=big(1000000001)//big(1000000000)
    exact_slope=2exact_scale/(2+exact_scale^2)
    @test isapprox(fit.model.coefficients[2],Float64(exact_slope);atol=1e-14,rtol=0)
    @test fit.model.coefficients[3:end]==ntuple(_->0.0,7)
    @test L.predict(fit.model,X)==reference.Z*reference.beta
    # Distinct weights on every coordinate detect column permutations in the
    # ordered eight-feature prediction contract, independently of fitting.
    ordered=L.NumericModel(recipe,:mixed8,zeros(8),ones(8),collect(0.0:8.0))
    @test L.predict(ordered,Matrix{Float64}(I,8,8))==collect(1.0:8.0)
    @test_throws ArgumentError L.fit_numeric(zeros(2,11),ones(2),recipe)
    @test_throws ArgumentError L.fit_numeric(zeros(1,8),ones(1),recipe)
    @test_throws ArgumentError L.fit_numeric(fill(Inf,2,8),ones(2),recipe)
    @test_throws ArgumentError L.fit_numeric(zeros(2,8),[1.,NaN],recipe)
    @test_throws ArgumentError L.predict(fit.model,zeros(1,11))
    @test_throws ArgumentError L.predict(fit.model,fill(NaN,1,8))
    huge=L.NumericModel(recipe,:mixed8,[-floatmax(Float64);zeros(7)],fill(1e-9,8),ones(9))
    @test_throws ArgumentError L.predict(huge,reshape([floatmax(Float64);zeros(7)],1,8))
end

@testset "Canonical model, recipe, layout and bit bindings" begin
    fit=L.fit_numeric(zeros(2,8),ones(2),L.Recipe());bytes=L.encode_model(fit.model)
    restored=L.decode_model(bytes)
    @test L.encode_model(restored)==bytes
    @test restored.coefficients==fit.model.coefficients
    @test F._recipe_bytes(Vector{UInt8}(codeunits(fit.model.recipe.json))) isa L.Recipe
    object=R.decode_canonical(bytes;limit=L.MAX_BYTES)
    for (field,bad) in (("schema","archive_classic_point_model_v1"),("family","retained_ridge_log1p"),
                       ("mode","ground11"),("center",fill("0000000000000000",11)),
                       ("scale",fill("0000000000000000",8)),("coefficients",fill("0000000000000000",8)))
        changed=deepcopy(object);changed[field]=bad
        @test_throws ArgumentError L.decode_model(R.canonical_bytes(changed;limit=L.MAX_BYTES))
    end
    for (field,bad) in (("intercept_penalty",false),("intercept_penalty",1),
                       ("penalty","0000000000000000"),("std_rule","sample_std_zero_nonfinite_to_one"),
                       ("epsilon","3ff0000000000000"),("feature_names",reverse(collect(L.FEATURES))))
        changed=deepcopy(object);changed["recipe"][field]=bad
        @test_throws ArgumentError L.decode_model(R.canonical_bytes(changed;limit=L.MAX_BYTES))
    end
    changed=deepcopy(object);changed["coefficients"][1]="7ff8000000000000"
    @test_throws ArgumentError L.decode_model(R.canonical_bytes(changed;limit=L.MAX_BYTES))
    @test_throws MethodError L.NumericModel()
    @test_throws ArgumentError L.NumericModel(L.Recipe(),:ground11,zeros(8),ones(8),zeros(9))
end

@testset "Typed fitting, strict readback and downstream static/shared path" begin
    mktempdir() do temporary
        root=realpath(temporary)
        for role in ("primary","independent")
            directory=joinpath(root,role);mkdir(directory)
            value=make_cases(realpath(directory),role,"mixed")
            recipe=F.LockedMixed8RidgeRecipe();start=START+Minute(40);stop=START+Minute(90);asof=START+Minute(211)
            arrays=Q.split_arrays(value,1;start,stop,asof)
            @test size(arrays.features,2)==8
            @test arrays.features[1,:]==[5.,5.,5.,0.,400.,-2.,2.,800.]
            expected=F.prepare_expected(value,1,recipe;start,stop,asof)
            calls=Ref(0)
            spy=(args...)->begin calls[]+=1;error("invalid binding must precede numeric fit") end
            wrong_source=F.ExpectedFit(expected.context_sha256,recipe.sha256,"0"^64,expected.library_sha256)
            @test_throws ArgumentError F._fit(value,1,recipe,wrong_source,start,stop,asof,spy)
            @test calls[]==0
            point=F.fit_generated(value,1,recipe,expected;start,stop,asof).capsule
            @test point.model isa L.NumericModel
            reference=historical_reference(arrays.features,log1p.(arrays.targets))
            @test F.predict_log(point,arrays.features)==reference.Z*reference.beta
            output=joinpath(directory,"published");pins=F.publish_fit(point,output)
            restored=F.read_published(output;expected,model_sha256=pins.model_sha256,
                receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
            python=get(ENV,"SOLARSINDY_RESEARCH_PYTHON",Sys.iswindows() ? "python" : "python3")
            checker=joinpath(@__DIR__,"..","tools","point_input_export","locked_mixed8_ridge.py")
            model_path=joinpath(output,"model.json")
            @test success(`$python -B -X utf8 $checker --model $model_path --model-sha256 $(pins.model_sha256) --recipe-sha256 $(recipe.sha256)`)
            @test F.model_bytes(restored.model)==F.model_bytes(point.model)
            @test F.predict_log(restored,arrays.features)==F.predict_log(point,arrays.features)
            @test R.decode_canonical(F.receipt_bytes(restored))["family"]==L.FAMILY
            wrong=F.ExpectedFit(expected.context_sha256,C.Recipe(kind=:ridge).sha256,expected.source_sha256,expected.library_sha256)
            @test_throws ArgumentError F.read_published(output;expected=wrong,model_sha256=pins.model_sha256,
                receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
            @test_throws ArgumentError F.fit_generated(value,1,C.Recipe(kind=:ridge),expected;start,stop,asof)
            @test_throws ArgumentError F.prepare_expected(value,1,recipe;start,stop=START+Minute(70)+Second(45),asof)
            @test_throws MethodError F.fit_archival(value,1,recipe,expected;start,stop,asof)
            generated=B.read_reverified(joinpath(directory,"case-request.json"),joinpath(directory,"cases");
                request_sha256=value.request_sha256,manifest_sha256=value.manifest_sha256,complete_sha256=value.complete_sha256)
            @test_throws ArgumentError F.fit_archival(generated,1,recipe,expected;start,stop,asof)
            # Deliberately invalid negative fixture: no archival grant or
            # qualification is created, and the non-2017 clock must fail first.
            flagged=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,value.labels,
                value.request_sha256,value.manifest_sha256,value.complete_sha256,true,false)
            invalid=B.ReverifiedCases(flagged,generated.admission_json,B._ReadToken())
            message=nothing
            try
                F._archival_request(invalid,1,DateTime(2016),DateTime(2017),DateTime(2018))
            catch error
                error isa ArgumentError || rethrow();message=error.msg
            end
            @test message=="archival fitting requires a strict 2017 training split"
            # A new well-formed numerical payload cannot inherit the original
            # fit receipt even when the outer model/completion pins are updated.
            changed=joinpath(directory,"tampered-model");cp(output,changed)
            object=R.decode_canonical(read(joinpath(changed,"model.json"));limit=L.MAX_BYTES)
            old=C.unbits(object["coefficients"][1]);object["coefficients"][1]=C.bits(old+.25)
            model_bytes=R.canonical_bytes(object;limit=L.MAX_BYTES)
            @test L.decode_model(model_bytes) isa L.NumericModel
            write(joinpath(changed,"model.json"),model_bytes)
            marker=R.decode_canonical(read(joinpath(changed,"COMPLETE.json"));limit=4096)
            marker["model_sha256"]=digest(model_bytes)
            marker["bytes_without_marker"]=sum(filesize(joinpath(changed,name)) for name in F._FIT_FILES[1:4])
            complete=R.canonical_bytes(marker;limit=4096);write(joinpath(changed,"COMPLETE.json"),complete)
            @test_throws ArgumentError F.read_published(changed;expected,model_sha256=digest(model_bytes),
                receipt_sha256=pins.receipt_sha256,complete_sha256=digest(complete))
            cal_start=START+Minute(90);cal_stop=START+Minute(135);issue=START+Minute(150)
            ce=S.K.prepare_expected(value,1,restored;start=cal_start,stop=cal_stop,asof,first_issue=issue)
            calibration=S.K.calibrate_generated(value,1,restored,ce;start=cal_start,stop=cal_stop,asof,first_issue=issue)
            static=S.B.calibrate_generated(value,1,restored,calibration,ce)
            E=SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
            subscribers=(E.SubscriberSpec(:phase,:phase;gamma=.01),E.SubscriberSpec(:fifo,:fifo;gamma=.01/31))
            result=S.replay_generated(value,1,restored,calibration,ce;start=issue,stop=START+Minute(166),asof,
                subscribers,nominal=.1,static_calibration=static)
            @test result.audit.mandatory_static_inventory_complete
            @test result.static_bounds.methods==(:het_gaussian,:raw_split,:normalized_split,:regime_split)
            @test all(entry->entry.issued==16,values(result.static_bounds.counts))
            @test !result.audit.actual_source_authority_granted && result.audit.actual_fits==0
            if role=="primary"
                future=joinpath(root,"changed-future");mkdir(future)
                changed=make_cases(realpath(future),role,"mixed";profile=:pulse_eval_missing)
                changed_arrays=Q.split_arrays(changed,1;start,stop,asof)
                @test F._data_sha(arrays)==F._data_sha(changed_arrays)
                changed_expected=F.prepare_expected(changed,1,recipe;start,stop,asof)
                changed_point=F.fit_generated(changed,1,recipe,changed_expected;start,stop,asof).capsule
                @test F.model_bytes(changed_point.model)==F.model_bytes(point.model)
                @test changed_expected.context_sha256!=expected.context_sha256
            end
        end
        ground=joinpath(root,"wrong-layout");mkdir(ground)
        value=make_cases(realpath(ground),"primary","ground-only")
        @test_throws ArgumentError F.prepare_expected(value,1,L.Recipe();start=START+Minute(40),stop=START+Minute(90),asof=START+Minute(211))
    end
end

@testset "Independent Python literal codec checker" begin
    python=get(ENV,"SOLARSINDY_RESEARCH_PYTHON",Sys.iswindows() ? "python" : "python3")
    script=joinpath(@__DIR__,"point_input_export","test_locked_mixed8_ridge.py")
    @test success(`$python -B -X utf8 -W error::ResourceWarning $script`)
end
end
