module ArchiveSourceBoundPointFitTests
using Dates,SHA,SolarSINDy,Test
const F=SolarSINDy.ArchiveSourceBoundPointFit
const P=SolarSINDy.ArchivePointNumericTrees
const C=SolarSINDy.ArchiveClassicPointModels
const B=SolarSINDy.ArchiveCatalogCaseReader
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const J=SolarSINDy.ArchiveCatalogSourceJob
const Q=SolarSINDy.ArchiveCatalogCaseJob
const START=DateTime(2020)
digest(x)=bytes2hex(sha256(x))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include("case_fixture.jl")
function recipe(;depth=2,rounds=2,bins=2,seed=27102026,eta=1.,min_weight=1.,lambda=0.,L2=0.)
    F.BoostRecipe(;depth,rounds,bins,seed,eta,min_weight,lambda,L2)
end

function run_generated_fit_tests(output)
@testset "Explicit recipe and expected pin admission" begin
    @test_throws ArgumentError recipe(depth=true)
    @test_throws ArgumentError recipe(rounds=0)
    @test_throws ArgumentError recipe(bins=256)
    @test_throws ArgumentError recipe(seed=-1)
    @test_throws ArgumentError recipe(eta=NaN)
    @test_throws ArgumentError recipe(min_weight=true)
    @test_throws ArgumentError recipe(lambda=-1.)
    @test_throws ArgumentError recipe(L2=Inf)
    @test_throws ArgumentError F.ExpectedFit("x",repeat("a",64),repeat("b",64),F.LIBRARY_SHA)
    @test_throws MethodError F.BoundFit()
end
results=Dict()
@testset "Verified source-bound generated fitting, independent rates and leakage controls" begin
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        let root=joinpath(output,role*"-"*layout)
            mkdir(root);root=realpath(root);value=make_cases(root,role,layout)
            start=START+Minute(40);stop=START+Minute(120);asof=START+Minute(211)
            r=recipe();prepared=F._prepare(value,1,r,start,stop,asof);a=prepared.arrays
            @test a.rows==collect(41:90)
            @test a.issues==collect(START+Minute(40):Minute(1):START+Minute(89))
            @test a.targets==vcat(fill(5.,20),fill(10.,30))
            @test prepared.y==vcat(fill(log(6.),20),fill(log(11.),30))
            @test size(a.features)==(50,layout=="mixed" ? 8 : 11)
            expected=F.prepare_expected(value,1,r;start,stop,asof)
            @test expected.context_sha256==prepared.context_sha256
            calls=Ref(0)
            spy=function(args...;kwargs...);calls[]+=1;error("Admission must precede fitting");end
            wrong=F.ExpectedFit(repeat("0",64),r.sha256,prepared.source,F.LIBRARY_SHA)
            @test_throws ArgumentError F._fit(value,1,r,wrong,start,stop,asof,spy)
            @test_throws ArgumentError F._fit(value,1,recipe(seed=1),expected,start,stop,asof,spy)
            @test_throws ArgumentError F._fit(value,2,r,expected,start,stop,asof,spy)
            @test_throws ArgumentError F._fit(value,1,r,expected,start,stop+Minute(1),asof,spy)
            @test calls[]==0
            mutator=function(X,y,recipe;kwargs...);X[1,1]+=1.;nothing;end
            @test_throws ArgumentError F._fit(value,1,r,expected,start,stop,asof,mutator)
            @test F._prepare(value,1,r,start,stop,asof).input==prepared.input
            fit=F.fit_generated(value,1,r,expected;start,stop,asof)
            @test fit.training_rows==50
            @test F.predict_log(fit.capsule,a.features)==SolarSINDy.gbm_predict(fit.native,a.features)
            receipt=R.decode_canonical(F.receipt_bytes(fit.capsule))
            @test receipt["all_training_native_predictions_exact"]===true
            @test receipt["actual_fit_performed"]===false && receipt["actual_source_access"]===false
            # Mutating the diagnostic native model cannot mutate the copied capsule.
            before=F.predict_log(fit.capsule,a.features);fit.native.trees[1].pred[1]+=10.0f0
            @test F.predict_log(fit.capsule,a.features)==before
            data=F._data_sha(a)[1]
            for field in (:features,:targets,:rows,:issues)
                altered=deepcopy(a)
                if field===:features;altered.features[1,1]=nextfloat(altered.features[1,1])
                elseif field===:targets;altered.targets[1]=nextfloat(altered.targets[1])
                elseif field===:rows;altered.rows[1]-=1
                else;altered.issues[1]-=Millisecond(1)
                end
                @test F._data_sha(altered)[1]!=data
            end
            @test F._prepare(value,1,r,start,stop,asof).input==data
            # Literal readiness: issue40 is ready at70:45, never at/before a strict70:45 cutoff.
            boundary=START+Minute(70)+Second(45)
            @test_throws ArgumentError F.prepare_expected(value,1,r;start,stop=boundary,asof)
            rejected=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,value.labels,
                value.request_sha256,value.manifest_sha256,value.complete_sha256,true,false)
            @test_throws ArgumentError F._fit(rejected,1,r,expected,start,stop,asof,spy)
            @test calls[]==0
            results[(role,layout)]=(input=data,model=digest(F.model_bytes(fit.capsule.model)),fit=fit,value=value,arrays=a,prediction=before,root=root)
        end
    end
    for layout in ("ground-only","mixed")
        @test results[("primary",layout)].input==results[("independent",layout)].input
        @test results[("primary",layout)].model==results[("independent",layout)].model
    end
end
return results
end

include("publication_tests.jl")
include("authority_tests.jl")
function run_scope(temporary)
    results=run_generated_fit_tests(realpath(temporary))
    run_publication_tests(results)
    run_classic_source_bound_tests(results)
    run_authority_tests(results)
    artifacts=Any[]
    for ((role,layout),entry) in sort!(collect(results);by=first)
        for name in vcat(["published"],role=="primary" ? ["classic-ridge","classic-boost","classic-forest"] : String[])
            directory=joinpath(entry.root,name)
            push!(artifacts,Dict("role"=>role,"layout"=>layout,"request_path"=>joinpath(entry.root,"case-request.json"),
                "case_directory"=>joinpath(entry.root,"cases"),"model_directory"=>directory,
                "context_sha256"=>digest(read(joinpath(directory,"context.json"))),
                "recipe_sha256"=>digest(read(joinpath(directory,"recipe.json"))),
                "model_sha256"=>digest(read(joinpath(directory,"model.json"))),
                "receipt_sha256"=>digest(read(joinpath(directory,"receipt.json"))),
                "complete_sha256"=>digest(read(joinpath(directory,"COMPLETE.json")))))
        end
    end
    @test length(artifacts)==10
    R._exclusive_write(joinpath(temporary,"FIT_TEST_REPORT.json"),R.canonical_bytes(Dict("complete"=>true,
        "runtime"=>string(VERSION),"artifacts"=>artifacts,"source_sha256"=>F.source_identity(),
        "actual_source_access"=>false,"actual_fits"=>0);limit=131072))
    nothing
end
requested=get(ENV,"SOLARSINDY_POINT_FIT_TEST_OUTPUT","")
if isempty(requested)
    mktempdir(run_scope)
else
    J._ordinary(dirname(abspath(requested));directory=true)
    @test !ispath(requested) && !islink(requested)
    mkdir(requested);run_scope(realpath(requested))
end
@test Sys.maxrss()<=8*1024^3
println("PACKAGE_SOURCE_BOUND_POINT_FIT_PASSED;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
