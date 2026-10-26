function repack_fit!(directory;receipt_change=(x)->nothing,marker_change=(x)->nothing)
    receipt=R.decode_canonical(read(joinpath(directory,"receipt.json"));limit=65536)
    marker=R.decode_canonical(read(joinpath(directory,"COMPLETE.json"));limit=4096)
    receipt_change(receipt);bytes=R.canonical_bytes(receipt;limit=65536)
    write(joinpath(directory,"receipt.json"),bytes)
    marker["receipt_sha256"]=digest(bytes)
    marker["bytes_without_marker"]=sum(filesize(joinpath(directory,name)) for name in F._FIT_FILES[1:4])
    marker_change(marker);complete=R.canonical_bytes(marker;limit=4096);write(joinpath(directory,"COMPLETE.json"),complete)
    (;model_sha256=digest(read(joinpath(directory,"model.json"))),receipt_sha256=digest(bytes),complete_sha256=digest(complete))
end
function run_publication_tests(results)
    @testset "Outside-bound generated point publication and complete independent reread" begin
        for ((role,layout),entry) in results
            value=entry.fit.capsule;output=joinpath(entry.root,"published");pins=F.publish_fit(value,output)
            read_fit(directory=output;expected=value.expected,model=pins.model_sha256,receipt=pins.receipt_sha256,complete=pins.complete_sha256)=
                F.read_published(directory;expected,model_sha256=model,receipt_sha256=receipt,complete_sha256=complete)
            restored=read_fit()
            @test Set(readdir(output))==Set(F._FIT_FILES)
            @test F.predict_log(restored,entry.arrays.features)==entry.prediction
            @test F.receipt_bytes(restored)==F.receipt_bytes(value) && restored.context==value.context && restored.recipe==value.recipe
            @test pins.actual_fit_performed===false
            @test F.model_bytes(restored.model)==F.model_bytes(value.model)
            owned=F.receipt_bytes(restored);owned[1]=0x00
            @test F.receipt_bytes(restored)[1]==UInt8('{')
            @test_throws ArgumentError F.publish_fit(value,output)
            @test_throws ArgumentError read_fit(;model="0"^64)
            @test_throws ArgumentError read_fit(;receipt="0"^64)
            @test_throws ArgumentError read_fit(;complete="0"^64)
            wrong=F.ExpectedFit("0"^64,value.expected.recipe_sha256,value.expected.source_sha256,F.LIBRARY_SHA)
            @test_throws ArgumentError read_fit(;expected=wrong)
            extra=joinpath(entry.root,"extra-member");cp(output,extra);write(joinpath(extra,"foreign.txt"),"unrecognized")
            @test_throws ArgumentError read_fit(extra)
            incomplete=joinpath(entry.root,"incomplete");mkdir(incomplete)
            for name in F._FIT_FILES[1:4];cp(joinpath(output,name),joinpath(incomplete,name));end
            @test_throws ArgumentError read_fit(incomplete)
            variants=[("bool-rows",(r)->(r["training_rows"]=true),(m)->nothing),
                ("wrong-input",(r)->(r["input_sha256"]="1"^64),(m)->nothing),
                ("actual-claim",(r)->(r["actual_source_access"]=true),(m)->nothing),
                ("wrong-family",(r)->(r["family"]="retained_ridge_log1p"),(m)->nothing),
                ("bool-members",(r)->nothing,(m)->(m["members"]=true)),
                ("wrong-actual-marker",(r)->nothing,(m)->(m["actual_fit_performed"]=true))]
            for (name,receipt_change,marker_change) in variants
                changed=joinpath(entry.root,name);cp(output,changed)
                newpins=repack_fit!(changed;receipt_change,marker_change)
                @test_throws ArgumentError read_fit(changed;receipt=newpins.receipt_sha256,complete=newpins.complete_sha256)
            end
            # A different well-formed model, even with its new outer model pin,
            # must not inherit the original independently pinned fit receipt.
            changed=joinpath(entry.root,"different-valid-model");cp(output,changed)
            object=R.decode_canonical(read(joinpath(changed,"model.json"));limit=8_388_608)
            prior=reinterpret(Float32,parse(UInt32,object["trees"][1]["prediction_bits"][1];base=16))
            object["trees"][1]["prediction_bits"][1]=string(reinterpret(UInt32,prior+0.25f0);base=16,pad=8)
            newmodel=R.canonical_bytes(object;limit=8_388_608)
            @test P.decode_model(newmodel) isa P.NumericModel
            write(joinpath(changed,"model.json"),newmodel)
            marker=R.decode_canonical(read(joinpath(changed,"COMPLETE.json"));limit=4096)
            marker["model_sha256"]=digest(newmodel)
            marker["bytes_without_marker"]=sum(filesize(joinpath(changed,name)) for name in F._FIT_FILES[1:4])
            newcomplete=R.canonical_bytes(marker;limit=4096);write(joinpath(changed,"COMPLETE.json"),newcomplete)
            @test_throws ArgumentError read_fit(changed;model=digest(newmodel),complete=digest(newcomplete))
            @test read(joinpath(output,"model.json"))==F.model_bytes(value.model)
        end
    end
end
function run_classic_source_bound_tests(results)
    @testset "Source-bound retained ridge, boost and forest publication" begin
        for layout in ("ground-only","mixed")
            entry=results[("primary",layout)];value=entry.value
            start=START+Minute(40);stop=START+Minute(120);asof=START+Minute(211)
            recipes=(C.Recipe(kind=:ridge,penalty=1.,bins=2),
                C.Recipe(kind=:boost,depth=1,trees=2,bins=2,rate=.5,minleaf=1),
                C.Recipe(kind=:forest,depth=1,trees=3,bins=2,minleaf=1,sample_rows=50,seed=17))
            for recipe in recipes
                expected=F.prepare_expected(value,1,recipe;start,stop,asof)
                fit=F.fit_generated(value,1,recipe,expected;start,stop,asof)
                @test fit.training_rows==50 && fit.capsule.model.recipe.json==recipe.json
                pred=F.predict_log(fit.capsule,entry.arrays.features)
                @test pred==C.predict(fit.capsule.model,entry.arrays.features) && all(isfinite,pred)
                output=joinpath(entry.root,"classic-"*String(recipe.kind));pins=F.publish_fit(fit.capsule,output)
                restored=F.read_published(output;expected,model_sha256=pins.model_sha256,
                    receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
                @test F.predict_log(restored,entry.arrays.features)==pred
                receipt=R.decode_canonical(F.receipt_bytes(restored);limit=65536)
                @test receipt["family"]=="retained_"*String(recipe.kind)*"_log1p"
                @test receipt["actual_source_access"]===false && receipt["actual_fit_performed"]===false
                if recipe.kind===:ridge;fit.native.coefficients[1]+=10.
                else;fit.native.trees[1].value[1]+=10.
                end
                @test F.predict_log(fit.capsule,entry.arrays.features)==pred
                @test_throws MethodError F.fit_archival(restored,1,recipe,expected;start,stop,asof)
            end
        end
    end
end
