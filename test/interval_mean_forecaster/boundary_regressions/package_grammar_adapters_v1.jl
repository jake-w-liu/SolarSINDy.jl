# Callable adapters over the already package-loaded F namespace.
# Caller loads producer_cases_frozen_v2.jl once in the same test namespace.
# No implementation files are loaded here. No ARGS/global environment mutation.
# Caller supplies its fresh synthetic model/checkpoint and existing owned output
# directory. realpath + retained mktempdir gives a physical portable parent.
using Test, Dates, SHA, TOML

function run_frozen_typed_codec_cases(F, model, checkpoint, output)
    isdir(output) || throw(ArgumentError("existing owned output parent required"))
    A=F.DstIntervalArtifacts
    OUT=mktempdir(realpath(output);prefix="typed-codec-",cleanup=false)
    BEFORE=Dict(F.source_identity())
    hashfile(path)=bytes2hex(sha256(read(path)))
    function observed(thunk)
        try
            value=thunk()
            Dict{String,Any}("accepted"=>true,"value_type"=>string(typeof(value)),"value_repr"=>repr(value))
        catch error
            Dict{String,Any}("accepted"=>false,"exception_type"=>string(typeof(error)),"message"=>sprint(showerror,error))
        end
    end

    cases = Dict{String,Any}[]
    for (label,text,strict) in (("valid","probe",true),("LF","probe\n",false),
        ("CRLF","probe\r\n",false),("doubleLF","probe\n\n",false),
        ("leadingLF","\nprobe",false),("tab","probe\t",false),("leadingdigit","1probe",false))
        symbol=observed(() -> A._decode(Dict("type"=>"symbol","value"=>text)))
        named=observed(() -> A._decode(Dict("type"=>"namedtuple","names"=>[text],
            "items"=>[Dict("type"=>"string","value"=>"synthetic marker")])))
        push!(cases,Dict("kind"=>"symbol","case"=>label,"strict_expected_accept"=>strict,"result"=>symbol))
        push!(cases,Dict("kind"=>"namedtuple_name","case"=>label,"strict_expected_accept"=>strict,"result"=>named))
    end
    for (label,text,strict) in (("valid","3ff0000000000000",true),("LF","3ff0000000000000\n",false),
        ("CRLF","3ff0000000000000\r\n",false),("doubleLF","3ff0000000000000\n\n",false),
        ("leadingLF","\n3ff0000000000000",false),("tab","3ff0000000000000\t",false),
        ("uppercase","3FF0000000000000",false),("long","3ff00000000000000",false))
        result=observed(() -> A._decode(Dict("type"=>"float64","bits"=>text)))
        push!(cases,Dict("kind"=>"float64_bits","case"=>label,"strict_expected_accept"=>strict,"result"=>result))
    end

    BASE_BUNDLE=joinpath(OUT,"valid-base")
    provenance=(scope="synthetic only",input_sha256=model.source_sha256,stage=:synthetic)
    saved=F.save_bundle(BASE_BUNDLE,model,checkpoint;provenance)
    baseline=F.load_bundle(BASE_BUNDLE;expected_manifest_sha256=saved.manifest_sha256,
        expected_sources=A.verify_source_identity())
    @assert isequal(baseline.model,model)
    @assert isequal(baseline.checkpoint,checkpoint)
    @assert isequal(baseline.provenance,provenance)
    public=Dict{String,Any}[]
    floatdoc=A._encode((probe=1.,));floatdoc["items"][1]["bits"]*="\n"
    symboldoc=A._encode((probe=:probe,));symboldoc["items"][1]["value"]*="\n"
    namedoc=A._encode((probe="synthetic marker",));namedoc["names"][1]*="\n"
    for (label,value) in (
        ("float64_LF",floatdoc),("symbol_LF",symboldoc),("namedtuple_name_LF",namedoc),
        )
        destination=joinpath(OUT,"public-"*label)
        cp(BASE_BUNDLE,destination)
        member=joinpath(destination,"provenance.toml")
        open(member,"w") do io;TOML.print(io,Dict("value"=>value);sorted=true);end
        manifestpath=joinpath(destination,"manifest.toml")
        manifest=TOML.parsefile(manifestpath)
        item=only(filter(x->x["name"]=="provenance.toml",manifest["members"]))
        item["bytes"]=filesize(member);item["sha256"]=hashfile(member)
        open(manifestpath,"w") do io;TOML.print(io,manifest;sorted=true);end
        pin=hashfile(manifestpath)
        result=observed(() -> F.load_bundle(destination;expected_manifest_sha256=pin,
            expected_sources=A.verify_source_identity()).provenance)
        push!(public,Dict("case"=>label,"strict_expected_accept"=>false,"result"=>result,
            "directory"=>destination,"manifest_sha256"=>pin,"member_sha256"=>hashfile(member)))
    end
    @assert Dict(F.source_identity())==BEFORE
    @testset "Strict typed grammar negative controls" begin
        for item in cases
            @test item["result"]["accepted"]==item["strict_expected_accept"]
        end
        for item in public
            @test item["result"]["accepted"]==item["strict_expected_accept"]
        end
    end
    (directory=OUT,direct_cases=length(cases),public_cases=length(public))
end

function run_frozen_producer_cases(F, model, checkpoint, output)
    isdir(output) || throw(ArgumentError("existing owned output parent required"))
    Q=F.DstIntervalArtifacts
    OUT=mktempdir(realpath(output);prefix="producer-",cleanup=false)
    BEFORE=Dict(F.source_identity())
    baseline=(model=model,checkpoint=checkpoint)
    hashfile(path)=bytes2hex(sha256(read(path)))
    RECORDS=Dict{String,Any}[]
    for case in producer_cases_frozen_v2()
        directory=joinpath(OUT,case.label)
        record=Dict{String,Any}("case"=>case.label,"valid"=>case.valid,"directory"=>directory)
        saved=try
            value=F.save_bundle(directory,baseline.model,baseline.checkpoint;provenance=case.value)
            record["save"]=Dict("accepted"=>true,"manifest_sha256"=>value.manifest_sha256)
            value
        catch error
            record["save"]=Dict("accepted"=>false,"exception_type"=>string(typeof(error)),"message"=>sprint(showerror,error))
            nothing
        end
        record["directory_exists"]=ispath(directory)
        record["manifest_exists"]=isfile(joinpath(directory,"manifest.toml"))
        if saved!==nothing
            record["members"]=Dict(name=>Dict("bytes"=>filesize(joinpath(directory,name)),"sha256"=>hashfile(joinpath(directory,name))) for name in readdir(directory))
            record["load"]=try
                loaded=F.load_bundle(directory;expected_manifest_sha256=saved.manifest_sha256,
                    expected_sources=Q.verify_source_identity())
                Dict("accepted"=>true,"provenance_identical"=>isequal(loaded.provenance,case.value),
                    "model_identical"=>isequal(loaded.model,baseline.model),
                    "checkpoint_identical"=>isequal(loaded.checkpoint,baseline.checkpoint))
            catch error
                Dict("accepted"=>false,"exception_type"=>string(typeof(error)),"message"=>sprint(showerror,error))
            end
        end
        push!(RECORDS,record)
    end
    @assert Dict(F.source_identity())==BEFORE
    open(joinpath(OUT,"observations.toml"),"w") do io
        TOML.print(io,Dict("observations_complete"=>true,"source_sha256"=>BEFORE,"julia_version"=>string(VERSION),
            "literal_casefile_sha256"=>hashfile(joinpath(@__DIR__,"producer_cases_frozen_v2.jl")),
            "records"=>RECORDS,"scope"=>"Synthetic public producer prepublication grammar/decoder/format-domain controls; not a runtime resource ceiling");sorted=true)
    end
    @testset "Producer rejects outside loader domain before path creation" begin
        for record in RECORDS
            @testset "$(record["case"])" begin
                if record["valid"]
                    @test record["save"]["accepted"]
                    @test record["load"]["accepted"]
                    @test record["load"]["provenance_identical"]
                    @test record["load"]["model_identical"]
                    @test record["load"]["checkpoint_identical"]
                else
                    @test !record["save"]["accepted"] && record["save"]["exception_type"]=="ArgumentError"
                    @test !record["directory_exists"]
                    @test !record["manifest_exists"]
                end
            end
        end
    end
    (directory=OUT,producer_cases=length(RECORDS))
end
