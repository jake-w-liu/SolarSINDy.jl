module Matched342CodecStorageTests
using Test,SHA,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
function run_tests()
    @testset "Matched342 bounded canonical codec preflight" begin
        valid=Dict{String,Any}("a"=>Any[1,true,"0000000000000000"],"b"=>Any[])
        bytes=M._canonical(valid)
        @test bytes==Vector{UInt8}(codeunits("{\"a\":[1,true,\"0000000000000000\"],\"b\":[]}\n"))
        @test M._decode_bytes(bytes)==valid
        count=Ref(0);backend=(args...)->begin count[]+=1;error("backend must not execute") end
        invalid=["{\"a\":1,\"a\":2}\n","{\"a\":0.0}\n","{\"a\":1e0}\n","{\"a\":truex}\n",
            "{\"a\":null}\n","{\"a\":NaN}\n","{\"a\":9223372036854775808}\n",
            "{\"a\":01}\n","{\"a\":\"\\u0061\"}\n","["^13*"0"*"]"^13*"\n",
            "{\"a\":\""*"x"^4097*"\"}\n","["*join(fill("0",500001),',')*"]\n"]
        for text in invalid
            @test_throws ArgumentError M._decode_bytes(Vector{UInt8}(codeunits(text));backend)
        end
        @test count[]==0
        for text in ("{ \"a\":1}\n","{\"b\":2,\"a\":1}\n","{\"a\":1}")
            @test_throws ArgumentError M._decode_bytes(Vector{UInt8}(codeunits(text)))
        end
        @test_throws ArgumentError M._canonical(Dict("a"=>1.0))
        @test_throws ArgumentError M._canonical(Dict("a"=>nothing))
        @test_throws ArgumentError M._canonical(Dict("a"=>"\\"))
        @test_throws ArgumentError M._preflight(fill(UInt8(' '),M.MAX_BYTES+1))
    end
    @testset "Matched342 immutable training rows through500000" begin
        rows=collect(1:500000);packed=M.TrainingRows342(rows)
        @test collect(packed)==rows
        @test sum(packed)==125000250000
        @test ncodeunits(packed.bytes)==2000000
        @test bytes2hex(sha256(codeunits(packed.bytes)))=="a3c3bdc832eb70d15ce111d4caceeda454d15b2de306cac505a3d36c7562cddb"
        @test typeof(packed)===typeof(M.TrainingRows342([1]))===typeof(M.TrainingRows342([2,500000]))
        @test isempty(typeof(packed).parameters)
        rows[1]=42;@test packed[1]==1
        @test packed[end]==500000
        @test_throws BoundsError packed[0]
        @test_throws BoundsError packed[500001]
        @test_throws ArgumentError (packed[1]=2)
        for bad in (Int[],[0],[500001],[1,1],[2,1],[true],[1.0],fill(1,500001))
            @test_throws ArgumentError M.TrainingRows342(bad)
        end
        io=IOBuffer();write(io,'[')
        for (k,i) in enumerate(packed);k>1&&write(io,',');print(io,i);end
        write(io,']');serialized=take!(io)
        @test length(serialized)==3388896
        @test bytes2hex(sha256(serialized))=="219fe943b443d7299cf528271ce09beeabaa323aa3d830d97cd8f962a3fef5df"
        encoder=M.Encoder342(packed,ntuple(_->0,165),ntuple(_->0.0,165),ntuple(_->0.0,342),ntuple(_->1.0,342),repeat("a",64))
        @test M._check_encoder(encoder)===nothing
        mktempdir() do dir
            dir=realpath(dir)
            path=joinpath(dir,"encoder.json");saved=M.save_encoder(path,encoder)
            @test saved.bytes<=M.MAX_BYTES
            @test saved.sha256==M.preprocessing_sha256(encoder)
            loaded=M.load_encoder(path;expected_sha256=saved.sha256,expected_training_design_sha256=repeat("a",64))
            @test loaded.training_rows==packed
            @test loaded.center===encoder.center
            @test M.preprocessing_sha256(loaded)==saved.sha256
            @test_throws ArgumentError M.save_encoder(path,encoder)
            @test_throws ArgumentError M.load_encoder(path;expected_sha256=repeat("b",64),expected_training_design_sha256=repeat("a",64))
            @test_throws ArgumentError M.load_encoder(path;expected_sha256=saved.sha256,expected_training_design_sha256=repeat("b",64))
            object=M._encoder_object(encoder);object["feature_count"]=true;changed=M._write_json(joinpath(dir,"bool.json"),object)
            @test_throws ArgumentError M.load_encoder(changed.path;expected_sha256=changed.sha256,expected_training_design_sha256=repeat("a",64))
            object=M._encoder_object(encoder);object["other"]=1;changed=M._write_json(joinpath(dir,"extra.json"),object)
            @test_throws ArgumentError M.load_encoder(changed.path;expected_sha256=changed.sha256,expected_training_design_sha256=repeat("a",64))
        end
    end
end
end
Matched342CodecStorageTests.run_tests()
