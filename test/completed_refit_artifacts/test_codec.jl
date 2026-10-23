@testset "completed refit canonical scalar and member codec" begin
    # Literal SHA and bytes were fixed independently before implementation.
    literal = "{\"a\":1,\"b\":\"3ff0000000000000\"}\n"
    @test R.digest(literal) == "5a87a869282fc8b8d3e930ab9de332b31d83e9efdf942472acee5182b21b361b"
    @test String(R.canonical_bytes(Dict("b"=>"3ff0000000000000", "a"=>1))) == literal
    @test R.decode_canonical(literal) == Dict("a"=>1,"b"=>"3ff0000000000000")
    @test String(R.canonical_bytes(Dict("ok"=>true,"off"=>false,"optional"=>nothing))) ==
        "{\"off\":false,\"ok\":true,\"optional\":null}\n"
    @test R.decode_canonical("{\"off\":false,\"ok\":true,\"optional\":null}\n")["ok"] === true
    for text in ("{\"a\":1,\"a\":2,\"b\":\"3ff0000000000000\"}\n",
                 "{\"b\":\"3ff0000000000000\",\"a\":1}\n",
                 "{ \"a\":1,\"b\":\"3ff0000000000000\"}\n",
                 "{\"a\":1,\"b\":\"3ff0000000000000\"}",
                 "1.0\n", "1e0\n", "01\n", "-0\n")
        @test_throws ArgumentError R.decode_canonical(text)
    end
    for (value, encoded) in ((0.0,"0000000000000000"),(-0.0,"8000000000000000"),
                            (nextfloat(0.0),"0000000000000001"),(floatmax(Float64),"7fefffffffffffff"))
        @test R.float_bits(value) == encoded
        @test reinterpret(UInt64,R.read_float_bits(encoded)) == reinterpret(UInt64,value)
    end
    for value in (true,NaN,Inf,-Inf,big"1e-400",big"1e400")
        @test_throws ArgumentError R.float_bits(value)
    end
    for encoded in ("7ff0000000000000","fff0000000000000","7ff8000000000000","3FF0000000000000","0")
        @test_throws ArgumentError R.read_float_bits(encoded)
    end
    @test_throws ArgumentError R.checked_integer(true,0,1)
    @test_throws ArgumentError R.checked_integer(Int128(typemax(Int64))+1,typemin(Int64),typemax(Int64))
    @test_throws ArgumentError R.canonical_bytes(Dict("notbits"=>1.0))
    @test_throws ArgumentError R.decode_canonical(UInt8[0xff,0x0a])
    @test_throws ArgumentError R.canonical_bytes(Dict(string(i)=>i for i in 1:33))
    @test_throws ArgumentError R.canonical_bytes(fill(0,8193))
    @test length(R.decode_canonical(R.canonical_bytes(fill(0,8192)))) == 8192
    @test_throws ArgumentError R.canonical_bytes(repeat("x",4097))
    @test R.decode_canonical(R.canonical_bytes(repeat("x",4096))) == repeat("x",4096)
    @test_throws ArgumentError R.canonical_bytes(Dict("a"=>1);limit=7)
    @test String(R.canonical_bytes(Dict("a"=>1);limit=8)) == "{\"a\":1}\n"
    # Empty containers at the maximum depth must not bypass the same bound.
    nested = Any[]
    for _ in 1:16
        nested = Any[nested]
    end
    @test_throws ArgumentError R.canonical_bytes(nested)
    @test_throws ArgumentError R.decode_canonical(repeat("[",17)*repeat("]",17)*"\n")
    boundary = 1
    for _ in 1:16
        boundary = Any[boundary]
    end
    @test R.decode_canonical(R.canonical_bytes(boundary)) isa Vector
    @test R.time_text(DateTime(1970)) == "0"
    @test R.time_text(DateTime(1970)-Millisecond(1)) == "-1"
    @test R.read_time_text("-1") == DateTime(1970)-Millisecond(1)
    for value in ("-0","+1","01","1.0","9223372036854775808")
        @test_throws ArgumentError R.read_time_text(value)
    end
    @test R.schema_fields(Dict("name"=>"x","issue_start_inclusive"=>"0","issue_stop_exclusive"=>"1",
        "label_ready_before"=>"2","target_admit_asof"=>"2","min_supervised_rows"=>3),"split")["name"] == "x"
    @test_throws ArgumentError R.schema_fields(Dict("name"=>"x"),"split")
end
