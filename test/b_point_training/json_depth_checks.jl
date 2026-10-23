function json_depth_checks()
    data=JSON3.read(read(joinpath(@__DIR__,"independent_depth_oracles_v1.json")))
    for fixture in data.fixtures
        body=hex2bytes(fixture.body_hex);@test T._sha(body)==fixture.sha256
        calls=Ref(0);stub=bytes->(calls[]+=1;:counted)
        if fixture.depth_admitted
            @test P._bounded_json_parse(body,stub)===:counted&&calls[]==1
            # Only independently depth-admitted, bounded normal inputs reach
            # the real backend. The byte expectation is frozen independently.
            parsed=P._bounded_json_parse(body)
            @test T._bytes(parsed)==body
        else
            @test_throws ArgumentError P._bounded_json_parse(body,stub)
            @test calls[]==0
        end
    end
    for fixture in data.overdepth_stub_only_recipes
        depth=fixture.array_count_inside_object
        body=collect(codeunits("{\"x\":"*repeat("[",depth)*"0"*repeat("]",depth)*"}\n"))
        @test length(body)==fixture.bytes&&T._sha(body)==fixture.sha256
        calls=Ref(0);stub=bytes->(calls[]+=1;:forbidden)
        @test_throws ArgumentError P._bounded_json_parse(body,stub)
        @test calls[]==fixture.parser_invocations_expected==0
    end
    for text in ("[}","{]","{","\"unterminated","\"unfinished\\")
        calls=Ref(0);stub=bytes->(calls[]+=1;:forbidden)
        @test_throws ArgumentError P._bounded_json_parse(collect(codeunits(text)),stub)
        @test calls[]==0
    end
    for exception in (InterruptException(),OutOfMemoryError(),StackOverflowError())
        # Constructed exception objects only; no recursion, memory exhaustion or
        # external interrupt is induced by this parser propagation test.
        calls=Ref(0);observed=nothing
        try P._bounded_json_parse(UInt8[0x30,0x0a],bytes->(calls[]+=1;throw(exception))) catch e;observed=e;end
        @test observed===exception&&calls[]==1
    end
    calls=Ref(0);stub=bytes->(calls[]+=1;:forbidden)
    @test_throws ArgumentError P._bounded_json_parse(fill(UInt8(' '),2*1024^2+1),stub)
    @test calls[]==0
    true
end
