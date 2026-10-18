# Loading-only adapter over the frozen independent API cases.
using Test, Dates

function run_digest_api_regressions(F)
    T=F.DstIntervalTraining
    S=F.DstIntervalStream
    # Independent byte grammar, frozen before proposed source changes. This oracle
    # neither imports a production validator nor uses its regular expression.
    literal_hash(value)=value isa AbstractString && ncodeunits(value)==64 &&
        all(byte->UInt8('0')<=byte<=UInt8('9') || UInt8('a')<=byte<=UInt8('f'),codeunits(value))
    CANONICAL="0123456789abcdef"^4
    CASES=(
        (:canonical,CANONICAL,true),(:zeros,"0"^64,true),(:fs,"f"^64,true),
        (:substring,SubString("x"*CANONICAL*"y",2,65),true),
        (:empty,"",false),(:short,"0"^63,false),(:long,"0"^65,false),
        (:uppercase,"A"^64,false),(:suffix_lf,CANONICAL*"\n",false),
        (:prefix_lf,"\n"*CANONICAL,false),(:suffix_crlf,CANONICAL*"\r\n",false),
        (:suffix_cr,CANONICAL*"\r",false),(:suffix_tab,CANONICAL*"\t",false),
        (:suffix_nul,CANONICAL*"\0",false),(:suffix_line_separator,CANONICAL*"\u2028",false),
        (:suffix_paragraph_separator,CANONICAL*"\u2029",false),
        (:prefix_space," "*CANONICAL,false),(:suffix_space,CANONICAL*" ",false),
        (:embedded_lf,CANONICAL[1:32]*"\n"*CANONICAL[33:64],false),
        (:unicode,"é"^64,false),(:bytes,collect(codeunits(CANONICAL)),false),
        (:missing,missing,false),(:nothing,nothing,false),(:boolean,true,false),(:integer,42,false))
    RAW=reshape([5.,15.,90.,4.,3.,-4.,5.,400.,5.,100000.,1.,2.,2.,5.,-20.,100.,.04],17,1)

    @testset "Strict digest grammar API regressions" begin
    @testset "Whole digest grammar helpers" begin
        for (label,value,expected) in CASES
            @testset "$label" begin
                @test literal_hash(value)===expected
                @test T._valid_hash(value)===expected
                @test S._hex(value)===expected
            end
        end
    end

    @testset "Public stream admission before sink side effects" begin
        for delay in (0,1)
            @testset "valid L$delay" begin
                input=F.HourlyInputs([DateTime(2019)],RAW;source_sha256=CANONICAL)
                rows=NamedTuple[]
                result=F.stream_basis(input;delay_hours=delay,taus=(2.,12.),sigma=1.,chunk_hours=1,
                    sink=chunk->append!(rows,chunk))
                @test length(rows)==1
                @test result.row_count==1
                @test rows[1].source_sha256==CANONICAL
                @test result.source_sha256==CANONICAL
                @test result.checkpoint.source_sha256==CANONICAL
                @test literal_hash(result.checkpoint.prefix_sha256)
            end
            for (label,value,expected) in CASES
                expected && continue
                value isa AbstractString || continue
                @testset "invalid $label L$delay" begin
                    input=F.HourlyInputs([DateTime(2019)],RAW;source_sha256=value)
                    rows=NamedTuple[]
                    callbacks=Ref(0)
                    observed=try
                        F.stream_basis(input;delay_hours=delay,taus=(2.,12.),sigma=1.,chunk_hours=1,
                            sink=chunk->(callbacks[]+=1;append!(rows,chunk)))
                        nothing
                    catch error
                        error
                    end
                    @test observed isa ArgumentError
                    @test isempty(rows)
                    @test callbacks[]==0
                end
            end
        end
    end

    @testset "Public training API rejects grammar at admission" begin
        rows=NamedTuple[]
        input=F.HourlyInputs([DateTime(2019)],RAW;source_sha256=CANONICAL)
        F.stream_basis(input;delay_hours=0,taus=(2.,12.),sigma=1.,chunk_hours=1,
            sink=chunk->append!(rows,chunk))
        for (label,value,expected) in CASES
            expected && continue
            value isa AbstractString || continue
            @testset "$label" begin
                altered=[merge(only(rows),(source_sha256=String(value),))]
                error=try
                    F.fit_gains(altered,DateTime(2019)+Hour(1000),first(F.settings());
                        delay_hours=0,source_sha256=value)
                    nothing
                catch caught
                    caught
                end
                @test error isa ArgumentError
                @test error isa ArgumentError && error.msg=="invalidsourceidentity"
            end
        end
    end

    end

end
