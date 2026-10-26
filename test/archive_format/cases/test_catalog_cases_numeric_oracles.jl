const WHOLE_NUMERIC_ORACLE=JSON3.read(read(joinpath(@__DIR__,"INDEPENDENT_WHOLE_NUMERIC_ORACLE_V1.json")),Any)
@testset "All generated-case floating fields against independent Python oracles" begin
    schedule=seal_catalog_features(G,collect(START:Minute(1):START+Minute(90));mode=:mixed8,wind=WIND)
    labels=audit_catalog_labels(schedule,G;asof=START+Minute(121))
    for expected in WHOLE_NUMERIC_ORACLE["rows"]
        index=expected["issue_minute"]+1
        f=feature_row(schedule,index);l=label_row(labels,index)
        @test f.issue==l.issue==START+Minute(index-1)
        @test f.label_ready==l.label_ready==START+Minute(index+29)+Second(45)
        @test f.accepted==expected["feature_accepted"]
        @test l.accepted==expected["label_accepted"]
        if expected["feature_accepted"]
            gold=expected["feature_numeric13"]
            @test isapprox(f.raw_now,gold[1];atol=1e-13,rtol=0)
            @test isapprox(f.scale,gold[2];atol=1e-13,rtol=0)
            for column in 1:8
                @test isapprox(f.features[column],gold[column+2];atol=1e-13,rtol=0)
            end
        else
            @test f.features===nothing && f.raw_now===nothing && f.scale===nothing
        end
        @test expected["target"] === nothing ? l.target === nothing : reinterpret(UInt64,l.target)==reinterpret(UInt64,Float64(expected["target"]))
    end
end
