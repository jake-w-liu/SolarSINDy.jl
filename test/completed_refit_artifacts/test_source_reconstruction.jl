@testset "mixed independent admission matches immutable CMR" begin
    issue=DateTime(2017,3,1,12)
    protocol=make_protocol(protocol_fixture(;mode="mixed8"))
    for overflow_summary in (false,true)
        records=Any[]
        for k in 0:30
            x=overflow_summary ? (mod(k,4)>=2 ? floatmax(Float64)/4 : 0.0) : Float64(k)
            push!(records,Dict{String,Any}("kind"=>"ground","time"=>R.time_text(issue-Minute(31)+Minute(k)),
                "values"=>R.float_bits.([x,0.0]),"body_id"=>"ground","body_sha256"=>repeat("1",64)))
        end
        if !overflow_summary
            push!(records,Dict{String,Any}("kind"=>"wind","time"=>R.time_text(issue-Minute(1)),
                "values"=>R.float_bits.([floatmax(Float64),-2.0,0.0,1.0]),"body_id"=>"wind","body_sha256"=>repeat("2",64)))
        end
        # This private structural snapshot is deliberately NOT a qualified
        # audited source; only admission arithmetic/order is exercised here.
        bundle=R._Bundle(R._VERIFIED,CODEC_TEST_DIRECTORY,repeat("3",64),repeat("4",64),"source_audit",protocol,"{}\n",
            ("source-audit-data.json"=>R._jsontext(records),),Int64(length(records)))
        source=R.AuditedSources(R._VERIFIED,bundle);index=R._source_index(source)
        primary,_=R._feature_row(source,index,issue;independent=false)
        rebuilt,_=R._feature_row(source,index,issue;independent=true)
        expected=overflow_summary ? "missing_omni_history" : "invalid_feature_summary"
        @test primary["accepted"]===false
        @test primary["reason"]==expected
        @test rebuilt["reason"]==expected
        @test rebuilt==primary
    end
end
