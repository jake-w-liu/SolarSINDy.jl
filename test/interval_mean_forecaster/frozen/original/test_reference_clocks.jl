include("test_training.jl")
@testset "referencefitclockshapeanddeclaredlatency" begin
    start,ending,original,ref=selection_fixture()
    cutoff=start-Hour(168)
    for delay in (0,1)
        function latency(row)
            clocks=Tuple(forecast_clock(row.issue,h,delay) for h in (4,6))
            merge(row,(anchor_tag=clocks[1].anchor_tag,anchor_complete=clocks[1].anchor_complete,
                anchor_receipt=clocks[1].anchor_receipt,target_receipt=Tuple(c.target_receipt for c in clocks),dst_delay_hours=delay))
        end
        panels=Dict(k=>latency.(rows) for (k,rows) in original)
        reference=merge(ref,(rows=latency.(ref.rows),delay_hours=delay,
            max_fit_target_complete=(cutoff-Hour(delay),cutoff-Hour(delay))))
        @test select_setting(panels,reference;validation_start=start,validation_end=ending,delay_hours=delay).status==:selected
        for bad in ((),(cutoff,),(cutoff,cutoff,cutoff),(true,cutoff),[cutoff,cutoff])
            for field in (:max_fit_target_complete,:max_fit_label_receipt)
                malformed=merge(reference,NamedTuple{(field,)}((bad,)))
                @test_throws ArgumentError select_setting(panels,malformed;validation_start=start,validation_end=ending,delay_hours=delay)
            end
        end
        for bad in (merge(reference,(max_fit_label_receipt=(cutoff+Millisecond(1),cutoff),)),
                merge(reference,(max_fit_target_complete=(cutoff+Hour(1),cutoff),)),
                merge(reference,(max_fit_label_receipt=(cutoff-Hour(1),cutoff),)),
                merge(reference,(source_sha256="not-a-digest",)))
            @test_throws ArgumentError select_setting(panels,bad;validation_start=start,validation_end=ending,delay_hours=delay)
        end
        wrongdelay=merge(reference,(max_fit_target_complete=(cutoff-Hour(1-delay),cutoff-Hour(1-delay)),))
        @test_throws ArgumentError select_setting(panels,wrongdelay;validation_start=start,validation_end=ending,delay_hours=delay)
    end
end
