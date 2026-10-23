@testset verbose=true "Shared literal clocks, grace and unavailable attempts" begin
    @test bytes2hex(sha256(read(joinpath(@__DIR__,"frozen","independent_oracles_v1.json"))))==
        "18ed0a66543852e21c9f17841b06d0f7118aa6a0ed059bdc90d4b52cfff2aa83"
    for grace in (0,1,60000,60001),late in (false,true)
        fixture=FIXTURES["grace_boundary_"*string(grace)];expected=fixture["expected"][late ? "strictly_after_expiry" : "at_expiry"]
        owner=state(;grace)
        issued=transact!(owner,I;issues=(request(),))
        window=only(owner.snapshot.pending)
        @test window.target_end==offset(1800000) && window.label_ready==offset(1845000)
        @test window.feedback_due==offset(1860000) && window.expires==offset(1860000+grace)
        @test window.expiry_timer==offset(1860001+grace)
        transact!(owner,offset(1800000);rows=rate_rows(1,29;positive_at=1))
        if grace==0 && !late
            result=transact!(owner,offset(1860000);rows=rate_rows(30,30),issues=(request(offset(1860000)),))
        else
            result=transact!(owner,offset(1860000);issues=(request(offset(1860000)),))
            @test only(result.issues).bounds[2].available==false
            @test only(result.issues).bounds[2].upper===nothing
            result=transact!(owner,offset(1860000+grace+Int(late));rows=rate_rows(30,30))
        end
        @test [e.id for e in result.events if e.kind===:completed_target]==expected["completed_ids"]
        @test [e.id for e in result.events if e.kind===:censored_target]==expected["censored_ids"]
        source=source_certificate(owner);phase=subscriber_certificate(owner,:phase)
        @test source.attempted==2 && source.completed_targets==expected["source_completed_count"]
        @test source.target_unknown==expected["source_target_unknown_total"]
        @test source.pending_targets==1 && phase.attempted==2
        @test phase.unavailable==(grace==0 && !late ? 0 : 1)
        @test phase.issued_bounds+phase.unavailable==phase.attempted
        @test phase.all_attempted_enclosure.issued==2
        @test phase.actual_bound_enclosure.issued==phase.issued_bounds
        @test source.study_pass==(phase.unavailable==0)
        @test safe_terminal_period(I;grace_ms=grace)==fixture["expected"]["earliest_safe_terminal_period_minutes"]
        if phase.unavailable>0
            # Even when this unissued row's target later completes, it has no hit/miss.
            known_before=phase.binary_resolved;scores_before=phase.completed_scores
            transact!(owner,I+Minute(62);rows=rate_rows(32,61))
            phase=subscriber_certificate(owner,:phase)
            @test source_certificate(owner).completed_targets==expected["source_completed_count"]+1
            @test phase.binary_resolved==known_before && phase.completed_scores==scores_before
            @test phase.binary_unknown==phase.attempted-phase.binary_resolved && !phase.study_pass
        end
    end
end

@testset verbose=true "Same-time receipts and deterministic completion orders" begin
    results=Any[]
    fixture=FIXTURES["same_time_overlapping_complete_windows"]
    for input in fixture["input"]["batches_in_two_input_orders"]
        owner=state(;grace=60000)
        transact!(owner,I;issues=(request(),));transact!(owner,I+Minute(1);issues=(request(I+Minute(1)),))
        result=transact!(owner,offset(input["received_offset_ms"]);rows=rows(input["observations"]))
        @test [e.id for e in result.events if e.kind===:completed_target]==[1,2]
        @test [e.target for e in result.events if e.kind===:completed_target]==[2.0,0.0]
        @test [e.error for e in result.subscriber_events if e.name===:fifo && e.kind===:feedback]==[true,false]
        @test [e.id for e in result.subscriber_events if e.name===:fifo && e.kind===:score]==[1,2]
        @test isapprox(only(subscriber(owner,:fifo).alpha),0.02;atol=1e-14,rtol=0) # Fraction1/50, frozen alpha tolerance.
        @test all(c.binary_resolved==2 && c.completed_scores==2 for c in (subscriber_certificate(owner,n) for n in (:static,:phase,:fifo,:early,:maturity)))
        push!(results,result)
    end
    @test isequal(results[1].events,results[2].events) && isequal(results[1].subscriber_events,results[2].subscriber_events)
    owner=state()
    transact!(owner,I;issues=(request(),));transact!(owner,I+Minute(30);rows=rate_rows(1,29;positive_at=1))
    result=transact!(owner,I+Minute(31);rows=rate_rows(30,30),issues=(request(I+Minute(31)),))
    @test [e.kind for e in result.events]==[:source_receipt,:completed_target,:issue]
    @test only(result.issues).bounds[2].upper==Inf
    @test isapprox(only(result.issues).bounds[2].alpha,0.01;atol=1e-14,rtol=0) #1/100 after the old miss.

    owner=state(;grace=180000)
    transact!(owner,I;issues=(request(),));transact!(owner,I+Minute(1);issues=(request(I+Minute(1)),))
    first=transact!(owner,I+Minute(32);rows=rate_rows(2,31),issues=(request(I+Minute(32)),))
    second=transact!(owner,I+Minute(33);rows=rate_rows(1,1;positive_at=1),issues=(request(I+Minute(33)),))
    @test [e.id for result in (first,second) for e in result.events if e.kind===:completed_target]==[2,1]
    @test [only(result.issues).bounds[3].upper for result in (first,second)]==[0.0,Inf]
    @test all(isapprox(only(result.issues).bounds[3].alpha,expected;atol=1e-14,rtol=0) for (result,expected) in zip((first,second),(.11,.02))) #11/100,1/50.
    @test [e.id for result in (first,second) for e in result.subscriber_events if e.name===:fifo && e.kind===:score]==[2,1]
    @test source_certificate(owner).completed_targets==2 && source_certificate(owner).pending_targets==2
end

@testset verbose=true "Early unknown maxima, signed zero and complete scores" begin
    owner=state(;subscribers=specs(;rolling=9))
    transact!(owner,I;issues=(request(),))
    first=transact!(owner,offset(105000);rows=rate_rows(1,1;positive_at=1))
    @test count(e->e.kind===:feedback,first.subscriber_events)==1
    @test subscriber_certificate(owner,:early).binary_resolved==1
    @test subscriber_certificate(owner,:maturity).binary_resolved==0
    partial=[r for r in rate_rows(2,30) if r.center!=I+Minute(15)]
    transact!(owner,I+Minute(31);rows=partial)
    expiry=transact!(owner,offset(1860001))
    @test only(expiry.events).kind===:censored_target
    @test only(expiry.events).missing_centers==(I+Minute(15),)
    @test source_certificate(owner).target_unknown==1 && source_certificate(owner).completed_targets==0
    @test subscriber_certificate(owner,:early).binary_unknown==0 && subscriber_certificate(owner,:maturity).binary_unknown==1
    @test all(subscriber_certificate(owner,n).completed_scores==0 for n in (:static,:phase,:fifo,:early,:maturity))
    late=transact!(owner,offset(1860002);rows=rate_rows(15,15))
    @test !any(e->e.kind===:completed_target,late.events) && source_certificate(owner).completed_targets==0
    zeros_results=Any[]
    for name in ("signed_zero_duplicate_plus_minus","signed_zero_duplicate_minus_plus")
        owner=state();transact!(owner,I;issues=(request(),))
        input=FIXTURES[name]["input"]["batch"]
        result=transact!(owner,offset(input["received_offset_ms"]);rows=rows(input["observations"]))
        @test length(result.events)==1 && only(result.events).kind===:source_receipt
        @test reinterpret(UInt64,only(result.events).rate)==0x0000000000000000
        @test Set(r.raw_bits for r in only(result.events).raw_identities)==Set([0x0000000000000000,0x8000000000000000])
        @test only(owner.snapshot.pending).rates[1]===0.0 && source_certificate(owner).pending_targets==1
        @test all(subscriber_certificate(owner,n).binary_resolved==0 for n in (:static,:phase,:fifo,:early,:maturity))
        push!(zeros_results,result)
    end
    @test isequal(zeros_results[1].events,zeros_results[2].events)
    owner=state();transact!(owner,I;issues=(request(;center=2,scale=2),))
    completed=transact!(owner,I+Minute(31);rows=rate_rows(1,30;positive_at=1))
    expected=parse(BigFloat,ORACLES["complete_score_oracles"]["score_target2_center2_scale2_decimal"])
    @test all(isapprox(e.score,expected;atol=1e-13,rtol=0) for e in completed.subscriber_events if e.kind===:score) # Independent Decimal80 ln3 oracle.
end
