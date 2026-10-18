function zero_prefix_case(R)
    f = fixture()
    initial = R.Checkpoint(f.origin, f.mean, f.covariance)
    intervals = [R.ForcingInterval("bin$j", f.origin+Hour(j-1), f.origin+Hour(j),
                                  -f.gains.*q, f.origin+Hour(j)) for (j,q) in enumerate(f.q)]
    function observations(records)
        [R.MeanObservation(o.revision_id, "bin$(Int(Dates.value(o.stop-f.origin) ÷ 3_600_000))",
            o.start, o.stop, o.value, o.R, o.receipt, o.verified) for o in records]
    end
    cache = R.ReplayCache(initial, f.lambdas, f.sigmas)
    boundary = f.origin+Hour(6)
    first = R.replay!(cache, intervals, observations(f.observations), boundary; asof=boundary+Millisecond(1))
    revision = merge(f.observations[5], (revision_id="revised-first-bin", value=-35.0,
                      receipt=boundary+Millisecond(10), verified=boundary+Millisecond(11)))
    revised_records = [f.observations; revision]
    revised_fixture = merge(f, (observations=revised_records,))
    selected_ids = ["revised-first-bin", "late-bin2", "initial-bin4", "future-receipt-bin5", "boundary-bin6"]
    expected = expected_posterior(revised_fixture, 6; selected_ids)
    result, failure, trace = nothing, nothing, ""
    try
        result = R.replay!(cache, intervals, observations(revised_records), boundary; asof=boundary+Millisecond(20))
    catch e
        failure = e
        trace = sprint(showerror, e, catch_backtrace())
    end
    return (; f, cache, first, result, failure, trace, expected)
end
