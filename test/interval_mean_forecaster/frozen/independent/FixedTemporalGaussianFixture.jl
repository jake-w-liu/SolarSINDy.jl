module FixedTemporalGaussianFixture

using Dates, LinearAlgebra
using ..IndependentGaussianOracle

export fixture, expected_posterior, expected_target

function fixture()
    origin = DateTime(2020, 1, 1)
    mean = [-12.0, 5.0]
    covariance = [9.0 -3.0; -3.0 16.0]
    lambdas = [log(2.0), 1/12]
    sigmas = [1.0, 2.5]
    gains = [2.5, 4.0]
    q = [0.0, 1.5, 0.25, 3.0, 0.75, 0.5, 9.0, 0.125]
    observation(bin, value, receipt, verified, revision_id) =
        (start=origin + Hour(bin-1), stop=origin + Hour(bin), value=Float64(value), R=1.25,
         receipt=origin + receipt, verified=origin + verified, revision_id=revision_id)
    observations = [
        observation(7, -22.0, Hour(7), Hour(7), "future-bin7"),
        observation(2, -23.0, Hour(5)+Minute(10), Hour(5)+Minute(10)+Millisecond(1), "late-bin2"),
        observation(1, -15.0, Hour(1)+Minute(5), Hour(1)+Minute(5)+Millisecond(1), "initial-bin1"),
        observation(6, -20.0, Hour(6), Hour(6), "boundary-bin6"),
        observation(1, -17.0, Hour(5)+Minute(30), Hour(5)+Minute(30)+Millisecond(1), "revision-bin1"),
        observation(4, -18.0, Hour(4)+Minute(2), Hour(4)+Minute(2)+Millisecond(1), "initial-bin4"),
        # Its verification finishes later, but capture is older than revision-bin1.
        observation(1, -99.0, Hour(4)+Minute(45), Hour(5)+Minute(59), "older-capture-late-verification"),
        observation(5, -25.0, Hour(6)+Millisecond(5), Hour(6)+Millisecond(6), "future-receipt-bin5")]
    # Manual eligibility/revision truth, not a copy of the candidate selector.
    selected = Dict(
        1 => String[],
        2 => ["initial-bin1"],
        3 => ["initial-bin1"],
        4 => ["initial-bin1"],
        5 => ["initial-bin1", "initial-bin4"],
        6 => ["revision-bin1", "late-bin2", "initial-bin4", "boundary-bin6"],
        7 => ["revision-bin1", "late-bin2", "initial-bin4", "future-receipt-bin5", "boundary-bin6", "future-bin7"],
        8 => ["revision-bin1", "late-bin2", "initial-bin4", "future-receipt-bin5", "boundary-bin6", "future-bin7"])
    return (; origin, mean, covariance, lambdas, sigmas, gains, q, observations, selected)
end

function expected_posterior(f, boundary_bin; gains=f.gains, selected_ids=f.selected[boundary_bin])
    joint = timeline_joint(f.mean, f.covariance, f.lambdas, f.sigmas,
                           [-gains .* q for q in f.q[1:boundary_bin]], ones(boundary_bin))
    selected = [only(filter(o -> o.revision_id == id, f.observations)) for id in selected_ids]
    rows = [2 + Int(Dates.value(o.stop - f.origin) ÷ 3_600_000) for o in selected]
    posterior = gaussian_condition(joint, [1, 2], rows, [o.value for o in selected],
                                   Matrix(Diagonal([o.R for o in selected])))
    return (; posterior..., selected_ids)
end

function expected_target(f, boundary_bin, horizon; gains=f.gains, selected_ids=f.selected[boundary_bin])
    posterior = expected_posterior(f, boundary_bin; gains, selected_ids)
    # Hold the last completed q; no later observed q enters this expectation.
    q = f.q[boundary_bin]
    future = timeline_joint(posterior.mean, posterior.covariance, f.lambdas, f.sigmas,
                            fill(-gains .* q, horizon), ones(horizon))
    return (mean=future.mean[2+horizon], variance=future.covariance[2+horizon, 2+horizon])
end

end
