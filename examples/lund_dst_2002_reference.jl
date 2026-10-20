using SolarSINDy

"Exercise the fixed published recurrence and exact chunk continuation, not skill."
function run_lund_dst_2002_reference()
    # A deterministic driver scenario. No observations or trained targets enter it.
    bz = vcat(fill(0.0, 24), fill(-15.0, 12), fill(5.0, 24))
    density = fill(8.0, length(bz))
    speed = vcat(fill(400.0, 24), fill(650.0, 12), fill(450.0, 24))
    direct = LundDst2002.rollout(bz, density, speed)
    first = LundDst2002.rollout(bz[1:30], density[1:30], speed[1:30])
    last = LundDst2002.rollout(bz[31:end], density[31:end], speed[31:end]; state=first.state)
    @assert vcat(first.dst_nt, last.dst_nt) == direct.dst_nt
    @assert last.state == direct.state
    @assert all(isfinite, direct.dst_nt)
    @assert all(x -> -283.015 <= x <= 104.375, direct.dst_nt)
    println("Published Lund 2002 reference: 60 deterministic hourly updates and chunk continuation PASS; no skill verdict")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lund_dst_2002_reference()
end
