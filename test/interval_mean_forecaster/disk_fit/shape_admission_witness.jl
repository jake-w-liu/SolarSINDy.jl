using Test, Dates, SHA, TOML
include(ARGS[1])
include(joinpath(@__DIR__,"DiskAffineGainFit.jl"))
const F=IntervalMeanForecaster
const T=F.DstIntervalTraining
const A=T.A
const D=DiskAffineGainFit
const SOURCE=repeat("a",64)
const SETTING=A.settings()[1]
const BOUNDARY=DateTime(2019,3,1)
function rows(kind)
    [begin
        issue=DateTime(2019,1,1)+Hour(i)
        c=Tuple(A.forecast_clock(issue,h,0) for h in (4,6))
        z=kind==:third_nonzero ? ((0.,0.,1.),(0.,0.,1.)) : ((1.,0.,7.),(0.,1.,7.))
        (;issue,driver_tag=c[1].driver_tag,driver_receipt=issue,
         anchor_tag=c[1].anchor_tag,anchor_complete=c[1].anchor_complete,
         anchor_receipt=c[1].anchor_receipt,anchor=-50.,
         target_tag=Tuple(x.target_tag for x in c),target_complete=Tuple(x.target_complete for x in c),
         target_receipt=Tuple(x.target_receipt for x in c),target=(-60.,-40.),
         scoreable=(true,true),model_available=true,offset=(-62.,-43.),coefficients=z,
         taus=SETTING.taus,sigma=SETTING.sigma,source_sha256=SOURCE,dst_delay_hours=0)
    end for i in 1:110]
end
mkdir(ARGS[2])
@testset "reject third coefficient instead of publishing two-column reinterpretation" begin
    for kind in (:extra_nonzero,:third_nonzero)
        input=rows(kind);destination=joinpath(ARGS[2],String(kind))
        @test_throws DimensionMismatch T.fit_gains(input,BOUNDARY,SETTING;delay_hours=0,source_sha256=SOURCE)
        accepted=false;receipt=nothing
        try
            receipt=D.export_fit_input(sink->sink(input),destination,BOUNDARY,SETTING;delay_hours=0,source_sha256=SOURCE)
            accepted=true
        catch error
            error isa ArgumentError || rethrow()
        end
        @test !accepted
        @test !isfile(joinpath(destination,"manifest.toml"))
        if accepted
            try
                result=D.solve_fit_input(destination;expected_manifest_sha256=receipt.manifest_sha256)
                println("MALFORMED_",kind,"_SOLVED_GAINS=",result.gains)
            catch error
                println("MALFORMED_",kind,"_LOAD_FAILED=",sprint(showerror,error))
            end
        end
    end
end
