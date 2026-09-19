module DailyReviewTests
using Test, Dates, JSON3, HTTP, SHA
include(joinpath(@__DIR__,"../examples/daily_review.jl"))
const R = DailyOperationalReview

@testset "Immutable daily operational review" begin
    reference = DateTime(2026,9,19,16)
    clock = ()->reference
    calls = String[]
    getter(url;kwargs...) = begin
        push!(calls,url)
        @test kwargs[:retries] == 0
        @test kwargs[:readtimeout] == 10
        @test kwargs[:connect_timeout] == 3
        data = endswith(url,"/api/health") ? (;status="ok") :
            endswith(url,"CMO") ? (;station="CMO",available=true,data_type="adjusted",forecast_status="disabled") :
            (;station="FRD",available=false,data_type="variation",forecast_status="disabled")
        HTTP.Response(200,JSON3.write(data))
    end
    mktempdir() do root
        claim = joinpath(root,"v2_4_live_claim_status.json")
        write(claim,JSON3.write((;generated_utc=R.utc(reference),integrity=(;violations=["historical finding"]),
            marginal=(;n_rows=10,coverage_90=.8,claim_ready=false),storm=(;claim_ready=false))))
        write(joinpath(root,"live_comparison_report.md"),"Same-row model comparison\n")
        screen = joinpath(root,"screen.json")
        write(screen,JSON3.write((;receipt_time_screen_pass=false)))
        output = R.write_daily_review(root;reference,http_get=getter,clock,screen_path=screen,
            dashboard_health_url="http://127.0.0.1:9000/api/health")
        @test length(calls) == 3
        @test all(startswith(url,"http://127.0.0.1:9000/") for url in calls)
        @test basename(output) == "2026-09-19"
        @test R.verify_review(output) == output
        record = JSON3.read(read(joinpath(output,"review.json")))
        @test record.claim.fresh
        @test !record.claim.payload.marginal.claim_ready
        @test record.claim.payload.integrity.violations == ["historical finding"]
        @test record.ground.CMO.payload.available
        @test !record.ground.FRD.payload.available
        @test record.claim.sha256 == bytes2hex(sha256(read(claim)))
        markdown = read(joinpath(output,"review.md"),String)
        @test occursin("A3 historical integrity findings: 1",markdown)
        @test occursin("measurement available=true; product=adjusted",markdown)
        @test occursin("measurement available=false; product=variation",markdown)
        @test occursin("CMO receipt-time screen: false",markdown)
        saved = read(joinpath(output,"review.json"))
        write(claim,"invalid replacement")
        @test R.write_daily_review(root;reference,http_get=(_;kwargs...)->error("unexpected refetch"),clock) == output
        @test read(joinpath(output,"review.json")) == saved
        @test !R.capture_file(claim;reference,check_age=true).available
        write(claim,JSON3.write((;generated_utc=R.utc(reference-Hour(3)))))
        @test !R.capture_file(claim;reference,check_age=true).fresh
        write(claim,JSON3.write((;generated_utc=R.utc(reference+Minute(1)))))
        @test !R.capture_file(claim;reference,check_age=true).fresh
        write(claim,JSON3.write((;generated_utc=R.utc(reference-Minute(130)))))
        @test R.capture_file(claim;reference,check_age=true).fresh
        @test !R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(503,"bad"),clock).available
        @test !R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(200,"invalid"),clock).available
        invalid = R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(200,"invalid"),clock)
        @test invalid.body == "invalid"
        @test invalid.sha256 == bytes2hex(sha256("invalid"))
        @test invalid.http_status == 200
        unavailable = R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(503,"offline"),clock)
        @test unavailable.http_status == 503 && unavailable.body == "offline"
        backwards = [reference,reference-Second(1)]
        skewed = R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(200,"{}"),
            clock=()->popfirst!(backwards))
        @test !skewed.available
        @test skewed.completed_utc == R.utc(reference-Second(1))
        oversized = R.capture_endpoint("http://test";http_get=(_;kwargs...)->HTTP.Response(200,repeat("x",1024*1024+1)),clock)
        @test !oversized.available && oversized.body === nothing
        write(claim,"broken JSON")
        broken = R.capture_file(claim;reference,check_age=true)
        @test broken.body == "broken JSON"
        @test broken.sha256 == bytes2hex(sha256("broken JSON"))
        write(joinpath(output,"review.md"),"tampered")
        @test_throws ErrorException R.verify_review(output)
        @test_throws ErrorException R.write_daily_review(root;reference,http_get=getter,clock)
        @test read(joinpath(output,"review.md"),String) == "tampered"
    end
    mktempdir() do root
        # Missing inputs are retained as unavailable, including a failed HTTP request.
        fail = (_;kwargs...)->error("source unavailable")
        output = R.write_daily_review(root;reference,http_get=fail,clock)
        record = JSON3.read(read(joinpath(output,"review.json")))
        @test !record.claim.available && !record.claim.fresh
        @test !record.dashboard.available
        @test occursin("source unavailable",record.dashboard.error)
        @test occursin("A3 matured rows: unavailable",read(joinpath(output,"review.md"),String))
        @test isempty(filter(name->startswith(name,".pending-"),readdir(dirname(output))))
        @test !isfile(joinpath(dirname(output),"daily-review.lock"))
    end
    mktempdir() do root
        reviews = joinpath(root,"reviews"); mkpath(reviews)
        partial = joinpath(reviews,".pending-interrupted"); mkpath(partial)
        write(joinpath(partial,"review.json"),"partial")
        output = R.write_daily_review(root;reference,http_get=getter,clock)
        @test R.verify_review(output) == output
        @test read(joinpath(partial,"review.json"),String) == "partial"
        if Sys.isunix()
            link = joinpath(reviews,string(Date(reference)+Day(1)))
            symlink(joinpath(root,"missing"),link)
            @test_throws ErrorException R.write_daily_review(root;reference=reference+Day(1),http_get=getter,clock)
        end
    end
    mktempdir() do root
        # Cooperative concurrent calls share the same immutable once-a-day receipt.
        slow = (url;kwargs...)->(yield();HTTP.Response(200,"{\"status\":\"ok\"}"))
        tasks = [@async R.write_daily_review(root;reference,http_get=slow,clock) for _ in 1:4]
        paths = fetch.(tasks)
        @test length(unique(paths)) == 1
        @test R.verify_review(first(paths)) == first(paths)
    end
    @test_throws ArgumentError R.write_daily_review(mktempdir();dashboard_health_url="http://test/wrong")
end
end
