using Dates, Test
isdefined(@__MODULE__, :InitialReferenceAudit) || error("Load the candidate reader before this frozen oracle")
const IRA = InitialReferenceAudit

@testset "Independent pre-code initial-reference clocks and cohorts" begin
    origin = DateTime(2019)
    times = [origin + Hour(i-1) for i in 1:1000]
    dst = -Float64.(1:1000)
    dst[[620,625]] .= NaN
    available = BitVector(i >= 25 for i in 1:1000)
    for (h,lastfit,lastvalidation,excluded,n,an) in
        ((4,328,728,[616,620,621,625],225,224),(6,326,726,[614,619,620,625],223,222))
        cohort = IRA.expected_cohorts(times,dst,available,h;
            initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
        @test cohort.fit == collect(25:lastfit)
        @test cohort.validation == setdiff(collect(500:lastvalidation),excluded)
        @test length(cohort.validation) == n
        @test cohort.a_join == cohort.validation
        later = IRA.expected_cohorts(times,dst,available,h;
            initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900),
            a_start=origin+Hour(501))
        @test later.a_join == setdiff(collect(501:lastvalidation),excluded)
        @test length(later.a_join) == an
        unavailable = IRA.expected_cohorts(times,dst,falses(1000),h;
            initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
        @test unavailable.validation == cohort.validation
        @test unavailable.a_join == cohort.a_join
        @test isempty(unavailable.fit)
        future = copy(dst); future[901:end] .= 123456
        unchanged = IRA.expected_cohorts(times,future,available,h;
            initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
        @test unchanged == cohort
        observed = collect(1.0:n)
        prediction = copy(observed); prediction[1:123] .+= h
        score = IRA.metrics(prediction,observed)
        @test score.n == n
        @test score.rmse_nt ≈ h*sqrt(123/n) rtol=5e-12 atol=0
        @test score.mae_nt == h*123/n
        @test score.bias_nt == h*123/n
        @test IRA.verify_metric(score,(n=n,rmse_nt=h*sqrt(123/n),mae_nt=h*123/n,bias_nt=h*123/n))
        @test_throws ArgumentError IRA.verify_metric(merge(score,(n=n-1,)),score)
    end
    clock = IRA.forecast_clock(origin+Hour(328),4)
    @test clock.origin_tag == clock.anchor_tag == clock.driver_tag == origin+Hour(327)
    @test clock.target_tag == origin+Hour(331)
    @test clock.anchor_complete == clock.anchor_receipt == origin+Hour(328)
    @test clock.target_complete == clock.target_receipt == origin+Hour(332)
    @test IRA.eligible(clock.issue,clock.target_complete,clock.target_receipt,origin+Hour(500))
    @test !IRA.eligible(clock.issue,clock.target_complete,clock.target_receipt+Millisecond(1),origin+Hour(500))
    @test_throws ArgumentError IRA.forecast_clock(origin,true)
    @test_throws ArgumentError IRA.forecast_clock(origin,5)
    @test_throws ArgumentError IRA.forecast_clock(origin+Millisecond(1),4)
    @test_throws ArgumentError IRA.shift_hour(origin,typemax(Int64))
    @test_throws ArgumentError IRA.expected_cohorts(times,dst,available,4;
        initial_boundary=origin+Hour(500)+Millisecond(1),selection_boundary=origin+Hour(900))
    @test_throws ArgumentError IRA.expected_cohorts(times,dst[1:end-1],available,4;
        initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
    duplicate = copy(times); duplicate[300] = duplicate[299]
    @test_throws ArgumentError IRA.expected_cohorts(duplicate,dst,available,4;
        initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
    bad = copy(dst); bad[10] = Inf
    @test_throws ArgumentError IRA.expected_cohorts(times,bad,available,4;
        initial_boundary=origin+Hour(500),selection_boundary=origin+Hour(900))
end

@testset "Strict original grid order, ties and metric validity" begin
    grid = Tuple((max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255))
    @test IRA.GRID == grid
    score = (n=20,rmse_nt=1.,mae_nt=.5,bias_nt=0.)
    candidates = [(configuration=g,metric=score) for g in grid]
    @test IRA.first_grid_choice(candidates) == 1
    candidates[8] = (configuration=grid[8],metric=merge(score,(rmse_nt=prevfloat(1.),)))
    @test IRA.first_grid_choice(candidates) == 8
    @test_throws ArgumentError IRA.first_grid_choice(candidates[1:7])
    @test_throws ArgumentError IRA.first_grid_choice(reverse(candidates))
    malformed = copy(candidates)
    malformed[1] = (configuration=grid[1],metric=merge(score,(rmse_nt=NaN,)))
    @test_throws ArgumentError IRA.first_grid_choice(malformed)
    @test_throws ArgumentError IRA.metrics(Float64[],Float64[])
    @test_throws ArgumentError IRA.metrics([1.],[Inf])
    @test_throws ArgumentError IRA.metrics([true],[1.])
    @test_throws ArgumentError IRA.verify_metric(merge(score,(n=true,)),score)
end
