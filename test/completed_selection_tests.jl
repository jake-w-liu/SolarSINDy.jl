# UNEXECUTED frozen pure-selection controls; no actual observations or fits.
module CompletedSelectionPrimitivesTests
using Test, Dates, JSON3, SolarSINDy
using SolarSINDy.CompletedSelectionPrimitives
const S = SolarSINDy.CompletedSelectionPrimitives
const FIXTURE = JSON3.read(read(joinpath(@__DIR__,"completed_selection","LITERAL_ORACLES_V1.json")),Dict{String,Any})

function literal_number(value)
    value === nothing && return nothing
    if occursin('/',value)
        numerator,denominator = split(value,'/')
        return Float64(parse(BigInt,numerator) // parse(BigInt,denominator))
    end
    parse(Float64,value)
end
literal_word(value) = reinterpret(Float64,parse(UInt64,value;base=16))
function gamma_row(row,schedule)
    GammaMetrics(;station=row["station"],schedule,grid=row["grid"],gamma=literal_word(row["gamma_bits"]),
        cohort=row["cohort"],n=row["n"],normalization=literal_number(row["normalization"]),
        normalized_pinball=literal_number(row["normalized_pinball"]),full_count=row["full"],
        empty_count=row["empty"],positive_inf_count=row["positive_inf"],negative_inf_count=row["negative_inf"],
        finite_negative_count=row["finite_negative"],nan_count=row["nan"],phase_period_minutes=31)
end
function gamma_expectation(schedule)
    GammaExpectation(schedule;cohorts=("literal-CMO","literal-FRD"),counts=(1,9),
        normalizations=(1.0,1.0),phase_period_minutes=31)
end
function gamma_case(case)
    schedule = length(case["priority"]) == 1 ? :phase : :fifo
    rows = [gamma_row(row,schedule) for row in case["rows"]]
    choose_gamma(rows,gamma_expectation(schedule))
end
function point_row(row)
    # Three is a literal fixture count, not an actual fitting/cohort claim.
    PointMetrics(;station="CMO",candidate_id=row["candidate_id"],family=Symbol(row["family"]),
        cohort=row["cohort"],n=3,selection_log_mse=literal_number(row["loss"]),phase_period_minutes=31)
end
point_expectation() = PointExpectation(;station="CMO",cohort="literal-ground-CMO",n=3,phase_period_minutes=31)
function point_case(case)
    choose_R2([point_row(row) for row in case["rows"]],point_expectation())
end
function privileges_are_false(result)
    !result.actual_source_access && result.actual_fits == 0 &&
    !result.source_bound_provenance_verified && !result.software_qualification &&
    !result.scientific_qualification && !result.actual_coordinator_implemented
end

@testset "Frozen independent gamma winners, rational objectives and exact ties" begin
    @test FIXTURE["fixture_only"] === true && FIXTURE["no_actual_source_or_fit_authority"] === true
    positive = filter(case -> !get(case,"expected_reject",false),FIXTURE["gamma_cases"])
    @test length(positive) == 8
    for case in positive
        result = gamma_case(case)
        @test result.selected_grid == case["expected_grid"]
        @test reinterpret(UInt64,result.gamma) == parse(UInt64,case["expected_gamma_bits"];base=16)
        @test result.objective == literal_number(case["expected_objective"])
        @test length(result.candidate_table) == (length(case["priority"]) == 1 ? 7 : 14)
        @test all(length(row.station_reports) == 2 for row in result.candidate_table)
        @test privileges_are_false(result) && result.primary_selection && !result.sensitivity_only
        schedule = length(case["priority"]) == 1 ? :phase : :fifo
        reversed = choose_gamma(reverse([gamma_row(row,schedule) for row in case["rows"]]),gamma_expectation(schedule))
        @test reversed.decision_sha256 == result.decision_sha256
    end
    # Literal example: per-station losses(0,1) average to1/2 despite1:9 row
    # counts; pooled loss9/10 would incorrectly select the(.6,.6) candidate.
    equal = only(filter(case -> case["name"] == "equal_station_weight_not_row_pooling",positive))
    result = gamma_case(equal)
    @test result.objective == 1/2
    @test reinterpret(UInt64,result.gamma) != parse(UInt64,equal["incorrect_row_pooled_winner_bits"];base=16)
    priority = only(filter(case -> case["name"] == "grid_priority_precedes_cross_grid_loss",positive))
    @test gamma_case(priority).selected_grid == "phase_declared"
    @test gamma_case(priority).objective == 10.0
end

@testset "Frozen gamma rejection cases retain a failed complete grid" begin
    negative = filter(case -> get(case,"expected_reject",false),FIXTURE["gamma_cases"])
    @test length(negative) == 11
    for case in negative
        if case["name"] == "all_ineligible"
            error = try
                gamma_case(case)
                nothing
            catch value
                value
            end
            @test error isa NoEligibleGamma
            if error isa NoEligibleGamma
                @test length(error.candidate_table) == 7
                @test all(!row.eligible && row.objective === nothing for row in error.candidate_table)
                @test all(length(row.station_reports) == 2 for row in error.candidate_table)
            end
        else
            @test_throws ArgumentError gamma_case(case)
        end
    end
end

@testset "Primary FIFO division30 and separately named division31 sensitivity" begin
    @test_throws ArgumentError GammaExpectation(:phase;cohorts=("a","b"),counts=(1,9),normalizations=(1.,1.),phase_period_minutes=30)
    case = first(filter(case -> !get(case,"expected_reject",false),FIXTURE["gamma_cases"]))
    raw = deepcopy(case["rows"])
    original = FIXTURE["gamma_bits_by_grid"]["phase_declared"]
    sensitivity = FIXTURE["gamma_bits_by_grid"]["clock_matched_1_over_31_sensitivity"]
    for row in raw
        index = findfirst(==(row["gamma_bits"]),original)
        row["grid"] = "clock_matched_1_over_31_sensitivity"
        row["gamma_bits"] = sensitivity[index]
    end
    rows = [gamma_row(row,:fifo_clock31_sensitivity) for row in raw]
    result = choose_gamma(rows,gamma_expectation(:fifo_clock31_sensitivity))
    @test result.sensitivity_only && !result.primary_selection && privileges_are_false(result)
    @test result.selected_grid == "clock_matched_1_over_31_sensitivity"
    @test_throws ArgumentError choose_gamma(rows,gamma_expectation(:phase))
    wrong_schedule = [gamma_row(row,:fifo) for row in case["rows"]]
    @test_throws ArgumentError choose_gamma(wrong_schedule,gamma_expectation(:phase))
    wrong_expected = GammaExpectation(:phase;cohorts=("literal-CMO","literal-FRD"),counts=(1,9),
        normalizations=(2.,1.),phase_period_minutes=31)
    @test_throws ArgumentError choose_gamma([gamma_row(row,:phase) for row in case["rows"]],wrong_expected)
    # Historical finite negative bounds can have a genuinely finite loss. It
    # remains retained but can never make the support-empty candidate eligible.
    negative = deepcopy(case["rows"])
    for row in negative
        row["gamma_bits"] == original[1] || continue
        row["empty"] = 1;row["finite_negative"] = 1;row["normalized_pinball"] = "0"
    end
    selected = choose_gamma([gamma_row(row,:phase) for row in negative],gamma_expectation(:phase))
    @test selected.gamma == literal_word(original[2])
    @test !selected.candidate_table[1].eligible
    @test selected.candidate_table[1].station_reports[1].normalized_pinball == 0.0
end

@testset "R2 independent first-in-grid and family ties retain all16" begin
    @test Tuple(row.id for row in S.POINT_GRID) == Tuple(row["id"] for row in FIXTURE["R2_grid"])
    # Complete literal parameter records from the historical 2017 fitter's
    # declared loops. IDs alone must not hide a changed depth/tree/penalty.
    declared = (
        (id="ridge-0.01",family=:ridge,depth=0,trees=0,penalty=0.01),
        (id="ridge-0.1",family=:ridge,depth=0,trees=0,penalty=0.1),
        (id="ridge-1",family=:ridge,depth=0,trees=0,penalty=1.0),
        (id="ridge-10",family=:ridge,depth=0,trees=0,penalty=10.0),
        (id="ridge-100",family=:ridge,depth=0,trees=0,penalty=100.0),
        (id="boost-2-50",family=:boost,depth=2,trees=50,penalty=0.0),
        (id="boost-2-100",family=:boost,depth=2,trees=100,penalty=0.0),
        (id="boost-2-200",family=:boost,depth=2,trees=200,penalty=0.0),
        (id="boost-3-50",family=:boost,depth=3,trees=50,penalty=0.0),
        (id="boost-3-100",family=:boost,depth=3,trees=100,penalty=0.0),
        (id="boost-3-200",family=:boost,depth=3,trees=200,penalty=0.0),
        (id="boost-4-50",family=:boost,depth=4,trees=50,penalty=0.0),
        (id="boost-4-100",family=:boost,depth=4,trees=100,penalty=0.0),
        (id="boost-4-200",family=:boost,depth=4,trees=200,penalty=0.0),
        (id="forest-6-64",family=:forest,depth=6,trees=64,penalty=0.0),
        (id="forest-9-64",family=:forest,depth=9,trees=64,penalty=0.0),
    )
    @test S.POINT_GRID == declared
    for case in FIXTURE["R2_cases"]
        if get(case,"expected_reject",false)
            @test_throws ArgumentError point_case(case)
            continue
        end
        result = point_case(case);expected = case["expected"]
        @test length(result.candidate_table) == 16 && length(result.family_winners) == 3
        @test Tuple(row.candidate_id for row in result.family_winners) ==
            (expected["ridge"],expected["boost"],expected["forest"])
        @test result.primary_family === Symbol(expected["primary"])
        @test result.selected_family_refits_required == 3 && privileges_are_false(result)
        @test choose_R2(reverse([point_row(row) for row in case["rows"]]),point_expectation()).decision_sha256 == result.decision_sha256
    end
    tied = point_case(only(filter(case -> case["name"] == "all_exact_ties",FIXTURE["R2_cases"])))
    @test Tuple(row.candidate_id for row in tied.family_winners) == ("ridge-0.01","boost-2-50","forest-6-64")
    @test tied.primary_family === :ridge && tied.primary_candidate_id == "ridge-0.01"
    raw = only(filter(case -> case["name"] == "all_exact_ties",FIXTURE["R2_cases"]))["rows"]
    frd = [PointMetrics(;station="FRD",candidate_id=row["candidate_id"],family=Symbol(row["family"]),
        cohort="literal-ground-FRD",n=3,selection_log_mse=literal_number(row["loss"]),phase_period_minutes=31) for row in raw]
    frd_expected = PointExpectation(;station="FRD",cohort="literal-ground-FRD",n=3,phase_period_minutes=31)
    @test Tuple(row.candidate_id for row in choose_R2(frd,frd_expected).family_winners) ==
        ("ridge-0.01","boost-2-50","forest-6-64")
    @test_throws ArgumentError choose_R2([point_row(row) for row in raw],frd_expected)
    wrong_n = PointExpectation(;station="CMO",cohort="literal-ground-CMO",n=4,phase_period_minutes=31)
    @test_throws ArgumentError choose_R2([point_row(row) for row in raw],wrong_n)
end

@testset "Legal nonwinner metric changes bind every retained candidate" begin
    equal = only(filter(case -> case["name"] == "equal_station_weight_not_row_pooling",FIXTURE["gamma_cases"]))
    before = gamma_case(equal)
    changed = deepcopy(equal)
    # The final .10/FRD candidate is a legal nonwinner with loss2. Raising its
    # loss to3 leaves the independently frozen .001 winner and objective1/2.
    changed["rows"][end]["normalized_pinball"] = "3"
    after = gamma_case(changed)
    @test after.selected_grid == before.selected_grid == "phase_declared"
    @test after.gamma == before.gamma == literal_word(equal["expected_gamma_bits"])
    @test after.objective == before.objective == 1/2
    @test after.input_sha256 != before.input_sha256
    @test after.decision_sha256 != before.decision_sha256
    tied = only(filter(case -> case["name"] == "all_exact_ties",FIXTURE["R2_cases"]))
    before_point = point_case(tied)
    changed_point = deepcopy(tied)
    # Forest depth9 is not the retained depth6 family winner under exact ties.
    changed_point["rows"][end]["loss"] = "2"
    after_point = point_case(changed_point)
    @test Tuple(row.candidate_id for row in after_point.family_winners) ==
        ("ridge-0.01","boost-2-50","forest-6-64")
    @test after_point.family_winners == before_point.family_winners
    @test after_point.primary_family === :ridge && before_point.primary_family === :ridge
    @test after_point.primary_candidate_id == before_point.primary_candidate_id == "ridge-0.01"
    @test after_point.input_sha256 != before_point.input_sha256
    @test after_point.decision_sha256 != before_point.decision_sha256
end

@testset "Strict2017 source-completion boundaries without a target argument" begin
    for row in FIXTURE["clock_cases"]
        issue = DateTime(first(split(row["issue"],'+')))
        boundary = DateTime(first(split(row["boundary"],'+')))
        extra = row["delay_ms"] - 1845000
        @test split_clock_eligible(issue,boundary;ground_extra_ms=extra) === row["eligible"]
    end
    @test split_clock_eligible(DateTime(2017,6,30,23,29),DateTime(2017,7,1))
    @test !split_clock_eligible(DateTime(2017,6,30,23,30),DateTime(2017,7,1))
    @test !split_clock_eligible(DateTime(2017,6,30,23,29),DateTime(2017,7,1);label_extra_ms=15000)
    @test !split_clock_eligible(DateTime(2025),DateTime(2025,2))
    @test_throws ArgumentError split_clock_eligible(DateTime(2017)+Second(1),DateTime(2017,2))
    @test_throws ArgumentError split_clock_eligible(DateTime(2017),DateTime(2017,2);ground_extra_ms=true)
    @test_throws ArgumentError split_clock_eligible(DateTime(2017),DateTime(2017,2);label_extra_ms=-1)
end

@testset "Malformed numeric metadata cannot acquire a valid decision" begin
    base = first(FIXTURE["gamma_cases"])["rows"][1]
    for field in ("n","full","empty","positive_inf","negative_inf","finite_negative","nan")
        row = deepcopy(base);row[field] = true
        @test_throws ArgumentError gamma_row(row,:phase)
    end
    row = deepcopy(base);row["normalized_pinball"] = "-1"
    @test_throws ArgumentError gamma_row(row,:phase)
    row = deepcopy(base);row["normalized_pinball"] = nothing
    @test_throws ArgumentError gamma_row(row,:phase)
    @test_throws ArgumentError PointExpectation(;station="OTHER",cohort="a",n=3,phase_period_minutes=31)
    @test_throws ArgumentError PointMetrics(;station="CMO",candidate_id="ridge-0.01",family=:ridge,
        cohort="literal-ground-CMO",n=3,selection_log_mse=Inf,phase_period_minutes=31)
    @test_throws ArgumentError PointMetrics(;station="CMO",candidate_id="ridge-0.01",family=:ridge,
        cohort="literal-ground-CMO",n=3,selection_log_mse=-1.,phase_period_minutes=31)
end

end # module CompletedSelectionPrimitivesTests
