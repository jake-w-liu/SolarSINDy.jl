module CMEGSMLabelsTests

using Test, Dates, TOML
include(joinpath(@__DIR__, "gsm_minute_labels.jl"))
const G = CMEGSMMinuteLabels
const SMALL_ATOL = 16eps(Float64) # A few scaled three-point sums, not a fitted tolerance.
const TAG = repeat("a",64)
const SOURCE_PARAMETERS = (source_group="generated-fixture",metadata_sha256=TAG)

function schedule(times, first, stop; id="generated-chunk")
    axis = G.time_phase_runs(times,first,stop;raw_id=id,raw_sha256=TAG,SOURCE_PARAMETERS...)
    G.freeze_phase_domains(axis)
end
function chunk(s, bz; counts=fill(1,length(bz)))
    G.minute_qc(s,s.axis.times,[(1.0,2.0,value) for value in bz],s.axis.times,counts)
end

struct ReadSpy <: AbstractVector{Any}
    values::Vector{Any}
    reads::Base.RefValue{Int}
end
Base.size(x::ReadSpy) = size(x.values)
Base.getindex(x::ReadSpy,i::Int) = (x.reads[] += 1; x.values[i])

@testset "Literal centred-support and Time-only regime oracles" begin
    # These six literal results are frozen in protocol df44b33c before the core.
    for (first,stop,phase,expected) in (
        (0,180000,0,[60000,120000]),
        (0,180000,30000,[30000,90000,150000]),
        (1,180000,30000,[90000,150000]),
        (0,59000,30000,Int[]),
        (30000,150000,0,[60000,120000]),
        (30000,150000,30000,[90000]))
        s = schedule([phase,phase+60000,phase+120000],-60000,240000)
        actual = G.expected_centres(first,stop,[s])
        @test actual.resolved && actual.centres == expected
    end
    negative = schedule([-150000,-90000,-30000],-180000,0)
    @test G.expected_centres(-180000,0,[negative]).centres == [-150000,-90000,-30000]
    @test_throws ArgumentError G.time_phase_runs([30000,30000],0,180000;raw_id="duplicate",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([90000,30000],0,180000;raw_id="reverse",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([30000.0],0,180000;raw_id="float",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([true],0,180000;raw_id="bool",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([180000],0,180000;raw_id="stop",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([30000],0,180000;raw_id="hash",raw_sha256="bad",SOURCE_PARAMETERS...)
    @test_throws ArgumentError G.time_phase_runs([30000],0,180000;raw_id="frame",raw_sha256=TAG,SOURCE_PARAMETERS...,frame="RTN")
    @test_throws ArgumentError G.time_phase_runs([30000],0,180000;raw_id="units",raw_sha256=TAG,SOURCE_PARAMETERS...,units="T")
    @test_throws ArgumentError G.time_phase_runs([30000],0,180000;raw_id="quality",raw_sha256=TAG,SOURCE_PARAMETERS...,quality_variable="OTHER")
    @test G.utc_millis("1970-01-01T00:00:00.001Z") == 1
    @test G.utc_millis("1969-12-31T23:59:59.999+00:00") == -1
    @test_throws ArgumentError G.utc_millis("1970-01-01T00:00:00+01:00")
    @test_throws ArgumentError G.utc_millis("1970-01-01T00:00:00.0001Z")
    huge = schedule([30000,90000],0,6000180000;id="bounded-allocation")
    @test_throws ArgumentError G.expected_centres(0,6000180000,[huge])
    @test_throws ArgumentError G.time_phase_runs(0:60000:6000000000,0,6000060000;
        raw_id="record-bound",raw_sha256=TAG,SOURCE_PARAMETERS...)

    axis = G.time_phase_runs([30000,90000,120000,180000],0,240000;raw_id="phase-change",raw_sha256=TAG,SOURCE_PARAMETERS...)
    @test [run.phase_ms for run in axis.runs] == [30000,0]
    @test axis.changes == [(old_phase_ms=30000,new_phase_ms=0,last_old_ms=90000,first_new_ms=120000)]
    unresolved = G.freeze_phase_domains(axis)
    @test !unresolved.resolved && unresolved.reason == :phase_change_boundary_unresolved
    blocked = G.expected_centres(0,240000,[unresolved])
    @test !blocked.resolved && :phase_change_boundary_unresolved in blocked.reasons
    # The boundary is a pre-value input, not inferred from vector outcomes.
    explicit = G.freeze_phase_domains(axis;domains=[G.PhaseDomain(0,120000,30000),G.PhaseDomain(120000,240000,0)])
    @test explicit.resolved
    @test G.expected_centres(0,240000,[explicit]).centres == [30000,90000,120000,180000]
    @test_throws ArgumentError G.freeze_phase_domains(axis;domains=[G.PhaseDomain(0,120000,0),G.PhaseDomain(120000,240000,30000)])
    @test_throws ArgumentError G.freeze_phase_domains(axis;domains=[G.PhaseDomain(0,100000,30000),G.PhaseDomain(110000,240000,0)])
    reads = Ref(0); spy = ReadSpy(Any[(1,2,-7) for _ in 1:4],reads)
    @test_throws ArgumentError G.minute_qc(unresolved,axis.times,spy,axis.times,ones(Int,4))
    @test reads[] == 0
    wrong_epoch = copy(axis.times); wrong_epoch[2] += 1
    @test_throws ArgumentError G.minute_qc(explicit,wrong_epoch,spy,axis.times,ones(Int,4))
    @test reads[] == 0
    @test_throws ArgumentError G.minute_qc(explicit,axis.times,spy,wrong_epoch,ones(Int,4))
    @test reads[] == 0
    short_spy = ReadSpy(spy.values[1:3],reads)
    @test_throws ArgumentError G.minute_qc(explicit,axis.times,short_spy,axis.times,ones(Int,4))
    @test reads[] == 0
    before = reads[]; axis.times[1] += 1
    @test_throws ArgumentError G.minute_qc(explicit,axis.times,spy,axis.times,ones(Int,4))
    @test reads[] == before
end

@testset "Same-Epoch vector/count QC and literal labels" begin
    s = schedule([30000,90000,150000],0,180000)
    event = G.mo_event("generated",0,180000,1,2)
    north = G.label_event(event,[s],[chunk(s,[5,2,1])])
    @test north.expected_count == 3 && north.valid_count == 3
    @test north.peak_bs_nt === 0.0 && north.admitted == (true,true,true)
    south = G.label_event(event,[s],[chunk(s,[-2,0,-7])])
    @test south.peak_bs_nt == 7 && south.extrema_supports == [(centre_ms=150000,support_start_ms=120000,support_stop_ms=180000)]
    @test all(row.valid for row in south.slots)
    @test south.source_chunks[1].raw_id == "generated-chunk"
    @test south.source_chunks[1].raw_sha256 == TAG
    @test G.label_event(event,[s],[chunk(s,[-7,-7,0])]).extrema_supports == [
        (centre_ms=30000,support_start_ms=0,support_stop_ms=60000),
        (centre_ms=90000,support_start_ms=60000,support_stop_ms=120000)]
    for component in 1:3
        for (value,reason) in ((-1e31,:fill),(-1e30,:pad),(NaN,:nonfinite),(Inf,:nonfinite),(-Inf,:nonfinite),
                (65535,:range),(-65535,:range),(true,:not_real),("bad",:not_real))
            bad = Any[1.0,2.0,-7.0]; bad[component] = value
            c = G.minute_qc(s,s.axis.times,Any[bad,(1,2,-2),(1,2,-3)],s.axis.times,[1,1,1])
            @test c.bz[1] === nothing
            @test Symbol("component$(component)_$(reason)") in c.reasons[1]
            label = G.label_event(event,[s],[c])
            @test label.valid_count == 2 && label.admitted == (false,false,false)
        end
    end
    for vector in ((),(1,2),(1,2,3,4),"not-a-vector")
        c = G.minute_qc(s,s.axis.times,Any[vector,(1,2,-2),(1,2,-3)],s.axis.times,[1,1,1])
        @test c.reasons[1] == [:vector_dimension]
    end
    # Independent native REAL4 bits from the metadata/IEEE oracle e0461651.
    for (bits,decimal,reason) in ((UInt32(0xf2fc6f7c),-1e31,:fill),(UInt32(0xf149f2ca),-1e30,:pad))
        native = reinterpret(Float32,bits); widened = Float64(native)
        @test widened != decimal
        for component in 1:3
            for value in (native,widened,decimal)
                vector = Any[1.0,2.0,-7.0]; vector[component] = value
                c = G.minute_qc(s,s.axis.times,Any[vector,(1,2,-2),(1,2,-3)],s.axis.times,[1,1,1])
                @test c.reasons[1] == [Symbol("component$(component)_$(reason)")]
                @test c.bz[1] === nothing
            end
            for value in (prevfloat(native),nextfloat(native),Float64(prevfloat(native)),Float64(nextfloat(native)),prevfloat(widened),nextfloat(widened))
                vector = Any[1.0,2.0,-7.0]; vector[component] = value
                c = G.minute_qc(s,s.axis.times,Any[vector,(1,2,-2),(1,2,-3)],s.axis.times,[1,1,1])
                @test c.reasons[1] == [Symbol("component$(component)_range")]
            end
        end
    end
    for (count,reason) in ((0,:point_count_zero),(-1,:point_count_fill),(-2,:point_count_range),
            (1501,:point_count_range),(1.0,:point_count_not_integer),(true,:point_count_not_integer))
        c = chunk(s,[-7,-2,-3];counts=Any[count,1,1])
        @test reason in c.reasons[1] && c.bz[1] === nothing
    end
    @test all(isempty,chunk(s,[-65534,65534,-1];counts=[1,1500,1]).reasons)
    sparse = schedule([30000,150000],0,180000;id="missing-centre")
    label = G.label_event(event,[sparse],[chunk(sparse,[-2,-7])])
    @test label.expected_count == 3 && label.valid_count == 2
    @test label.slots[2].centre_ms == 90000 && label.slots[2].reasons == [:missing]
    @test label.slots[2].raw_id == "missing-centre"
    @test label.admitted == (false,false,false)
    @test !G.expected_centres(0,180000,typeof(s)[]).resolved
    @test_throws ArgumentError G.expected_centres(0,180000,[s,s])
    changed = chunk(s,[-2,0,-7]); changed.bz[1] = -65534
    @test_throws ArgumentError G.label_event(event,[s],[changed])
    other_axis = G.time_phase_runs(s.axis.times,0,180000;raw_id=s.axis.raw_id,raw_sha256=TAG,
        source_group="generated-fixture",metadata_sha256=repeat("b",64))
    other_schedule = G.freeze_phase_domains(other_axis)
    @test_throws ArgumentError G.label_event(event,[s],[chunk(other_schedule,[-2,0,-7])])
    # Only counts that meet the exact fixed ratios pass; no float threshold.
    for (ratio,valid,expected) in (((3,4),3,4),((9,10),9,10),((19,20),19,20))
        @test G.coverage_admitted(valid,expected,ratio)
        @test !G.coverage_admitted(valid-1,expected,ratio)
    end
    @test !G.coverage_admitted(1,1,(9,10))
    @test !G.coverage_admitted(0,0,(9,10))
    @test_throws ArgumentError G.coverage_admitted(3,2,(9,10))
    @test_throws ArgumentError G.coverage_admitted(true,2,(9,10))
    @test_throws ArgumentError G.coverage_admitted(2,2,(4,5))
    mktempdir() do temporary
        root = realpath(temporary)
        population = joinpath(root,"population.json"); provider = joinpath(root,"provider.csv")
        write(population,"{}"); write(provider,"not,the,frozen,provider\n")
        @test_throws ArgumentError G.load_frozen_events(population,provider)
    end
end

@testset "Independent paired, rank, calendar-weight and endpoint arithmetic" begin
    @test G.weighted_mid_ranks([4,1,4,2],ones(Int,4)) == [3.5,1.0,3.5,2.0]
    @test G.weighted_pearson([1,2,3],[1,2,3],[1,1,1]).value ≈ 1 atol=SMALL_ATOL rtol=0
    @test G.weighted_pearson([1,2,3],[1,3,2],[1,1,1]).value ≈ 1/2 atol=SMALL_ATOL rtol=0
    raw = G.weighted_pearson([1,2,3],[1,3,2],[2,1,1])
    @test raw.value ≈ 7/11 atol=SMALL_ATOL rtol=0
    weighted = G.paired_correlations([1,2,3],[3,2,1],[1,3,2],[2,1,1])
    @test weighted.spearman_field.value ≈ 7/9 atol=SMALL_ATOL rtol=0
    @test G.weighted_mid_ranks([1,2,3],[2,1,1]) == [1.5,3,4]
    @test G.weighted_mid_ranks([1,3,2],[2,1,1]) == [1.5,4,3]
    opposite = G.paired_correlations([1,2,3],[3,2,1],[1,2,3])
    @test opposite.pearson_contrast ≈ 0 atol=SMALL_ATOL rtol=0
    @test opposite.spearman_contrast ≈ 0 atol=SMALL_ATOL rtol=0
    @test !G.weighted_pearson([4,4,4],[1,2,3],[1,1,1]).valid
    @test G.weighted_pearson([1,2],[1,2],[0,0]).reason == :insufficient_support
    @test_throws ArgumentError G.weighted_pearson([1,NaN],[1,2],[1,1])
    @test_throws ArgumentError G.weighted_pearson([1,2],[1,2],[1,-1])
    @test_throws ArgumentError G.weighted_pearson([1,2],[1,2],[true,1])
    # Empty matched blocks stay in the universe and consume their selected slots.
    @test G.year_weights([1995,1995,1997],[1995,1996,1997],[2,2,1]) == [1,1,0]
    @test G.year_weights([1995,1995,1997],[1995,1996,1997],[3,3,1]) == [1,1,2]
    @test_throws ArgumentError G.year_weights([1995],[1995,1996],[1])
    @test_throws ArgumentError G.year_weights([1994],[1995,1996],[1,2])
    @test_throws ArgumentError G.year_weights([1995],[1995,1996],[1,3])
    @test G.percentile_type7([0,10,20,30,40],1//40) == 1
    @test G.percentile_type7([0,10,20,30,40],39//40) == 39
    @test G.percentile_type7([7],1//40) == 7
    @test_throws ArgumentError G.percentile_type7(Float64[],1//40)
    @test_throws ArgumentError G.percentile_type7([0,Inf],1//40)
    for invalid in (100,101)
        draws = [(draw=i,valid=i>invalid,pearson_contrast=i>invalid ? 0.5 : nothing,
            spearman_contrast=i>invalid ? 0.5 : nothing,reasons=i>invalid ? Symbol[] : [:zero_variance]) for i in 1:10000]
        result = G.contrast_summary(draws)
        @test result.invalid_count == invalid && result.valid_count == 10000-invalid
        @test result.uncertainty_gate == (invalid==100)
        @test result.pearson_interval == (0.5,0.5) && result.spearman_interval == (0.5,0.5)
        @test_throws ArgumentError G.contrast_summary(draws[1:9999])
    end
end

"A genuine decoded-array numerical route on generated inputs, never a CDF/source admission."
function generated_fixture(; invalid_after=551, unmatched_first=0, unresolved_first=false)
    metadata = TOML.parsefile(G.Bootstrap.DEFAULT_INPUT)
    events = G.MOEvent[]; schedules = G.PhaseSchedule[]; chunks = G.QCChunk[]
    year_serial = Dict{Int,Int}()
    for (index,row) in enumerate(metadata["event"])
        y = row["utc_year"]; serial = get(year_serial,y,0)+1; year_serial[y] = serial
        # Generated, disjoint three-minute windows retain the exact frozen IDs,
        # order and all31 start years; they are not the actual catalogue windows.
        first = Dates.value(DateTime(y,1,1)-DateTime(1970,1,1)) + serial*600000
        stop = first+180000; times = [first+30000,first+90000,first+150000]
        field = index <= unmatched_first ? nothing : Float64(index)
        speed = Float64(mod(index-1+275,551)+1)
        push!(events,G.mo_event(row["id"],first,stop,field,speed))
        if index == 1 && unresolved_first
            axis = G.time_phase_runs([first+30000,first+90000,first+120000],first,stop;
                raw_id="generated-$index",raw_sha256=TAG,SOURCE_PARAMETERS...)
            push!(schedules,G.freeze_phase_domains(axis))
        else
            s = schedule(times,first,stop;id="generated-$index"); push!(schedules,s)
            push!(chunks,chunk(s,fill(-Float64(index),3);counts=fill(index <= invalid_after ? 1 : 0,3)))
        end
    end
    population = G.generated_population(events)
    labels = G.label_population(population,schedules,chunks)
    (; population,schedules,chunks,labels=labels.labels,result=labels)
end

function run_generated_population_fixture()
    @testset "All551 generated labels to all10000 retained paired contrasts" begin
        fixture = generated_fixture()
        @test length(fixture.labels) == 551
        @test [label.id for label in fixture.labels] == collect(G.Bootstrap.load_population().ids)
        @test [label.peak_bs_nt for label in fixture.labels] == Float64.(1:551)
        @test all(label -> label.expected_count==3 && label.valid_count==3 && label.admitted==(true,true,true),fixture.labels)
        @test fixture.result.generated_inputs_only && !fixture.result.source_authority_granted && !fixture.result.scientific_qualification
        # Literal synthetic indices: each of the31 declared years occurs once
        # in every row. This is not the native RNG artifact or a control pass.
        indices = repeat(reshape(collect(1:31),1,31),10000,1)
        report = G.comparison_report(fixture.population,fixture.labels,indices;index_sha256=repeat("f",64))
        @test length(report.summaries) == length(report.uncertainties) == 3
        for summary in report.summaries
            @test summary.admitted_labels == length(summary.indices) == 551
            @test summary.full_population_gate && summary.matched_population_gate
            # For the cyclic permutation by275 of1:551, exact Pearson and
            # Spearman are -1/2; the field is the label, so both contrasts1/2.
            @test summary.statistics.pearson_field.value ≈ 1 atol=2e-13 rtol=0
            @test summary.statistics.pearson_speed.value ≈ -1/2 atol=2e-13 rtol=0
            @test summary.statistics.spearman_field.value ≈ 1 atol=2e-13 rtol=0
            @test summary.statistics.spearman_speed.value ≈ -1/2 atol=2e-13 rtol=0
        end
        for uncertainty in report.uncertainties
            @test length(uncertainty.draws)==10000 && uncertainty.valid_count==10000 && uncertainty.invalid_count==0
            @test all(draw -> draw.valid,uncertainty.draws)
            @test all(value -> isapprox(value,0.5;atol=2e-13,rtol=0),uncertainty.pearson_interval)
            @test all(value -> isapprox(value,0.5;atol=2e-13,rtol=0),uncertainty.spearman_interval)
        end
        @test report.numerical_support_gate && report.generated_inputs_only
        @test !report.source_authority_granted && !report.scientific_qualification
        @test_throws ArgumentError G.bootstrap_contrasts(fixture.population,fixture.labels,indices[1:9999,:];index_sha256=repeat("f",64))
        bad = copy(indices); bad[1,1]=0
        @test_throws ArgumentError G.bootstrap_contrasts(fixture.population,fixture.labels,bad;index_sha256=repeat("f",64))
        for (limit,expected_gate) in ((496,true),(495,false))
            partial = generated_fixture(;invalid_after=limit)
            summary = G.paired_statistics(partial.population,partial.labels)
            @test length(partial.labels)==551 && summary.admitted_labels==limit
            @test summary.full_population_gate==expected_gate
        end
        for (unmatched,expected_gate) in ((59,true),(60,false))
            partial = generated_fixture(;unmatched_first=unmatched)
            summary = G.paired_statistics(partial.population,partial.labels)
            @test length(partial.labels)==551 && summary.admitted_labels==551
            @test length(summary.indices)==551-unmatched && summary.matched_population_gate==expected_gate
        end
        unresolved = generated_fixture(;unresolved_first=true)
        @test length(unresolved.labels)==551 && unresolved.labels[1].expected_count===nothing
        @test :phase_change_boundary_unresolved in unresolved.labels[1].reasons
        @test unresolved.labels[1].admitted==(false,false,false)
        duplicate = collect(fixture.population.events); duplicate[2]=duplicate[1]
        @test_throws ArgumentError G.generated_population(duplicate)
        @test_throws ArgumentError G.generated_population(collect(fixture.population.events[1:550]))
        @test_throws ArgumentError G.paired_statistics(fixture.population,reverse(fixture.labels))
    end
    println("CME_GSM_GENERATED551_NUMERICAL_FIXTURE_ONLY;ACTUAL_CDF=0;SOURCE_AUTHORITY=false;SCIENCE=false")
    nothing
end

end
