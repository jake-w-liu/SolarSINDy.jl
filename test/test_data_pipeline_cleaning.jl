using CSV
using DataFrames

@testset "Data Pipeline And Cleaning" begin

    @testset "Original-observation provenance masks" begin
        nrow_test = 15
        df = DataFrame(
            V=fill(400.0, nrow_test), Bz=fill(-5.0, nrow_test),
            By=fill(1.0, nrow_test), n=fill(5.0, nrow_test),
            Pdyn=fill(2.0, nrow_test),
            T=fill(1.0e5, nrow_test), Dst=collect(-1.0:-1.0:-15.0),
            AE=fill(100.0, nrow_test), AL=fill(-100.0, nrow_test),
            AU=fill(100.0, nrow_test),
        )
        df.Dst[8] = NaN
        add_original_observation_flags!(df)
        @test !df.Dst_observed[8]
        # A later interpolation must not rewrite provenance.
        df.Dst[8] = -8.0
        @test !df.Dst_observed[8]
        @test_throws ArgumentError add_original_observation_flags!(df)

        mask = original_sindy_mask(df, 1:nrow_test; smooth_window=5)
        # A five-point centered smoother followed by a centered derivative has
        # an interior dependency radius of three samples.
        @test all(!mask[i] for i in 5:11)
        @test all(mask[i] for i in vcat(1:4, 12:15))
        df.AE_observed[3] = false
        mask_ae = original_sindy_mask(df, 1:nrow_test;
                                      smooth_window=5, require_ae=true)
        @test !mask_ae[1] && !mask_ae[6]
        @test_throws ArgumentError original_sindy_mask(
            select(df, DataFrames.Not(:V_observed)), 1:nrow_test)
        @test_throws ArgumentError original_sindy_mask(df, 1:nrow_test;
                                                       smooth_window=4)
        @test_throws ArgumentError original_sindy_mask(df, [1, 3, 4])

        parsed = select(df, collect(SolarSINDy.OMNI_OBSERVATION_COLUMNS))
        parsed.Dst[8] = NaN
        SolarSINDy._clean_parsed_omni!(parsed)
        @test :Dst_observed in propertynames(parsed)
        @test !parsed.Dst_observed[8]
        @test isfinite(parsed.Dst[8])
        @test !original_sindy_mask(parsed, 1:nrow_test; smooth_window=5)[8]

        incomplete = select(df, DataFrames.Not(:AU, r"_observed$"))
        @test_throws ArgumentError add_original_observation_flags!(incomplete)
        @test all(!endswith(name, "_observed") for name in names(incomplete))
    end

    @testset "A: parse_omni2_csv converts fill values and datetimes" begin
        mktempdir() do tmp
            csv_path = joinpath(tmp, "omni_extract.csv")
            open(csv_path, "w") do io
                println(io, "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU")
                println(io, "1964,60,12,1.5,-2.5,500000,4.0,450,3.50,-40,120,-80,90")
                println(io, "1964,61,0,999.9,999.9,9999999,999.9,9999,99.99,-99999,9999,-99999,99999")
                println(io, "2026,1,0,0.0,-1.0,100000,5.0,500,2.00,-20,80,-50,60")
            end

            df = parse_omni2(csv_path; year_start=1964, year_end=1964)

            @test nrow(df) == 2
            # 1964 is a leap year, so day-of-year 60 is Feb 29.
            @test df.datetime[1] == DateTime(1964, 2, 29, 12)
            @test df.datetime[2] == DateTime(1964, 3, 1, 0)
            @test df.V[1] == 450.0
            @test isnan(df.By[2])
            @test isnan(df.Bz[2])
            @test isnan(df.T[2])
            @test isnan(df.n[2])
            @test isnan(df.Pdyn[2])
            @test isnan(df.Dst[2])
            @test isnan(df.AE[2])
            @test isnan(df.AL[2])
            @test isnan(df.AU[2])
        end
    end

    @testset "A: extract_omni2_columns preserves requested column ordering" begin
        mktempdir() do tmp
            raw_path = joinpath(tmp, "omni_raw.dat")
            extracted_path = joinpath(tmp, "omni_extract.csv")

            open(raw_path, "w") do io
                first_row = collect(1:54)
                first_row[1:3] = [1964, 60, 12]
                second_row = collect(101:154)
                second_row[1:3] = [1964, 60, 13]
                println(io, join(first_row, ' '))
                println(io, join(second_row, ' '))
            end

            extract_omni2_columns(raw_path, extracted_path)
            out = CSV.read(extracted_path, DataFrame)

            @test names(out) == ["year", "doy", "hour", "By", "Bz", "T", "n", "V", "Pdyn", "Dst", "AE", "AL", "AU"]
            @test Tuple(out[1, :]) == (1964, 60, 12, 16, 17, 23, 24, 25, 29, 41, 42, 53, 54)
            @test Tuple(out[2, :]) == (1964, 60, 13, 116, 117, 123, 124, 125, 129, 141, 142, 153, 154)
        end
    end

    @testset "A: load_omni2_csv parses saved datetime strings" begin
        mktempdir() do tmp
            path = joinpath(tmp, "cleaned.csv")
            src = DataFrame(
                datetime = ["2025-01-01T00:00:00", "2025-01-01T01:00:00"],
                V = [400.0, 420.0],
                Bz = [-5.0, -4.0],
                By = [1.0, 2.0],
                n = [5.0, 6.0],
                Pdyn = [2.0, 2.5],
                T = [1.0e5, 1.1e5],
                Dst = [-20.0, -25.0],
                AE = [100.0, 120.0],
                AL = [-50.0, -70.0],
                AU = [80.0, 90.0],
            )
            CSV.write(path, src)

            df = load_omni2_csv(path)

            @test df.datetime[1] == DateTime(2025, 1, 1, 0)
            @test df.datetime[2] == DateTime(2025, 1, 1, 1)
        end
    end

    @testset "A/D: clean_omni_data! computes derived fields and preserves long gaps" begin
        df = DataFrame(
            datetime = [DateTime(2020, 1, 1) + Hour(i - 1) for i in 1:8],
            V = [400.0, NaN, NaN, 700.0, 900.0, NaN, NaN, NaN],
            Bz = [-5.0, -5.0, -5.0, -5.0, NaN, NaN, NaN, NaN],
            By = [1.0, NaN, NaN, 4.0, 2.0, 2.0, 2.0, 2.0],
            n = [5.0, NaN, NaN, 8.0, 10.0, 10.0, 10.0, 10.0],
            Pdyn = [NaN, NaN, NaN, NaN, 8.0, 8.0, 8.0, 8.0],
            T = [1.0e5, NaN, NaN, 1.3e5, 1.4e5, 1.4e5, 1.4e5, 1.4e5],
            Dst = [-40.0, -45.0, -50.0, -55.0, -60.0, -65.0, -70.0, -75.0],
            AE = [100.0, NaN, NaN, 160.0, 180.0, 180.0, 180.0, 180.0],
            AL = [-50.0, NaN, NaN, -80.0, -90.0, -90.0, -90.0, -90.0],
            AU = [80.0, NaN, NaN, 120.0, 130.0, 130.0, 130.0, 130.0],
        )

        clean_omni_data!(df)

        @test df.V[2] ≈ 500.0 atol=1e-12  # Linear interpolation between 400 and 700.
        @test df.V[3] ≈ 600.0 atol=1e-12
        @test df.Pdyn[1] ≈ 1.6726e-6 * 5.0 * 400.0^2 atol=1e-12
        @test df.Bs[1] == 5.0
        @test df.theta_c[1] ≈ atan(abs(1.0), -5.0) atol=1e-12
        @test df.BT[4] ≈ sqrt(4.0^2 + (-5.0)^2) atol=1e-12
        @test df.Dst_star[4] ≈ df.Dst[4] - 7.26 * sqrt(df.Pdyn[4]) + 11.0 atol=1e-12
        @test isnan(df.Bz[6])  # Gap of length 4 is longer than interpolation limit.
        @test df.quality[6] == 0

        wide = DataFrame(
            datetime=[DateTime(2020, 1, 1)], V=[400.0], Bz=[1.0e308],
            By=[1.0e308], n=[5.0], Pdyn=[2.0], T=[1.0e5], Dst=[-20.0],
            AE=[100.0], AL=[-50.0], AU=[80.0],
        )
        clean_omni_data!(wide)
        @test wide.BT[1] == hypot(wide.By[1], wide.Bz[1])
        @test isfinite(wide.BT[1])
        @test wide.quality[1] == 1

        out_of_range = copy(wide)
        out_of_range.By[1] = floatmax(Float64)
        out_of_range.Bz[1] = floatmax(Float64)
        clean_omni_data!(out_of_range)
        @test isnan(out_of_range.BT[1])
        @test out_of_range.quality[1] == 0
    end

    @testset "NF-CAUSAL-01: causal cleaning forward-fills with the pre-gap value only, never fills Dst" begin
        # Short interior gaps in every measured column. Causal mode carries the
        # last PRE-gap value forward (last-observation-carried-forward); it must
        # never inject a post-gap or centered-interpolated value, so issue-time
        # replay inputs stay strictly causal. Dst is excluded from causal filling:
        # a missing anchor/target is left NaN, never persisted or interpolated.
        df = DataFrame(
            datetime = [DateTime(2022, 1, 1) + Hour(i - 1) for i in 1:4],
            V    = [400.0, NaN, NaN, 700.0],
            Bz   = [-5.0, NaN, -10.0, -10.0],
            By   = [1.0, NaN, NaN, 4.0],
            n    = [5.0, NaN, NaN, 8.0],
            Pdyn = [2.0, NaN, NaN, 3.0],
            T    = [1.0e5, NaN, NaN, 1.3e5],
            Dst  = [-40.0, NaN, NaN, -55.0],
            AE   = [100.0, NaN, NaN, 160.0],
            AL   = [-50.0, NaN, NaN, -80.0],
            AU   = [80.0, NaN, NaN, 120.0],
        )

        clean_omni_data!(df; causal=true)

        # Every gap hour takes the pre-gap value (forward-fill), never the future bound.
        @test df.V[2] == 400.0
        @test df.V[3] == 400.0
        @test df.By[2] == 1.0
        @test df.By[3] == 1.0
        @test df.Bz[2] == -5.0
        # Mutation sensitivity: centered interpolation (the non-causal fallback)
        # would use the post-gap bound and yield 500/600 (V) and -7.5 (Bz). If
        # causal mode ever fell back to centered interpolation these would fire.
        @test df.V[2] != 500.0        # centered value of the 400→700 gap at hour 2
        @test df.V[3] != 600.0        # centered value of the 400→700 gap at hour 3
        @test df.Bz[2] != -7.5        # centered value of the -5→-10 gap at hour 2
        # Dst is never causally filled: the short gap stays NaN (not persisted,
        # not interpolated), so the pressure-corrected Dst* is NaN on those rows.
        @test isnan(df.Dst[2])
        @test isnan(df.Dst[3])
        @test isnan(df.Dst_star[2])
        @test isnan(df.Dst_star[3])
        # Rows with an observed Dst still yield a finite pressure-corrected Dst*.
        @test isfinite(df.Dst_star[1])
        @test isfinite(df.Dst_star[4])
    end

    @testset "A/B: build_storm_catalog finds storm window and assigns split" begin
        @test SolarSINDy._solar_cycle(Date(1964, 9, 30)) == 19
        @test SolarSINDy._solar_cycle(Date(1964, 10, 1)) == 20
        @test SolarSINDy._solar_cycle(Date(1976, 2, 29)) == 20
        @test SolarSINDy._solar_cycle(Date(1976, 3, 1)) == 21
        @test SolarSINDy._solar_cycle(Date(1986, 9, 1)) == 22
        @test SolarSINDy._solar_cycle(Date(1996, 8, 1)) == 23
        @test SolarSINDy._solar_cycle(Date(2008, 12, 1)) == 24
        @test SolarSINDy._solar_cycle(Date(2019, 12, 1)) == 25
        @test SolarSINDy._assign_split(19) == "exclude"
        n = 40
        df = DataFrame(
            datetime = [DateTime(2020, 1, 1) + Hour(i - 1) for i in 1:n],
            V = fill(500.0, n),
            Bz = fill(-5.0, n),
            By = fill(0.0, n),
            n = fill(6.0, n),
            Pdyn = fill(2.5, n),
            T = fill(1.0e5, n),
            Dst = fill(0.0, n),
            AE = fill(100.0, n),
            AL = fill(-50.0, n),
            AU = fill(80.0, n),
            quality = ones(Int, n),
            Dst_star = vcat(fill(0.0, 10), [-10.0, -30.0, -55.0, -80.0, -60.0, -30.0, -10.0], fill(0.0, n - 17)),
        )

        catalog = build_storm_catalog(df; dst_thresh=-50.0, window_pre=2, window_post=10, min_separation=5)

        @test length(catalog) == 1
        entry = catalog[1]
        @test entry.min_dst_star == -80.0
        @test entry.min_dst_star_time == df.datetime[14]
        # Legacy property aliases remain readable for package compatibility.
        @test entry.min_dst == -80.0
        @test entry.min_dst_time == df.datetime[14]
        @test entry.solar_cycle == 25
        @test entry.split == "test"
        @test entry.onset_idx == 10
        @test entry.end_idx == 40

        @test_throws ArgumentError build_storm_catalog(df; dst_thresh=NaN)
        @test_throws ArgumentError build_storm_catalog(df; window_pre=-1)
        @test_throws ArgumentError build_storm_catalog(df; window_post=-1)
        @test_throws ArgumentError build_storm_catalog(df; min_separation=-1)
        @test_throws ArgumentError build_storm_catalog(select(df, Not(:quality)))
        gapped = copy(df)
        gapped.datetime[20] += Hour(1)
        @test_throws ArgumentError build_storm_catalog(gapped)
        infinite_dst = copy(df)
        infinite_dst.Dst_star[5] = Inf
        @test_throws ArgumentError build_storm_catalog(infinite_dst)
    end

    @testset "D/G: extract_storm_data, extract_all_storms, and catalog roundtrip" begin
        df = DataFrame(
            datetime = [DateTime(2015, 1, 1) + Hour(i - 1) for i in 1:4],
            V = [NaN, 400.0, NaN, 500.0],
            Bz = [NaN, -5.0, -6.0, NaN],
            By = [1.0, NaN, 2.0, 3.0],
            n = [5.0, NaN, 6.0, 7.0],
            Pdyn = [2.0, NaN, 3.0, NaN],
            Dst = [NaN, -30.0, -40.0, -45.0],
            Dst_star = [NaN, -32.0, -42.0, -47.0],
        )
        entry = StormCatalogEntry(
            7,
            df.datetime[2],
            -47.0,
            df.datetime[4],
            df.datetime[4],
            3.0,
            24,
            "val",
            1,
            4,
        )

        swd = extract_storm_data(df, entry)
        @test swd.V == [400.0, 400.0, 400.0, 500.0]
        @test swd.Bz == [-5.0, -5.0, -6.0, -6.0]
        @test swd.Dst_star == [-32.0, -32.0, -42.0, -47.0]
        @test swd.t == [0.0, 1.0, 2.0, 3.0]

        datasets, entries = extract_all_storms(df, [entry]; split="val")
        @test length(datasets) == 1
        @test length(entries) == 1
        @test datasets[1].Dst_star[end] == -47.0

        mktempdir() do tmp
            path = joinpath(tmp, "storm_catalog.csv")
            save_storm_catalog([entry], path)
            saved = CSV.read(path, DataFrame)
            @test :min_dst_star in propertynames(saved)
            @test :min_dst_star_time in propertynames(saved)
            @test !(:min_dst in propertynames(saved))
            loaded = load_storm_catalog(path)
            @test loaded == [entry]

            legacy_path = joinpath(tmp, "legacy_storm_catalog.csv")
            legacy = copy(saved)
            rename!(legacy, :min_dst_star => :min_dst,
                            :min_dst_star_time => :min_dst_time)
            CSV.write(legacy_path, legacy)
            @test load_storm_catalog(legacy_path) == [entry]

            duplicate_path = joinpath(tmp, "duplicate_storm_catalog.csv")
            CSV.write(duplicate_path, vcat(saved, saved))
            @test_throws ArgumentError load_storm_catalog(duplicate_path)
        end
    end

    @testset "NF-DATA-01: Pdyn is proton-only from filled n·V² (OMNI word-29 not kept)" begin
        # n, V, and Pdyn share a 2-hour interior gap. After cleaning, n and V are
        # interpolated and Pdyn is recomputed proton-only from them (the n·V²
        # identity holds on every row); the native OMNI word-29 pressure is dropped
        # so training and serving share the proton-only convention.
        df = DataFrame(
            datetime = [DateTime(2021, 1, 1) + Hour(i - 1) for i in 1:5],
            V   = [400.0, NaN, NaN, 700.0, 800.0],
            Bz  = fill(-5.0, 5),
            By  = fill(1.0, 5),
            n   = [5.0, NaN, NaN, 11.0, 12.0],
            Pdyn = [NaN, NaN, NaN, NaN, 13.0],   # native (alpha-inclusive) value is NOT kept
            T   = fill(1.0e5, 5),
            Dst = [-30.0, -32.0, -34.0, -36.0, -38.0],
            AE  = fill(100.0, 5), AL = fill(-50.0, 5), AU = fill(80.0, 5),
        )
        clean_omni_data!(df)
        for i in 1:5   # every row: proton-only from (interpolated) n, V
            @test df.Pdyn[i] ≈ 1.6726e-6 * df.n[i] * df.V[i]^2 atol=1e-12
        end
        # Row 5's native 13.0 is overwritten with the proton-only value (≈12.85).
        @test df.Pdyn[5] ≈ 1.6726e-6 * 12.0 * 800.0^2 atol=1e-12
        @test df.Pdyn[5] != 13.0
    end

    @testset "NF-DATA-02: Pdyn-missing Dst* fallback carries the last known pressure forward" begin
        # Trailing n/V gap → not interpolated → Pdyn stays NaN → Dst* uses the
        # carried-forward last known pressure (row-1 proton-only), NOT a +11-only
        # (Pdyn=0) baseline. The corrected fallback sits strictly below Dst + 11.
        df = DataFrame(
            datetime = [DateTime(2021, 2, 1) + Hour(i - 1) for i in 1:3],
            V   = [400.0, NaN, NaN],
            Bz  = fill(-5.0, 3),
            By  = fill(1.0, 3),
            n   = [5.0, NaN, NaN],
            Pdyn = [2.0, NaN, NaN],
            T   = fill(1.0e5, 3),
            Dst = [-30.0, -40.0, -50.0],
            AE  = fill(100.0, 3), AL = fill(-50.0, 3), AU = fill(80.0, 3),
        )
        clean_omni_data!(df)
        @test isnan(df.Pdyn[2]); @test isnan(df.Pdyn[3])
        pdyn1 = 1.6726e-6 * 5.0 * 400.0^2               # row-1 proton-only pressure, carried forward
        @test df.Pdyn[1] ≈ pdyn1 atol=1e-12
        @test df.Dst_star[2] ≈ -40.0 - 7.26 * sqrt(pdyn1) + 11.0 atol=1e-9
        @test df.Dst_star[3] ≈ -50.0 - 7.26 * sqrt(pdyn1) + 11.0 atol=1e-9
        @test df.Dst_star[2] < df.Dst[2] + 11.0          # not the physically-impossible Pdyn=0 fallback
    end

    @testset "NF-DATA-02b: quiet-pressure default when no prior Pdyn is available" begin
        # No finite Pdyn anywhere (plasma out from the start) → the fallback reverts
        # to the climatological quiet-time default, not +11 (Pdyn=0).
        df = DataFrame(
            datetime = [DateTime(2021, 3, 1) + Hour(i - 1) for i in 1:2],
            V   = [NaN, NaN],
            Bz  = fill(-5.0, 2),
            By  = fill(1.0, 2),
            n   = [NaN, NaN],
            Pdyn = [NaN, NaN],
            T   = fill(1.0e5, 2),
            Dst = [-30.0, -40.0],
            AE  = fill(100.0, 2), AL = fill(-50.0, 2), AU = fill(80.0, 2),
        )
        clean_omni_data!(df)
        q = SolarSINDy.QUIET_PDYN_NPA
        @test df.Dst_star[1] ≈ -30.0 - 7.26 * sqrt(q) + 11.0 atol=1e-9
        @test df.Dst_star[2] ≈ -40.0 - 7.26 * sqrt(q) + 11.0 atol=1e-9
    end

    @testset "NF-DATA-03: AU/AL fill threshold preserves real values in [9999,99999)" begin
        mktempdir() do tmp
            csv_path = joinpath(tmp, "omni_au.csv")
            open(csv_path, "w") do io
                println(io, "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU")
                # Active hour: real AU=10000, AL=-12000 (magnitude >9999, below the
                # 99999 fill value) must survive — a 9999 threshold would NaN them.
                println(io, "1964,60,12,1.5,-2.5,500000,4.0,450,3.50,-40,120,-12000,10000")
                println(io, "1964,61,0,1.0,-1.0,100000,5.0,500,2.00,-20,80,-99999,99999")
            end
            df = parse_omni2(csv_path; year_start=1964, year_end=1964)
            @test df.AU[1] == 10000.0    # real value preserved (threshold is 99999, not 9999)
            @test df.AL[1] == -12000.0
            @test isnan(df.AU[2])        # 99999 fill → NaN
            @test isnan(df.AL[2])        # -99999 fill → NaN (abscheck)
        end
    end

    @testset "CACHE-1: _last_valid_omni_date skips trailing fill padding" begin
        # NASA pads the current year to Dec 31 with fill rows, so the file's last
        # row always reads as full-year coverage. The cache validator must instead
        # find the last row whose Dst is a real observation.
        mktempdir() do tmp
            # Raw archive form: whitespace-separated, Dst is word 41, fill 99999.
            raw = joinpath(tmp, "raw.dat")
            open(raw, "w") do io
                r1 = collect(1:54); r1[1]=2025; r1[2]=100; r1[3]=5;  r1[41]=-42
                r2 = collect(1:54); r2[1]=2025; r2[2]=100; r2[3]=6;  r2[41]=-55
                f1 = collect(1:54); f1[1]=2025; f1[2]=100; f1[3]=7;  f1[41]=99999
                f2 = collect(1:54); f2[1]=2025; f2[2]=365; f2[3]=23; f2[41]=99999
                for r in (r1, r2, f1, f2); println(io, join(r, ' ')); end
            end
            @test SolarSINDy._last_valid_omni_date(raw; delim=nothing, dst_field=41) ==
                  DateTime(2025, 1, 1) + Day(99) + Hour(6)

            # Extracted CSV form: comma-separated, Dst is column 10, header skipped.
            csv = joinpath(tmp, "ex.csv")
            open(csv, "w") do io
                println(io, "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU")
                println(io, "2025,100,5,1,-2,5e5,4,450,3.5,-42,120,-80,90")
                println(io, "2025,100,6,1,-2,5e5,4,450,3.5,99999,9999,-99999,99999")
            end
            @test SolarSINDy._last_valid_omni_date(csv; delim=',', dst_field=10) ==
                  DateTime(2025, 1, 1) + Day(99) + Hour(5)

            # An all-fill tail yields nothing (treated as unverifiable → stale).
            allfill = joinpath(tmp, "allfill.csv")
            open(allfill, "w") do io
                println(io, "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU")
                println(io, "2025,1,0,1,-2,5e5,4,450,3.5,99999,9999,-99999,99999")
            end
            @test SolarSINDy._last_valid_omni_date(allfill; delim=',', dst_field=10) === nothing
        end
    end

    @testset "CACHE-2: _omni_cache_stale re-fetches only when coverage is short" begin
        now_utc = DateTime(2026, 3, 15, 12)
        old   = DateTime(2025, 11, 1)   # cache fetched long ago
        fresh = DateTime(2026, 3, 14)   # cache fetched yesterday

        # Completed past year fully covered through its final hour → keep cache.
        @test SolarSINDy._omni_cache_stale(DateTime(2025, 12, 31, 23), 2025, now_utc, old) == false
        # Completed past year covered only to October, stale cache → re-fetch.
        @test SolarSINDy._omni_cache_stale(DateTime(2025, 10, 1), 2025, now_utc, old) == true
        # Same short coverage but a freshly fetched file → do not thrash.
        @test SolarSINDy._omni_cache_stale(DateTime(2025, 10, 1), 2025, now_utc, fresh) == false
        # Current year: coverage end unknowable, gate on freshness.
        @test SolarSINDy._omni_cache_stale(DateTime(2026, 3, 4, 19), 2026, now_utc, old) == true
        @test SolarSINDy._omni_cache_stale(DateTime(2026, 3, 4, 19), 2026, now_utc, fresh) == false
        # No readable Dst record → always stale.
        @test SolarSINDy._omni_cache_stale(nothing, 2025, now_utc, fresh) == true
    end

    @testset "CACHE-3: failed extraction preserves the last good cache" begin
        mktempdir() do tmp
            raw = joinpath(tmp, "empty_raw.dat")
            extracted = joinpath(tmp, "omni_extract.csv")
            write(raw, "")
            previous = "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU\n2025,1,0,1,-2,1e5,5,400,2,-20,50,-30,20\n"
            write(extracted, previous)
            @test_throws ErrorException extract_omni2_columns(raw, extracted)
            @test read(extracted, String) == previous

            missing_raw = joinpath(tmp, "missing.dat")
            @test_throws ArgumentError extract_omni2_columns(missing_raw, extracted)
            @test read(extracted, String) == previous

            malformed_raw = joinpath(tmp, "malformed_raw.dat")
            write(malformed_raw, "garbage\n")
            @test_throws ErrorException extract_omni2_columns(malformed_raw, extracted)
            @test read(extracted, String) == previous

            invalid_time = joinpath(tmp, "invalid_time.dat")
            bad = collect(1:54)
            bad[1:3] = [2025, 0, 99]
            bad[41] = -20
            write(invalid_time, join(bad, ' ') * "\n")
            @test SolarSINDy._last_valid_omni_date(
                invalid_time; delim=nothing, dst_field=41, min_fields=54,
            ) === nothing

            download_target = joinpath(tmp, "download_cache.dat")
            write(download_target, previous)
            malformed_download = (_, path) -> (write(path, "not OMNI\n"); path)
            @test_throws ErrorException download_omni2(
                download_target; download_fn=malformed_download,
            )
            @test read(download_target, String) == previous

            truncated_target = joinpath(tmp, "truncated_download.dat")
            truncated_download = function (_, path)
                open(path, "w") do io
                    first_time = DateTime(1964, 1, 1)
                    for offset in 0:8_759
                        timestamp = first_time + Hour(offset)
                        fields = fill("0", 54)
                        fields[1] = string(year(timestamp))
                        fields[2] = string(dayofyear(Date(timestamp)))
                        fields[3] = string(hour(timestamp))
                        fields[41] = "-20"
                        println(io, join(fields, ' '))
                    end
                end
                path
            end
            @test_throws ErrorException download_omni2(
                truncated_target;
                year_end=2025,
                now_utc=DateTime(2026, 1, 2),
                download_fn=truncated_download,
            )
            @test !ispath(truncated_target)

            adversarial_download = function (_, path)
                open(path, "w") do io
                    for _ in 1:9_000
                        println(io, "garbage payload")
                    end
                    tail = collect(1:54)
                    tail[1:3] = [2025, 365, 23]
                    tail[41] = -20
                    println(io, join(tail, ' '))
                end
                path
            end
            @test_throws ErrorException download_omni2(
                download_target; download_fn=adversarial_download,
            )
            @test read(download_target, String) == previous

            # A large, apparently current cache with a plausible final row must
            # not bypass validation merely because its tail looks valid.
            invalid_cached_raw = joinpath(tmp, "invalid_cached_raw.dat")
            open(invalid_cached_raw, "w") do io
                write(io, repeat("garbage payload\n", 80_000))
                tail = collect(1:54)
                tail[1:3] = [2025, 365, 23]
                tail[41] = -20
                println(io, join(tail, ' '))
            end
            @test filesize(invalid_cached_raw) > 1_000_000
            attempted_replacement = Ref(false)
            failed_replacement = function (_, _)
                attempted_replacement[] = true
                error("replacement unavailable")
            end
            cached_bytes = read(invalid_cached_raw)
            @test_throws ErrorException download_omni2(
                invalid_cached_raw;
                year_end=2025,
                now_utc=DateTime(2026, 1, 2),
                download_fn=failed_replacement,
            )
            @test attempted_replacement[]
            @test read(invalid_cached_raw) == cached_bytes

            # The extracted-cache reuse path receives the same full semantic
            # validation and rebuilds an invalid body from a valid raw archive.
            valid_raw = joinpath(tmp, "one_valid_raw.dat")
            raw_row = collect(1:54)
            raw_row[1:3] = [2025, 1, 0]
            raw_row[41] = -20
            write(valid_raw, join(raw_row, ' ') * "\n")
            invalid_cached_csv = joinpath(tmp, "invalid_cached_extract.csv")
            open(invalid_cached_csv, "w") do io
                println(io, SolarSINDy.OMNI_EXTRACTED_HEADER)
                write(io, repeat("garbage,row\n", 10_000))
                println(io, "2025,365,23,1,-2,1e5,5,400,2,-20,50,-30,20")
            end
            @test filesize(invalid_cached_csv) > 100_000
            extract_omni2_columns(valid_raw, invalid_cached_csv;
                                  year_end=2025,
                                  now_utc=DateTime(2026, 1, 2))
            rebuilt = SolarSINDy._validate_extracted_omni(invalid_cached_csv)
            @test rebuilt.n_rows == 1
            @test rebuilt.first_time == DateTime(2025, 1, 1)

            nonfinite_csv = joinpath(tmp, "nonfinite_extract.csv")
            write(nonfinite_csv,
                  SolarSINDy.OMNI_EXTRACTED_HEADER * "\n" *
                  "2025,1,0,1,-2,Inf,5,400,2,-20,50,-30,20\n")
            @test_throws ErrorException SolarSINDy._validate_extracted_omni(nonfinite_csv)
        end
    end

    @testset "NF-DATA-04: non-finite and nonphysical critical inputs are invalid" begin
        df = DataFrame(
            datetime=[DateTime(2025, 1, 1), DateTime(2025, 1, 1, 1)],
            V=[400.0, Inf], Bz=[-5.0, Inf], By=[1.0, Inf], n=[5.0, -1.0],
            Pdyn=[2.0, -2.0], T=[1.0e5, Inf], Dst=[-20.0, Inf],
            AE=[50.0, Inf], AL=[-30.0, -Inf], AU=[20.0, Inf],
        )
        clean_omni_data!(df)
        @test df.quality == [1, 0]
        @test isnan(df.V[2])
        @test isnan(df.Bz[2])
        @test isnan(df.Pdyn[2])
        @test isnan(df.Dst_star[2])
    end
end
