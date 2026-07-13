using CSV
using DataFrames

@testset "Data Pipeline And Cleaning" begin

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
                println(io, join(1:54, ' '))
                println(io, join(101:154, ' '))
            end

            extract_omni2_columns(raw_path, extracted_path)
            out = CSV.read(extracted_path, DataFrame)

            @test names(out) == ["year", "doy", "hour", "By", "Bz", "T", "n", "V", "Pdyn", "Dst", "AE", "AL", "AU"]
            @test Tuple(out[1, :]) == (1, 2, 3, 16, 17, 23, 24, 25, 29, 41, 42, 53, 54)
            @test Tuple(out[2, :]) == (101, 102, 103, 116, 117, 123, 124, 125, 129, 141, 142, 153, 154)
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
    end

    @testset "A/B: build_storm_catalog finds storm window and assigns split" begin
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
        @test entry.min_dst == -80.0
        @test entry.min_dst_time == df.datetime[14]
        @test entry.solar_cycle == 25
        @test entry.split == "test"
        @test entry.onset_idx == 10
        @test entry.end_idx == 40
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
            loaded = load_storm_catalog(path)
            @test loaded == [entry]
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
end
