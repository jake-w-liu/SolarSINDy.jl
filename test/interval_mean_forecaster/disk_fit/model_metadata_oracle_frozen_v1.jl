using Test, Dates, SHA, TOML
if !isdefined(@__MODULE__, :DISK_FIT_NAMESPACE)
    include(ARGS[1])
    const DISK_FIT_NAMESPACE = IntervalMeanForecaster
end
const MF = DISK_FIT_NAMESPACE
const MA = MF.DstIntervalAdapter
const MROOT = abspath(ARGS[2])
mkdir(MROOT)
const MSOURCE = repeat("a", 64)
const MSTART = DateTime(2019)
const MBOUNDARY = MSTART+Hour(1000)

function metadata_rows(setting, delay)
    rows = NamedTuple[]
    for i in 1:191
        # 181 informative origins, ten exact persistence fallbacks, with the
        # last informative origin exactly at the received-label embargo edge.
        informative = i <= 181
        issue = i == 181 ? MBOUNDARY-Hour(174+delay) :
            MSTART+Hour(i <= 180 ? i-1 : i-2)
        clocks = Tuple(MA.forecast_clock(issue, h, delay) for h in (4, 6))
        designs = ((1.,0.), (0.,1.), (1.,1.))
        z = informative ? (designs[mod1(i,3)], designs[mod1(i+1,3)]) : ((0.,0.),(0.,0.))
        labels = informative ? (-60.,-40.) : (-50.,-50.)
        offset = informative ? Tuple(labels[j]-2z[j][1]-3z[j][2] for j in 1:2) : (-55.,-55.)
        push!(rows, (;issue, driver_tag=clocks[1].driver_tag, driver_receipt=issue,
            anchor_tag=clocks[1].anchor_tag, anchor_complete=clocks[1].anchor_complete,
            anchor_receipt=clocks[1].anchor_receipt, anchor=-55.,
            target_tag=Tuple(c.target_tag for c in clocks),
            target_complete=Tuple(c.target_complete for c in clocks),
            target_receipt=Tuple(c.target_receipt for c in clocks), target=labels,
            scoreable=(true,true), model_available=informative, offset, coefficients=z,
            taus=setting.taus, sigma=setting.sigma, source_sha256=MSOURCE, dst_delay_hours=delay))
    end
    sort!(rows; by=x->x.issue)
    rows
end

@testset "independent canonical disk-fit metadata and loss" begin
    for delay in (0,1), index in (1,2)
        setting = MF.settings()[index]
        rows = metadata_rows(setting,delay)
        input = MF.export_fit_input(joinpath(MROOT,"input-L$delay-R$index"),
            MBOUNDARY,setting;delay_hours=delay,source_sha256=MSOURCE) do sink
            for i in 1:11:length(rows)
                sink(rows[i:min(i+10,end)])
            end
        end
        manifest = TOML.parsefile(joinpath(input.directory,"manifest.toml"))
        model = MF.fit_gains_from_input(input.directory;
            expected_manifest_sha256=input.manifest_sha256)
        @test model.gains[1] ≈ 2. atol=1e-12 rtol=0
        @test model.gains[2] ≈ 3. atol=1e-12 rtol=0
        @test model.fit_count == 191
        @test model.informative_count == 181
        @test model.storm_count == 201
        @test model.fit_mask_sha256 == manifest["fit_mask_sha256"]
        @test model.fit_data_sha256 == manifest["fit_data_sha256"]
        @test model.fit_mask_sha256 != model.fit_data_sha256
        @test model.max_fit_issue == MBOUNDARY-Hour(174+delay)
        @test model.max_target_complete == MBOUNDARY-Hour(168+delay)
        @test model.max_label_receipt == MBOUNDARY-Hour(168)
        @test model.source_sha256 == MSOURCE
        @test Dict(model.source_hashes) == manifest["source_hashes"]
        @test model.setting_index == index
        @test (model.taus,model.sigma,model.eta) == (setting.taus,setting.sigma,setting.eta)
        # Ten fallback origins contribute two residuals of 5; all 201 storm
        # labels include their twenty labels. Informative residuals vanish.
        @test model.solver.weighted_sse ≈ 500/382+500setting.eta/201 atol=2e-14 rtol=0
        names = Tuple(keys(model))[1:end-1]
        base = NamedTuple{names}(Tuple(model)[1:end-1])
        @test model.model_id == bytes2hex(sha256(repr(base)))
    end
end
