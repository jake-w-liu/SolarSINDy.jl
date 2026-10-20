using Test, SHA, TOML, JSON3, LinearAlgebra, SolarSINDy
BLAS.set_num_threads(1)
const F=SolarSINDy.IntervalMeanForecaster
const T=F.DstIntervalTraining
const A=F.DstIntervalAdapter
const R=A.Replay
const S=F.DstIntervalStream
const B=F.DstIntervalArtifacts
include(joinpath(@__DIR__,"..","..","tools","initial_reference","DstInitialGBMReference.jl"))
const G=DstInitialGBMReference
const ORACLE=joinpath(@__DIR__,"literal_oracles_v1.json")
length(ARGS)>=2 || error("MODE OUTPUT [CASES_SHA256]")
mode=ARGS[1];root=ARGS[2]

if mode=="oracles"
    @test bytes2hex(sha256(read(ORACLE)))=="fab77d16251ad2cf91c9cc604d8373a3dbad3419d8f526b4c3beddbdcbec9821"
    oracle=JSON3.read(read(ORACLE),Dict{String,Any})
    @test !isdefined(Main,:DateTime)
    payload=("context",T.DateTime(2019),T.DateTime(2019,1,1,4),(-0.0,1.25),"source")
    reference_keys=((repeat("a",64),T.DateTime(2019),4,T.DateTime(2019,1,1,4)),
        (repeat("a",64),T.DateTime(2019),6,T.DateTime(2019,1,1,6)))
    parts=(T.DateTime(2019),(1.25,-0.0),"receipt")
    before=(training=T._hash(payload),replay=R._fingerprint(parts...),reference=G._hash(reference_keys))
    @testset "Golden hashes before importing Dates into Main" begin
        for name in keys(before)
            @test before[name]==oracle["canonical_expected_sha256"][String(name)]
        end
    end
    Core.eval(Main,:(using Dates))
    after=(training=T._hash(payload),replay=R._fingerprint(parts...),reference=G._hash(reference_keys))
    @testset "Golden hashes retain the original Dates-imported representation" begin
        for name in keys(after)
            @test after[name]==oracle["canonical_expected_sha256"][String(name)]
        end
        @test before==after
    end
elseif mode=="produce"
    Core.eval(Main,:(using Dates))
    boundary=T.DateTime(2019)+T.Dates.Hour(1000);recipe=A.settings()[1];source=repeat("a",64)
    rows=NamedTuple[]
    for i in 1:191
        available=i<=181
        issue=i==181 ? boundary-T.Dates.Hour(174) : T.DateTime(2019)+T.Dates.Hour(i<=180 ? i-1 : i-2)
        clocks=Tuple(A.forecast_clock(issue,h,0) for h in (4,6))
        designs=((1.,0.),(0.,1.),(1.,1.))
        coefficients=available ? (designs[mod1(i,3)],designs[mod1(i+1,3)]) : ((0.,0.),(0.,0.))
        target=available ? (-60.,-40.) : (-50.,-50.)
        offset=available ? Tuple(target[j]-2coefficients[j][1]-3coefficients[j][2] for j in 1:2) : (-55.,-55.)
        push!(rows,(;issue,driver_tag=clocks[1].driver_tag,driver_receipt=issue,
            anchor_tag=clocks[1].anchor_tag,anchor_complete=clocks[1].anchor_complete,anchor_receipt=clocks[1].anchor_receipt,
            anchor=-55.,target_tag=Tuple(c.target_tag for c in clocks),target_complete=Tuple(c.target_complete for c in clocks),
            target_receipt=Tuple(c.target_receipt for c in clocks),target,scoreable=(true,true),model_available=available,
            offset,coefficients,taus=recipe.taus,sigma=recipe.sigma,source_sha256=source,dst_delay_hours=0))
    end
    sort!(rows;by=x->x.issue)
    model=T.fit_gains(rows,boundary,recipe;delay_hours=0,source_sha256=source)
    @test isapprox(model.gains[1],2.;atol=1e-12,rtol=0)
    @test isapprox(model.gains[2],3.;atol=1e-12,rtol=0)
    @test model.fit_count==191 && model.informative_count==181 && model.storm_count==201
    cursor=boundary+T.Dates.Hour(1)
    checkpoint=S.StreamCheckpoint(1001,cursor,source,repeat("b",64),0,recipe.taus,recipe.sigma,
        R.Checkpoint(cursor,[0.,0.],zeros(2,2)),((0.,0.),(0.,0.)),nothing,false,0,cursor)
    # This deliberately inactive synthetic checkpoint tests the typed codec;
    # it is not a claim that these affine hand rows came from a raw archive.
    bundle=B.save_bundle(joinpath(root,"bundle"),model,checkpoint;provenance=(synthetic_context_fixture=true,scientific_verdict=false))
    selected=[first(rows),first(row for row in rows if !row.model_available)]
    predicted=T.predict_rows(model,selected)
    cases=(rows=Tuple(selected),predictions=Tuple(row.prediction for row in predicted),model_id=model.model_id,
        bundle_manifest_sha256=bundle.manifest_sha256)
    B._write_new(joinpath(root,"cases.toml"),B._typed_payload_bytes(cases))
    @test B.load_bundle(bundle.directory;expected_manifest_sha256=bundle.manifest_sha256,expected_sources=B.verify_source_identity()).model==model
elseif mode in ("load","load-shadow","load-using-dates")
    @test !isdefined(Main,:DateTime)
    mode=="load-shadow" && Core.eval(Main,:(struct DateTime end))
    mode=="load-using-dates" && Core.eval(Main,:(using Dates))
    length(ARGS)==3 || error("explicit cases digest required")
    path=joinpath(root,"cases.toml");@test bytes2hex(sha256(read(path)))==ARGS[3]
    cases=B._decode(TOML.parsefile(path)["value"])
    loaded=B.load_bundle(joinpath(root,"bundle");expected_manifest_sha256=cases.bundle_manifest_sha256,expected_sources=B.verify_source_identity())
    @test loaded.model.model_id==cases.model_id
    @test Tuple(row.prediction for row in T.predict_rows(loaded.model,collect(cases.rows)))==cases.predictions
    @test loaded.provenance==(synthetic_context_fixture=true,scientific_verdict=false)
else
    error("unknown hash-context worker mode")
end
