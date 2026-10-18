# Portable additional public-value comparison against the untouched baseline.
# Values are not printed or interpreted as forecast-skill evidence.
using Dates,SHA,Test,TOML
length(ARGS)==3 || error("baseline-root candidate-root new-output-root")
const BASELINE=abspath(ARGS[1]);const CANDIDATE=abspath(ARGS[2]);const OUT=abspath(ARGS[3]);mkdir(OUT)
const MODULES=[Module(gensym(:BaselineBits)),Module(gensym(:CandidateBits))]
for (wrapper,root) in zip(MODULES,(BASELINE,CANDIDATE))
    for name in ("DstIntervalAdapter.jl","DstIntervalStream.jl","DstIntervalTraining.jl","DstIntervalArtifacts.jl")
        Base.include(wrapper,joinpath(root,name))
    end
end
const B=MODULES[1];const C=MODULES[2]
const BS=B.DstIntervalStream;const CS=C.DstIntervalStream
const BT=B.DstIntervalTraining;const CT=C.DstIntervalTraining
const BF=B.DstIntervalArtifacts;const CF=C.DstIntervalArtifacts
function token(io,x)
    if x isa Float64
        write(io,UInt8('F'),htol(reinterpret(UInt64,x)))
    elseif x isa DateTime
        write(io,UInt8('D'),htol(Int64(Dates.value(x))))
    elseif x isa Bool
        write(io,UInt8('B'),UInt8(x))
    elseif x isa Integer
        write(io,UInt8('I'),htol(Int64(x)))
    elseif x isa AbstractString || x isa Symbol
        value=String(x);write(io,UInt8(x isa Symbol ? 'Y' : 'S'),htol(UInt64(ncodeunits(value))),codeunits(value))
    elseif x===nothing
        write(io,UInt8('0'))
    elseif x isa NamedTuple
        write(io,UInt8('N'),htol(UInt64(length(x))))
        for (name,value) in pairs(x);token(io,name);token(io,value);end
    elseif x isa Tuple || x isa AbstractArray
        write(io,UInt8(x isa Tuple ? 'T' : 'A'),htol(UInt64(length(x))))
        x isa AbstractArray && token(io,size(x))
        for value in x;token(io,value);end
    elseif isstructtype(typeof(x))
        token(io,nameof(typeof(x)))
        for name in fieldnames(typeof(x));token(io,name);token(io,getfield(x,name));end
    else
        error("unsupported public bit token $(typeof(x))")
    end
end
bytes(x)=begin io=IOBuffer();token(io,x);take!(io) end
without(x,names)=NamedTuple{Tuple(k for k in keys(x) if !(k in names))}(Tuple(x[k] for k in keys(x) if !(k in names)))
const ID=bytes2hex(sha256("synthetic-training-fixture"));const N=1000;const T=DateTime(2018)
const TIMES=[T+Hour(i-1) for i in 1:N];const VALUES=ones(17,N)
VALUES[8,:].=400.;VALUES[6,:]=[-5+3sin(i/43) for i in 1:N]
VALUES[12,:].=2.;VALUES[15,:]=[-60+40sin(i/61) for i in 1:N]
const INPUTS=(BS.HourlyInputs(TIMES,VALUES;source_sha256=ID),CS.HourlyInputs(TIMES,VALUES;source_sha256=ID))
const SOURCE_BEFORE=(Dict(BF.verify_source_identity()),Dict(CF.verify_source_identity()))
const RECORDS=Dict{String,Any}[]
@testset "all18 settings both latencies training/prediction/artifact public bits" begin
    @test B.DstIntervalAdapter.settings()==C.DstIntervalAdapter.settings()
    for setting in B.DstIntervalAdapter.settings(), delay in (0,1)
        rows=(NamedTuple[],NamedTuple[])
        summaries=map(zip((BS,CS),INPUTS,rows)) do (S,input,panel)
            S.stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,sink=part->append!(panel,part))
        end
        @test bytes(rows[1])==bytes(rows[2])
        @test bytes(summaries[1].checkpoint)==bytes(summaries[2].checkpoint)
        models=(BT.fit_gains(rows[1],T+Hour(700),setting;delay_hours=delay,source_sha256=ID),
                CT.fit_gains(rows[2],T+Hour(700),setting;delay_hours=delay,source_sha256=ID))
        @test bytes(without(models[1],(:source_hashes,:model_id)))==bytes(without(models[2],(:source_hashes,:model_id)))
        @test models[1].model_id!=models[2].model_id
        source_maps=Dict.(getproperty.(models,:source_hashes))
        @test sort([name for name in keys(source_maps[1]) if source_maps[1][name]!=source_maps[2][name]])==["DstIntervalAdapter.jl","DstIntervalObserver.jl"]
        predictions=(BT.predict_rows(models[1],rows[1]),CT.predict_rows(models[2],rows[2]))
        @test bytes([without(row,(:model_id,)) for row in predictions[1]])==bytes([without(row,(:model_id,)) for row in predictions[2]])
        for (index,(F,model,summary)) in enumerate(zip((BF,CF),models,summaries))
            dest=joinpath(OUT,"setting$(setting.index)-L$delay-version$index")
            saved=F.save_bundle(dest,model,summary.checkpoint;provenance=(scope="synthetic bit equivalence",input_sha256=ID,stage=:synthetic))
            loaded=F.load_bundle(dest;expected_manifest_sha256=saved.manifest_sha256,expected_sources=F.verify_source_identity())
            @test bytes(loaded.model)==bytes(model)
            @test bytes(loaded.checkpoint)==bytes(summary.checkpoint)
        end
        push!(RECORDS,Dict("setting_index"=>setting.index,"delay"=>delay,"rows"=>N,
            "model_public_bits_except_explicit_source_provenance_equal"=>true,"prediction_bits_except_model_identity_equal"=>true))
    end
    @test (Dict(BF.verify_source_identity()),Dict(CF.verify_source_identity()))==SOURCE_BEFORE
end
open(joinpath(OUT,"receipt.toml"),"w") do io
    TOML.print(io,Dict("complete"=>true,"julia_version"=>string(VERSION),"records"=>RECORDS,
        "baseline_source_sha256"=>SOURCE_BEFORE[1],"candidate_source_sha256"=>SOURCE_BEFORE[2],
        "driver_sha256"=>bytes2hex(sha256(read(@__FILE__)))),sorted=true)
end
