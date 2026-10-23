module IsolatedBridgeSources
const REPO=normpath(joinpath(@__DIR__,"..",".."))
function with_sources(f;stream_transform=identity,tracker_transform=identity)
    mktempdir(;prefix="bridge-isolated-source-") do root
        src=joinpath(root,"src");mkdir(src)
        interval=joinpath(src,"interval_mean_forecaster");mkdir(interval)
        for name in readdir(joinpath(REPO,"src","interval_mean_forecaster"))
            endswith(name,".jl")||continue
            original=joinpath(REPO,"src","interval_mean_forecaster",name)
            body=read(original,String)
            name=="DstIntervalStream.jl"&&(body=stream_transform(body))
            write(joinpath(interval,name),body)
        end
        for name in ("duration_conditioned_forcing.jl","duration_current_state_stream.jl")
            body=read(joinpath(REPO,"src",name),String)
            name=="duration_current_state_stream.jl"&&(body=tracker_transform(body))
            write(joinpath(src,name),body)
        end
        parent=Module(gensym(:IsolatedBridge))
        Base.include(parent,joinpath(interval,"IntervalMeanForecaster.jl"))
        Base.include(parent,joinpath(src,"duration_conditioned_forcing.jl"))
        Base.include(parent,joinpath(src,"duration_current_state_stream.jl"))
        Base.invokelatest(f,parent,src)
    end
end
end
