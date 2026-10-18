# Independent 256-bit moment oracle. This deliberate test-only method replacement
# checks that a more accurate mean cannot invalidate legitimate saved scale state.
isdefined(@__MODULE__,:IndependentFeatureArithmetic) || error("load feature arithmetic first")
@eval IndependentFeatureArithmetic begin
    function fit_scaling(X,available,rowpath)
        rows=parse.(Int,readlines(rowpath))
        length(rows)>=2 && issorted(rows) && allunique(rows) &&
            all(i->1<=i<=size(X,1) && available[i],rows) || error("invalid exact oracle fit rows")
        center=zeros(29);scale=zeros(29)
        setprecision(BigFloat,256) do
            for j in 1:29
                values=BigFloat.(X[rows,j]);m=sum(values)/length(values)
                variance=sum(abs2,value-m for value in values)/(length(values)-1)
                center[j]=Float64(m);scale[j]=iszero(variance) ? 1. : Float64(sqrt(variance))
            end
        end
        center,scale
    end
end
