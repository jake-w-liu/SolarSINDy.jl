# Fixed ADC18+lag11 arithmetic and sample scaling; no author or ML imports.
using Statistics

const NAMES = ["bz0","bz1","bz2","bz_mean6","bz_sd6","by0","bperp0",
    "bperp_mean6","v0","dv6","logn0","logn_mean6","pdyn0","vbs0",
    "vbs_mean3","south_run","dst0","ddst1","dst_lag1","dst_lag2",
    "dst_lag3","dst_lag4","dst_lag5","dst_lag6","dst_lag12","dst_lag24",
    "vbs_lag1","vbs_lag2","vbs_lag3"]
const DST_LAGS = (1,2,3,4,5,6,12,24)

function features29(raw::Matrix{Float64})
    size(raw,1) == 17 || error("expected exactly17 input channels")
    n = size(raw,2)
    X = fill(NaN,n,29)
    available = falses(n)
    for i in 25:n
        isfinite(raw[15,i]) && isfinite(raw[15,i-1]) || continue
        all(lag -> all(j -> isfinite(raw[j,i-lag]),(5,6,8,9,12)) && raw[9,i-lag]>0.,0:6) || continue
        all(lag -> isfinite(raw[15,i-lag]),DST_LAGS) || continue
        bzsum = 0.; btsum = 0.; logsum = 0.
        for lag in 0:5
            bzsum += raw[6,i-lag]
            btsum += hypot(raw[5,i-lag],raw[6,i-lag])
            logsum += log(raw[9,i-lag])
        end
        bzmean = bzsum/6
        varsum = 0.
        for lag in 0:5;varsum += abs2(raw[6,i-lag]-bzmean);end
        vbssum = 0.
        for lag in 0:2;vbssum += raw[8,i-lag]*max(0.,-raw[6,i-lag])/1000.;end
        south = 0
        for lag in 0:11
            r = i-lag
            r>=1 && isfinite(raw[6,r]) && raw[6,r]<0. || break
            south += 1
        end
        X[i,1]=raw[6,i];X[i,2]=raw[6,i-1];X[i,3]=raw[6,i-2]
        X[i,4]=bzmean;X[i,5]=sqrt(varsum/6);X[i,6]=raw[5,i]
        X[i,7]=hypot(raw[5,i],raw[6,i]);X[i,8]=btsum/6
        X[i,9]=raw[8,i];X[i,10]=raw[8,i]-raw[8,i-6]
        X[i,11]=log(raw[9,i]);X[i,12]=logsum/6;X[i,13]=raw[12,i]
        X[i,14]=raw[8,i]*max(0.,-raw[6,i])/1000.;X[i,15]=vbssum/3
        X[i,16]=south;X[i,17]=raw[15,i];X[i,18]=raw[15,i]-raw[15,i-1]
        for (j,lag) in enumerate(DST_LAGS);X[i,18+j]=raw[15,i-lag];end
        for lag in 1:3;X[i,26+lag]=raw[8,i-lag]*max(0.,-raw[6,i-lag])/1000.;end
        all(isfinite,view(X,i,:)) || error("nonfinite computed GBM feature")
        available[i]=true
    end
    X,available
end

function fit_scaling(X, available, rowpath)
    rows = parse.(Int,readlines(rowpath))
    length(rows)>=2 && issorted(rows) && allunique(rows) &&
        all(i -> 1<=i<=size(X,1) && available[i],rows) || error("invalid exact scaling fit rows")
    values = Matrix{Float64}(X[rows,:])
    center = vec(mean(values;dims=1))
    scale = vec(std(values;dims=1,corrected=true))
    all(isfinite,center) && all(isfinite,scale) || error("nonfinite sample scaling")
    scale[scale.==0.].=1.
    center,scale
end

function main(args)
    length(args)==5 || error("input17.f64 N NEW_output_directory fit_h4.txt fit_h6.txt")
    inputpath,ntext,outdir,fit4,fit6 = args
    n=parse(Int,ntext)
    1<=n<=1_000_000 || error("hour count outside declared resource budget")
    isfile(inputpath) && !islink(inputpath) && filesize(inputpath)==17*n*8 || error("invalid input byte count/path")
    ENDIAN_BOM==0x04030201 || error("little-endian execution required")
    raw=reshape(reinterpret(Float64,read(inputpath)),17,n) |> Matrix{Float64}
    X,available=features29(raw)
    scales = (fit_scaling(X,available,fit4),fit_scaling(X,available,fit6))
    ispath(outdir) && error("output already exists")
    mkdir(outdir)
    open(joinpath(outdir,"features29.f64"),"w") do io
        for i in 1:n,j in 1:29;write(io,X[i,j]);end
    end
    open(joinpath(outdir,"availability.u8"),"w") do io;write(io,UInt8.(available));end
    for (h,(center,scale)) in zip((4,6),scales)
        open(joinpath(outdir,"gbm29_scaling_h$(h).csv"),"w") do io
            println(io,"feature,center,scale")
            for j in 1:29;println(io,NAMES[j],',',repr(center[j]),',',repr(scale[j]));end
        end
    end
    println("independent Base/Statistics GBM arithmetic complete; rows=",n," available=",count(available))
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    main(ARGS)
end
