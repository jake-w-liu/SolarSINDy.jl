"Immutable source-length independent ordered training indices."
struct TrainingRows342 <: AbstractVector{Int}
    bytes::String
    count::Int
    function TrainingRows342(rows)
        _need((rows isa AbstractVector||rows isa Tuple)&&1<=length(rows)<=MAX_ROWS,"bounded training indices required")
        Base.require_one_based_indexing(rows)
        bytes=Vector{UInt8}(undef,4length(rows));previous=0
        for (k,i) in enumerate(rows)
            _need(i isa Integer&&!(i isa Bool)&&previous<i<=MAX_ROWS,"strictly ordered bounded training indices required")
            value=UInt32(i)
            for j in 0:3;bytes[4(k-1)+j+1]=UInt8((value>>(8j))&0xff);end
            previous=Int(i)
        end
        new(String(bytes),length(rows))
    end
end
Base.size(rows::TrainingRows342)=(rows.count,)
Base.IndexStyle(::Type{TrainingRows342})=IndexLinear()
function Base.getindex(rows::TrainingRows342,k::Int)
    @boundscheck checkbounds(rows,k)
    offset=4(k-1)
    @inbounds value=UInt32(codeunit(rows.bytes,offset+1))|
        UInt32(codeunit(rows.bytes,offset+2))<<8|
        UInt32(codeunit(rows.bytes,offset+3))<<16|
        UInt32(codeunit(rows.bytes,offset+4))<<24
    Int(value)
end
Base.setindex!(::TrainingRows342,value,k)=throw(ArgumentError("immutable training indices"))
Base.copy(rows::TrainingRows342)=rows
_training_rows(rows)=TrainingRows342(rows)
function _check_training_rows(rows::TrainingRows342)
    _need(1<=length(rows)<=MAX_ROWS&&ncodeunits(rows.bytes)==4length(rows),"training index storage differs")
    previous=0
    for i in rows;_need(previous<i<=MAX_ROWS,"encoder training rows differ");previous=i;end
    nothing
end
"Training-only encoder with immutable parameters and explicit ordered row identity."
struct Encoder342
    training_rows::TrainingRows342
    finite_counts::Tuple{Vararg{Int}}
    imputation::Tuple{Vararg{Float64}}
    center::Tuple{Vararg{Float64}}
    scale::Tuple{Vararg{Float64}}
    training_design_sha256::String
end
function _design(X)
    _need(size(X,2)==WIDTH&&1<=size(X,1)<=MAX_ROWS,"bounded342-column design required")
    Base.require_one_based_indexing(X)
    _need(all(x->x isa Real&&!(x isa Bool)&&(isfinite(x)||isnan(x)),X),"design Inf/Bool rejected")
    matrix=Matrix{Float64}(X)
    _need(all(x->isfinite(x)||isnan(x),matrix),"design conversion overflow")
    for i in axes(matrix,1)
        for j in 1:BASE_WIDTH
            _need(matrix[i,j+BASE_WIDTH]==Float64(isnan(matrix[i,j])),"pre-imputation missing flag differs")
        end
        _need(all(isfinite,@view(matrix[i,BASE_WIDTH+1:WIDTH])),"non-base design must be finite")
    end
    matrix
end
function _train_sha(matrix,rows)
    io=IOBuffer();write(io,"matched342-training-design-v1\n")
    for i in rows
        write(io,htol(Int64(i)))
        for j in 1:WIDTH
            x=matrix[i,j];write(io,htol(isnan(x) ? reinterpret(UInt64,NaN) : reinterpret(UInt64,x)))
        end
    end
    bytes2hex(sha256(take!(io)))
end
"Fit165 finite-value means and342 sample-standard-deviation parameters using only explicit ordered training rows; constants/one-row scales are1."
function fit_encoder(X,training_rows)
    source_identity();matrix=_design(X)
    _need(!isempty(training_rows)&&all(i->i isa Integer&&!(i isa Bool)&&1<=i<=size(matrix,1),training_rows),"nonempty bounded training indices required")
    rows=collect(Int,training_rows);_need(issorted(rows)&&allunique(rows),"unique ordered training indices required")
    counts=Int[];imputation=Float64[]
    for j in 1:BASE_WIDTH
        values=Float64[matrix[i,j] for i in rows if isfinite(matrix[i,j])]
        push!(counts,length(values));value=isempty(values) ? 0. : mean(values)
        _need(isfinite(value),"imputation mean overflow");push!(imputation,value)
    end
    selected=matrix[rows,:]
    for i in axes(selected,1),j in 1:BASE_WIDTH
        isnan(selected[i,j])&&(selected[i,j]=imputation[j])
    end
    center=vec(mean(selected;dims=1))
    scale=length(rows)==1 ? ones(WIDTH) : vec(std(selected;dims=1,corrected=true))
    _need(all(isfinite,center)&&all(isfinite,scale),"training moment overflow")
    scale[scale.==0.].=1.
    _need(all(>(0.),scale),"positive training scales required")
    Encoder342(_training_rows(rows),Tuple(counts),Tuple(imputation),Tuple(center),Tuple(scale),_train_sha(matrix,rows))
end
function _check_encoder(encoder::Encoder342)
    _check_training_rows(encoder.training_rows)
    _need(length(encoder.finite_counts)==length(encoder.imputation)==BASE_WIDTH&&length(encoder.center)==length(encoder.scale)==WIDTH,
        "encoder dimensions differ")
    _need(all(n->0<=n<=length(encoder.training_rows),encoder.finite_counts)&&all(isfinite,encoder.imputation)&&
        all(isfinite,encoder.center)&&all(x->isfinite(x)&&x>0,encoder.scale)&&_hex(encoder.training_design_sha256),"encoder domain differs")
    _need(all(n!=0||iszero(value) for (n,value) in zip(encoder.finite_counts,encoder.imputation)),"allmissing imputer must be zero")
    _need(length(encoder.training_rows)!=1||all(==(1.),encoder.scale),"one-row primitive scale must be one")
    nothing
end
"Return a finite copied342-column matrix using saved imputation/centering/scaling; pre-imputation flags must match."
function transform(encoder::Encoder342,X)
    _check_encoder(encoder);result=_design(X)
    for i in axes(result,1),j in 1:WIDTH
        isnan(result[i,j])&&(result[i,j]=encoder.imputation[j])
        result[i,j]=(result[i,j]-encoder.center[j])/encoder.scale[j]
        _need(isfinite(result[i,j]),"normalized design overflow")
    end
    result
end
