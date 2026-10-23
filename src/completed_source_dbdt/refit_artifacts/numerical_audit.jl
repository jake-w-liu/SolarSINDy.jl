# Directed MPFR intervals are confined to single-threaded offline checking.
# Every endpoint operation restores the previous rounding mode on exit.
struct _Iv
    lo::BigFloat
    hi::BigFloat
end
const _TOY_ORACLE_SHA="f7792e079256c51f3511febfd79ac36b83cd278bb49cf48295effd31b755a9ae"
function _toy_ridge_check()
    X=[0.0 5.0;1.0 5.0;2.0 5.0];y=[1.0,2.0,3.0]
    for (mixed,expected) in ((false,[2.0,2/3,0.0]),(true,[1.5,0.6666666668888889069052675683111637,0.0]))
        recipe=Dict{String,Any}("penalty"=>float_bits(1.0),"intercept_penalty"=>mixed,
            "std_rule"=>(mixed ? "sample_std_plus_1e_minus9" : "sample_std_zero_nonfinite_to_one"))
        _,Z=_ridge_preprocess(X,recipe);A,b,_=_normal_statistics(Z,y,recipe)
        require(all(abs.(A\b.-expected).<=1e-13),"independent frozen toy coefficient control failed")
    end
    _TOY_ORACLE_SHA
end
_down(f)=setrounding(f,BigFloat,RoundDown)
_up(f)=setrounding(f,BigFloat,RoundUp)
_iv(x::Real)=_Iv(BigFloat(x),BigFloat(x))
_add(a::_Iv,b::_Iv)=_Iv(_down(()->a.lo+b.lo),_up(()->a.hi+b.hi))
_neg(a::_Iv)=_Iv(-a.hi,-a.lo)
_sub(a::_Iv,b::_Iv)=_add(a,_neg(b))
function _mul(a::_Iv,b::_Iv)
    lower=[_down(()->x*y) for x in (a.lo,a.hi),y in (b.lo,b.hi)]
    upper=[_up(()->x*y) for x in (a.lo,a.hi),y in (b.lo,b.hi)]
    _Iv(minimum(lower),maximum(upper))
end
function _div(a::_Iv,b::_Iv)
    require(!(b.lo<=0<=b.hi),"interval divisor contains zero")
    _mul(a,_Iv(_down(()->inv(b.hi)),_up(()->inv(b.lo))))
end
_abs(a::_Iv)=a.lo>=0 ? a : a.hi<=0 ? _neg(a) : _Iv(BigFloat(0),max(-a.lo,a.hi))
_sum(values)=foldl(_add,values;init=_iv(0))
_inside(inner::_Iv,outer::_Iv)=outer.lo<=inner.lo<=inner.hi<=outer.hi
function _norminf(values::AbstractVector{_Iv})
    absolutes=_abs.(values)
    _Iv(maximum(x.lo for x in absolutes),maximum(x.hi for x in absolutes))
end
function _norminf(values::AbstractMatrix{_Iv})
    _norminf([_sum(_abs.(view(values,i,:))) for i in axes(values,1)])
end
function _matvec(A::AbstractMatrix{_Iv},x::AbstractVector{_Iv})
    [_sum(_mul(A[i,j],x[j]) for j in axes(A,2)) for i in axes(A,1)]
end
function _matmul(A::AbstractMatrix{_Iv},B::AbstractMatrix{_Iv})
    [_sum(_mul(A[i,k],B[k,j]) for k in axes(A,2)) for i in axes(A,1),j in axes(B,2)]
end
function _dyadic(x::BigFloat)
    require(isfinite(x),"nonfinite diagnostic endpoint")
    x==0 && return "0p0"
    rational=Rational{BigInt}(x);numerator_value=numerator(rational);denominator_value=denominator(rational)
    exponent=-ndigits(denominator_value;base=2)+1
    while iseven(numerator_value);numerator_value>>=1;exponent+=1;end
    mantissa=string(abs(numerator_value);base=16)
    require(length(mantissa)<=128 && abs(exponent)<=16384,"diagnostic endpoint exceeds grammar")
    (numerator_value<0 ? "-" : "")*mantissa*"p"*string(exponent)
end
function _interval_record(x::_Iv,precision_bits)
    require(isfinite(x.lo)&&isfinite(x.hi)&&x.lo<=x.hi,"invalid diagnostic interval")
    Dict{String,Any}("codec"=>"dyadic_interval_v1","lower_dyadic"=>_dyadic(x.lo),
        "upper_dyadic"=>_dyadic(x.hi),"precision_bits"=>precision_bits)
end
function _read_dyadic(value,bits)
    text=bounded_string(value;limit=160,nonempty=true)
    text=="0p0" && return BigFloat(0;precision=bits)
    require(occursin(r"^-?[1-9a-f][0-9a-f]*p-?(0|[1-9][0-9]*)$",text),"invalid dyadic diagnostic grammar")
    parts=split(text,'p');mantissa_text=first(parts);digits=startswith(mantissa_text,"-") ? mantissa_text[2:end] : mantissa_text
    require(length(digits)<=128,"diagnostic mantissa bound exceeded")
    mantissa=parse(BigInt,mantissa_text;base=16);require(isodd(mantissa),"dyadic mantissa must be odd")
    exponent=tryparse(Int,last(parts));require(exponent!==nothing && abs(BigInt(exponent))<=16384,"diagnostic exponent bound exceeded")
    number=setprecision(BigFloat,bits) do
        ldexp(BigFloat(mantissa),exponent)
    end
    require(isfinite(number) && _dyadic(number)==text,"noncanonical/nonfinite dyadic diagnostic")
    number
end
function _read_interval_record(value)
    _schema(value,"dyadic_interval")
    require(value["codec"]=="dyadic_interval_v1","unsupported diagnostic interval codec")
    bits=checked_integer(value["precision_bits"],256,512)
    require(bits in (256,512),"unsupported diagnostic precision")
    lo=_read_dyadic(value["lower_dyadic"],bits);hi=_read_dyadic(value["upper_dyadic"],bits)
    require(lo<=hi,"reversed diagnostic interval")
    _Iv(lo,hi)
end
function _fit_diagnostic_sanity(payload,n,p)
    checked_integer(payload["precision_bits"],512,512)
    eta=_read_interval_record(payload["backward_eta"]);condition=_read_interval_record(payload["condition_estimate"])
    forward=_read_interval_record(payload["forward_enclosure"]);residual=_read_interval_record(payload["normal_residual"])
    require(all(x->x.lo>=0,(eta,condition,forward,residual)) && condition.hi>=1,"invalid nonnegative fit diagnostic bounds")
    setprecision(BigFloat,512) do
        u=_iv(BigFloat(2)^(-53));k=n+2p+4
        limit=_mul(_iv(8),_div(_mul(_iv(k),u),_sub(_iv(1),_mul(_iv(k),u))))
        require(eta.hi<=limit.lo,"saved backward residual exceeds frozen feature-count gate")
        product=_mul(condition,eta);require(product.hi<1,"saved condition/residual product is not below1")
        required=_div(_mul(_iv(2),product),_sub(_iv(1),product))
        require(forward.hi>=required.hi && forward.hi<=_down(()->BigFloat(1)/BigFloat(10)^6),"saved forward enclosure is insufficient/above frozen cap")
    end
    nothing
end

function _prediction_enclosure(x,mu,scale,beta,z,prediction,bits)
    setprecision(BigFloat,bits) do
        u=_iv(BigFloat(2)^(-53));eta=_iv(BigFloat(2)^(-1074));m=length(beta)
        exact=[_div(_sub(_iv(x[j]),_iv(mu[j])),_iv(scale[j])) for j in eachindex(x)]
        normal_loss=_sum(_mul(_abs(_iv(beta[j+1])),_abs(_sub(_iv(z[j]),exact[j]))) for j in eachindex(x))
        reference=_add(_iv(beta[1]),_sum(_mul(_iv(beta[j+1]),exact[j]) for j in eachindex(x)))
        products=[_abs(_mul(_iv(beta[j+1]),_iv(z[j]))) for j in eachindex(x)]
        S=_add(_abs(_iv(beta[1])),_sum(products))
        M=_sum(_add(_mul(u,v),_div(eta,_iv(2))) for v in products)
        denominator=_sub(_iv(1),_mul(_iv(m),u));require(denominator.lo>0,"invalid reduction bound")
        gamma=_div(_mul(_iv(m),u),denominator)
        ED=_add(M,_add(_mul(gamma,_add(S,M)),_div(_mul(_iv(m),eta),_mul(_iv(2),denominator))))
        require(_add(_add(S,M),ED).hi<BigFloat(floatmax(Float64)),"portable no-overflow enclosure cannot be established")
        bound=_add(normal_loss,ED);error=_abs(_sub(_iv(prediction),reference))
        require(error.hi<=bound.hi,"native prediction is outside directed exact-reference bound")
        (;reference,normal_loss,ED,bound,error)
    end
end
function _check_prediction_bound(x,mu,scale,beta,z,prediction)
    require(!Base.get_zero_subnormals(),"flush-to-zero arithmetic is unsupported")
    require(nextfloat(0.0)!=0 && nextfloat(0.0)*1.0==nextfloat(0.0),"flush-to-zero arithmetic is unsupported")
    low=_prediction_enclosure(x,mu,scale,beta,z,prediction,256)
    high=_prediction_enclosure(x,mu,scale,beta,z,prediction,512)
    for field in (:reference,:normal_loss,:ED,:bound,:error)
        require(_inside(getproperty(high,field),getproperty(low,field)),"prediction interval refinement does not nest")
    end
    high
end

function _independent_design(X,recipe,preprocessing)
    _schema(preprocessing,"ridge_preprocessing")
    n,p=size(X);center=vec(mean(X;dims=1));scale=vec(std(X;dims=1))
    constants=Int[]
    for j in 1:p
        all(X[i,j]==X[1,j] for i in 1:n) && push!(constants,j)
        if recipe["std_rule"]=="sample_std_plus_1e_minus9"
            scale[j]=scale[j]+1e-9
        elseif !isfinite(scale[j]) || scale[j]==0
            scale[j]=1.0
        end
    end
    require(preprocessing["convention"]==recipe["std_rule"] && preprocessing["mean"]==float_bits.(center) &&
        preprocessing["scale"]==float_bits.(scale) && preprocessing["constant_columns"]==constants,"independent preprocessing reconstruction differs")
    Z=Matrix{Float64}(undef,n,p+1);Z[:,1].=1.0
    for j in 1:p,i in 1:n;Z[i,j+1]=(X[i,j]-center[j])/scale[j];end
    require(all(isfinite,Z),"independent normalization is nonfinite")
    Z
end

function _directed_system(Z,y,weights)
    n,p=size(Z);ZI=_iv.(Z);yi=_iv.(y)
    A=[_add(_sum(_mul(ZI[k,i],ZI[k,j]) for k in 1:n),_iv(i==j ? weights[i] : 0.0)) for i in 1:p,j in 1:p]
    b=[_sum(_mul(ZI[k,i],yi[k]) for k in 1:n) for i in 1:p]
    A,b
end
function _ridge_enclosure(Z,y,weights,beta,independent_qr,bits)
    require(!Base.get_zero_subnormals(),"ridge proof requires gradual underflow")
    setprecision(BigFloat,bits) do
        A,b=_directed_system(Z,y,weights);p=length(beta);n=length(y)
        midpointA=[(v.lo+v.hi)/2 for v in A];midpointb=[(v.lo+v.hi)/2 for v in b]
        V=inv(midpointA);require(all(isfinite,V),"independent inverse is nonfinite")
        VI=_iv.(V);AV=_matmul(A,VI)
        inverse_residual=_norminf([_sub(_iv(i==j ? 1 : 0),AV[i,j]) for i in 1:p,j in 1:p])
        require(inverse_residual.hi<1,"directed inverse-residual check is inconclusive")
        inverse_bound=_div(_norminf(VI),_sub(_iv(1),inverse_residual))
        # This theorem supplies an upper bound, not a lower bound on inv(A).
        inverse_norm=_Iv(BigFloat(0),inverse_bound.hi)
        condition=_mul(_norminf(A),inverse_norm)
        residual=_norminf([_sub(v,w) for (v,w) in zip(_matvec(A,_iv.(beta)),b)])
        denominator=_add(_mul(_norminf(A),_norminf(_iv.(beta))),_norminf(b))
        if denominator.hi==0
            require(all(==(0),beta) && all(v->v.lo==v.hi==0,b),"zero system does not have exact zero coefficients")
            return (;eta=_iv(0),condition,forward=_iv(0),residual)
        end
        require(denominator.lo>0,"backward-error denominator is not positive")
        # p here is the feature count; the design has m=p+1 coefficients.
        eta=_div(residual,denominator);u=_iv(BigFloat(2)^(-53));k=n+2*(p-1)+4
        gamma=_div(_mul(_iv(k),u),_sub(_iv(1),_mul(_iv(k),u)))
        require(eta.hi<=_mul(_iv(8),gamma).lo,"ridge backward error exceeds frozen bound")
        product=_mul(condition,eta);require(product.hi<1,"condition/backward-error product is not below1")
        forward=_div(_mul(_iv(2),product),_sub(_iv(1),product))
        require(forward.hi<=_down(()->BigFloat(1)/BigFloat(10)^6),"ridge forward-error enclosure exceeds frozen cap")
        # A second BigFloat solve is independently enclosed through its residual.
        independent=midpointA\midpointb
        independent_residual=_norminf([_sub(v,w) for (v,w) in zip(_matvec(A,_iv.(independent)),b)])
        independent_error=_mul(inverse_norm,independent_residual)
        exact_norm_lower=_down(()->maximum(abs,independent)-independent_error.hi)
        if exact_norm_lower<=0
            require(all(v->v.lo==v.hi==0,b) && all(==(0),beta),"independent zero case is inconclusive")
        else
            observed=_up(()->(maximum(abs.(BigFloat.(beta).-independent))+independent_error.hi)/exact_norm_lower)
            require(observed<=forward.hi,"native coefficient error is outside verified forward enclosure")
            qr_residual=_norminf([_sub(v,w) for (v,w) in zip(_matvec(A,_iv.(independent_qr)),b)])
            qr_denominator=_add(_mul(_norminf(A),_norminf(_iv.(independent_qr))),_norminf(b))
            require(qr_denominator.lo>0 && _div(qr_residual,qr_denominator).hi<=_mul(_iv(8),gamma).lo,"independent augmented QR fails residual gate")
        end
        (;eta,condition,forward,residual)
    end
end

function _draft_member(draft::_Draft,name)
    values=filter(p->first(p)==name,draft.members);require(length(values)==1,"draft member is absent/duplicated")
    decode_canonical(last(only(values)))
end

"""
    check_ridge_fit(draft, input; protocol)

Independently reconstruct preprocessing and augmented QR, then certify the saved
native coefficients with directed256/512-bit residual and inverse-residual
enclosures. An inconclusive numerical gate rejects the entire draft.
"""
function check_ridge_fit(draft::_Draft,input::FitInput;protocol::RefitProtocol)
    require(draft.kind=="refit_point" && draft.protocol.sha256==protocol.sha256==_bundle(input).protocol.sha256,"fit check identity differs")
    values=_input_values(input);recipe=_draft_member(draft,"recipe.json");pre=_draft_member(draft,"preprocessing.json")
    model=_draft_member(draft,"model.json");receipt=_draft_member(draft,"fit-receipt.json")
    _schema(recipe,"ridge_recipe");_schema(model,"ridge_model");_schema(receipt,"fit_receipt")
    require(receipt["input_manifest_sha256"]==_pin(input) && receipt["input_data_sha256"]==_context(input)["kind_context"]["input_data_sha256"],"fit receipt input differs")
    require(receipt["recipe_sha256"]==digest(_jsontext(recipe)) && receipt["preprocessing_sha256"]==digest(_jsontext(pre)) &&
        receipt["model_payload_sha256"]==digest(_jsontext(model)),"fit receipt numerical bytes differ")
    Z=_independent_design(values.X,recipe,pre);penalty=read_float_bits(recipe["penalty"])
    weights=vcat(recipe["intercept_penalty"] ? penalty : 0.0,fill(penalty,size(Z,2)-1))
    beta=read_float_bits.(_array(model["coefficients"]))
    require(model["kind"]=="ridge" && length(beta)==size(Z,2),"fit-check coefficient dimension differs")
    augmented=vcat(Z,Matrix(Diagonal(sqrt.(weights))))
    independent_qr=qr(augmented)\vcat(values.y,zeros(length(weights)))
    require(all(isfinite,independent_qr),"independent augmented QR is nonfinite")
    A=Z'*Z+Diagonal(weights);b=Z'*values.y
    statistics=_native_statistics_bytes(A,b,_spec(protocol)["layout"])
    require(receipt["diagnostic_payload_sha256"]==digest(statistics),"saved sufficient-statistic digest differs")
    require(length(draft.chunks)==1 && only(draft.chunks)==statistics,"immutable as-solved statistics snapshot differs")
    lower=_ridge_enclosure(Z,values.y,weights,beta,independent_qr,256)
    higher=_ridge_enclosure(Z,values.y,weights,beta,independent_qr,512)
    for field in (:eta,:condition,:forward,:residual)
        require(_inside(getproperty(higher,field),getproperty(lower,field)),"ridge audit refinement does not nest")
    end
    Dict{String,Any}("backward_eta"=>_interval_record(higher.eta,512),"condition_estimate"=>_interval_record(higher.condition,512),
        "forward_enclosure"=>_interval_record(higher.forward,512),"normal_residual"=>_interval_record(higher.residual,512))
end
