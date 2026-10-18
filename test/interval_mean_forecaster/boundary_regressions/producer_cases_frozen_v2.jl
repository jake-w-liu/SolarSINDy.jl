# Frozen literal producer-domain cases. Preserve v1 unchanged.
# Invalid producer inputs must raise ArgumentError before destination creation.
function producer_cases_frozen_v2()
    wide_names=Tuple(Symbol("field",i) for i in 1:129)
    wide=NamedTuple{wide_names}(ntuple(identity,129))
    deep=1
    for _ in 1:32
        deep=(deep,)
    end
    cases=NamedTuple[
        (label="valid_lowercase",value=(scope="synthetic marker",stage=:synthetic,probe=1.0),valid=true),
        (label="valid_uppercase",value=(UPPERCASE=:UPPERCASE,),valid=true),
        (label="valid_underscore",value=(_under_score_0=:_Under_0,),valid=true),
    ]
    # These token expectations are literal, not obtained from production regexes.
    for (label,token) in (("hyphen","probe-bad"),("LF","probe\n"),
        ("CRLF","probe\r\n"),("empty",""),("leading_digit","1probe"))
        push!(cases,(label="symbol_"*label,value=(probe=Symbol(token),),valid=false))
        push!(cases,(label="name_"*label,value=NamedTuple{(Symbol(token),)}(("synthetic marker",)),valid=false))
    end
    append!(cases,[
        (label="width129",value=wide,valid=false),
        (label="decode_depth33",value=(probe=deep,),valid=false),
        (label="member_over_1000000",value=(probe=repeat("a",1_000_001),),valid=false),
    ])
    cases
end
