module InitialAInputs

using Dates, SHA, SolarSINDy
using ..InitialReferenceImport
const I=InitialReferenceImport
const B=I.B
const F=SolarSINDy.IntervalMeanForecaster
const S=F.DstIntervalStream
const A=F.DstIntervalAdapter
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
const PARENT_LOADED_SHA256=getfield(parentmodule(@__MODULE__),:LOADED_SOURCE_SHA256)
const ACTUAL_SOURCE="593354e93e8f6c1486dc46de3ae96d4281d1129f9d5d39933347c2702122d248"
export PrefixInput, load_prefix, recheck, input_digest, code_identity

need(ok,message)=ok || throw(ArgumentError(message))
function code_identity()
    own=I.code_identity()
    for (name,expected) in (("InitialAInputs.jl",LOADED_SOURCE_SHA256),("ActualAChain.jl",PARENT_LOADED_SHA256))
        path=joinpath(@__DIR__,name)
        need(I.sha(path)==expected,"loaded A input adapter source changed")
        own["tools/actual_a_chain/"*name]=expected
    end
    own
end

"""Hash every time and raw value; this does not grant fitting permission."""
function input_digest(input::S.HourlyInputs)
    need(ENDIAN_BOM==0x04030201,"little-endian input execution required")
    count=length(input.time)
    need(1<=count<=500_000 && size(input.values)==(17,count),"bounded seventeen-channel input required")
    need(B.hex(input.source_sha256) && input.availability_kind==:assumed_completed_hour,"raw source/availability identity differs")
    context=SHA.SHA2_256_CTX()
    SHA.update!(context,codeunits("A-prefix-input-v1|"*input.source_sha256*"|"*string(count)*"|"))
    buffer=IOBuffer()
    for (index,t) in enumerate(input.time)
        need(t==A._shift(first(input.time),index-1),"raw prefix hour sequence changed")
        write(buffer,htol(Int64(Dates.value(t))))
        if index%4096==0;SHA.update!(context,take!(buffer));end
    end
    position(buffer)>0 && SHA.update!(context,take!(buffer))
    SHA.update!(context,reinterpret(UInt8,vec(input.values)))
    bytes2hex(SHA.digest!(context))
end

"""A mutable raw matrix protected by immutable input/proof digests.

This source adapter performs no fitting, state replay, selection or scoring.
Actual fit stages must additionally require their separately qualified permit.
"""
struct PrefixInput
    reference::I.ArtifactReference
    input::S.HourlyInputs
    evidence_mode::String
    source_rows_from_audit::Int
    parser_rows::Int
    parser_source_bytes::Int
    parsed_matrix_sha256::String
    prefix_matrix_sha256::String
    input_sha256::String
    own::Dict{String,String}
    sealed_sha256::String
end

function fingerprint(reference,input,mode,source_rows,parser_rows,source_bytes,parsed,prefix,digest,own)
    data=(reference.directory,reference.completion_sha256,reference.audit_completion_sha256,
        input.source_sha256,mode,source_rows,parser_rows,source_bytes,parsed,prefix,digest,
        Tuple(sort!(collect(own);by=first)))
    bytes2hex(sha256(repr(data)))
end
function prepared_plan(proof)
    path=B.resolved(proof.roots,"audit","prepared/audit-plan.json")
    B.json_file(path,proof.pinned[path],I.LIMITS)
end

function load_prefix(reference::I.ArtifactReference)
    own=code_identity()
    proof=I.materialize(reference).proof;meta=proof.meta;plan=prepared_plan(proof)
    n=Int(meta["prefix_rows"]);start=I.datehour(meta["prefix_start"])
    need(B.integer(plan["full_rows"],n) && plan["full_rows"]<=500_000,"bounded original-source rows required")
    mode=String(meta["evidence_mode"])
    need(ENDIAN_BOM==0x04030201,"little-endian raw matrix required")
    rawpath=B.resolved(proof.roots,"reference","prefix_raw17.f64")
    expected_prefix=String(proof.reference["output_sha256"]["prefix_raw17.f64"])
    need(filesize(rawpath)==17*n*8 && I.sha(rawpath)==expected_prefix,"pinned prefix matrix differs")
    source=String(meta["source_sha256"])
    parser_rows=n;source_bytes=filesize(rawpath);parsed_matrix=expected_prefix
    frame=if mode=="exposed_development_reconstruction"
        need(reference.admitted!==nothing && source==ACTUAL_SOURCE,"matching actual source admission required")
        admitted=reference.admitted;B.recheck(admitted)
        record=admitted.plan["support"]["raw"]
        path=B.resolved(admitted.roots,record["root"],record["path"])
        need(record["sha256"]==source && I.sha(path)==source,"actual raw parser input differs")
        need(I.datehour(meta["training_boundary"])==DateTime(2020) &&
             I.datehour(meta["validation_start"])==DateTime(2020) &&
             I.datehour(meta["validation_end"])==DateTime(2023),"locked actual A boundaries differ")
        full=F.read_author_hourly(path;expected_sha256=source)
        need(length(full.time)==430248 && length(full.time)==plan["full_rows"] &&
             first(full.time)==DateTime(1976,11,1) && last(full.time)==DateTime(2025,11,30,23),
             "actual parser coverage differs")
        parser_rows=length(full.time);source_bytes=filesize(path)
        bytes=reinterpret(UInt8,vec(full.values))
        parsed_matrix=bytes2hex(sha256(bytes))
        need(parsed_matrix==admitted.plan["expected"]["raw_hourly_matrix_sha256"] &&
             parsed_matrix==proof.reference["actual_full_raw_matrix_sha256"],"actual full raw matrix identity differs")
        need(bytes2hex(sha256(view(bytes,1:17*n*8)))==expected_prefix,"actual parser prefix differs from audited reference")
        need(first(full.time)==start && A._shift(full.time[n],1)==I.datehour(plan["selection_boundary"]),
             "actual prefix clock differs")
        B.recheck(admitted)
        S.HourlyInputs(full.time[1:n],full.values[:,1:n];source_sha256=source)
    else
        need(mode=="synthetic_only" && reference.admitted===nothing,"unknown generated input mode")
        words=reinterpret(Float64,read(rawpath))
        values=reshape(copy(words),17,n)
        times=[A._shift(start,i-1) for i in 1:n]
        S.HourlyInputs(times,values;source_sha256=source)
    end
    need(A._shift(last(frame.time),1)==I.datehour(plan["selection_boundary"]),"retained prefix stop differs")
    prefix=bytes2hex(sha256(reinterpret(UInt8,vec(frame.values))))
    need(prefix==expected_prefix,"retained raw prefix identity differs")
    digest=input_digest(frame)
    seal=fingerprint(reference,frame,mode,Int(plan["full_rows"]),parser_rows,source_bytes,parsed_matrix,prefix,digest,own)
    token=PrefixInput(reference,frame,mode,Int(plan["full_rows"]),parser_rows,source_bytes,parsed_matrix,prefix,digest,own,seal)
    recheck(token)
    token
end

function recheck(token::PrefixInput)
    need(code_identity()==token.own,"A input implementation identity changed")
    need(fingerprint(token.reference,token.input,token.evidence_mode,token.source_rows_from_audit,
        token.parser_rows,token.parser_source_bytes,token.parsed_matrix_sha256,token.prefix_matrix_sha256,
        token.input_sha256,token.own)==token.sealed_sha256,"A input metadata was mutated")
    need(input_digest(token.input)==token.input_sha256,"raw A prefix values/times changed")
    proof=I.verified_metadata(token.reference);plan=prepared_plan(proof)
    need(token.evidence_mode==proof.meta["evidence_mode"] && token.source_rows_from_audit==plan["full_rows"] &&
         length(token.input.time)==proof.meta["prefix_rows"] && token.input.source_sha256==proof.meta["source_sha256"],
         "A prefix and reference provenance differ")
    need(token.prefix_matrix_sha256==proof.reference["output_sha256"]["prefix_raw17.f64"] &&
         bytes2hex(sha256(reinterpret(UInt8,vec(token.input.values))))==token.prefix_matrix_sha256,
         "raw A matrix no longer matches the audited prefix")
    need(first(token.input.time)==I.datehour(proof.meta["prefix_start"]) &&
         A._shift(last(token.input.time),1)==I.datehour(plan["selection_boundary"]),"A prefix clock coverage changed")
    if token.evidence_mode=="exposed_development_reconstruction"
        need(token.parser_rows==430248 && token.parsed_matrix_sha256==proof.reference["actual_full_raw_matrix_sha256"],
             "actual parser/full-matrix proof differs")
        need(token.input.source_sha256==ACTUAL_SOURCE && I.datehour(proof.meta["training_boundary"])==DateTime(2020) &&
             I.datehour(proof.meta["validation_start"])==DateTime(2020) && I.datehour(proof.meta["validation_end"])==DateTime(2023),
             "actual source or locked boundaries differ")
        B.recheck(token.reference.admitted)
        raw=token.reference.admitted.plan["support"]["raw"]
        path=B.resolved(token.reference.admitted.roots,raw["root"],raw["path"])
        need(raw["sha256"]==token.input.source_sha256 && token.parser_source_bytes==filesize(path),
             "actual parser source-byte count differs")
    else
        need(token.parser_rows==length(token.input.time) && token.parsed_matrix_sha256==token.prefix_matrix_sha256,
             "generated prefix was mislabeled as a parsed full archive")
        need(token.parser_source_bytes==17*length(token.input.time)*8,"generated parser byte count differs")
    end
    I.guard_files(proof.pinned)
    need(code_identity()==token.own,"A input implementation changed during recheck")
    true
end

end # module
