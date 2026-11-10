"""Post-native, data-only issuer of measured generated full-shape evidence."""
module Matched342ResourceIssuer
include("run_matched342_full_shape_profile.jl")
const Profile=Matched342FullShapeProfile
using Dates,SHA,JSON3
const FC=Profile.FC;const T=Profile.T;const P=Profile.P;const Audit=Profile.Audit
const Entry=Profile.Entry;const M=Profile.M
const need=Profile.need;const fields=Profile.fields;const same=Profile.same
const ISSUER_FIELDS=(:schema,:operation,:head,:source_proof,:root_qualification,:component_qualification,:fixture_spec,
    :profile_result,:native,:raw,:prelaunch_admission,:measurement,:limits,:budgets,:output_directory,:request_id)
function validate_issuer_metadata(d)
    fields(d,ISSUER_FIELDS);FC._check_id(d,:request_id)
    need(d.schema=="matched342_resource_issuance_request_v1"&&d.operation=="issue_initial32_full_shape_resource_qualification"&&
        d.head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",d.head),"resource issuance request header differs")
    for k in (:source_proof,:root_qualification,:component_qualification,:profile_result,:native,:raw,:prelaunch_admission,:measurement)
        Profile.ref_grammar(getproperty(d,k))
    end
    fields(d.fixture_spec,(:member,:path,:sha256));need(d.fixture_spec.member==Profile.SPEC_MEMBER,"fixed fixture member required")
    Profile.ref_grammar((;path=d.fixture_spec.path,sha256=d.fixture_spec.sha256))
    Profile.validate_limits(d.limits);Profile.validate_budgets(d.budgets);Profile.absolute_grammar(d.output_directory);d
end
"Independent calendar calculation, with one-based source row ranges."
function derive_full_cells()
    origin=DateTime(1976,11,1)
    b=Int(Dates.value(DateTime(2020)-origin)÷3_600_000)
    v=Int(Dates.value(DateTime(2023)-origin)÷3_600_000)
    Tuple(begin
        training_rows=(1+L):(b-168-h-L);validation_rows=b:(v-168-h-L);union_rows=b:(v-168-4-L)
        need(!isempty(training_rows)&&!isempty(validation_rows)&&last(union_rows)+6<=500000,"full calendar out of bounds")
        (;delay_hours=L,horizon=h,training_rows,validation_rows,union_rows,training_count=length(training_rows),
            validation_count=length(validation_rows),union_count=length(union_rows))
    end for L in (0,1) for h in (4,6))
end
"Separate literal implementation of the frozen raw generator; no generator call."
function independent_value(channel,hour)
    j=T._int(channel,1,17);t=T._int(hour,0,499999)
    j==6 ? Float64((t÷16)%9-4) : j==8 ? Float64(350+4*((t÷8)%31)) :
        j==12 ? 1.0+Float64((t÷4)%16)/8.0 : j==15 ? -80.0+Float64(t%256)/4.0 : Float64((13*t+7*j)%16)/4.0
end
function check_full_geometry(g)
    fields(g,(:delay_hours,:horizon,:training_rows,:validation_rows,:union_rows,:training_count,:validation_count,:union_count))
    L=T._int(g.delay_hours,0,1);h=T._int(g.horizon,4,6);need(h in (4,6),"full horizon differs")
    expected=only(filter(x->x.delay_hours==L&&x.horizon==h,derive_full_cells()))
    for k in keys(expected)
        actual=getproperty(g,k);target=getproperty(expected,k)
        if k in (:training_rows,:validation_rows,:union_rows)
            need(actual isa UnitRange{Int}&&actual==target,"full ordered calendar range differs")
        else;T._int(actual,target,target)
        end
    end
    expected
end
function independently_scan_fixture(ref,controls)
    raw=FC._pinned(ref,controls;limit=68_000_000);need(filesize(raw.path)==68_000_000,"full fixture cannot be padded/small")
    dst=Vector{Float64}(undef,500000);ctx=SHA.SHA2_256_CTX();buf=Vector{UInt8}(undef,136*256);offset=0
    open(raw.path,"r") do io
        while !eof(io)
            n=readbytes!(io,buf,length(buf));need(n>0&&n%136==0,"full fixture boundary differs")
            SHA.update!(ctx,@view(buf[1:n]))
            for word in reinterpret(UInt64,@view(buf[1:n]))
                j=mod1(offset+1,17);t=offset÷17;expected=independent_value(j,t)
                need(ltoh(word)==reinterpret(UInt64,expected),"independent full fixture byte formula differs")
                j==15&&(dst[t+1]=expected);offset+=1
            end
        end
    end
    need(offset==8_500_000&&bytes2hex(SHA.digest!(ctx))==raw.sha256&&P._file_sha(raw.path)[1]==raw.sha256,"full fixture scan/pin differs")
    dst
end
function response_extrema(dst,geometry)
    check_full_geometry(geometry)
    need(dst isa AbstractVector{Float64}&&length(dst)==500000&&all(isfinite,dst),"exact full finite Dst source required")
    L=geometry.delay_hours;h=geometry.horizon;lo=Inf;hi=-Inf
    for i in geometry.training_rows
        y=dst[i+h]-dst[i-L];need(isfinite(y),"full response overflow");lo=min(lo,y);hi=max(hi,y)
    end
    need(lo<hi,"constant full-shape response rejected")
    (;response_min_f64_bits=Audit.bits(lo),response_max_f64_bits=Audit.bits(hi))
end
function check_full_encoder(o,g;expected_preprocessing_sha256,expected_training_design_sha256)
    check_full_geometry(g)
    fields(o,Audit.ENCODERKEYS)
    need(o.schema=="matched342_encoder_v1"&&o.feature_count==342&&o.feature_schema_sha256==M.FEATURE_SCHEMA_SHA256&&
        o.feature_contract_sha256==M.FEATURE_CONTRACT_SHA256&&o.training_design_sha256==expected_training_design_sha256&&
        T._hex(expected_training_design_sha256)&&T._sha(T._bytes(o))==expected_preprocessing_sha256,"saved full encoder identity differs")
    need(length(o.training_rows)==g.training_count&&all(T._int(x,1,500000)==y for (x,y) in zip(o.training_rows,g.training_rows)),"narrow/padded encoder training cohort rejected")
    need(length(o.finite_counts)==165&&length(o.imputation_f64_bits)==165&&length(o.center_f64_bits)==342&&length(o.scale_f64_bits)==342,"saved encoder dimensions differ")
    for n in o.finite_counts;T._int(n,0,g.training_count);end
    for x in (o.imputation_f64_bits...,o.center_f64_bits...);need(isfinite(Audit.f64(x)),"saved encoder finite bits differ");end
    for x in o.scale_f64_bits;need(Audit.f64(x)>0,"saved encoder scale must be finite positive");end
    o
end
function saved_model_facts(path,sha256,expected,setting)
    need(expected isa NamedTuple,"literal saved-model expected header required")
    fields(expected,(:schema,:feature_count,:feature_schema_sha256,:evotrees_version,:evotrees_source_sha256,:cell_delay,:cell_horizon,
        :setting_index,:max_depth,:nrounds,:nbins,:seed,:training_mask_sha256,:preprocessing_sha256,:input_admission_sha256,
        :feature_contract_sha256,:loss_contract_sha256))
    state,id=Audit.decode_model(path,sha256,expected,T._int(setting,1,8));splits=0
    for tree in state.trees[2:end]
        pending=[1]
        while !isempty(pending)
            i=pop!(pending)
            if tree.split[i]
                splits+=1;push!(pending,2i,2i+1)
            end
        end
    end
    need(splits>0,"full-shape model has no reachable nonbias split")
    (;model_id=id,nonbias_split_count=splits)
end
function check_measurement_metadata(m;head,proof_sha,spec_sha,fixture_sha,runtime_sha,pid,budgets,limits)
    fields(m,FC.MEASUREMENT_FIELDS);FC._check_id(m,:measurement_id);b=Profile.validate_budgets(budgets)
    Profile.validate_limits(limits);T._int(pid,1,typemax(Int64))
    need(head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",head)&&all(T._hex,(proof_sha,spec_sha,fixture_sha,runtime_sha)),"measurement outside binding grammar differs")
    need(m.schema=="matched342_julia_process_measurement_v1"&&m.complete===true&&m.observed===true&&m.evidence_mode=="synthetic_only"&&
        m.head==head&&m.source_proof_sha256==proof_sha&&m.fixture_spec_sha256==spec_sha&&m.fixture_sha256==fixture_sha&&
        m.runtime_sha256==runtime_sha&&T._int(m.julia_pid,pid,pid)==pid&&m.scope=="initial32_producer_and_independent_saved_state_audit"&&
        m.peak_rss_method=="Julia Sys.maxrss bytes; OS process-lifetime maximum"&&
        m.allocation_method=="Julia Base.@timed.bytes; GC_Diff.allocd managed cumulative bytes"&&
        m.elapsed_method=="Julia time_ns difference around complete timed body","measured Julia scope/provenance differs")
    T._int(m.peak_rss_bytes,1,b.peak_rss_bytes);T._int(m.julia_gc_allocated_bytes,1,b.julia_gc_allocated_bytes)
    T._int(m.elapsed_ns,1,limits.stage_wall_seconds*10^9-1);d=m.descriptors
    fields(d,(:kind,:method,:before,:after,:supported,:observed))
    allowed=Dict("windows_handles"=>"Windows GetProcessHandleCount on Julia current process",
        "linux"=>"Linux /proc/self/fd count on Julia process","macos"=>"macOS /dev/fd count on Julia process")
    need(d.supported===true&&d.observed===true&&((d.kind=="windows_handles"&&d.method==allowed["windows_handles"])||
        (d.kind=="posix_file_descriptors"&&d.method in (allowed["linux"],allowed["macos"]))),"descriptor method unsupported/unobserved")
    before=T._int(d.before,0,typemax(Int64));after=T._int(d.after,0,typemax(Int64));need(after-before<=b.descriptor_growth,"descriptor growth exceeded")
    m
end
function completion_summary(rawpath)
    found=Any[];prefix="MATCHED342_PROFILE_COMPLETE ";line=UInt8[]
    function consume()
        text=String(copy(line));endswith(text,"\r")&&(text=chop(text;tail=1))
        startswith(text,prefix)&&push!(found,P._bounded_json_parse(Vector{UInt8}(codeunits(text[length(prefix)+1:end]))))
        need(length(found)<=1,"duplicate profile completion line");empty!(line)
    end
    open(rawpath,"r") do io
        while !eof(io)
            block=read(io,65536)
            for byte in block
                if byte==0x0a;consume()
                else;push!(line,byte);need(length(line)<=T.MAX_META,"oversized native output line")
                end
            end
        end
    end
    isempty(line)||consume()
    need(length(found)==1,"one genuine profile completion line required");s=only(found)
    fields(s,(:profile_request,:julia_pid,:profile_result_sha256,:measurement_sha256,:fixture_sha256,:runtime_sha256,:source_proof_sha256))
    Profile.ref_grammar(s.profile_request);s
end
function model_expected(cell,s)
    g=M.GRID342[s]
    (;schema="matched_gbm_scalar342_model_v1",feature_count=342,feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,
        evotrees_version="0.18.7",evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,cell_delay=cell.delay_hours,
        cell_horizon=cell.horizon,setting_index=s,input_admission_sha256=cell.context_id,training_mask_sha256=cell.model_training_mask_sha256,
        preprocessing_sha256=cell.preprocessing_sha256,max_depth=g.max_depth,nrounds=g.nrounds,nbins=g.nbins,seed=22022026,
        feature_contract_sha256=M.FEATURE_CONTRACT_SHA256,loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256)
end
function main(request_path,request_sha)
    controls=Dict{String,String}();iref=FC._pinned((;path=request_path,sha256=request_sha),controls)
    d=FC._json_ref(iref,controls;fields=ISSUER_FIELDS);validate_issuer_metadata(d)
    sources=Entry.source_identity();proof=FC.recheck_source_proof(d.source_proof;expected_head=String(d.head),expected_sources=sources)
    FC._source_member("tools/actual_b_chain/issue_matched342_resource_qualification.jl",proof.source_sha256["tools/actual_b_chain/issue_matched342_resource_qualification.jl"],proof,controls)
    FC._root_qualification(d.root_qualification,String(d.head),String(d.source_proof.sha256),controls)
    FC._component_qualification(d.component_qualification,String(d.head),sources,controls;root_sha=String(d.root_qualification.sha256),proof_sha=String(d.source_proof.sha256),mode="synthetic_only")
    spec=FC._resource_spec((;fixture_spec=d.fixture_spec),proof,controls)
    native=FC._native_receipt(d.native,d.raw,d.limits,controls);summary=completion_summary(native.raw.path)
    original=Profile.read_profile_request(summary.profile_request.path;expected_sha256=summary.profile_request.sha256);o=original.data
    need(o.qualified_head==d.head&&same(o.source_proof,d.source_proof)&&same(o.component_qualification,d.component_qualification)&&
        same(o.fixture_spec,d.fixture_spec)&&same(o.limits,d.limits)&&same(o.budgets,d.budgets),"original raw-bound profile request/budgets differ")
    result=FC._json_ref(d.profile_result,controls;fields=Profile.RESULT_FIELDS);FC._check_id(result,:profile_id)
    need(result.schema=="matched342_full_shape_profile_result_v1"&&result.complete===true&&result.evidence_mode=="synthetic_only"&&
        result.head==d.head&&result.source_proof_sha256==d.source_proof.sha256&&result.fixture_spec_sha256==d.fixture_spec.sha256&&
        normpath(dirname(String(d.profile_result.path)))==o.output_directory&&same(result.measurement,d.measurement)&&
        summary.profile_result_sha256==d.profile_result.sha256&&summary.measurement_sha256==d.measurement.sha256&&
        summary.julia_pid==native.pid&&summary.fixture_sha256==result.fixture_data.sha256&&summary.runtime_sha256==result.runtime_sha256&&
        summary.source_proof_sha256==d.source_proof.sha256,"raw/result/original request linkage differs")
    runtime=FC._check_runtime(result.runtime);need(same(result.runtime,o.runtime)&&runtime.runtime_sha256==result.runtime_sha256,"profile runtime differs")
    pref=FC._ref(d.prelaunch_admission);need(pref.path==joinpath(String(o.owner_directory),"PRELAUNCH.json"),"profile original owner prelaunch path differs")
    FC._prelaunch(pref,String(d.head),String(d.source_proof.sha256),String(d.fixture_spec.sha256),runtime.runtime_sha256,d.limits,controls;mode=:synthetic_only,pid=native.pid)
    measure=FC._json_ref(d.measurement,controls;fields=FC.MEASUREMENT_FIELDS)
    check_measurement_metadata(measure;head=d.head,proof_sha=d.source_proof.sha256,spec_sha=d.fixture_spec.sha256,
        fixture_sha=result.fixture_data.sha256,runtime_sha=runtime.runtime_sha256,pid=native.pid,budgets=d.budgets,limits=d.limits)
    dst=independently_scan_fixture(result.fixture_data,controls)
    pair=Profile.admit_sources(result.sources;mode=:synthetic_only).pair
    need(all(x.raw_prefix_sha256==result.fixture_data.sha256 for x in FC.source_pair_metadata(pair).source_bindings),"admitted source pair differs from rescanned fixture")
    checked=Audit.read_component_audit(pair,String(result.component_audit.path),String(result.component_audit.sha256);
        expected_component_sha256=String(result.component.sha256),expected_sources=sources,expected_audit_contract_sha256=Profile.AUDIT_SHA)
    Audit.recheck_component_audit(checked)
    c=FC._json_ref(result.component,controls;fields=FC.COMPONENT_FIELDS);ca=FC._json_ref(result.component_audit,controls;fields=FC.CA_FIELDS)
    need(c.evidence_mode=="synthetic_only"&&ca.evidence_mode=="synthetic_only"&&length(c.cells)==4&&c.baseline_qualification_sha256==d.component_qualification.sha256,"generated component qualification differs")
    report=FC._json_ref((;path=joinpath(dirname(String(result.component_audit.path)),"audit-report.json"),sha256=String(ca.report_sha256)),controls;fields=FC.REPORT_FIELDS)
    root=dirname(String(result.component.path));cells=NamedTuple[];models=NamedTuple[]
    for (k,g) in enumerate(derive_full_cells())
        cell=c.cells[k];ac=report.cells[k];fields(cell,FC.COMPONENT_CELL_FIELDS);fields(ac,FC.AUDIT_CELL_FIELDS)
        need(cell.delay_hours==g.delay_hours&&cell.horizon==g.horizon&&length(cell.models)==8&&length(ac.models)==8,"full cell order differs")
        for row in (cell,ac);need(row.training_count==g.training_count&&row.validation_count==g.validation_count&&row.union_count==g.union_count&&row.validation_persistence_count==0,"small/narrow/full cohort differs");end
        epath=FC._member_record(root,cell.encoder_member,controls;limit=4*1024^2)
        encoder,_=Audit.readjson(epath;expected=String(cell.encoder_member.sha256),names=Audit.ENCODERKEYS,limit=4*1024^2)
        check_full_encoder(encoder,g;expected_preprocessing_sha256=cell.preprocessing_sha256,expected_training_design_sha256=cell.training_design_sha256)
        extrema=response_extrema(dst,g)
        push!(cells,(;delay_hours=g.delay_hours,horizon=g.horizon,context_id=String(cell.context_id),training_count=g.training_count,
            validation_count=g.validation_count,union_count=g.union_count,extrema...,encoder_member=cell.encoder_member,
            fit_input_sha256=String(cell.fit_input_sha256),model_training_mask_sha256=String(cell.model_training_mask_sha256),
            reference_training_mask_sha256=String(cell.reference_training_mask_sha256),reference_validation_cohort_sha256=String(cell.reference_validation_cohort_sha256)))
        for s in 1:8
            cm=cell.models[s];am=ac.models[s];mpath=FC._member_record(root,cm.model_member,controls;limit=4*1024^2)
            facts=saved_model_facts(mpath,String(cm.model_member.sha256),model_expected(cell,s),s)
            receipt=FC._json_ref((;path=FC._member_record(root,cm.fit_receipt_member,controls),sha256=String(cm.fit_receipt_member.sha256)),controls;fields=FC.FIT_RECEIPT_FIELDS)
            FC._check_id(receipt,:receipt_id);T._int(receipt.fit_elapsed_ns,1,d.limits.fit_wall_seconds*10^9-1)
            need(receipt.complete===true&&receipt.evidence_mode=="synthetic_only"&&receipt.execution_authority_sha256===nothing&&
                receipt.runtime_sha256==runtime.runtime_sha256&&receipt.context_id==cell.context_id&&receipt.setting_index==s&&
                receipt.delay_hours==g.delay_hours&&receipt.horizon==g.horizon&&receipt.union_count==g.union_count&&
                receipt.model_id==facts.model_id==cm.model_id==am.model_id&&receipt.model_sha256==cm.model_member.sha256&&
                receipt.encoder_sha256==cell.encoder_member.sha256&&receipt.fit_input_sha256==cell.fit_input_sha256,"genuine saved fit receipt differs")
            need(cm.native_increment_sha256==cm.reloaded_increment_sha256==receipt.native_increment_sha256==receipt.reloaded_increment_sha256==
                am.native_increment_sha256==am.reloaded_increment_sha256==am.audited_increment_sha256,"full native/reloaded/audited bits differ")
            push!(models,(;delay_hours=g.delay_hours,horizon=g.horizon,setting_index=s,model_id=facts.model_id,model_member=cm.model_member,
                fit_receipt_member=cm.fit_receipt_member,encoder_sha256=String(cell.encoder_member.sha256),training_count=g.training_count,
                validation_count=g.validation_count,union_count=g.union_count,fit_input_sha256=String(cell.fit_input_sha256),
                nonbias_split_count=facts.nonbias_split_count,native_increment_sha256=String(cm.native_increment_sha256),
                reloaded_increment_sha256=String(cm.reloaded_increment_sha256),audited_increment_sha256=String(am.audited_increment_sha256)))
        end
    end
    need(length(models)==32&&length(unique(x.model_id for x in models))==32,"all32 distinct saved fitted models required")
    forbidden=[Profile.REPO,String(o.output_directory),String(o.owner_directory),iref.path,String(d.source_proof.path)]
    output=Profile.fresh_directory(String(d.output_directory),forbidden)
    process=Profile.identified((;schema="matched342_measured_julia_process_resources_v1",complete=true,evidence_mode="synthetic_only",head=String(d.head),
        source_proof_sha256=String(d.source_proof.sha256),fixture_spec_sha256=String(d.fixture_spec.sha256),fixture_sha256=String(result.fixture_data.sha256),
        runtime_sha256=runtime.runtime_sha256,julia_pid=native.pid,julia_binary_sha256=runtime.runtime.julia_binary_sha256,
        profiler=(;member=Profile.MEMBER,sha256=String(proof.source_sha256[Profile.MEMBER])),prelaunch_admission=d.prelaunch_admission,
        native=d.native,raw=d.raw,measurement=d.measurement,budgets=d.budgets),:record_id)
    FC._check_controls(controls);FC._check_controls(original.controls)
    pr=Profile.write_json(joinpath(output,"process-resources.json"),process)
    readback=Profile.identified((;schema="matched342_initial_full_shape_readback_v1",complete=true,evidence_mode="synthetic_only",head=String(d.head),
        implementation_sources=o.implementation_sources,source_proof_sha256=String(d.source_proof.sha256),fixture_spec_sha256=String(d.fixture_spec.sha256),
        fixture_sha256=String(result.fixture_data.sha256),fixture_data=result.fixture_data,component=result.component,component_audit=result.component_audit,
        runtime=runtime.runtime,runtime_sha256=runtime.runtime_sha256,cells=Tuple(cells),models=Tuple(models),encoder_count=4,fit_count=32,
        native=d.native,raw=d.raw,process_resources=pr),:readback_id)
    rb=Profile.write_json(joinpath(output,"readback.json"),readback)
    q=(;schema="matched342_initial_full_shape_resource_qualification_v1",complete=true,evidence_mode="synthetic_only",head=String(d.head),
        implementation_sources=o.implementation_sources,source_proof_sha256=String(d.source_proof.sha256),fixture_sha256=String(result.fixture_data.sha256),
        fixture_spec=d.fixture_spec,source_rows=500000,feature_count=342,cell_order=((0,4),(0,6),(1,4),(1,6)),
        cells=Tuple((;delay_hours=g.delay_hours,horizon=g.horizon,training_count=g.training_count,validation_count=g.validation_count,union_count=g.union_count) for g in derive_full_cells()),
        native_fit_count=32,encoder_count=4,limits=d.limits,native=d.native,raw=d.raw,readback=rb,process_resources=pr)
    # The final qualification name is installed only after compatibility and all rechecks.
    FC._check_controls(controls);Audit.recheck_component_audit(checked)
    pending=Profile.write_json(joinpath(output,"QUALIFICATION_PENDING.json"),q);compat=copy(controls)
    FC._resource_qualification(pending,String(d.head),sources,proof,String(d.source_proof.sha256),String(d.root_qualification.sha256),d.limits,compat)
    FC.recheck_source_proof(d.source_proof;expected_head=String(d.head),expected_sources=sources);FC._check_controls(controls)
    FC._check_controls(compat);final=joinpath(output,"RESOURCE_QUALIFICATION.json");need(!ispath(final),"qualification final already exists")
    mv(pending.path,final;force=false);saved=(;path=final,sha256=pending.sha256)
    println("MATCHED342_RESOURCE_QUALIFICATION_COMPLETE ",JSON3.write(saved));flush(stdout);nothing
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==2||error("issuer requires pinned request path and SHA")
    Matched342ResourceIssuer.main(ARGS...)
end
