module ReceiptMaturityIntegrationTests
using Dates,SHA,CSV,DataFrames,JSON3,Test
const ROOT=mktempdir(;cleanup=false)
include(joinpath(@__DIR__,"..","examples","live_receipt_maturity.jl"))
const E=LiveReceiptMaturity.LiveObservationEvidence
const R=LiveReceiptMaturity
const MANIFEST=R.CURRENT_MANIFEST
const MODEL=R.CURRENT_MODEL
const ORIGIN=DateTime(2020,1,1,9)
const T=DateTime(2020,1,1,10)
function synthetic_row(issue,target,step,point;observed=missing,manifest=MANIFEST)
    (;issue_time_utc=string(issue),latest_dst_time_utc=string(ORIGIN),target_time_utc=string(target),
        model_step_hours=step,sub_hourly_model_version=MODEL,superlearner_manifest_sha256=manifest,
        served_pred_dst_nt=Float64(point),served_pred_dst_ci05_nt=Float64(point-10),served_pred_dst_ci95_nt=Float64(point+10),
        observation_dst_nt=observed,persistence_dst_nt=Float64(point-1),burton_dst_nt=Float64(point-2),
        burton_full_dst_nt=Float64(point-3),obrien_dst_nt=Float64(point-4),sindy_core_pred_dst_nt=Float64(point-5))
end

function fixture()
    root=mktempdir(ROOT;prefix="synthetic-",cleanup=false);log=joinpath(root,"log.csv");store=joinpath(root,"store")
    table=DataFrame([
        synthetic_row(ORIGIN+Minute(30),T,1,10),
        synthetic_row(ORIGIN+Minute(30),T+Hour(1),2,-3;observed=-4.),
        synthetic_row(ORIGIN+Minute(30),T+Hour(2),3,1),
        synthetic_row(ORIGIN+Minute(40),T+Hour(1),2,-20;manifest=repeat("b",64))])
    table.observation_dst_nt=Union{Missing,Float64}[missing,-4.,missing,missing]
    CSV.write(log,table)
    (;root,log,store,table)
end
function capture_commit!(f,row,value,received;verified=received+Millisecond(2),write_csv=true)
    body=JSON3.write([Dict("time_tag"=>string(DateTime(f.table.target_time_utc[row]))*"Z","dst"=>value)])
    reference=E.capture_response!(f.store,(;status=200,body,headers=["x-fixture"=>"explicit synthetic mock; not historical evidence"]);
        request_started_utc=received,receipt_completed_utc=received,
        monotonic_started_ns=UInt64(Dates.value(received)),monotonic_completed_ns=UInt64(Dates.value(received)),
        consumer_source_sha256=repeat("c",64))
    prior=ismissing(f.table.observation_dst_nt[row]) ? nothing : f.table.observation_dst_nt[row]
    binding=E.forecast_binding(f.table[row,:])
    intent=E.prepare_verification!(f.store,reference,[(;binding,previous_observation=prior,observation=value)];
        log_path=f.log,verification_started_utc=verified-Millisecond(1))
    if write_csv
        f.table.observation_dst_nt[row]=value;CSV.write(f.log,f.table)
    else
        @test prior==value
    end
    E.commit_verification!(f.store,intent;log_path=f.log,verification_completed_utc=verified)
end
report(f,asof;kwargs...)=R.read_owned_snapshot(f.log,f.store;owned_root=f.root,asof,kwargs...)

@testset "Independent manual synthetic clock and metric expectations" begin
    f=fixture();original=read(f.log)
    missing_journal=report(f,T+Hour(1))
    @test missing_journal.counts["legacy_maturity_unknown"]==1
    @test missing_journal.pending==3
    @test !missing_journal.journal_available
    @test !isdir(f.store)
    @test missing_journal.qualified_scientific_support==0 && !missing_journal.claim_gate_ready
    @test R.nominal_end(T)==T+Hour(1)
    first=capture_commit!(f,1,15.,T+Hour(1)-Millisecond(1);verified=T+Hour(1)+Millisecond(1))
    early=report(f,T+Hour(1)+Millisecond(2))
    @test early.counts["preliminary_actual_receipt"]==1
    @test early.counts["completed_actual_receipt"]==0
    @test early.counts["legacy_maturity_unknown"]==1
    @test early.completed_primary_current_identity.n==0
    @test early.records[1].latest_eligible_event_key!=""
    @test early.records[1].latest_eligible_capture_provenance!==nothing
    @test report(f,T+Year(10)).counts==early.counts
    @test report(f,T+Year(10)).completed_primary_current_identity.n==0
    before_same=read(f.log)
    exact=capture_commit!(f,1,15.,T+Hour(1);verified=T+Hour(1)+Millisecond(4),write_csv=false)
    @test read(f.log)==before_same
    equal_asof=report(f,T+Hour(1)+Millisecond(4))
    @test equal_asof.counts["preliminary_actual_receipt"]==1
    @test equal_asof.counts["completed_actual_receipt"]==0
    complete=report(f,T+Hour(1)+Millisecond(5))
    @test complete.counts["completed_actual_receipt"]==1
    @test complete.counts["preliminary_actual_receipt"]==0
    @test complete.counts["legacy_maturity_unknown"]==1
    @test complete.pending==2
    @test complete.completed_primary_current_identity.n==1
    @test complete.completed_primary_current_identity.rmse==5.
    @test complete.completed_primary_current_identity.mae==5.
    @test complete.completed_primary_current_identity.bias==5.
    @test complete.completed_primary_current_identity.coverage_90==1.
    @test complete.completed_primary_current_identity.mean_width==20.
    @test complete.completed_primary_current_identity.mean_interval_score_90==20.
    @test all(!r.issuance_source_qualified && !r.scientific_claim_eligible for r in complete.records)
    @test !complete.claim_gate_ready && complete.qualified_scientific_support==0
    primary=complete.records[1].primary_event_key
    duplicate=capture_commit!(f,1,15.,T+Hour(1)+Minute(1);write_csv=false)
    after_duplicate=report(f,T+Hour(1)+Minute(2))
    @test after_duplicate.completed_primary_current_identity.n==1
    @test after_duplicate.records[1].primary_event_key==primary
    @test after_duplicate.records[1].completed_event_count==2
    @test after_duplicate.completed_primary_current_identity.rmse==5.
    capture_commit!(f,1,18.,T+Hour(2))
    after_revision=report(f,T+Hour(2)+Minute(1))
    @test after_revision.records[1].primary_event_key==primary
    @test after_revision.records[1].primary_observation==15.
    @test after_revision.records[1].latest_completed_observation==18.
    @test after_revision.completed_primary_current_identity.rmse==5.
    @test after_revision.completed_latest_revision_current_identity.rmse==8.
    @test after_revision.descriptive_current_served_label.rmse≈sqrt((64+1)/2)
    @test after_revision.descriptive_current_served_label.n==2
    @test after_revision.descriptive_exact_identity.n==2
    past=report(f,T+Hour(1)+Millisecond(5))
    @test past.completed_primary_current_identity==complete.completed_primary_current_identity
    @test past.records[1].primary_event_key==primary
    capture_commit!(f,4,-18.,T+Hour(2)+Minute(1))
    foreign=report(f,T+Hour(2)+Minute(2))
    @test foreign.counts["completed_actual_receipt"]==2
    @test foreign.completed_primary_all_identities.n==2
    @test foreign.completed_primary_all_identities.rmse≈sqrt((25+4)/2)
    @test foreign.completed_primary_current_identity.n==1
    @test !foreign.records[4].model_identity_matches
    @test foreign.records[4].category=="completed_actual_receipt"
    @test foreign.counts["legacy_maturity_unknown"]==1
    @test foreign.pending==1
    @test sum(values(foreign.counts))+foreign.pending==4
    for comparator in foreign.completed_primary_current_matched_comparators
        @test comparator.served.n==comparator.comparator.n==1
        @test comparator.matched_keys==foreign.completed_primary_current_identity.forecast_keys
        @test isempty(comparator.missing_keys)
    end
    @test !foreign.claim_gate_ready
    @test foreign.manifest==MANIFEST
    @test occursin("057",foreign.claim_gate_reason)
    @test read(f.log)!=original
end

@testset "Independent invalid evidence and unsupported mutation controls" begin
    f=fixture();capture_commit!(f,1,15.,T+Hour(1))
    asof=T+Hour(2);baseline=report(f,asof);csvbytes=read(f.log);head=read(joinpath(f.store,"head.json"))
    headpath=joinpath(f.store,"head.json");savedhead=joinpath(f.root,"preserved_head.json")
    mv(headpath,savedhead;force=false)
    @test_throws ArgumentError report(f,asof)
    mv(savedhead,headpath;force=false)
    @test_throws ArgumentError R.read_owned_snapshot(f.log,f.store;owned_root=joinpath(f.root,"other"),asof)
    @test_throws ArgumentError report(f,asof;before_recheck=()->write(f.log,vcat(csvbytes,UInt8(0x20))))
    write(f.log,csvbytes)
    @test_throws ArgumentError report(f,asof;before_recheck=()->write(joinpath(f.store,"head.json"),vcat(head,UInt8(0x20))))
    write(joinpath(f.store,"head.json"),head)
    f.table.observation_dst_nt[1]=99.;CSV.write(f.log,f.table)
    @test_throws ArgumentError report(f,asof)
    write(f.log,csvbytes);f.table.observation_dst_nt[1]=15.
    f.table.served_pred_dst_nt[1]+=1;CSV.write(f.log,f.table)
    @test_throws ArgumentError report(f,asof)
    write(f.log,csvbytes);f.table.served_pred_dst_nt[1]-=1
    raw=only(readdir(joinpath(f.store,"raw");join=true));rawbytes=read(raw);write(raw,vcat(rawbytes,UInt8(0x20)))
    @test_throws ArgumentError report(f,asof)
    write(raw,rawbytes)
    duplicate=vcat(f.table,f.table[2:2,:]);CSV.write(f.log,duplicate)
    dupreport=report(f,asof)
    @test dupreport.counts["invalid_or_conflicting_evidence"]==2
    @test dupreport.counts["completed_actual_receipt"]==1
    @test sum(values(dupreport.counts))+dupreport.pending==5
    write(f.log,csvbytes)
    f.table.observation_dst_nt[2]=NaN;CSV.write(f.log,f.table)
    nanreport=report(f,asof)
    @test nanreport.counts["invalid_or_conflicting_evidence"]==1
    @test !nanreport.claim_gate_ready
    f.table.observation_dst_nt[2]=-4.;write(f.log,csvbytes)
    f.table.persistence_dst_nt[1]=NaN;CSV.write(f.log,f.table)
    missingcomparison=report(f,asof)
    cmp=only(c for c in missingcomparison.completed_primary_current_matched_comparators if c.method=="persistence")
    @test cmp.served.n==cmp.comparator.n==0
    @test cmp.missing_keys==baseline.completed_primary_current_identity.forecast_keys
    f.table.persistence_dst_nt[1]=9.;write(f.log,csvbytes)
    body=JSON3.write([Dict("time_tag"=>string(T)*"Z","dst"=>15.)])
    ref=E.capture_response!(f.store,(;status=200,body,headers=[]);request_started_utc=asof,
        receipt_completed_utc=asof,monotonic_started_ns=UInt64(Dates.value(asof)),monotonic_completed_ns=UInt64(Dates.value(asof)))
    binding=E.forecast_binding(f.table[1,:])
    E.prepare_verification!(f.store,ref,[(;binding,previous_observation=15.,observation=15.)];
        log_path=f.log,verification_started_utc=asof+Millisecond(1))
    @test_throws ArgumentError report(f,asof+Minute(1))
end

@testset "Production snapshot is read only and detects competing source changes" begin
    f=fixture();capture_commit!(f,1,15.,T+Hour(1));asof=T+Hour(2)
    before=R._inventory(f.store;max_files=10000,max_bytes=10000000,max_member_bytes=10000000);csv=read(f.log)
    actual=R.snapshot_report(f.log,f.store;asof,snapshot_parent=ROOT)
    @test actual.payload.counts["completed_actual_receipt"]==1
    @test actual.payload.completed_primary_current_identity.rmse==5.
    @test R._inventory(f.store;max_files=10000,max_bytes=10000000,max_member_bytes=10000000)==before
    @test read(f.log)==csv
    @test actual.snapshot_root!=f.root && isdir(actual.snapshot_root)
    claim=R.trusted_claim_cohort(actual)
    @test claim.manifest_sha256==R.LEGACY_CLAIM_MANIFEST
    @test claim.support==0 && !claim.gate_ready && !claim.identity_retargeted
    @test isempty(claim.forecast_keys) && isempty(claim.observations)
    @test_throws MethodError R.trusted_claim_cohort(actual;issuance_ledger=Dict("fake"=>true))
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,snapshot_parent=f.store)
    if Sys.iswindows()
        aliased=lowercase(f.store)
        @test isdir(aliased)
        @test R._inside(aliased,f.store)
        before_alias=R._inventory(f.store;max_files=10000,max_bytes=10000000,max_member_bytes=10000000)
        @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,snapshot_parent=aliased)
        @test R._inventory(f.store;max_files=10000,max_bytes=10000000,max_member_bytes=10000000)==before_alias
        output_alias=joinpath(aliased,"new_output")
        @test_throws ArgumentError R.publish_report(actual,output_alias)
        @test !ispath(output_alias)
        @test !R._inside(f.store*"_sibling",f.store)
    end
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_files=true)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_bytes=0)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_member_bytes=1)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_bytes=length(csv)-1)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_files=1)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_entries=1)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,max_entries=false)
    manydirs=joinpath(f.root,"manydirs");mkdir(manydirs)
    for i in 1:12;mkdir(joinpath(manydirs,"directory-$i"));end
    @test_throws ArgumentError R._inventory(manydirs;max_files=4,max_bytes=1000,max_member_bytes=100)
    inventory=R._inventory(manydirs;max_files=4,max_entries=12,max_bytes=1000,max_member_bytes=100)
    @test inventory.entries==length(inventory.directories)==12
    @test isempty(inventory.files)
    bounded=joinpath(f.root,"bounded-read.bin")
    write(bounded,UInt8[1,2,3,4,5,6,7,8])
    @test R._bounded_read(bounded,8)==UInt8[1,2,3,4,5,6,7,8]
    write(bounded,UInt8[1,2,3,4,5,6,7,8,9])
    @test_throws ArgumentError R._bounded_read(bounded,8)
    hook=stage->stage==:after_copy && write(f.log,vcat(csv,UInt8(0x20)))
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,snapshot_hook=hook)
    write(f.log,csv)
    headpath=joinpath(f.store,"head.json");head=read(headpath)
    hook=stage->stage==:after_inventory && write(headpath,vcat(head,UInt8(0x20)))
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,snapshot_hook=hook)
    write(headpath,head)
    hook=stage->stage==:before_source_recheck && write(headpath,vcat(head,UInt8(0x20)))
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof,snapshot_hook=hook)
    write(headpath,head)
    busy=joinpath(f.store,"journal.lock");mkdir(busy)
    @test_throws ArgumentError R.snapshot_report(f.log,f.store;asof)
    rm(busy)
    if Sys.iswindows()
        # Native file-symlink creation failed EPERM in the retained before run.
        # This distinct unprivileged directory-junction fixture covers linked parents only.
        targetdir=joinpath(f.root,"owned_target");mkdir(targetdir)
        targetfile=joinpath(targetdir,"log.csv");write(targetfile,csv)
        junction=joinpath(f.root,"owned_junction")
        @test startswith(targetdir,f.root*string(Base.Filesystem.path_separator))
        @test startswith(junction,f.root*string(Base.Filesystem.path_separator))
        quote_ps(x)="'"*replace(x,"'"=>"''")*"'"
        script="New-Item -ItemType Junction -Path "*quote_ps(junction)*" -Target "*quote_ps(targetdir)*" | Out-Null"
        run(Cmd(["C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe","-NoProfile","-NonInteractive","-Command",script]))
        @test isdir(junction)
        @test_throws ArgumentError R.snapshot_report(joinpath(junction,"log.csv"),f.store;asof)
        @test read(targetfile)==csv
        println("Native directory-junction guard exercised; file-symlink branch remains unverified on this Windows EPERM environment.")
        # Keep the owned junction for inspection; never recursively remove/traverse it.
    else
        source_link=joinpath(f.root,"linked.csv");symlink(f.log,source_link)
        @test_throws ArgumentError R.snapshot_report(source_link,f.store;asof)
        rm(source_link)
    end
    @test read(f.log)==csv
    @test R._inventory(f.store;max_files=10000,max_bytes=10000000,max_member_bytes=10000000)==before
    absent=fixture();missingroot=joinpath(absent.root,"not_existing")
    absent_report=R.snapshot_report(absent.log,missingroot;asof)
    @test absent_report.payload.counts["legacy_maturity_unknown"]==1
    @test !ispath(missingroot)
end

@testset "Exclusive new publication and fixed empty057 claim cohort" begin
    f=fixture();capture_commit!(f,1,15.,T+Hour(1));asof=T+Hour(2)
    actual=R.snapshot_report(f.log,f.store;asof)
    parent=mktempdir(ROOT);destination=joinpath(parent,"report")
    publication=R.publish_report(actual,destination)
    @test R.verify_publication(destination;expected_completion_sha256=publication.completion_sha256)
    @test isfile(joinpath(destination,"COMPLETED.json"))
    @test JSON3.read(read(joinpath(destination,"legacy057_claim_cohort.json"))).support==0
    @test_throws ArgumentError R.publish_report(actual,destination)
    @test_throws ArgumentError R.publish_report(actual,joinpath(f.store,"report"))
    @test_throws ArgumentError R.verify_publication(destination;expected_completion_sha256=repeat("0",64))
    markdown=joinpath(destination,"report.md");original=read(markdown);write(markdown,vcat(original,UInt8(0x20)))
    @test_throws ArgumentError R.verify_publication(destination)
    write(markdown,original)
    @test R.verify_publication(destination)
    contested=joinpath(parent,"competitor")
    @test_throws Base.IOError R.publish_report(actual,contested;before_claim=()->write(contested,"first writer"))
    @test read(contested,String)=="first writer"
    @test !isdir(contested)
    broken=joinpath(parent,"incomplete");mkdir(broken)
    @test_throws ArgumentError R.verify_publication(broken)
    claimpath=joinpath(destination,"legacy057_claim_cohort.json");completionpath=joinpath(destination,"COMPLETED.json")
    claimbytes=read(claimpath);completionbytes=read(completionpath)
    function rebound_claim!(edit)
        claim=JSON3.read(claimbytes,Dict{String,Any});edit(claim);write(claimpath,JSON3.write(claim))
        completion=JSON3.read(completionbytes,Dict{String,Any})
        completion["outputs"]["legacy057_claim_cohort.json"]=R.hashfile(claimpath)
        write(completionpath,JSON3.write(completion))
    end
    rebound_claim!(c->c["forecast_keys"]=["invented_forecast"])
    @test_throws ArgumentError R.verify_publication(destination)
    rebound_claim!(c->c["observations"]=[1.])
    @test_throws ArgumentError R.verify_publication(destination)
    rebound_claim!(c->c["source_csv_sha256"]=repeat("0",64))
    @test_throws ArgumentError R.verify_publication(destination)
    write(claimpath,claimbytes);write(completionpath,completionbytes)
    @test R.verify_publication(destination)
    actual.payload.counts["completed_actual_receipt"]=999
    @test_throws ArgumentError R.trusted_claim_cohort(actual)
    @test_throws ArgumentError R.publish_report(actual,joinpath(parent,"mutated"))
end

@testset "Read-only CLI options and completed output" begin
    include(joinpath(@__DIR__,"..","examples","live_receipt_maturity_report.jl"))
    @test receipt_maturity_cli(["--help"])==0
    @test_throws ArgumentError receipt_maturity_cli(String[])
    @test_throws ArgumentError receipt_maturity_cli(["--log","x","--log","x"])
    @test_throws ArgumentError receipt_maturity_cli(["--unknown","x"])
    f=fixture();capture_commit!(f,1,15.,T+Hour(1));csv=read(f.log)
    output=joinpath(mktempdir(ROOT),"cli-report")
    @test receipt_maturity_cli(["--log",f.log,"--evidence-root",f.store,"--output",output,"--as-of",string(T+Hour(2))*"Z"])==0
    @test R.verify_publication(output)
    @test read(f.log)==csv
end

end
