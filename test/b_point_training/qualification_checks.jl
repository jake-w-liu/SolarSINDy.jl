# All qualification fixtures below are parser-only synthetic metadata.
# No actual execution permission or component qualification is created.
function qualification_checks(root)
data=plain(JSON3.read(read(joinpath(root,"matched-good","qualification.json"))));head=data["head"];sources=T._hashmap(pairs(data["source_sha256"]))
body(x)=collect(codeunits(JSON3.write(x)*"\n"))
check(bytes)=P._software_qualification(bytes;schema="b_reference_component_software_qualification_v1",evidence_mode="synthetic_only",head,sources)
@test check(body(data)).complete===true
pretty=IOBuffer();JSON3.pretty(pretty,data);@test check(take!(pretty)).complete===true
point=deepcopy(data);point["schema"]="b_point_execution_software_qualification_v1";point["evidence_mode"]="software_qualified";point["source_sha256"]=Dict(T._hashmap(P.source_identity()))
@test P._software_qualification(body(point);schema="b_point_execution_software_qualification_v1",evidence_mode="software_qualified",head,sources=T._hashmap(P.source_identity())).complete===true
@test_throws ArgumentError check(body(point))
@test_throws ArgumentError P._software_qualification(body(data);schema="b_point_execution_software_qualification_v1",evidence_mode="software_qualified",head,sources=T._hashmap(P.source_identity()))
changes=["schema"=>"wrong","complete"=>false,"evidence_mode"=>"software_qualified","head"=>repeat("b",40),"source_sha256"=>Dict("contracts/features.json"=>repeat("0",64)),"root_qualification_sha256"=>repeat("0",64),"source_proof_sha256"=>repeat("0",64),"canonical_harness_sha256"=>repeat("0",64),"software_qualification_only"=>false,"actual_panel_fit_or_outcome_qualified"=>true,"serving_promotion"=>true]
results=NamedTuple[]
function negative(label,changed)
    rejected=false;error=""
    encoded=body(changed);P._exclusive(joinpath(root,"mutant-$label.json"),encoded)
    try check(encoded) catch e;rejected=e isa ArgumentError;error=sprint(showerror,e);end
    push!(results,(;label,rejected,error));write(joinpath(root,"progress.json"),JSON3.write((;positive_wrapper=true,positive_point_metadata=true,namespace_cross_rejected=true,results))*"\n");@test rejected
end
for (field,value) in changes;x=deepcopy(data);x[field]=value;negative(field,x);end
for (label,mutate) in (
    "extra_root"=>(x->(x["extra"]=0)),
    "extra_harness"=>(x->(x["harness"]["extra"]=0)),
    "extra_counts"=>(x->(x["harness"]["counts"]["extra"]=0)),
    "all17_false"=>(x->(x["harness"]["all17_independent_workers_complete"]=false)),
    "Pkg_false"=>(x->(x["harness"]["unfiltered_pkg_test_complete"]=false)),
    "experiments_false"=>(x->(x["harness"]["experiments_entry_complete"]=false)),
    "PASS_Bool"=>(x->(x["harness"]["counts"]["PASS"]=false)),
    "PASS_Float"=>(x->(x["harness"]["counts"]["PASS"]=438.)),
    "WARN_negative"=>(x->(x["harness"]["counts"]["WARN"]=-1)),
    "FAIL_Float"=>(x->(x["harness"]["counts"]["FAIL"]=0.)),
    "missing_native"=>(x->pop!(x["native_jobs"])),
    "duplicate_native"=>(x->(x["native_jobs"][2]["name"]=x["native_jobs"][1]["name"])),
    "failed_native"=>(x->(x["native_jobs"][1]["conclusion"]="failure")),
    "zero_raw_log"=>(x->(x["native_jobs"][1]["log_sha256"]=repeat("0",64))),
    "extra_cell"=>(x->(x["native_jobs"][1]["extra"]=true)),
    "failed_docs"=>(x->(x["docs_and_containers"][1]["conclusion"]="failure")))
    x=deepcopy(data);mutate(x);negative(label,x)
end
@test P._qualification_integer(collect(codeunits("{\"harness\":{\"counts\":{\"PASS\":9223372036854775807}}}")),("harness","counts","PASS"))==typemax(Int64)
for literal in ("9223372036854775808","99999999999999999999","00","-0","1e0","1.0")
    @test_throws ArgumentError P._qualification_integer(collect(codeunits("{\"harness\":{\"counts\":{\"PASS\":"*literal*"}}}")),("harness","counts","PASS"))
end

true
end
