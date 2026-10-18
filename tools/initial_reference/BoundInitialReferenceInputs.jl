module BoundInitialReferenceInputs

using SHA, JSON3
using ..DstInitialGBMReference
const R = DstInitialGBMReference
const LOADED_SHA256 = bytes2hex(sha256(read(@__FILE__)))
export admit, recheck, Limits

need(x, message) = x || throw(ArgumentError(message))
hex(x) = x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z", x)
role_name(x) = x isa String && occursin(r"\A[a-z][a-z0-9_]*\z",x)
integer(x, minimum=0) = x isa Integer && !(x isa Bool) && x >= minimum
zero(x) = integer(x) && x == 0
getv(x, k) = (need(x isa AbstractDict && haskey(x,k), "missing object field: $k"); x[k])
same(x,y,message) = need(x == y, message)

Base.@kwdef struct Limits
    file_bytes::Int = 512 * 1024^2
    json_bytes::Int = 64 * 1024^2
    total_bytes::Int = 8 * 1024^3
    files::Int = 4096
    nodes::Int = 4_000_000
    depth::Int = 64
end
function valid(l::Limits)
    need(all(x->integer(x,1),(l.file_bytes,l.json_bytes,l.total_bytes,l.files,l.nodes,l.depth)),"positive budgets required")
    need(l.json_bytes<=l.file_bytes<=l.total_bytes,"inconsistent byte budgets")
end

# Windows junctions and all other reparse points are rejected as well as links.
function reparse(path)
    Sys.iswindows() || return islink(path)
    flags = ccall((:GetFileAttributesW,"kernel32"), stdcall, UInt32, (Cwstring,), path)
    need(flags != typemax(UInt32), "cannot inspect Windows path attributes")
    (flags & UInt32(0x400)) != 0
end
function safe_components(name)
    need(name isa String && !isempty(name) && !occursin(r"[\\:\x00-\x1f]",name),"unsafe relative path")
    parts=split(name,'/';keepempty=true)
    need(all(p->!isempty(p) && p!="." && p!=".." && !endswith(p,".") && !endswith(p," ") &&
        !occursin(r"(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",p),parts),"unsafe path component")
    parts
end
canonical(path) = Sys.iswindows() ? lowercase(replace(abspath(path),'\\'=>'/')) : abspath(path)
function actual_spelling(path)
    # Long-name expansion preserves caller case on Windows. Resolve the actual
    # canonical spelling before comparing it with the declared path.
    realpath(path)
end
function root_path(path)
    need(path isa AbstractString && isabspath(path) && isdir(path),"explicit existing absolute root required")
    absolute=abspath(path)
    cursor=absolute
    while true
        need(!islink(cursor) && !reparse(cursor),"root has a link/reparse ancestor")
        parent=dirname(cursor); parent==cursor && break; cursor=parent
    end
    same(actual_spelling(absolute),absolute,"root case/short-name alias")
    same(canonical(realpath(absolute)),canonical(absolute),"root alias differs from actual location")
    absolute
end
function resolved(roots, root, name)
    need(root isa String && haskey(roots,root),"unknown explicit root")
    parts=safe_components(name); base=roots[root]; cursor=base
    for part in parts
        cursor=joinpath(cursor,part)
        need(ispath(cursor),"missing path member")
        need(!islink(cursor) && !reparse(cursor),"link/reparse path member")
        same(actual_spelling(cursor),cursor,"case/short-name path alias")
    end
    need(isfile(cursor),"bound artifact must be a regular file")
    actual=realpath(cursor)
    need(startswith(canonical(actual),canonical(base)*"/"),"artifact escaped its explicit root")
    same(canonical(actual),canonical(cursor),"artifact path alias")
    cursor
end

function digest_file(path, limit)
    before=stat(path)
    need(isfile(path) && !islink(path) && !reparse(path) && before.size<=limit,"file byte budget or link")
    context=SHA.SHA2_256_CTX(); count=0
    open(path,"r") do io
        while !eof(io)
            # At most one byte beyond the remaining cap is requested.
            bytes=read(io,min(65536,limit-count+1)); count+=length(bytes)
            need(count<=limit,"file grew beyond byte budget")
            SHA.update!(context,bytes)
        end
    end
    after=stat(path)
    need(count==before.size==after.size && before.mtime==after.mtime && before.ctime==after.ctime,
        "file changed during bounded read")
    (sha256=bytes2hex(SHA.digest!(context)),bytes=count)
end

# Scan syntax/keys/budgets before materializing JSON records. JSON3 subsequently
# validates primitives and string escapes; no source path is followed by parsing.
function lexical_json(bytes, limits)
    n=length(bytes); at=Ref(1); nodes=Ref(0)
    whitespace() = (while at[]<=n && bytes[at[]] in (0x20,0x09,0x0a,0x0d);at[]+=1;end)
    function string_token()
        need(at[]<=n && bytes[at[]]==0x22,"JSON string expected")
        start=at[];at[]+=1
        while at[]<=n
            byte=bytes[at[]];at[]+=1
            if byte==0x22
                return JSON3.read(String(bytes[start:at[]-1]),String)
            elseif byte==0x5c
                need(at[]<=n,"truncated JSON escape");at[]+=1
            else
                need(byte>=0x20,"JSON control character")
            end
        end
        throw(ArgumentError("unterminated JSON string"))
    end
    function value(depth)
        need(depth<=limits.depth,"JSON depth budget")
        nodes[]+=1;need(nodes[]<=limits.nodes,"JSON node budget");whitespace()
        need(at[]<=n,"missing JSON value")
        token=bytes[at[]]
        if token==0x7b || token==0x5b
            object=token==0x7b;ending=object ? 0x7d : 0x5d
            at[]+=1;whitespace();keys=Set{String}()
            if at[]<=n && bytes[at[]]==ending;at[]+=1;return;end
            while true
                if object
                    key=string_token();need(!(key in keys),"duplicate decoded JSON object key");push!(keys,key)
                    whitespace();need(at[]<=n && bytes[at[]]==0x3a,"JSON colon expected");at[]+=1
                end
                value(depth+1);whitespace();need(at[]<=n,"unterminated JSON container")
                byte=bytes[at[]];at[]+=1
                byte==ending && break
                need(byte==0x2c,"JSON comma expected");whitespace()
            end
        elseif token==0x22
            string_token()
        else
            start=at[]
            while at[]<=n && !(bytes[at[]] in (0x20,0x09,0x0a,0x0d,0x2c,0x7d,0x5d));at[]+=1;end
            need(at[]>start,"invalid JSON token")
        end
    end
    value(1);whitespace();need(at[]==n+1,"trailing JSON bytes")
    nothing
end
function json_file(path,digest,limits)
    need(hex(digest),"invalid pinned JSON digest")
    before=stat(path);need(before.size<=limits.json_bytes,"JSON byte budget before materialization")
    bytes=open(path,"r") do io
        data=read(io,limits.json_bytes+1);need(length(data)<=limits.json_bytes,"JSON grew beyond byte budget");data
    end
    after=stat(path)
    need(length(bytes)==before.size==after.size && before.mtime==after.mtime && before.ctime==after.ctime,
        "JSON changed during exact-byte read")
    same(bytes2hex(sha256(bytes)),digest,"pinned JSON bytes differ")
    lexical_json(bytes,limits)
    result=JSON3.read(bytes,Dict{String,Any})
    need(result isa AbstractDict,"JSON object root required")
    result
end

function own_guard()
    same(bytes2hex(sha256(read(@__FILE__))),LOADED_SHA256,"loader loaded/disk source differs")
    R.verify_source_identity()
end
function token_fingerprint(token)
    # NamedTuple fields are fixed; seal every mutable record/root/inventory value
    # so an accidental caller mutation cannot retain a successful recheck.
    payload=(roots=token.roots,files=token.files,
        limits=Tuple(getfield(token.limits,key) for key in fieldnames(Limits)),own=token.own,
        plan=token.plan,records=token.records,mode=token.mode,
        new_model_fitting=token.new_model_fitting,scientific_verdict=token.scientific_verdict)
    bytes2hex(sha256(JSON3.write(payload)))
end

function semantics_check(t,c,a,runtime,closure,s,plan,receipts,support,records)
    e=plan["expected"]; mode=plan["mode"]
    same(e["configuration"]["mode"],mode,"manifest/expected mode")
    R.validate_receipts(t,c,a,e)
    same(t["runtime_gate_sha256"],receipts["runtime"]["sha256"],"original runtime receipt binding")
    need(runtime["success"]===true && runtime["source_unchanged"]===true,"runtime success")
    same(runtime["files"],t["package_files"],"runtime full file map")
    same(runtime["git_commit"],t["package_commit"],"runtime package commit")
    runs=runtime["runs"]
    same([r["name"] for r in runs],["serving","maturity","package","experiments"],"all runtime native stages")
    need(all(r->zero(r["native_exit"]),runs),"runtime native failure")
    for (index,run) in enumerate(t["runs"])
        same(run["log_sha256"],support["original_log_$index"]["sha256"],"original native log binding")
    end
    for (index,run) in enumerate(runs)
        same(run["log_sha256"],support["runtime_log_$index"]["sha256"],"runtime native log binding")
    end
    for (role,field) in (("driver","driver_sha256"),("julia","julia_binary_sha256"),
        ("native","native_library_sha256"),("raw","input_sha256"),("format","format_sha256"),
        ("synthetic_gate","synthetic_gate_sha256"))
        same(support[role]["sha256"],e[field],"actual support/expected identity: $role")
    end
    same(e["completion_sha256"],receipts["completion"]["sha256"],"actual source completion digest")
    need(haskey(t["source_sha256"],"tools/dst_forecaster/audit_sandu_matched_outputs.py"),"legacy checker source absent")
    same(e["checker_sha256"],t["source_sha256"]["tools/dst_forecaster/audit_sandu_matched_outputs.py"],"legacy checker identity")
    same(plan["files"]["source"],t["source_sha256"],"actual source13 map")
    same(plan["files"]["package"],t["package_files"],"actual runtime388 map")
    same(plan["files"]["artifacts"],merge(c["output_sha256"],Dict("completion.json"=>e["completion_sha256"])),"actual complete artifact map")
    same(c["raw_hourly_matrix_sha256"],c["output_sha256"]["input_hourly_channels.f64"],"actual source raw matrix bytes")
    need(mode!="exposed_development" || length(plan["files"]["reader"])==12,"complete production enhanced reader source group")
    for (key,digest) in t["source_sha256"]
        same(getv(c["source_sha256"],key),digest,"original completion source13 relation")
    end
    extras=setdiff(Set(keys(c["source_sha256"])),Set(keys(t["source_sha256"])))
    need(length(extras)==5,"exactly five completion package dependencies")
    for name in ("Project.toml","Manifest.toml","src/SolarSINDy.jl","src/analog_features.jl","src/analog_gbm.jl")
        keys_for=[key for key in extras if key==name || endswith(key,"/"*name)]
        need(length(keys_for)==1,"completion dependency name: $name")
        same(c["source_sha256"][only(keys_for)],t["package_files"][name],"completion dependency digest")
    end
    need(closure["schema"]=="sandu_completed_gbm_independent_closure_v1" && closure["complete"]===true &&
        zero(closure["native_exit"]),"new actual enhanced closure required")
    for flag in ("new_model_fitting","source_run_modified","independent_confirmation","scientific_verdict")
        need(closure[flag]===false,"enhanced closure scope flag: $flag")
    end
    for (field,key) in (("original_terminal_sha256","terminal"),("original_completion_sha256","completion"),
        ("additive_receipt_sha256","semantics"))
        same(closure[field],receipts[key]["sha256"],"enhanced closure same-run binding")
    end
    same(closure["original_source_sha256"],t["source_sha256"],"closure original source map")
    same(closure["original_runtime_sha256"],t["package_files"],"closure original runtime map")
    same(closure["original_execution_source_sha256"],c["source_sha256"],"closure original execution map")
    same(closure["reader_source_sha256"],plan["files"]["reader"],"actual reader source group")
    p=records["closure_plan"]
    need(p["schema"]=="sandu_additive_real_closure_inputs_v1","original enhanced input-plan schema")
    same(closure["plan_sha256"],receipts["closure_plan"]["sha256"],"enhanced input-plan actual bytes")
    same(closure["verified_input_plan"],p,"enhanced verified plan content")
    for (field,role) in (("terminal","terminal"),("completion","completion"),("legacy_oracle","legacy"),
        ("runtime_gate","runtime"),("qualification","qualification"),("full_harness","harness"))
        same(p[field]["sha256"],receipts[role]["sha256"],"enhanced original plan receipt role: $field")
    end
    same(p["reader_sources"],plan["files"]["reader"],"enhanced original plan reader map")
    q=records["qualification"];h=records["harness"]
    need(q["complete"]===true && q["real_output_or_raw_read"]===false && q["original_fixture_unchanged"]===true &&
        q["all13_runner_sources_unchanged"]===true && q["all388_runtime_files_unchanged"]===true,"enhanced portable qualification scope")
    same(q["source_sha256"],plan["files"]["reader"],"enhanced qualified reader sources")
    need(length(q["cases"])>=25 && all(x->x["expectation_met"]===true && x["expected_rejection"] isa Bool &&
        integer(x["native_exit"]) && (x["expected_rejection"] ? x["native_exit"]!=0 : zero(x["native_exit"])),q["cases"]),
        "all enhanced portable mutation cases")
    need(length(q["unit_runs"])>=3 && all(x->zero(x["native_exit"]),q["unit_runs"]),"all enhanced portable unit runs")
    need(h["success"]===true && zero(h["native_exit"]) && isempty(h["changed_package_files"]) &&
        isempty(h["changed_reader_files"]) && isempty(h["changed_support_files"]) &&
        length(h["package_files_before"])==395,"unshortened enhanced full harness")
    same(h["reader_source_sha256"],plan["files"]["reader"],"full harness reader sources")
    same(h["qualification_sha256"],receipts["qualification"]["sha256"],"full harness portable qualification")
    same(h["log_sha256"],support["harness_log"]["sha256"],"full harness native log")
    same(closure["log_sha256"],support["closure_log"]["sha256"],"actual additive native log")
    same(closure["driver_sha256"],support["closure_driver"]["sha256"],"actual closure driver bytes")
    need(s["schema"]=="sandu_completed_gbm_independent_semantics_v1" && s["complete"]===true && s["mode"]==mode,
        "enhanced semantics successful matching mode")
    same(s["completion_sha256"],e["completion_sha256"],"semantics same completion")
    same(s["artifact_sha256"],plan["files"]["artifacts"],"semantics complete actual artifacts")
    same(s["original_execution_source_sha256"],c["source_sha256"],"semantics source execution map")
    core=("independent_completed_gbm.py","bounded_bson.py","independent_gbm_math.jl","FROZEN_ORACLES.json","ARITHMETIC_ADDENDUM.md")
    same(s["source_sha256"],Dict(name=>getv(plan["files"]["reader"],name) for name in core),"exact semantic core source group")
    same(s["julia_binary_sha256"],e["julia_binary_sha256"],"semantic Julia binary")
    helpers=s["helper_runs"];need(length(helpers)==2,"both native arithmetic helpers")
    for (index,run) in enumerate(helpers)
        need(zero(run["native_exit"]) && run["binary_sha256"]==e["julia_binary_sha256"],"math native exit/binary")
        same(run["log_sha256"],support["helper_log_$index"]["sha256"],"math native log")
    end
    need(Set(keys(s["stages"]))==Set(("initial","final")),"initial and final enhanced stages")
    for stage in ("initial","final")
        item=s["stages"][stage]
        need(Set(keys(item["fit_count"]))==Set(("4","6")) && all(v->integer(v,1),values(item["fit_count"])),"positive full per-lead fit counts")
        for (field,role) in (("input_sha256","imputed"),("features_sha256","features"),("availability_sha256","availability"))
            same(item[field],support[stage*"_"*role]["sha256"],"actual arithmetic helper stage input")
        end
    end
    for field in ("checks","gbm_predictions","gbm_fallbacks")
        need(integer(s[field],field=="gbm_fallbacks" ? 0 : 1),"semantic positive integer counts")
        same(closure[field],s[field],"closure/semantics count")
    end
    same(closure["stages"],s["stages"],"closure/semantics stages")
    same(closure["original_artifact_sha256"],s["artifact_sha256"],"closure full artifact inventory")
    for flag in ("new_model_fitting","scientific_verdict","author_or_ml_library_imported","native_training_availability_independently_reconstructed")
        need(s[flag]===false,"semantic scope/limitation: $flag")
    end
    true
end

function admit(manifest_path, manifest_sha256; roots, limits=Limits())
    valid(limits);own=own_guard()
    # The manifest itself is an explicit file argument, never read through an untrusted receipt path.
    parent=root_path(dirname(abspath(manifest_path)))
    manifest=resolved(Dict("manifest"=>parent),"manifest",basename(manifest_path))
    plan=json_file(manifest,manifest_sha256,limits)
    need(plan["schema"]=="bound_initial_reference_inputs_v1" && plan["mode"] in ("synthetic_fixture","exposed_development"),"manifest schema/mode")
    roles=plan["root_roles"]
    need(roles isa AbstractVector && length(roles)<=64 && all(role_name,roles) &&
        allunique(roles) && Set(keys(roots))==Set(roles) &&
        issubset(Set(("source","package","artifacts","reader","closure")),Set(roles)),"exact bounded explicit root roles")
    bound_roots=Dict(String(k)=>root_path(v) for (k,v) in roots)
    paths=collect(values(bound_roots))
    for i in eachindex(paths),j in 1:i-1
        a=canonical(paths[i]);b=canonical(paths[j])
        need(a!=b && !startswith(a,b*"/") && !startswith(b,a*"/"),"overlapping or aliased roots")
    end
    # The manifest is a bound input and consumes both budgets from the start.
    # Its exact-byte bounded read above is necessarily the first metadata read.
    entries=Dict{String,Tuple{String,String}}(canonical(manifest)=>(manifest,String(manifest_sha256)))
    total=stat(manifest).size
    need(total<=limits.total_bytes && total<=limits.file_bytes,"manifest total/file byte budget")
    function add(root,name,digest)
        need(hex(digest),"invalid file digest")
        path=resolved(bound_roots,root,name);key=canonical(path)
        if haskey(entries,key)
            same(entries[key],(path,digest),"conflicting or aliased file identity");return
        end
        need(length(entries)<limits.files,"file count budget")
        size=stat(path).size;need(size<=limits.file_bytes && size<=limits.total_bytes-total,"file/total byte budget")
        entries[key]=(path,String(digest));total+=size
    end
    files=plan["files"]
    need(Set(keys(files))==Set(("source","package","artifacts","reader")),"exact inventory groups")
    for (root,group) in files
        need(group isa AbstractDict && !isempty(group),"nonempty inventory")
        for (name,digest) in group;add(root,name,digest);end
    end
    receipts=plan["receipts"]
    need(Set(keys(receipts))==Set(("terminal","completion","legacy","runtime","closure","semantics",
        "closure_plan","qualification","harness")),"all receipt roles required")
    support=plan["support"]
    roles=Set(("driver","julia","native","raw","format","synthetic_gate","closure_driver","closure_log",
        "original_log_1","original_log_2","runtime_log_1","runtime_log_2","runtime_log_3","runtime_log_4",
        "helper_log_1","helper_log_2","initial_imputed","initial_features","initial_availability",
        "final_imputed","final_features","final_availability","harness_log"))
    need(Set(keys(support))==roles,"exact supporting role set")
    for group in (receipts,support),(_,record) in group
        need(Set(keys(record))==Set(("root","path","sha256")),"strict file record fields")
        add(record["root"],record["path"],record["sha256"])
    end
    # Every path and budget is accepted before content hashing or receipt loads.
    for (path,digest) in values(entries)
        same(digest_file(path,limits.file_bytes).sha256,digest,"actual bound file differs")
    end
    records=Dict(key=>json_file(resolved(bound_roots,item["root"],item["path"]),item["sha256"],limits)
        for (key,item) in receipts)
    semantics_check(records["terminal"],records["completion"],records["legacy"],records["runtime"],
        records["closure"],records["semantics"],plan,receipts,support,records)
    metadata=(roots=bound_roots,files=entries,limits=limits,own=own,plan=plan,records=records,
        mode=plan["mode"],new_model_fitting=false,scientific_verdict=false)
    token=(metadata...,sealed_sha256=token_fingerprint(metadata))
    recheck(token)
    token
end

function recheck(token)
    same(token_fingerprint(token),token.sealed_sha256,"admitted in-memory metadata changed")
    same(own_guard(),token.own,"loaded reference sources changed")
    for (key,(path,digest)) in token.files
        same(canonical(path),key,"bound path identity changed")
        parent=root_path(dirname(path))
        checked=resolved(Dict("file"=>parent),"file",basename(path))
        same(digest_file(checked,token.limits.file_bytes).sha256,digest,"admitted file changed")
    end
    true
end

end # module
