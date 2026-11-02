"""Synthetic-only nonlinear source fixture shared by tests and the complete example."""
module CandidateValidationGeneratedFixture
using SolarSINDy,Dates,SHA
const J=SolarSINDy.ArchiveCatalogSourceJob
const Q=SolarSINDy.ArchiveCatalogCaseJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const START=DateTime(2020)
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
digest(bytes)=bytes2hex(sha256(bytes))
function make_cases(root;future_delta=0)
    root=realpath(root);input=joinpath(root,"inputs");mkdir(input);descriptors=Any[]
    for (s,station) in enumerate(("CMO","FRD"))
        lines=[" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
            " IAGA CODE              $station |"," Reported               XYZF |",
            " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
            "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |"]
        x=10000;y=1000
        for minute in 0:650
            # Multiple changing rate regimes in training AND validation; all16
            # original recipes retain bins64/minleaf64or32/sample20000/seed.
            k=1+mod(minute÷37+s,5)
            minute>0 && (x+=3k;y+=4k)
            xx=x+(minute>=570 ? future_delta : 0)
            push!(lines,"$(Dates.format(START+Minute(minute),dateformat"yyyy-mm-dd HH:MM:SS.sss")) 001 $xx $y 0 0")
        end
        name="generated-"*station*".txt";path=joinpath(input,name)
        bytes=Vector{UInt8}(codeunits(join(lines,"\n")*"\n"));R._exclusive_write(path,bytes)
        push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>station,
            "stream_labels"=>[station],"product"=>"quasi-definitive","start_inclusive"=>clock(START),
            "stop"=>clock(START+Minute(650)),"stop_inclusive"=>true,"bytes"=>length(bytes),"sha256"=>digest(bytes),
            "role"=>"generated_fixture","provenance"=>"generated_fixture"))
    end
    request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>"ground-only",
        "role"=>"primary","runtime"=>string(VERSION),"data_root"=>input,"inputs"=>descriptors)
    path=joinpath(root,"source-request.json");R._exclusive_write(path,R.canonical_bytes(request));source=joinpath(root,"source");J.reconstruct(path,source)
    request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated","layout"=>"ground-only","role"=>"primary",
        "runtime"=>string(VERSION),"source"=>Dict("directory"=>source,"manifest_sha256"=>digest(read(joinpath(source,"manifest.json"))),
        "complete_sha256"=>digest(read(joinpath(source,"COMPLETE.json")))),"tables"=>[Dict("id"=>station*"-generated","station"=>station,
        "start_inclusive"=>clock(START),"stop_exclusive"=>clock(START+Minute(651)),"label_asof"=>clock(START+Minute(681))) for station in ("CMO","FRD")])
    path=joinpath(root,"case-request.json");R._exclusive_write(path,R.canonical_bytes(request));request_pin=digest(read(path));destination=joinpath(root,"cases")
    Q.publish(path,destination;request_sha256=request_pin)
    Q.read_verified(path,destination;request_sha256=request_pin,manifest_sha256=digest(read(joinpath(destination,"manifest.json"))),
        complete_sha256=digest(read(joinpath(destination,"COMPLETE.json"))))
end
function run_chain(N,root)
    root=realpath(root);case_root=joinpath(root,"source-case");mkdir(case_root);cases=make_cases(realpath(case_root))
    train=N.CandidateSplit(START+Minute(40),START+Minute(310),START+Minute(681))
    validation=N.CandidateSplit(START+Minute(320),START+Minute(530),START+Minute(681))
    refit=N.CandidateSplit(train.start,validation.stop,train.asof)
    stations=Any[]
    for index in 1:2
        station_root=joinpath(root,"station-$index");mkdir(station_root);verified=N.VerifiedCandidate[];fits=Any[];validations=Any[];completed=Any[]
        for spec in N.S.POINT_GRID
            id=spec.id;input=N.prepare_candidate_fit_input_generated(cases,index,id,train,joinpath(station_root,id*"-input"))
            fit=N.fit_candidate_generated(cases,index,id,train,input.input;directory=joinpath(station_root,id*"-point"))
            scored=N.validate_candidate_generated(cases,index,id,validation,fit.point,joinpath(station_root,id*"-validation"))
            check=N.verify_candidate_validation_generated(cases,index,id,validation,fit,scored.artifact.directory,scored.pins)
            witness_directory=joinpath(station_root,id*"-independent-witness");witness_pins=N.publish_generated_verification(check,witness_directory)
            raw_path=joinpath(station_root,id*"-completion-fixture.log")
            raw=Vector{UInt8}(codeunits("GENERATED FIXTURE ONLY; no native execution or actual authority.\n"));R._exclusive_write(raw_path,raw)
            owner=Dict("schema"=>"generated_candidate_completion_fixture_v1","actual_authority"=>false,"native_execution_evidence"=>false,
                "declared_fixture_native_exit"=>0,"timed_out"=>false,"job_sha256"=>N.job_identity(),"output_witness_sha256"=>check.witness_sha256,
                "output_manifest_sha256"=>witness_pins.manifest_sha256,"output_complete_sha256"=>witness_pins.complete_sha256,"raw_sha256"=>digest(raw))
            owner_path=joinpath(station_root,id*"-completion-fixture.json");owner_bytes=N.canonical(owner);R._exclusive_write(owner_path,owner_bytes)
            completion=N.NativeEvidence(owner_path,digest(owner_bytes),raw_path,digest(raw))
            restored=N.read_completed_generated_candidate(cases,index,id,validation,fit,scored.artifact.directory,scored.pins,
                witness_directory,witness_pins;completion)
            @assert restored.witness_sha256==check.witness_sha256 && !restored.actual
            push!(verified,restored);push!(fits,fit);push!(validations,scored);push!(completed,(;witness_directory,witness_pins,completion))
        end
        decision=N.choose_verified_R2(verified);refits=Any[]
        for winner in decision.decision.family_winners
            id=winner.candidate_id;input=N.prepare_candidate_fit_input_generated(cases,index,id,refit,joinpath(station_root,String(winner.family)*"-refit-input"))
            fit=N.refit_selected_generated(decision,cases,index,winner.family,refit,input.input;
                directory=joinpath(station_root,String(winner.family)*"-selected-refit"))
            push!(refits,fit)
        end
        push!(stations,(;verified,fits,validations,decision,refits,completed))
    end
    (;cases,train,validation,refit,stations)
end
end
