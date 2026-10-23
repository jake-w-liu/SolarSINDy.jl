function consistency_checks(root,result)
    stage=result.stage.directory
    for case in (:point_model_id,:support_scope,:support_count,:journal_model)
        clone=joinpath(root,"stage-consistency-"*String(case));cp(stage,clone)
        data=plain(JSON3.read(read(joinpath(clone,"completion.json"))))
        if case==:point_model_id
            data["models"][1]["points"]["model_id"]=repeat("0",64)
        elseif case in (:support_scope,:support_count)
            path=joinpath(clone,"support.json");support=plain(JSON3.read(read(path)))
            if case==:support_scope;support["accepted"]=false
            else;support["settings"][1]["informative_origins"]+=1
            end
            write(path,T._bytes(support))
        else
            path=joinpath(clone,"gain-journal","setting-1.json");journal=plain(JSON3.read(read(path)));journal["fit_count"]=1;write(path,T._bytes(journal))
        end
        tree=B.BInitialPointFits._tree(clone)
        data["output_sha256"]=tree.sha256;data["output_bytes"]=tree.bytes;data["output_file_count"]=tree.files;data["output_directory_count"]=tree.directories
        body=T._bytes(data);write(joinpath(clone,"completion.json"),body)
        @test_throws ArgumentError B.BInitialPointFits.verify_initial_l0(clone,result.admission,result.reference;expected_completion_sha256=T._sha(body),expected_sources=P.source_identity(),chunk_hours=2)
    end
    true
end
