function tree_checks(base)
    path=joinpath(@__DIR__,"tree_inventory_oracles_v1.json")
    @test T._sha(read(path))=="e829beee61d48333689fea77527c5f6593e86412a1a21d66ab84ab917a4ebc96"
    O=JSON3.read(read(path))
    for case in O.parent_cases
        parent=dirname(String(case.path));observed=String[]
        while !isempty(parent)&&parent!="."
            push!(observed,replace(parent,'\\'=>'/'));parent=dirname(parent)
        end
        @test sort(observed)==String.(case.parents)
    end
    root=joinpath(base,"inventory");mkdir(root)
    for name in O.directories
        mkpath(joinpath(root,split(String(name),'/')...))
    end
    for name in O.files;P._exclusive(P._member(root,String(name)),UInt8[]);end
    tree=B.BInitialPointFits._tree(root)
    @test tree.files==423&&tree.directories==116&&tree.bytes==0
    @test tree.sha256==O.tree_sha256
    P._exclusive(joinpath(root,"completion.json"),UInt8[])
    @test B.BInitialPointFits._tree(root).sha256==O.tree_sha256
    @test B.BInitialPointFits._tree(root;exclude_completion=false).files==424
    @test B.BInitialPointFits._tree(root;exclude_completion=false).sha256==O.with_completion_tree_sha256
    for name in O.rejected_members
        @test_throws ArgumentError P._member(root,String(name))
    end
    @test_throws ArgumentError P._member(root,abspath(joinpath(base,"absolute-file")))
    @test P._member(root,"alpha/beta/file.bin")==abspath(joinpath(root,"alpha","beta","file.bin"))
    @test_throws Exception B.BInitialPointFits._tree(joinpath(base,"missing-root"))
    extra=joinpath(root,"extra.bin");P._exclusive(extra,UInt8[])
    @test_throws ArgumentError B.BInitialPointFits._tree(root)
    rm(extra)
    extra_dir=joinpath(root,"extra-dir");mkdir(extra_dir)
    @test_throws ArgumentError B.BInitialPointFits._tree(root)
    rm(extra_dir)
    missing=joinpath(root,"points","setting-36.bin");rm(missing)
    @test_throws ArgumentError B.BInitialPointFits._tree(root)
    P._exclusive(missing,UInt8[])
    @test B.BInitialPointFits._tree(root).sha256==O.tree_sha256
    input=joinpath(base,"input-boundary");mkdir(input)
    admission=(;roots=Dict("input"=>input))
    for path in (input,joinpath(input,"child"),dirname(input),B.BInitialPointFits.REPO)
        @test_throws ArgumentError B.BInitialPointFits._outside_inputs(admission,path)
    end
    @test B.BInitialPointFits._outside_inputs(admission,joinpath(base,"new-output"))===nothing
true
end
