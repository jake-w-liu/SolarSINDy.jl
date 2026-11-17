function tree_checks(base)
    root=joinpath(base,"path-safety");mkdir(root)
    for name in ("../file","./file","alpha//file","alpha/./file",
                 "alpha/../file","/absolute/file","C:/absolute/file","alpha\\file")
        @test_throws ArgumentError P._member(root,String(name))
    end
    @test_throws ArgumentError P._member(root,abspath(joinpath(base,"absolute-file")))
    @test P._member(root,"alpha/beta/file.bin")==abspath(joinpath(root,"alpha","beta","file.bin"))
    input=joinpath(base,"input-boundary");mkdir(input)
    admission=(;roots=Dict("input"=>input))
    for path in (input,joinpath(input,"child"),dirname(input),B.BInitialPointFits.REPO)
        @test_throws ArgumentError B.BInitialPointFits._outside_inputs(admission,path)
    end
    @test B.BInitialPointFits._outside_inputs(admission,joinpath(base,"new-output"))===nothing
    true
end
