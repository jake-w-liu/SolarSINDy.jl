function core_checks()
    oracle=JSON3.read(read(joinpath(@__DIR__,"independent_literals_v1.json")))
    @test T._sha(read(joinpath(@__DIR__,"independent_literals_v1.json")))=="736e8e052c513647dd1c712e447fb801fb4d452381e1dfa7d9ec3ebbabe74b58"
    @test length(T.settings())==36
    @test length(P.PROVENANCE_FIELDS)==22
    @test all(s.index==i&&s.beta==(isodd(i) ? 5 : 20) for (i,s) in enumerate(T.settings()))
    for value in (true,false,1.,"1",-1,500001);@test_throws ArgumentError T._int(value,0,500000);end
    @test T._int(500000,0,500000)==500000
    for name in ("../x","./x","x/../y","x//y","/x","C:/x","x\\y");@test_throws ArgumentError P._member(pwd(),name);end
    @test P._qualification_counter(collect(codeunits("{\"harness\":{\"counts\":{\"FAIL\":0}}}")))
    for token in ("0.0","0e0","-0","false","true","\"0\"")
        @test_throws ArgumentError P._qualification_counter(collect(codeunits("{\"harness\":{\"counts\":{\"FAIL\":"*token*"}}}")))
    end
    issue=T._clock(string(oracle.clock.latest_l0_training_issue_ms));ctx=repeat("6",64)
    design=T.DesignValues(1,issue,-45.,(-42.,-63.),((1.,0.),(0.,1.)),true,true,:model,:available,repeat("1",64),repeat("2",64),repeat("3",64),repeat("4",64),repeat("5",64),ctx)
    row=T.BTrainingRow(T.BTrainingDesignRow(design),(-40.,-60.))
    io=IOBuffer();P._header(io,"SSBPBAS1",248);write(io,P._basis_bytes(row));encoded=take!(io)
    @test bytes2hex(encoded)==oracle.codec.basis_literal_hex
    @test T._sha(encoded)==oracle.codec.basis_literal_sha256
    # Data-only primitive IDs; this does not construct a duration model/artifact.
    stream=IOBuffer(encoded);P._check_header(stream,"SSBPBAS1",248)
    decoded=P._read_basis_row(stream,(;context_id=ctx),(;model_id=repeat("4",64)),1,:training,1)
    @test eof(stream)&&P._basis_bytes(decoded)==P._basis_bytes(row)
    negzero=T.DesignValues(1,issue,-45.,(-42.,-63.),((1.,-0.),(0.,1.)),true,true,:model,:available,repeat("1",64),repeat("2",64),repeat("3",64),repeat("4",64),repeat("5",64),ctx)
    io=IOBuffer();P._header(io,"SSBPBAS1",248);write(io,P._basis_bytes(T.BTrainingRow(T.BTrainingDesignRow(negzero),(-40.,-60.))))
    @test T._sha(take!(io))==oracle.codec.negative_zero_record_sha256
    metric=I._reference_metric([-3.,1.,-2.,4.],zeros(4))
    @test metric==(;n=4,rmse_nt=sqrt(7.5),mae_nt=2.5,bias_nt=0.)
    @test_throws MethodError T.predict_point_rows(nothing,[row])
    mktempdir() do root
        root=realpath(root)
        p=joinpath(root,"one.json");P._exclusive(p,T._bytes((;x=1)))
        @test_throws ArgumentError P._exclusive(p,T._bytes((;x=2)))
        @test P._json(p)[1].x==1
        @test_throws ArgumentError P._json(p;fields=(:x,:extra))
        @test json_depth_checks()
    end
    true
end
function nnls_literal_checks()
    literal=JSON3.read(read(joinpath(@__DIR__,"independent_literals_v1.json")))
    n=548;t0=DateTime(1900);boundary=t0+Hour(274);raw=ones(17,n)
    for i in 1:n;raw[6,i]=(i-1)%10<5 ? 1. : -2.5;raw[8,i]=400.;raw[15,i]=-60.;end
    source=T._sha(reinterpret(UInt8,vec(raw)))
    input=T.A.HourlyInputs([t0+Hour(i) for i in 0:n-1],raw;source_sha256=source)
    wind=[T.D.WindHour(t,t+Hour(1),t+Hour(1),raw[6,i],max(raw[8,i]*max(-raw[6,i],0.)/1000-.49,0.)) for (i,t) in enumerate(input.time)]
    view=T._point_input_view(input,wind,(;evidence_mode=:synthetic_only,input_admission_sha256=T._sha("generated numerical unit admission"),contract_sha256=T.CONTRACT))
    law=T.D.fit_duration(wind,boundary;beta=5,source_sha256=source)
    mask=T._masks(view,boundary,boundary,t0+Hour(n),0).training
    @test findall(mask)==collect(1:100)
    binding=T._make_binding(view,law,mask,"numeric_matrix_unit",boundary,0)
    Z=((1.,0.),(0.,1.),(1.,1.),(2.,1.));residual=(1.,2.,4.,0.)
    for unequal in (false,true)
        labels=unequal ? (-40.,-40.,-55.,-20.) : (-40.,-60.,-55.,-20.)
        offsets=ntuple(k->labels[k]-residual[k],4);rows=T.BTrainingRow[]
        for i in 1:100
            k=isodd(i) ? 1 : 3
            d=T.DesignValues(i,t0+Hour(i),0.,(offsets[k],offsets[k+1]),(Z[k],Z[k+1]),true,true,:model,:available,repeat("1",64),repeat("2",64),repeat("3",64),law.model_id,repeat("5",64),binding.context_id)
            push!(rows,T.BTrainingRow(T.BTrainingDesignRow(d),(labels[k],labels[k+1])))
        end
        # These are declared numerical matrix units, not claimed A-state replay
        # or model artifacts. Full raw-source replay is tested separately.
        panel=T.BTrainingPanel(rows,binding,law,1,T._sha("numerical matrix unit"),:synthetic_only)
        for index in (1,3)
            setting=T.settings()[index];problem=T._fit_problem(panel,setting)
            @test problem.storm==(unequal ? 50 : 100)
            solver=T.Adapter.nnls2(problem.Z,problem.rhs;weights=problem.weights)
            expected=setting.eta==0 ? 2. : unequal ? 2.5 : 2.25
            loss=setting.eta==0 ? 2.25 : unequal ? 3. : 43/16
            @test isapprox(solver.gains[1],0.;rtol=literal.nnls.gain_rtol,atol=literal.nnls.gain_atol)
            @test isapprox(solver.gains[2],expected;rtol=literal.nnls.gain_rtol,atol=literal.nnls.gain_atol)
            @test abs(solver.weighted_sse-loss)<=literal.nnls.weighted_sse_atol
            full_gradient=[2sum(problem.weights[i]*problem.Z[i,j]*(sum(problem.Z[i,k]*solver.gains[k] for k in 1:2)-problem.rhs[i]) for i in eachindex(problem.rhs)) for j in 1:2]
            expected_gradient=setting.eta==0||unequal ? .5 : 7/16
            @test abs(full_gradient[1]-expected_gradient)<=literal.nnls.weighted_sse_atol&&abs(full_gradient[2])<=literal.nnls.weighted_sse_atol
            @test T._verify_solver(solver,problem.Z,problem.rhs,problem.weights)
        end
    end
    true
end
