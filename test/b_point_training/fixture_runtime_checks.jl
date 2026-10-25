function fixture_runtime_checks()
    blas=F.BLAS
    caller=blas.get_num_threads()
    try
        blas.set_num_threads(2)
        @test F.with_single_blas(()->blas.get_num_threads())==1
        @test blas.get_num_threads()==2
        @test_throws ErrorException F.with_single_blas(()->error("generated fixture restoration control"))
        @test blas.get_num_threads()==2
    finally
        blas.set_num_threads(caller)
    end
    @test blas.get_num_threads()==caller==1
    true
end
