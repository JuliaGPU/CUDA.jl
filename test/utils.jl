@testset "test utilities" begin
    mutable struct NoThrowTestSet <: Test.AbstractTestSet
        results::Vector
        NoThrowTestSet(desc) = new([])
    end
    Test.record(ts::NoThrowTestSet, t::Test.Result) = (push!(ts.results, t); t)
    Test.finish(ts::NoThrowTestSet) = ts.results
    fails = @testset NoThrowTestSet begin
        # OK
        @test_throws_cuerror CUDA.ERROR_UNKNOWN throw(CuError(CUDA.ERROR_UNKNOWN))
        # Fail, wrong CuError
        @test_throws_cuerror CUDA.ERROR_UNKNOWN throw(CuError(CUDA.ERROR_INVALID_VALUE))
        # Fail, wrong Exception
        @test_throws_cuerror CUDA.ERROR_UNKNOWN error()
    end
    @test isa(fails[1], Test.Pass)
    @test isa(fails[2], Test.Fail)
    @test isa(fails[3], Test.Fail)
end

@testset "@ssync" begin
  t = Base.@elapsed ret = CUDA.@sync begin
    # TODO: do something that takes a while on the GPU
    #       (need to wrap clock64 for that)
    42
  end
  @test t >= 0
  @test ret == 42
end
