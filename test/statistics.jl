using Statistics

@testset "std" begin
    @test testf(std, rand(10))
    @test testf(std, rand(10,1,2))
    @test testf(std, rand(10,1,2); corrected=true)
    @test testf(std, rand(10,1,2); dims=1)
end

@testset "var" begin
    @test testf(var, rand(10))
    @test testf(var, rand(10,1,2))
    @test testf(var, rand(10,1,2); corrected=true)
    @test testf(var, rand(10,1,2); dims=1)
    @test testf(var, rand(10,1,2); dims=[1])
    @test testf(var, rand(10,1,2); dims=(1,))
    @test testf(var, rand(10,1,2); dims=[2,3])
    @test testf(var, rand(10,1,2); dims=(2,3))
end

@testset "mean" begin
    @test testf(mean, rand(2,2))
    @test testf(mean, rand(2,2); dims=2)
    @test testf(mean, rand(2,2,2); dims=[1,3])
    @test testf(x->mean(sin, x), rand(2,2))
    @test testf(x->mean(sin, x; dims=2), rand(2,2))
    @test testf(x->mean(sin, x; dims=[1,3]), rand(2,2,2))
end
