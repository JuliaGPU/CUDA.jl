using JLD2: JLD2
using CUDA: CUDA
using Test: @test, @testset

@testset "JLD2 serialization" begin
    mktemp() do file, _
        a = CUDA.rand(10)
        JLD2.save_object(file, a)
        b = JLD2.load_object(file)
        @test a == b
        @test typeof(a) == typeof(b)
    end
end
