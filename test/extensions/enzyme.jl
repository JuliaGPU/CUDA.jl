using Enzyme, EnzymeCore
using GPUCompiler
using Test
using CUDA

@testset "compiler_job_from_backend" begin
    @test EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(()->nothing), Tuple{}) isa GPUCompiler.CompilerJob
end

function square_kernel!(x)
    i = threadIdx().x
    x[i] *= x[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x)
    @cuda blocks = 1 threads = length(x) square_kernel!(x)
    return nothing
end

@testset "Forward Kernel" begin
    A = CUDA.rand(64)
    dA = CUDA.ones(64)
    A .= (1:1:64)
    dA .= 1
    Enzyme.autodiff(Forward, square!, Duplicated(A, dA))
    @test all(dA .≈ (2:2:128))

    A = CUDA.rand(32)
    dA = CUDA.ones(32)
    dA2 = CUDA.ones(32)
    A .= (1:1:32)
    dA .= 1
    dA2 .= 3
    Enzyme.autodiff(Forward, square!, BatchDuplicated(A, (dA, dA2)))
    @test all(dA .≈ (2:2:64))
    @test all(dA2 .≈ 3*(2:2:64))
end

@testset "Forward Fill!" begin
    A = CUDA.ones(64)
    dA = CUDA.ones(64)
    Enzyme.autodiff(Forward, fill!, Duplicated(A, dA), Duplicated(2.0, 3.0))
    @test all(A .≈ 2.0)
    @test all(dA .≈ 3.0)
end

@testset "Reverse Fill!" begin
    A = CUDA.zeros(64)
    dA = CUDA.ones(64)
    res = Enzyme.autodiff(Reverse, fill!, Const, Duplicated(A, dA), Active(1.0))[1][2]
    @test res ≈ 64
    @test all(A .≈ 1)
    @test all(dA .≈ 0)
end

alloc(x) = CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}(undef, (x,))

@testset "Forward allocate" begin
    dup = Enzyme.autodiff(Forward, alloc, Duplicated, Const(10))
    @test all(dup[2] .≈ 0.0)
    
    dup = Enzyme.autodiff(Forward, alloc, DuplicatedNoNeed, Const(10))
    @test all(dup[1] .≈ 0.0)
end

@testset "Reverse allocate" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(alloc)}, Duplicated, Const{Int})
    tape, prim, shad = fwd(Const(alloc), Const(10))
    @test all(shad .≈ 0.0)
end

firstsum(x, y) = first(x .+ y)
@testset "Forward broadcast" begin
    x = CuArray(5*ones(5))
    y = CuArray(3*ones(5))
    dx = CuArray([1.0, 0.0, 0.0, 0.0, 0.0])
    dy = CuArray([0.2, 0.0, 0.1, 0.0, 0.0])
    # TODO enable once cuMemcpy derivatives are implemented
    #res = CUDA.@allowscalar autodiff(Forward, firstsum, Duplicated, Duplicated(x, dx), Duplicated(y, dy))
    #@test res[1] ≈ 8
    #@test res[2] ≈ 1.2
end

# TODO once reverse kernels are in
# function togpu(x)
#     x = CuArray(x)
#     square!(x)
# end
