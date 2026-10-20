using GPUCompiler
using KernelAbstractions

function copy3d!(B, A)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if i <= size(A,1) && j <= size(A,2) && k <= size(A,3)
        @inbounds B[i,j,k] = A[i,j,k]
    end
    return
end

@testset "kernels" begin
    tt = Tuple{CuDeviceArray{Float32,3,AS.Global}, CuDeviceArray{Float32,3,AS.Global}}

    # with facts about the extents of the arguments, indices are computed in 32 bits
    @test @filecheck begin
        @check "call void @llvm.assume"
        @check "[[IDX:%nidx[0-9]*]] = zext nneg i32"
        @check "getelementptr float, {{.*}} i64 [[IDX]]"
        CUDA.code_llvm(copy3d!, tt; kernel=true, argument_ranges=CUDACore.candidate_index_ranges(typeof(copy3d!), tt))
    end
    @test @filecheck begin
        @check_not "%nidx"
        CUDA.code_llvm(copy3d!, tt; kernel=true, argument_ranges=())
    end

    # `@cuda` derives the facts from the arguments
    A = CUDA.rand(Float32, 7, 5, 3)
    B = similar(A)
    k = @cuda threads=(7, 5, 3) copy3d!(B, A)
    @test !isempty(k.specialization)
    @test Array(B) == Array(A)
end

# with facts, `last(r) > typemax(Int32)` is folded to `false`
function exceeds(out, r)
    out[1] = last(r) > typemax(Int32)
    return
end

@kernel function ka_exceeds(out, r)
    out[1] = last(r) > typemax(Int32)
end

# Scalar data stays unbounded; explicit range endpoints are specialization candidates.
function exceeds_signed(out, x, r)
    out[1] = x > typemax(Int32) || first(r) < typemin(Int32)
    return
end

store_first(out, xs) = (out[1] = first(xs); return)
function launch_allocations(k, args...)
    k(args...)
    @allocated k(args...)
end

world_helper() = 1
function world_kernel(out, r)
    out[1] = world_helper() + (last(r) > typemax(Int32))
    return
end

@testset "launch" begin
    small = Base.OneTo(10)
    huge = Base.OneTo(2^40)
    out = CuArray([false])

    k = @cuda exceeds(out, huge)
    @test isempty(k.specialization)
    @test Array(out) == [true]

    # compiled with facts, and launched with arguments that don't fit them
    k = @cuda launch=false exceeds(out, small)
    @test !isempty(k.specialization)
    k(out, small)
    @test Array(out) == [false]
    k(out, huge)
    @test Array(out) == [true]
    fallback = CUDACore._unbounded_kernels[k.fun]
    k(out, huge)
    @test Array(out) == [true]
    @test CUDACore._unbounded_kernels[k.fun] === fallback

    # the facts belong to the compiled function, also when reconstructing the kernel
    k′ = typeof(k)(k.f, k.fun, k.state)
    @test !isempty(k′.specialization)
    k′(out, huge)
    @test Array(out) == [true]

    # captured by the kernel function, and replaced after compilation
    captured(r) = out -> exceeds(out, r)
    k = @cuda launch=false captured(small)(out)
    @test !isempty(k.specialization)
    k′ = typeof(k)(captured(huge), k.fun, k.state)
    k′(out)
    @test Array(out) == [true]

    # captured by the kernel function
    k = @cuda (out -> exceeds(out, huge))(out)
    @test isempty(k.specialization)

    # kernels compiled from types assume that the arguments fit, and check them on launch
    fill!(out, false)
    ka_exceeds(CUDABackend())(out, huge; ndrange=1)
    @test Array(out) == [true]
    ka_exceeds(CUDABackend())(out, small; ndrange=1)
    @test Array(out) == [false]

    # Scalar data does not force fallback; range endpoints are checked in signed 32 bits.
    k = @cuda launch=false exceeds_signed(out, -1, -1:1)
    @test !isempty(k.specialization)
    k(out, 1, -1:1)
    @test Array(out) == [false]
    k(out, 2^40, -1:1)
    @test Array(out) == [true]
    k(out, 1, -2^40:1)
    @test Array(out) == [true]
    @test (@cuda exceeds_signed(out, -2^40, 1:1)).specialization !== ()

    # Unrelated Int fields add no guard work or allocations.
    many = ntuple(identity, 100)
    k = @cuda launch=false store_first(out, many)
    @test !isempty(k.specialization)
    k′ = @cuda launch=false specialize_indices=false store_first(out, many)
    @test launch_allocations(k, out, many) <= launch_allocations(k′, out, many)

    # An explicitly empty set of facts suppresses automatic classification too.
    @test isempty((@cuda launch=false argument_ranges=() exceeds(out, small)).specialization)

    # the fallback is compiled in the world of the original kernel
    out = CuArray([0])
    k = @cuda launch=false world_kernel(out, small)
    @eval world_helper() = 2
    Base.invokelatest(k, out, huge)
    @test Array(out) == [2]
end

function typed_index(out, ::Type{T}, r) where {T}
    out[1] = T(last(r))
    return
end

@testset "type-valued arguments" begin
    out = CuArray([0f0])
    k = @cuda typed_index(out, Float32, Base.OneTo(10))
    @test !isempty(k.specialization)
    @test Array(out) == [10f0]
end

function copy_view!(out, view)
    i = threadIdx().x
    if i <= length(view)
        @inbounds out[i] = view[i]
    end
    return
end

@testset "strided views" begin
    parent = CUDA.rand(Float32, 7, 5)
    v = view(parent, 1:2:7, :)
    converted = cudaconvert(v)
    @test converted isa SubArray
    paths = first.(CUDACore.index_ranges(typeof(converted)))
    @test (:offset1,) in paths
    @test (:stride1,) in paths
    @test (:indices, 1, :step) in paths
    out = similar(parent, size(v))
    k = @cuda threads=length(v) copy_view!(out, v)
    @test !isempty(k.specialization)
    @test Array(out) == Array(v)
end
