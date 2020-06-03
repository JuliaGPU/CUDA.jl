@testset "device pointer" begin

# inner constructors

voidptr_a = CuPtr{Cvoid}(Int(0xDEADBEEF))
generic_voidptr_a = CUDA.DevicePtr{Cvoid,AS.Generic}(voidptr_a)
global_voidptr_a = CUDA.DevicePtr{Cvoid,AS.Global}(voidptr_a)
local_voidptr_a = CUDA.DevicePtr{Cvoid,AS.Local}(voidptr_a)

voidptr_b = CuPtr{Cvoid}(Int(0xCAFEBABE))
generic_voidptr_b = CUDA.DevicePtr{Cvoid,AS.Generic}(voidptr_b)
global_voidptr_b = CUDA.DevicePtr{Cvoid,AS.Global}(voidptr_b)
local_voidptr_b = CUDA.DevicePtr{Cvoid,AS.Local}(voidptr_b)

intptr_b = convert(CuPtr{Int}, voidptr_b)
generic_intptr_b = CUDA.DevicePtr{Int,AS.Generic}(intptr_b)
global_intptr_b = CUDA.DevicePtr{Int,AS.Global}(intptr_b)
local_intptr_b = CUDA.DevicePtr{Int,AS.Local}(intptr_b)

# outer constructors
@test CUDA.DevicePtr{Cvoid}(voidptr_a) == generic_voidptr_a
@test CUDA.DevicePtr(voidptr_a) == generic_voidptr_a

# getters
@test eltype(generic_voidptr_a) == Cvoid
@test eltype(global_intptr_b) == Int
@test addrspace(generic_voidptr_a) == AS.Generic
@test addrspace(global_voidptr_a) == AS.Global
@test addrspace(local_voidptr_a) == AS.Local

# comparisons
@test generic_voidptr_a != global_voidptr_a
@test generic_voidptr_a != generic_intptr_b


@testset "conversions" begin

# between host and device pointers

@test convert(CuPtr{Cvoid}, generic_voidptr_a) == voidptr_a
@test convert(CUDA.DevicePtr{Cvoid}, voidptr_a) == generic_voidptr_a
@test convert(CUDA.DevicePtr{Cvoid,AS.Global}, voidptr_a) == global_voidptr_a


# between device pointers

@test_throws ArgumentError convert(typeof(local_voidptr_a), global_voidptr_a)
@test convert(typeof(generic_voidptr_a), generic_voidptr_a) == generic_voidptr_a
@test convert(typeof(global_voidptr_a), global_voidptr_a) == global_voidptr_a
@test Base.unsafe_convert(typeof(local_voidptr_a), global_voidptr_a) == local_voidptr_a

@test convert(typeof(global_voidptr_a), global_intptr_b) == global_voidptr_b
@test convert(typeof(generic_voidptr_a), global_intptr_b) == generic_voidptr_b
@test convert(typeof(global_voidptr_a), generic_intptr_b) == global_voidptr_b

@test convert(CUDA.DevicePtr{Cvoid}, global_intptr_b) == global_voidptr_b

end

@testset "unsafe_load & unsafe_store!" begin

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64, Int128, Float32, Float64),
             cached in (false, true)
    d_a = CuArray(ones(T))
    d_b = CuArray(zeros(T))

    ptr_a = CUDA.DevicePtr{T,AS.Global}(pointer(d_a))
    ptr_b = CUDA.DevicePtr{T,AS.Global}(pointer(d_b))
    @test Array(d_a) != Array(d_b)

    let ptr_a=ptr_a, ptr_b=ptr_b #JuliaLang/julia#15276
        if cached && capability(device()) >= v"3.2"
            @on_device unsafe_store!(ptr_b, unsafe_cached_load(ptr_a))
        else
            @on_device unsafe_store!(ptr_b, unsafe_load(ptr_a))
        end
    end
    @test Array(d_a) == Array(d_b)
end

@testset "indexing" begin
    function kernel(src, dst)
        unsafe_store!(dst, unsafe_cached_load(src, 4))
        return
    end

    T = Complex{Int8}
    # this also tests the fallback for unsafe_cached_load

    src = CuArray([T(1) T(9); T(3) T(4)])
    dst = CuArray([0])

    @cuda kernel(
        CUDA.DevicePtr{T,AS.Global}(pointer(src)),
        CUDA.DevicePtr{T,AS.Global}(pointer(dst))
    )

    @test Array(src)[4] == Array(dst)[1]
end

end

@testset "reinterpret(Nothing, nothing)" begin
    kernel(ptr) = Base.unsafe_load(ptr)
    @cuda kernel(reinterpret(CUDA.DevicePtr{Nothing,AS.Global}, C_NULL))
end

end
