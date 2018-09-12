@testset "pointer (on device)" begin

@testset "unsafe_load & unsafe_store!" begin

@eval struct LoadableStruct
    a::Int64
    b::UInt8
end
Base.one(::Type{LoadableStruct}) = LoadableStruct(1,1)
Base.zero(::Type{LoadableStruct}) = LoadableStruct(0,0)

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64, Int128,
                   Float32,Float64,
                   LoadableStruct),
             cached in (false, true)
    d_a = Mem.upload(ones(T))
    d_b = Mem.upload(zeros(T))

    ptr_a = CUDAnative.DevicePtr{T,AS.Global}(Base.unsafe_convert(Ptr{T}, d_a))
    ptr_b = CUDAnative.DevicePtr{T,AS.Global}(Base.unsafe_convert(Ptr{T}, d_b))

    @test Mem.download(T, d_a) != Mem.download(T, d_b)
    if cached && capability(dev) >= v"3.2"
        @on_device unsafe_store!($ptr_b, unsafe_cached_load($ptr_a))
    else
        @on_device unsafe_store!($ptr_b, unsafe_load($ptr_a))
    end
    @test Mem.download(T, d_a) == Mem.download(T, d_b)
end

@testset "indexing" begin
    @eval function issue_221(src, dst)
        unsafe_store!(dst, CUDAnative.unsafe_cached_load(src, 4))
    end

    T = Complex{Int8}

    src = Mem.upload([T(1) T(9); T(3) T(4)])
    dst = Mem.upload([0])

    @cuda issue_221(
        CUDAnative.DevicePtr{T,AS.Global}(Ptr{T}(src.ptr)),
        CUDAnative.DevicePtr{T,AS.Global}(Ptr{T}(dst.ptr))
    )

    @test Mem.download(T, src, 4)[4] == Mem.download(T, dst)[1]
end

end

end
