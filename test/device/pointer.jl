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
    if cached
        @on_device unsafe_store!($ptr_b, unsafe_cached_load($ptr_a))
    else
        @on_device unsafe_store!($ptr_b, unsafe_load($ptr_a))
    end
    @test Mem.download(T, d_a) == Mem.download(T, d_b)
end

end

end
