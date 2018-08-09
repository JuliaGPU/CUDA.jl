@testset "memory" begin

let
    a,b = Mem.info()
    # NOTE: actually testing this is pretty fragile on CI
    #=@test a == =# Mem.free()
    #=@test b == =# Mem.total()
    #=@test b-a == =# Mem.used()
end

# pointer-based
for async in [false, true]
    src = 42

    buf1 = Mem.alloc(sizeof(src))

    Mem.set!(buf1, UInt32(0), sizeof(Int)÷sizeof(UInt32); async=async)

    Mem.upload!(buf1, Ref(src), sizeof(src); async=async)

    dst1 = Ref(0)
    Mem.download!(dst1, buf1, sizeof(src); async=async)
    async && synchronize()
    @test src == dst1[]

    buf2 = Mem.alloc(sizeof(src))

    Mem.transfer!(buf2, buf1, sizeof(src); async=async)

    dst2 = Ref(0)
    Mem.download!(dst2, buf2, sizeof(src); async=async)
    async && synchronize()
    @test src == dst2[]

    Mem.free(buf2)
    Mem.free(buf1)
end

# array-based
for async in [false, true]
    src = [42]

    buf1 = Mem.alloc(src)

    Mem.upload!(buf1, src; async=async)

    dst1 = similar(src)
    Mem.download!(dst1, buf1; async=async)
    async && synchronize()
    @test src == dst1

    buf2 = Mem.upload(src)

    dst2 = similar(src)
    Mem.download!(dst2, buf2; async=async)
    async && synchronize()
    @test src == dst2

    Mem.free(buf1)
end

# type-based
for async in [false, true]
    buf = Mem.alloc(Int)

    # there's no type-based upload, duh
    src = [42]
    Mem.upload!(buf, src)

    dst = Mem.download(eltype(src), buf; async=async)
    async && synchronize()
    @test src == dst
end

let
    @test_throws ArgumentError Mem.alloc(Function, 1)   # abstract
    @test_throws ArgumentError Mem.alloc(Array{Int}, 1) # UnionAll
    @test_throws ArgumentError Mem.alloc(Integer, 1)    # abstract
    # TODO: can we test for the third case?
    #       !abstract && leaftype seems to imply UnionAll nowadays...

    # zero-width allocations should be permitted
    null = Mem.alloc(Int, 0)
    Mem.free(null)

    # double-free should throw (we rely on it for CuArray finalizer tests)
    x = Mem.alloc(1)
    Mem.free(x)
    @test_throws_cuerror CUDAdrv.ERROR_INVALID_VALUE Mem.free(x)
end

let
    @eval mutable struct MutablePtrFree
        foo::Int
        bar::Int
    end
    buf = Mem.alloc(MutablePtrFree)
    Mem.upload!(buf, [MutablePtrFree(0,0)])
    Mem.free(buf)
end

let
    @eval mutable struct MutableNonPtrFree
        foo::Int
        bar::String
    end
    @test_throws ArgumentError Mem.alloc(MutableNonPtrFree)
end

end
