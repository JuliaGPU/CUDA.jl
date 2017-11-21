@testset "memory" begin

let
    a,b = Mem.info()
    @test a == Mem.free()
    @test b == Mem.total()
    @test b-a == Mem.used()
end

# pointer-based
let
    src = 42

    buf1 = Mem.alloc(sizeof(src))

    Mem.set!(buf1, Cuint(0), sizeof(Int)÷sizeof(Cuint))

    Mem.upload!(buf1, Ref(src), sizeof(src))

    dst1 = Ref(0)
    Mem.download!(Ref(dst1), buf1, sizeof(src))
    @test src == dst1[]

    buf2 = Mem.alloc(sizeof(src))

    Mem.transfer!(buf2, buf1, sizeof(src))

    dst2 = Ref(0)
    Mem.download!(Ref(dst2), buf2, sizeof(src))
    @test src == dst2[]

    Mem.free(buf2)
    Mem.free(buf1)
end

# array-based
let
    src = [42]

    buf = Mem.alloc(src)

    Mem.upload!(buf, src)

    dst = similar(src)
    Mem.download!(dst, buf)
    @test src == dst

    Mem.free(buf)
end

# type-based
let
    buf = Mem.alloc(Int)

    # there's no type-based upload, duh
    src = [42]
    Mem.upload!(buf, src)

    @test src == Mem.download(eltype(src), buf)
end

# various
let
    @test_throws ArgumentError Mem.alloc(Function)   # abstract
    @test_throws ArgumentError Mem.alloc(Array{Int}) # UnionAll
    @test_throws ArgumentError Mem.alloc(Integer)    # abstract
    # TODO: can we test for the third case?
    #       !abstract && leaftype seems to imply UnionAll nowadays...
    @test_throws ArgumentError Mem.alloc(0)

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
