@testset "memory" begin

let
    a,b = Mem.info()
    @test a == Mem.free()
    @test b == Mem.total()
    @test b-a == Mem.used()
end

# pointer-based
let
    obj = 42

    buf = Mem.alloc(sizeof(obj))

    Mem.set(buf, Cuint(0), sizeof(Int)÷sizeof(Cuint))

    Mem.upload(buf, Ref(obj), sizeof(obj))

    obj_copy = Ref(0)
    Mem.download(Ref(obj_copy), buf, sizeof(obj))
    @test obj == obj_copy[]

    ptr2 = Mem.alloc(sizeof(obj))
    Mem.transfer(ptr2, buf, sizeof(obj))
    obj_copy2 = Ref(0)
    Mem.download(Ref(obj_copy2), ptr2, sizeof(obj))
    @test obj == obj_copy2[]

    Mem.free(ptr2)
    Mem.free(buf)
end

let
    dev_array = CuArray{Int32}(10)
    Mem.set(dev_array.buf, UInt32(0), 10)
    host_array = Array(dev_array)

    @test all(x -> x==0, host_array)
end

# object-based
let
    obj = 42

    buf = Mem.alloc(typeof(obj))

    Mem.upload(buf, obj)

    obj_copy = Mem.download(typeof(obj), buf)
    @test obj == obj_copy

    ptr2 = Mem.upload(obj)

    obj_copy2 = Mem.download(typeof(obj), ptr2)
    @test obj == obj_copy2
end

let
    @test_throws ArgumentError Mem.alloc(Function, 1)   # abstract
    @test_throws ArgumentError Mem.alloc(Array{Int}, 1) # UnionAll
    @test_throws ArgumentError Mem.alloc(Integer, 1)    # abstract
    # TODO: can we test for the third case?
    #       !abstract && leaftype seems to imply UnionAll nowadays...
    @test_throws ArgumentError Mem.alloc(Int, 0)

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
    buf = Mem.alloc(MutablePtrFree, 1)
    Mem.upload(buf, MutablePtrFree(0,0))
    Mem.free(buf)
end

let
    @eval mutable struct MutableNonPtrFree
        foo::Int
        bar::String
    end
    buf = Mem.alloc(MutableNonPtrFree, 1)
    @test_throws ArgumentError Mem.upload(buf, MutableNonPtrFree(0,""))
    Mem.free(buf)
end

end
