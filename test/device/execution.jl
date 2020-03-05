@testset "execution" begin

############################################################################################

dummy() = return

@testset "@cuda" begin

@test_throws UndefVarError @cuda undefined()
@test_throws MethodError @cuda dummy(1)


@testset "low-level interface" begin
    k = cufunction(dummy)
    k()
    k(; threads=1)

    CUDAnative.version(k)
    CUDAnative.memory(k)
    CUDAnative.registers(k)
    CUDAnative.maxthreads(k)
end


@testset "launch configuration" begin
    @cuda dummy()

    @cuda threads=1 dummy()
    @cuda threads=(1,1) dummy()
    @cuda threads=(1,1,1) dummy()

    @cuda blocks=1 dummy()
    @cuda blocks=(1,1) dummy()
    @cuda blocks=(1,1,1) dummy()

    @cuda config=(kernel)->() dummy()
    @cuda config=(kernel)->(threads=1,) dummy()
    @cuda config=(kernel)->(blocks=1,) dummy()
    @cuda config=(kernel)->(shmem=0,) dummy()
end


@testset "compilation params" begin
    @cuda dummy()

    @test_throws CuError @cuda threads=2 maxthreads=1 dummy()
    @cuda threads=2 dummy()
end


@testset "reflection" begin
    CUDAnative.code_lowered(dummy, Tuple{})
    CUDAnative.code_typed(dummy, Tuple{})
    CUDAnative.code_warntype(devnull, dummy, Tuple{})
    CUDAnative.code_llvm(devnull, dummy, Tuple{})
    CUDAnative.code_ptx(devnull, dummy, Tuple{})
    CUDAnative.code_sass(devnull, dummy, Tuple{})

    @device_code_lowered @cuda dummy()
    @device_code_typed @cuda dummy()
    @device_code_warntype io=devnull @cuda dummy()
    @device_code_llvm io=devnull @cuda dummy()
    @device_code_ptx io=devnull @cuda dummy()
    @device_code_sass io=devnull @cuda dummy()

    mktempdir() do dir
        @device_code dir=dir @cuda dummy()
    end

    @test_throws ErrorException @device_code_lowered nothing

    # make sure kernel name aliases are preserved in the generated code
    @test occursin("julia_dummy", sprint(io->(@device_code_llvm io=io optimize=false @cuda dummy())))
    @test occursin("julia_dummy", sprint(io->(@device_code_llvm io=io @cuda dummy())))
    @test occursin("julia_dummy", sprint(io->(@device_code_ptx io=io @cuda dummy())))
    @test occursin("julia_dummy", sprint(io->(@device_code_sass io=io @cuda dummy())))

    # make sure invalid kernels can be partially reflected upon
    let
        invalid_kernel() = throw()
        @test_throws CUDAnative.KernelError @cuda invalid_kernel()
        @test_throws CUDAnative.KernelError @grab_output @device_code_warntype @cuda invalid_kernel()
        out, err = @grab_output begin
            try
                @device_code_warntype @cuda invalid_kernel()
            catch
            end
        end
        @test occursin("Body::Union{}", err)
    end

    let
        range_kernel() = (0.0:0.1:100.0; nothing)

        @test_throws CUDAnative.InvalidIRError @cuda range_kernel()
    end

    # set name of kernel
    @test occursin("julia_mykernel", sprint(io->(@device_code_llvm io=io begin
        k = cufunction(dummy, name="mykernel")
        k()
    end)))
end


@testset "shared memory" begin
    @cuda shmem=1 dummy()
end


@testset "streams" begin
    s = CuStream()
    @cuda stream=s dummy()
end


@testset "external kernels" begin
    @eval module KernelModule
        export external_dummy
        external_dummy() = return
    end
    import ...KernelModule
    @cuda KernelModule.external_dummy()
    @eval begin
        using ...KernelModule
        @cuda external_dummy()
    end

    @eval module WrapperModule
        using CUDAnative
        @eval dummy() = return
        wrapper() = @cuda dummy()
    end
    WrapperModule.wrapper()
end


@testset "calling device function" begin
    @noinline child(i) = sink(i)
    function parent()
        child(1)
        return
    end

    @cuda parent()
end

end


############################################################################################

@testset "argument passing" begin

dims = (16, 16)
len = prod(dims)

@testset "manually allocated" begin
    function kernel(input, output)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        val = unsafe_load(input, i)
        unsafe_store!(output, val, i)

        return
    end

    input = round.(rand(Float32, dims) * 100)
    output = similar(input)

    input_dev = CuArray(input)
    output_dev = CuArray(output)

    @cuda threads=len kernel(pointer(input_dev), pointer(output_dev))
    @test input ≈ Array(output_dev)
end


@testset "scalar through single-value array" begin
    function kernel(a, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            _val = unsafe_load(a, i)
            unsafe_store!(x, _val)
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda threads=len kernel(pointer(arr_dev), pointer(val_dev))
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "scalar through single-value array, using device function" begin
    function child(a, i)
        return unsafe_load(a, i)
    end
    @noinline function parent(a, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        max = gridDim().x * blockDim().x
        if i == max
            _val = child(a, i)
            unsafe_store!(x, _val)
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = CuArray(arr)
    val_dev = CuArray(val)

    @cuda threads=len parent(pointer(arr_dev), pointer(val_dev))
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "tuples" begin
    # issue #7: tuples not passed by pointer

    function kernel(keeps, out)
        if keeps[1]
            unsafe_store!(out, 1)
        else
            unsafe_store!(out, 2)
        end
        return
    end

    keeps = (true,)
    d_out = CuArray(zeros(Int))

    @cuda kernel(keeps, pointer(d_out))
    @test Array(d_out)[] == 1
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)
    c = similar(a)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    @eval struct ExecGhost end

    function kernel(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)
        return
    end
    @cuda threads=len kernel(ExecGhost(), pointer(d_a), pointer(d_b), pointer(d_c))
    @test a+b == Array(d_c)


    # bug: ghost type function parameters confused aggregate type rewriting

    function kernel(ghost, out, aggregate)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(out, aggregate[1], i)
        return
    end
    @cuda threads=len kernel(ExecGhost(), pointer(d_c), (42,))

    @test all(val->val==42, Array(d_c))
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    function kernel(ptr, b)
        unsafe_store!(ptr, imag(b))
        return
    end

    arr = CuArray(zeros(Float32))
    x = ComplexF32(2,2)

    @cuda kernel(pointer(arr), x)
    @test Array(arr)[] == imag(x)
end


@testset "automatic recompilation" begin
    arr = CuArray(zeros(Int))

    function kernel(ptr)
        unsafe_store!(ptr, 1)
        return
    end

    @cuda kernel(pointer(arr))
    @test Array(arr)[] == 1

    function kernel(ptr)
        unsafe_store!(ptr, 2)
        return
    end

    @cuda kernel(pointer(arr))
    @test Array(arr)[] == 2
end


@testset "automatic recompilation (bis)" begin
    arr = CuArray(zeros(Int))

    @eval doit(ptr) = unsafe_store!(ptr, 1)

    function kernel(ptr)
        doit(ptr)
        return
    end

    @cuda kernel(pointer(arr))
    @test Array(arr)[] == 1

    @eval doit(ptr) = unsafe_store!(ptr, 2)

    @cuda kernel(pointer(arr))
    @test Array(arr)[] == 2
end


@testset "non-isbits arguments" begin
    function kernel1(T, i)
        sink(i)
        return
    end
    @cuda kernel1(Int, 1)

    function kernel2(T, i)
        sink(unsafe_trunc(T,i))
        return
    end
    @cuda kernel2(Int, 1.)
end


@testset "splatting" begin
    function kernel(out, a, b)
        unsafe_store!(out, a+b)
        return
    end

    out = [0]
    out_dev = CuArray(out)
    out_ptr = pointer(out_dev)

    @cuda kernel(out_ptr, 1, 2)
    @test Array(out_dev)[1] == 3

    all_splat = (out_ptr, 3, 4)
    @cuda kernel(all_splat...)
    @test Array(out_dev)[1] == 7

    partial_splat = (5, 6)
    @cuda kernel(out_ptr, partial_splat...)
    @test Array(out_dev)[1] == 11
end

@testset "object invoke" begin
    # this mimics what is generated by closure conversion

    @eval struct KernelObject{T} <: Function
        val::T
    end

    function (self::KernelObject)(a)
        unsafe_store!(a, self.val)
        return
    end

    function outer(a, val)
       inner = KernelObject(val)
       @cuda inner(a)
    end

    a = [1.]
    a_dev = CuArray(a)

    outer(pointer(a_dev), 2.)

    @test Array(a_dev) ≈ [2.]
end

@testset "closures" begin
    function outer(a_dev, val)
       function inner(a)
            # captures `val`
            unsafe_store!(a, val)
            return
       end
       @cuda inner(pointer(a_dev))
    end

    a = [1.]
    a_dev = CuArray(a)

    outer(a_dev, 2.)

    @test Array(a_dev) ≈ [2.]
end

@testset "conversions" begin
    @eval struct Host   end
    @eval struct Device end

    Adapt.adapt_storage(::CUDAnative.Adaptor, a::Host) = Device()

    Base.convert(::Type{Int}, ::Host)   = 1
    Base.convert(::Type{Int}, ::Device) = 2

    out = [0]

    # convert arguments
    out_dev = CuArray(out)
    let arg = Host()
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg))
            return
        end
        @cuda kernel(arg, pointer(out_dev))
        @test Array(out_dev) ≈ [2]
    end

    # convert tuples
    out_dev = CuArray(out)
    let arg = (Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg[1]))
            return
        end
        @cuda kernel(arg, pointer(out_dev))
        @test Array(out_dev) ≈ [2]
    end

    # convert named tuples
    out_dev = CuArray(out)
    let arg = (a=Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg.a))
            return
        end
        @cuda kernel(arg, pointer(out_dev))
        @test Array(out_dev) ≈ [2]
    end

    # don't convert structs
    out_dev = CuArray(out)
    @eval struct Nested
        a::Host
    end
    let arg = Nested(Host())
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg.a))
            return
        end
        @cuda kernel(arg, pointer(out_dev))
        @test Array(out_dev) ≈ [1]
    end
end

@testset "argument count" begin
    val = [0]
    val_dev = CuArray(val)
    cuda_ptr = pointer(val_dev)
    ptr = CUDAnative.DevicePtr{Int}(cuda_ptr)
    for i in (1, 10, 20, 35)
        variables = ('a':'z'..., 'A':'Z'...)
        params = [Symbol(variables[j]) for j in 1:i]
        # generate a kernel
        body = quote
            function kernel($(params...))
                unsafe_store!($ptr, $(Expr(:call, :+, params...)))
                return
            end
        end
        eval(body)
        args = [j for j in 1:i]
        call = Expr(:call, :kernel, args...)
        cudacall = :(@cuda $call)
        eval(cudacall)
        @test Array(val_dev)[1] == sum(args)
    end
end

@testset "keyword arguments" begin
    @eval inner_kwargf(foobar;foo=1, bar=2) = nothing

    @cuda (()->inner_kwargf(42;foo=1,bar=2))()

    @cuda (()->inner_kwargf(42))()

    @cuda (()->inner_kwargf(42;foo=1))()

    @cuda (()->inner_kwargf(42;bar=2))()

    @cuda (()->inner_kwargf(42;bar=2,foo=1))()
end

@testset "captured values" begin
    function f(capture::T) where {T}
        function kernel(ptr)
            unsafe_store!(ptr, capture)
            return
        end

        arr = CuArray(zeros(T))
        @cuda kernel(pointer(arr))

        return Array(arr)[1]
    end

    using Test
    @test f(1) == 1
    @test f(2) == 2
end

end

############################################################################################

@testset "exceptions" begin

@testset "stack traces at different debug levels" begin

script = """
    function kernel(ptr, val)
        unsafe_store!(ptr, Int(val))
        return
    end

    cpu = zeros(Int)
    gpu = CUDAdrv.Mem.alloc(CUDAdrv.Mem.Device, sizeof(cpu))
    gpu_ptr = convert(CUDAdrv.CuPtr{Int}, gpu)
    @cuda kernel(gpu_ptr, 1.2)
    unsafe_copyto!(pointer(cpu), gpu_ptr, 1)
"""

let (code, out, err) = julia_script(script, `-g0`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test isempty(out)
end

let (code, out, err) = julia_script(script, `-g1`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
    @test occursin("Run Julia on debug level 2 for device stack traces", out)
end

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
    if VERSION < v"1.3.0-DEV.270"
        @test occursin("[1] Type at float.jl", out)
    else
        @test occursin("[1] Int64 at float.jl", out)
    end
    @test occursin("[2] kernel at none:2", out)
end

end

@testset "#329" begin

script = """
    @noinline foo(a, i) = a[1] = i
    bar(a) = (foo(a, 42); nothing)

    ptr = CUDAnative.DevicePtr{Int,AS.Global}(0)
    arr = CuDeviceArray{Int,1,AS.Global}((0,), ptr)

    @cuda bar(arr)
    CUDAdrv.synchronize()
"""

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
    @test occursin("foo at none:1", out)
    @test occursin("bar at none:2", out)
end

end

end

############################################################################################

@testset "shmem divergence bug" begin

@testset "trap" begin
    function trap()
        ccall("llvm.trap", llvmcall, Cvoid, ())
    end

    function kernel(input::Int32, output::CUDAnative.DevicePtr{Int32}, yes::Bool=true)
        i = threadIdx().x

        temp = @cuStaticSharedMem(Cint, 1)
        if i == 1
            yes || trap()
            temp[1] = input
        end
        sync_threads()

        yes || trap()
        unsafe_store!(output, temp[1], i)

        return nothing
    end

    input = rand(Cint(1):Cint(100))
    N = 2

    let output = CuArray(zeros(Cint, N))
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr)
        @test Array(output) == fill(input, N)
    end

    let output = CuArray(zeros(Cint, N))
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr, true)
        @test Array(output) == fill(input, N)
    end
end

@testset "unreachable" begin
    function unreachable()
        @cuprintln("go home ptxas you're drunk")
        Base.llvmcall("unreachable", Cvoid, Tuple{})
    end

    function kernel(input::Int32, output::CUDAnative.DevicePtr{Int32}, yes::Bool=true)
        i = threadIdx().x

        temp = @cuStaticSharedMem(Cint, 1)
        if i == 1
            yes || unreachable()
            temp[1] = input
        end
        sync_threads()

        yes || unreachable()
        unsafe_store!(output, temp[1], i)

        return nothing
    end

    input = rand(Cint(1):Cint(100))
    N = 2

    let output = CuArray(zeros(Cint, N))
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr)
        @test Array(output) == fill(input, N)
    end

    let output = CuArray(zeros(Cint, N))
        ptr = pointer(output)
        @cuda threads=N kernel(input, ptr, true)
        @test Array(output) == fill(input, N)
    end
end

@testset "mapreduce (full)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}; dims = :, init...) where {T, N}
        OT = Float32
        v0 =  0.0f0

        threads = 256
        out = CuArray{OT}(undef, (1,))
        @cuda threads=threads reduce_kernel(f, op, v0, A, Val{threads}(), out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, ::Val{LMEM}, result) where {T, LMEM}
        tmp_local = @cuStaticSharedMem(T, LMEM)
        global_index = threadIdx().x
        acc = v0

        # Loop sequentially over chunks of input vector
        while global_index <= length(A)
            element = f(A[global_index])
            acc = op(acc, element)
            global_index += blockDim().x
        end

        # Perform parallel reduction
        local_index = threadIdx().x - 1
        @inbounds tmp_local[local_index + 1] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if local_index < offset
                other = tmp_local[local_index + offset + 1]
                mine = tmp_local[local_index + 1]
                tmp_local[local_index + 1] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if local_index == 0
            result[blockIdx().x] = @inbounds tmp_local[1]
        end

        return
    end

    A = rand(Float32, 1000)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (full, complex)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}; dims = :, init...) where {T, N}
        OT = Complex{Float32}
        v0 =  0.0f0+0im

        threads = 256
        out = CuArray{OT}(undef, (1,))
        @cuda threads=threads reduce_kernel(f, op, v0, A, Val{threads}(), out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, ::Val{LMEM}, result) where {T, LMEM}
        tmp_local = @cuStaticSharedMem(T, LMEM)
        global_index = threadIdx().x
        acc = v0

        # Loop sequentially over chunks of input vector
        while global_index <= length(A)
            element = f(A[global_index])
            acc = op(acc, element)
            global_index += blockDim().x
        end

        # Perform parallel reduction
        local_index = threadIdx().x - 1
        @inbounds tmp_local[local_index + 1] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if local_index < offset
                other = tmp_local[local_index + offset + 1]
                mine = tmp_local[local_index + 1]
                tmp_local[local_index + 1] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if local_index == 0
            result[blockIdx().x] = @inbounds tmp_local[1]
        end

        return
    end

    A = rand(Complex{Float32}, 1000)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (reduced)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuArray{T, N}) where {T, N}
        OT = Int
        v0 = 0

        out = CuArray{OT}(undef, (1,))
        @cuda threads=64 reduce_kernel(f, op, v0, A, out)
        Array(out)[1]
    end

    function reduce_kernel(f, op, v0::T, A, result) where {T}
        tmp_local = @cuStaticSharedMem(T, 64)
        acc = v0

        # Loop sequentially over chunks of input vector
        i = threadIdx().x
        while i <= length(A)
            element = f(A[i])
            acc = op(acc, element)
            i += blockDim().x
        end

        # Perform parallel reduction
        @inbounds tmp_local[threadIdx().x] = acc
        sync_threads()

        offset = blockDim().x ÷ 2
        while offset > 0
            @inbounds if threadIdx().x <= offset
                other = tmp_local[(threadIdx().x - 1) + offset + 1]
                mine = tmp_local[threadIdx().x]
                tmp_local[threadIdx().x] = op(mine, other)
            end
            sync_threads()
            offset = offset ÷ 2
        end

        if threadIdx().x == 1
            result[blockIdx().x] = @inbounds tmp_local[1]
        end

        return
    end

    A = rand(1:10, 100)
    dA = CuArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

end

############################################################################################

@testset "dynamic parallelism" begin

@testset "basic usage" begin
    function hello()
        @cuprint("Hello, ")
        @cuda dynamic=true world()
        return
    end

    @eval function world()
        @cuprint("World!")
        return
    end

    _, out = @grab_output begin
        @cuda hello()
        synchronize()
    end
    @test out == "Hello, World!"
end

@testset "anonymous functions" begin
    function hello()
        @cuprint("Hello, ")
        world = () -> (@cuprint("World!"); nothing)
        @cuda dynamic=true world()
        return
    end

    _, out = @grab_output begin
        @cuda hello()
        synchronize()
    end
    @test out == "Hello, World!"
end

if VERSION >= v"1.1" # behavior of captured variables (box or not) has improved over time
@testset "closures" begin
    function hello()
        x = 1
        @cuprint("Hello, ")
        world = () -> (@cuprint("World $(x)!"); nothing)
        @cuda dynamic=true world()
        return
    end

    _, out = @grab_output begin
        @cuda hello()
        synchronize()
    end
    @test out == "Hello, World 1!"
end
end

@testset "argument passing" begin
    ## padding

    function kernel(a, b, c)
        @cuprint("$a $b $c")
        return
    end

    for args in ((Int16(1), Int32(2), Int64(3)),    # padding
                 (Int32(1), Int32(2), Int32(3)),    # no padding, equal size
                 (Int64(1), Int32(2), Int16(3)),    # no padding, inequal size
                 (Int16(1), Int64(2), Int32(3)))    # mixed
        _, out = @grab_output begin
            @cuda kernel(args...)
            synchronize()
        end
        @test out == "1 2 3"
    end

    ## conversion

    function kernel(a)
        increment(a) = (a[1] += 1; nothing)

        a[1] = 1
        increment(a)
        @cuda dynamic=true increment(a)

        return
    end

    dA = CuArray{Int}(undef, (1,))
    @cuda kernel(dA)
    A = Array(dA)
    @test A == [3]
end

@testset "self-recursion" begin
    @eval function kernel(x::Bool)
        if x
            @cuprint("recurse ")
            @cuda dynamic=true kernel(false)
        else
            @cuprint("stop")
        end
       return
    end

    _, out = @grab_output begin
        @cuda kernel(true)
        synchronize()
    end
    @test out == "recurse stop"
end

@testset "deep recursion" begin
    @eval function kernel_a(x::Bool)
        @cuprint("a ")
        @cuda dynamic=true kernel_b(x)
        return
    end

    @eval function kernel_b(x::Bool)
        @cuprint("b ")
        @cuda dynamic=true kernel_c(x)
        return
    end

    @eval function kernel_c(x::Bool)
        @cuprint("c ")
        if x
            @cuprint("recurse ")
            @cuda dynamic=true kernel_a(false)
        else
            @cuprint("stop")
        end
        return
    end

    _, out = @grab_output begin
        @cuda kernel_a(true)
        synchronize()
    end
    @test out == "a b c recurse a b c stop"
end

@testset "streams" begin
    function hello()
        @cuprint("Hello, ")
        s = CuDeviceStream()
        @cuda dynamic=true stream=s world()
        CUDAnative.unsafe_destroy!(s)
        return
    end

    @eval function world()
        @cuprint("World!")
        return
    end

    _, out = @grab_output begin
        @cuda hello()
        synchronize()
    end
    @test out == "Hello, World!"
end

end

############################################################################################

if capability(device()) >= v"6.0" && attribute(device(), CUDAdrv.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1

@testset "cooperative groups" begin
    function kernel_vadd(a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        grid_handle = this_grid()
        c[i] = a[i] + b[i]
        sync_grid(grid_handle)
        c[i] = c[1]
        return nothing
    end

    a = round.(rand(Float32, (300, 40)) * 100)
    b = round.(rand(Float32, (300, 40)) * 100)
    c = zeros(Float32, (300, 40))
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)  # output array
    @cuda cooperative=true threads=600 blocks=20 kernel_vadd(d_a, d_b, d_c)
    c = Array(d_c)
    @test all(c[1] .== c)
end

end

############################################################################################

end
