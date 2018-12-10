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
    @test occursin("ptxcall_dummy", sprint(io->(@device_code_llvm io=io @cuda dummy())))
    @test occursin("ptxcall_dummy", sprint(io->(@device_code_ptx io=io @cuda dummy())))
    @test occursin("ptxcall_dummy", sprint(io->(@device_code_sass io=io @cuda dummy())))
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

    input_dev = Mem.upload(input)
    output_dev = Mem.alloc(input)

    @cuda threads=len kernel(Base.unsafe_convert(Ptr{Float32}, input_dev),
                             Base.unsafe_convert(Ptr{Float32}, output_dev))
    Mem.download!(output, output_dev)
    @test input ≈ output
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

    arr_dev = Mem.upload(arr)
    val_dev = Mem.upload(val)

    @cuda threads=len kernel(Base.unsafe_convert(Ptr{Float32}, arr_dev),
                             Base.unsafe_convert(Ptr{Float32}, val_dev))
    @test arr[dims...] ≈ Mem.download(eltype(val), val_dev)[1]
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

    arr_dev = Mem.upload(arr)
    val_dev = Mem.upload(val)

    @cuda threads=len parent(Base.unsafe_convert(Ptr{Float32}, arr_dev),
                             Base.unsafe_convert(Ptr{Float32}, val_dev))
    @test arr[dims...] ≈ Mem.download(eltype(val), val_dev)[1]
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
    d_out = Mem.alloc(Int)

    @cuda kernel(keeps, Base.unsafe_convert(Ptr{Int}, d_out))
    @test Mem.download(Int, d_out) == [1]
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)
    c = similar(a)

    d_a = Mem.upload(a)
    d_b = Mem.upload(b)
    d_c = Mem.alloc(c)

    @eval struct ExecGhost end

    function kernel(ghost, a, b, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(c, unsafe_load(a,i)+unsafe_load(b,i), i)
        return
    end
    @cuda threads=len kernel(ExecGhost(),
                             Base.unsafe_convert(Ptr{Float32}, d_a),
                             Base.unsafe_convert(Ptr{Float32}, d_b),
                             Base.unsafe_convert(Ptr{Float32}, d_c))
    Mem.download!(c, d_c)
    @test a+b == c


    # bug: ghost type function parameters confused aggregate type rewriting

    function kernel(ghost, out, aggregate)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unsafe_store!(out, aggregate[1], i)
        return
    end
    @cuda threads=len kernel(ExecGhost(), Base.unsafe_convert(Ptr{Float32}, d_c), (42,))

    Mem.download!(c, d_c)
    @test all(val->val==42, c)
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    function kernel(ptr, b)
        unsafe_store!(ptr, imag(b))
        return
    end

    buf = Mem.upload([0f0])
    x = ComplexF32(2,2)

    @cuda kernel(Base.unsafe_convert(Ptr{Float32}, buf), x)
    @test Mem.download(Float32, buf) == [imag(x)]
end


@testset "automatic recompilation" begin
    buf = Mem.alloc(Int)

    function kernel(ptr)
        unsafe_store!(ptr, 1)
        return
    end

    @cuda kernel(Base.unsafe_convert(Ptr{Int}, buf))
    @test Mem.download(Int, buf) == [1]

    function kernel(ptr)
        unsafe_store!(ptr, 2)
        return
    end

    @cuda kernel(Base.unsafe_convert(Ptr{Int}, buf))
    @test Mem.download(Int, buf) == [2]
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
    out_dev = Mem.upload(out)
    out_ptr = Base.unsafe_convert(Ptr{eltype(out)}, out_dev)

    @cuda kernel(out_ptr, 1, 2)
    @test Mem.download(eltype(out), out_dev)[1] == 3

    all_splat = (out_ptr, 3, 4)
    @cuda kernel(all_splat...)
    @test Mem.download(eltype(out), out_dev)[1] == 7

    partial_splat = (5, 6)
    @cuda kernel(out_ptr, partial_splat...)
    @test Mem.download(eltype(out), out_dev)[1] == 11
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
    a_dev = Mem.upload(a)

    outer(Base.unsafe_convert(Ptr{Float64}, a_dev), 2.)

    @test Mem.download(eltype(a), a_dev) ≈ [2.]
end

@testset "closures" begin
    function outer(a_dev, val)
       function inner(a)
            # captures `val`
            unsafe_store!(a, val)
            return
       end
       @cuda inner(Base.unsafe_convert(Ptr{Float64}, a_dev))
    end

    a = [1.]
    a_dev = Mem.upload(a)

    outer(a_dev, 2.)

    @test Mem.download(eltype(a), a_dev) ≈ [2.]
end

@testset "conversions" begin
    @eval struct Host   end
    @eval struct Device end

    Adapt.adapt_storage(::CUDAnative.Adaptor, a::Host) = Device()

    Base.convert(::Type{Int}, ::Host)   = 1
    Base.convert(::Type{Int}, ::Device) = 2

    out = [0]

    # convert arguments
    out_dev = Mem.upload(out)
    let arg = Host()
        @test Mem.download(eltype(out), out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg))
            return
        end
        @cuda kernel(arg, Base.unsafe_convert(Ptr{Int}, out_dev))
        @test Mem.download(eltype(out), out_dev) ≈ [2]
    end

    # convert tuples
    out_dev = Mem.upload(out)
    let arg = (Host(),)
        @test Mem.download(eltype(out), out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg[1]))
            return
        end
        @cuda kernel(arg, Base.unsafe_convert(Ptr{Int}, out_dev))
        @test Mem.download(eltype(out), out_dev) ≈ [2]
    end

    # convert named tuples
    out_dev = Mem.upload(out)
    let arg = (a=Host(),)
        @test Mem.download(eltype(out), out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg.a))
            return
        end
        @cuda kernel(arg, Base.unsafe_convert(Ptr{Int}, out_dev))
        @test Mem.download(eltype(out), out_dev) ≈ [2]
    end

    # don't convert structs
    out_dev = Mem.upload(out)
    @eval struct Nested
        a::Host
    end
    let arg = Nested(Host())
        @test Mem.download(eltype(out), out_dev) ≈ [0]
        function kernel(arg, out)
            unsafe_store!(out, convert(Int, arg.a))
            return
        end
        @cuda kernel(arg, Base.unsafe_convert(Ptr{Int}, out_dev))
        @test Mem.download(eltype(out), out_dev) ≈ [1]
    end
end

@testset "argument count" begin
    val = [0]
    val_dev = Mem.upload(val)
    ptr = Base.unsafe_convert(Ptr{Int}, val_dev)
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
        @test Mem.download(eltype(val), val_dev)[1] == sum(args)
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

        buf = Mem.alloc(T)
        @cuda kernel(Base.unsafe_convert(Ptr{T}, buf))

        val = Mem.download(T, buf)[1]
        Mem.free(buf)
        return val
    end

    using Test
    @test f(1) == 1
    @test f(2) == 2
end

end

############################################################################################

@testset "shmem divergence bug" begin

@testset "trap" begin
    function trap()
        ccall("llvm.trap", llvmcall, Cvoid, ())
    end

    function kernel(input::Int32, output::Ptr{Int32}, yes::Bool=true)
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

    let output = Mem.alloc(Cint, N)
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        @cuda threads=N kernel(input, convert(Ptr{eltype(input)}, output.ptr))
        @test Mem.download(Cint, output, N) == repeat([input], N)
    end

    let output = Mem.alloc(Cint, N)
        @cuda threads=N kernel(input, convert(Ptr{eltype(input)}, output.ptr), true)
        @test Mem.download(Cint, output, N) == repeat([input], N)
    end
end

@testset "unreachable" begin
    function unreachable()
        @cuprintf("go home ptxas you're drunk")
        Base.llvmcall("unreachable", Cvoid, Tuple{})
    end

    function kernel(input::Int32, output::Ptr{Int32}, yes::Bool=true)
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

    let output = Mem.alloc(Cint, N)
        # defaulting to `true` embeds this info in the PTX module,
        # allowing `ptxas` to emit validly-structured code.
        @cuda threads=N kernel(input, convert(Ptr{eltype(input)}, output.ptr))
        @test Mem.download(Cint, output, N) == repeat([input], N)
    end

    let output = Mem.alloc(Cint, N)
        @cuda threads=N kernel(input, convert(Ptr{eltype(input)}, output.ptr), true)
        @test Mem.download(Cint, output, N) == repeat([input], N)
    end
end

@testset "mapreduce (full)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuTestArray{T, N}; dims = :, init...) where {T, N}
        OT = Float32
        v0 =  0.0f0

        threads = 256
        out = CuTestArray{OT,1}((1,))
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
    dA = CuTestArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (full, complex)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuTestArray{T, N}; dims = :, init...) where {T, N}
        OT = Complex{Float32}
        v0 =  0.0f0+0im

        threads = 256
        out = CuTestArray{OT,1}((1,))
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
    dA = CuTestArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

@testset "mapreduce (reduced)" begin
    function mapreduce_gpu(f::Function, op::Function, A::CuTestArray{T, N}) where {T, N}
        OT = Int
        v0 = 0

        out = CuTestArray{OT,1}((1,))
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
    dA = CuTestArray(A)

    @test mapreduce(identity, +, A) ≈ mapreduce_gpu(identity, +, dA)
end

end

############################################################################################

end
