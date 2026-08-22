using CUDA, Adapt

## GPU-compatible unidirectional (device->host) I/O type
## backed by a circular buffer in host memory

# TODO:
# - use an additional head index to reduce number of system atomics,
#   initially only synchronizing device-side untill a flush happens.

mutable struct GPUIO <: IO
    const ctx::CuContext

    const data::CUDA.HostMemory     # UInt8
    const indices::CUDA.HostMemory  # UInt[head, tail]

    open::Bool

    monitor::Task

    function GPUIO(ctx = context(); capacity = 1024)
        dev, data, indices = context!(ctx) do
            device(),
                CUDA.alloc(CUDA.HostMemory, capacity),
                CUDA.alloc(CUDA.HostMemory, 2 * sizeof(UInt))
        end
        obj = new(ctx, data, indices, true)

        unsafe_store!(obj.headptr, 1)
        unsafe_store!(obj.tailptr, 1)

        obj.monitor = @async begin
            while !eof(io)
                try
                    line = readline(io)
                    println("      From device $(deviceid(dev)): $line")
                catch err
                    isa(err, EOFError) && break
                    rethrow()
                end
            end
        end

        finalizer(close, obj)

        return obj
    end
end

Base.length(io::GPUIO) = sizeof(io.data)

function Base.getproperty(io::GPUIO, name::Symbol)
    if name == :dataptr
        convert(Ptr{UInt8}, io.data)
    elseif name == :headptr
        convert(Ptr{UInt}, io.indices)
    elseif name == :tailptr
        convert(Ptr{UInt}, io.indices) + sizeof(UInt)
    else
        getfield(io, name)
    end
end

Base.isopen(io::GPUIO) = io.open
function Base.close(io::GPUIO)
    if io.open
        io.open = false

        # give any pending messages a change to be printed
        if !istaskdone(io.monitor)
            try
                wait(io.monitor)
            catch err
                @error "Exception occured during GPU I/O processing" exception = (err, catch_backtrace())
            end
        end

        context!(io.ctx; skip_destroyed = true) do
            CUDA.free(io.data)
            CUDA.free(io.indices)
        end
    end
end

# to be allocated in device memory
struct GPUDeviceIO <: IO
    dataptr::Core.LLVMPtr{UInt8, AS.Global}
    size::UInt
    headptr::Core.LLVMPtr{UInt, AS.Global}
    tailptr::Core.LLVMPtr{UInt, AS.Global}
end

Base.length(io::GPUDeviceIO) = io.size

# we assume that a CPU-side I/O object will be available when a kernel is running
Base.isopen(io::GPUDeviceIO) = true

function Adapt.adapt_storage(to::CUDA.KernelAdaptor, io::GPUIO)
    @assert isopen(io)
    return GPUDeviceIO(
        reinterpret(Core.LLVMPtr{UInt8, AS.Global}, io.dataptr),
        sizeof(io.data),
        reinterpret(Core.LLVMPtr{UInt, AS.Global}, io.headptr),
        reinterpret(Core.LLVMPtr{UInt, AS.Global}, io.tailptr),
    )
end


## GPU writer

Base.isreadable(io::GPUDeviceIO) = false
Base.iswritable(io::GPUDeviceIO) = true

function Base.write(io::GPUDeviceIO, c::UInt8)
    while true
        head = unsafe_load(io.headptr)
        tail = unsafe_load(io.tailptr)

        # block if full
        if mod1(head + 1, length(io)) == tail
            continue
        end

        # write a byte
        new_head = mod1(head + one(UInt), length(io))
        if CUDA.atomic_cas!(io.headptr, head, new_head) == head
            unsafe_store!(io.dataptr, c, head)
            threadfence_system()
            break
        end
    end
    return 1
end

function Base.unsafe_write(io::GPUDeviceIO, ptr::Ptr{UInt8}, n::UInt)
    nwritten = zero(UInt)
    while nwritten < n
        head = unsafe_load(io.headptr)
        tail = unsafe_load(io.tailptr)

        # block if full
        if mod1(head + 1, length(io)) == tail
            continue
        end

        # how many bytes can be written (contiguously)
        nwriteable = tail - head - 1
        if head >= tail
            nwriteable += length(io)
        end
        ncontiguous = length(io) - head + 1 # overestimation clamped below

        # write bytes
        ntodo = min(n - nwritten, nwriteable)
        new_head = mod1(head + ntodo, length(io))
        if head == CUDA.atomic_cas!(io.headptr, head, new_head) == head
            # split in two if wrapping
            if ntodo > ncontiguous
                unsafe_copyto!(io.dataptr + head - 1, ptr + nwritten, ncontiguous)
                unsafe_copyto!(io.dataptr, ptr + nwritten + ncontiguous, ntodo - ncontiguous)
            else
                unsafe_copyto!(io.dataptr + head - 1, ptr + nwritten, ntodo)
            end
            threadfence_system()
            nwritten += ntodo
        end
    end
    return nwritten
end

# get rid of locking and exceptions, and force specialization
function Base.print(io::GPUDeviceIO, x::T) where {T}
    show(io, x)
    return
end
function Base.print(io::GPUDeviceIO, xs::Vararg{Any, N}) where {N}
    for x in xs
        print(io, x)
    end
    return
end
## ambiguities with char
Base.print(io::GPUDeviceIO, c::Char) = (write(io, c); nothing)
## ambiguties with string
Base.print(io::GPUDeviceIO, s::AbstractString) = for c in s; print(io, c); end


## CPU reader

Base.isreadable(io::GPUIO) = true
Base.iswritable(io::GPUIO) = false

# closed and no bytes left
Base.eof(io::GPUIO) = !isopen(io) && unsafe_load(io.headptr) == unsafe_load(io.tailptr)

function Base.read(io::GPUIO, ::Type{UInt8})
    while true
        head = unsafe_load(io.headptr)
        tail = unsafe_load(io.tailptr)

        # block if empty
        if head == tail
            sleep(0.1)
            continue
        end

        # read a byte
        val = unsafe_load(io.dataptr, tail)
        unsafe_store!(io.tailptr, mod1(tail + 1, length(io)))
        unsafe_load(io.tailptr)
        return val
    end
end

function Base.unsafe_read(io::GPUIO, ptr::Ptr{UInt8}, n::UInt)
    nread = zero(UInt)
    while nread < n
        head = io.headptr
        tail = io.tailptr

        # block if empty
        if head == tail
            yield()
            continue
        end

        # how many bytes can be read (contiguously)
        nreadable = head - tail
        if head < tail
            nreadable += length(io)
        end
        ncontiguous = length(io) - tail + 1 # overestimation clamped below

        # read bytes
        ntodo = min(n - nread, nreadable)
        ## split in two if wrapping
        if ntodo > ncontiguous
            unsafe_copyto!(ptr + nread, io.dataptr + tail - 1, ncontiguous)
            unsafe_copyto!(ptr + nread + ncontiguous, io.dataptr, ntodo - ncontiguous)
        else
            unsafe_copyto!(ptr + nread, io.dataptr + tail - 1, ntodo)
        end
        nread += ntodo
        unsafe_store!(io.tailptr, mod1(tail + ntodo, length(io)))
    end
    return nread
end


# test

#using StaticStrings

io = GPUIO()

function kernel(dio)
    # working: bytes
    # Tuple(Vector{UInt8}("Hello, World!"))
    for c in (0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x2c, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x21)
        write(dio, c)
    end
    write(dio, 0x0a)  # newline

    # working: chars
    print(dio, 'H')
    print(dio, 'e')
    print(dio, 'l')
    print(dio, 'l')
    print(dio, 'o')
    print(dio, '\n')

    # working: series of chars
    print(dio, 'H', 'e', 'l', 'l', 'o', '\n')

    # todo: static strings
    #println(dio, static"Hello world!")
    return
end

@device_code_llvm dump_module = true @cuda kernel(io)
synchronize()
close(io)
