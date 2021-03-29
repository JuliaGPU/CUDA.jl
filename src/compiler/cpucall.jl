const CPUCALL_AREA = "hostcall_area"

const cpucall_areas = Dict{CuContext, Mem.HostBuffer}()
has_hostcalls(ctx::CuContext) = haskey(cpucall_areas, ctx)

usleep(usecs) = ccall(:usleep, Cint, (Cuint,), usecs)

abstract type Poller end

struct AlwaysPoller <: Poller end
struct ConstantPoller <: Poller
    dur :: UInt32
end
mutable struct VarPoller <: Poller

end


function wait_and_kill_watcher(poller::P, event::CuEvent, ctx::CuContext) where {P<:Poller}
    empty!(host_refs)
    t = @async begin
        has_hostcalls(ctx) || return
        # reset!(timer)

        println("Start the watcher!")
        try
            launch_poller(poller, event, ctx)
        catch e
            println("Failed $e")
            stacktrace()
        end
        println("Killed the watcher!")
    end

    while !istaskstarted(t)
        yield()
    end

    return t
end


function launch_poller(::AlwaysPoller, e::CuEvent, ctx::CuContext)
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, cpucall_areas[ctx].ptr)
    count = cpucall_area_count()-1
    size = cpucall_area_size()

    while !query(e)
        for i in 0:count
            if handle_cpucall(llvmptr + i * size)
                println("handled $i")
            end
        end
        yield()
    end
end


function launch_poller(poller::ConstantPoller, e::CuEvent, ctx::CuContext)
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, cpucall_areas[ctx].ptr)
    count = cpucall_area_count()-1
    size = cpucall_area_size()

    while !query(e)
        for i in 0:count
            handle_cpucall(llvmptr + i * size)
        end

        usleep(poller.dur)
    end
end


function launch_poller(poller::VarPoller, e::CuEvent, ctx::CuContext)
    error("Not yet supported")
end


cpucall_area_count() = 20
cpucall_area_size() = sizeof(UInt8) * 1024

@eval @inline cpucall_area() =
    Base.llvmcall(
        $("""@hostcall_area = weak externally_initialized global i$(WORD_SIZE) 0
             define i64 @entry() #0 {
                 %ptr = load i$(WORD_SIZE), i$(WORD_SIZE)* @hostcall_area, align 8
                 ret i$(WORD_SIZE) %ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), Ptr{Cvoid}, Tuple{})



dump_memory(ty=UInt8) = dump_memory{UInt8}(ty)
dump_memory(ty::Type{T}, size=cpucall_area_size() ÷ sizeof(ty) * cpucall_area_count(), ctx::CuContext=context()) where {T} = dump_memory(ty, size, ctx)
function dump_memory(::Type{T}, size::Int64, ctx::CuContext=context()) where {T}
    ptr    = convert(CuPtr{T}, cpucall_areas[ctx])
    cuarray = unsafe_wrap(CuArray{T}, ptr, size)
    println("Dump $cuarray")
end


function reset_cpucall_area!(ctx::CuContext)
    has_hostcalls(ctx) || return

    println("resetting")
    ptr    = convert(Ptr{Int64}, cpucall_areas[ctx])

    unsafe_store!(ptr, 0, 1)
    unsafe_store!(ptr, 0, 2)
end


function create_cpucall_area!(mod::CuModule)
    flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, CPUCALL_AREA)
    cpucall_area = get!(cpucall_areas, mod.ctx,
        Mem.alloc(Mem.Host, cpucall_area_size() * cpucall_area_count(), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED))
    flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, cpucall_area))

    ptr    = convert(CuPtr{UInt8}, cpucall_area)
    cuarray = unsafe_wrap(CuArray{UInt8}, ptr, cpucall_area_size() * cpucall_area_count())
    fill!(cuarray, 0)

    return
end



time_ms() = round(Int64, time() * 1000)


mutable struct Timer
    sample::Int64
    total_duration::Float64
    total_duration_useless::Float64
    count::Int64
    count_useless::Int64
    last_start::Union{Nothing,Float64}
end

function start_sample!(timer::Timer)
    timer.sample += 1
end

const timer = Timer(0,0,0,0,0,nothing)

function start!(timer::Timer)
    sample = time()
    timer.last_start = sample
end


function stop!(timer::Timer, useless=True)
    sample = time()
    if isa(timer.last_start, Float64)
        delta = sample - timer.last_start
        if useless
            timer.total_duration_useless += delta
            timer.count_useless += 1
        end
        timer.total_duration += delta

        timer.count += 1
    else
        println("Something fucked")
    end
    timer.last_start = nothing
end

function Base.show(io::IO, timer::Timer)
    percentage_time = timer.total_duration_useless/timer.total_duration * 100
    percentage_count = Float32(timer.count_useless) / Float32(timer.count) * 100
    @printf(io, "<Timer tried: %d (%.3f%% useless) in %.3fs (%.3f%% useless) samples=%d>",
        timer.count / timer.sample, percentage_count,timer.total_duration / timer.sample, percentage_time, timer.sample)
end

function reset!(timer::Timer)
    timer.last_start = nothing
    timer.count = 0
    timer.sample = 0
    timer.count_useless = 0
    timer.total_duration = 0
    timer.total_duration_useless = 0
end

function time_it(f::Function)
    reset!(timer)

    f(timer)

    return timer
end


function handle_cpucall(llvmptr::Core.LLVMPtr{Int64,AS.Global})
    start!(timer)
    ptr     = reinterpret(Ptr{Int64}, llvmptr)
    ptr_u8  = reinterpret(Ptr{UInt8}, llvmptr)

    v = atomic_cas!(llvmptr, CPU_CALL, CPU_HANDLING)

    if v != CPU_CALL
        stop!(timer, true)
        return false
    end

    # this code is very fragile
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory
    # 'Note that atomic functions (see Atomic Functions) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices.'
    while atomic_cas!(llvmptr, CPU_CALL, CPU_HANDLING) == CPU_CALL
        unsafe_store!(ptr, CPU_HANDLING)
    end


    # Fetch this cpucall
    cpucall = unsafe_load(ptr+8)
    # update!(timer2)


    try
        ## Handle cpucall
        handle_cpucall(Val(cpucall), ptr_u8+16)
    catch e
        println("ERROR ERROR")
        println(e)
    end

    # this code is very fragile
    while atomic_cas!(llvmptr, CPU_HANDLING, CPU_DONE) == CPU_HANDLING
        unsafe_store!(ptr, CPU_DONE)
    end

    stop!(timer, false)
    return true
end



struct TypeCache{T, I}
    stash::Dict{T, I}
    vec::Vector{T}
end
const cpufunctions = TypeCache{Tuple{Symbol, Expr}, Int64}(Dict(), Vector())

function type_to_int!(cache::TypeCache{T, I}, type::T) where {T, I}
    if haskey(cache.stash, type)
        return cache.stash[type]
    else
        push!(cache.vec, type)
        cache.stash[type] = length(cache.vec)
    end
end

int_to_type(cache::TypeCache{T, I}, index::I) where {T, I} = cache.vec[index]


function handle_cpucall(::Val{N}, kwargs...) where {N}
    println("Syscall $N not yet supported")
end

function exec_hostmethode(::Type{A}, ::Type{R}, func::Function, ptr::Ptr{UInt8}) where {A, R}
    arg_tuple_ptr = reinterpret(Ptr{A}, ptr)
    arg_tuple = unsafe_load(arg_tuple_ptr)
    args = map(invcudaconvert, arg_tuple)

    # Actually call function
    ret = cudaconvert(func(args...))
    ret_ptr = reinterpret(Ptr{R}, ptr)
    unsafe_store!(ret_ptr, ret)
end


struct HostRef
    index::Int32
end

const host_refs = Vector{WeakRef}()
const host_refs_lk = ReentrantLock()

Base.show(io::IO, t::HostRef) = print(io, "HostRef to $(host_refs[t.index])")
Base.convert(::Type{HostRef}, t::HostRef) = t

function Base.convert(::Type{HostRef}, t::T) where {T}
    lock(host_refs_lk) do
        push!(host_refs, WeakRef(t))
    end

    return HostRef(length(host_refs))
end

Base.convert(::Type{Any}, t::HostRef) = host_refs[t.index].value
Adapt.adapt(::InvAdaptor, t::HostRef) = host_refs[t.index].value


@inline function acquire_lock(ptr::Core.LLVMPtr{Int64,AS.Global}, expected::Int64, value::Int64, count)
    ## try capture lock

    area_size = cpucall_area_size()
    i = 0

    while atomic_cas!(ptr + (i % count) * area_size, expected, value) != expected
        nanosleep(UInt32(16))
        i += 1
    end

    return ptr + (i % count) * area_size
end


const IDLE = Int64(0)
const CPU_DONE = Int64(1)
const LOADING = Int64(2)
const CPU_CALL = Int64(3)
const CPU_HANDLING = Int64(4)


function prettier_string(thing)
    lines = split(string(thing), "\n")
    lines = filter(x -> strip(x)[1] != '#', lines)
    return join(lines, "\n")
end


# unused
@generated function volatile_store!(ptr::Ptr{T}, value::T) where T
    JuliaContext() do ctx
        ptr_type = convert(LLVMType, Ptr{T}, ctx)
        lt = convert(LLVMType, T, ctx)

        ir = """
            %ptr = inttoptr $ptr_type %0 to $lt*
            store volatile $lt %1, $lt* %ptr
            ret void
            """
        :(Core.Intrinsics.llvmcall($ir, Cvoid, Tuple{$(Ptr{T}), $T}, ptr, value))
    end
end


# unused
@generated function volatile_load(ptr::Ptr{T}) where T
    JuliaContext() do ctx
        ptr_type = convert(LLVMType, Ptr{T}, ctx)
        lt = convert(LLVMType, T, ctx)

        ir = """
            %ptr = inttoptr $ptr_type %0 to $lt*
            %value = load volatile $lt, $lt* %ptr
            ret $lt %value
            """

        :(Base.llvmcall($ir, T, Tuple{Ptr{T}}, ptr))
    end
end


@inline function call_hostcall(::Type{R}, n, args::A) where {R, A}
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, cpucall_area())

    ## Store args
    llvmptr = acquire_lock(llvmptr, IDLE, LOADING, cpucall_area_count())


    ptr     = reinterpret(Ptr{UInt8}, llvmptr)
    ptr_64  = reinterpret(Ptr{Int64}, llvmptr)
    args_ptr =  reinterpret(Ptr{A}, ptr + 16)
    unsafe_store!(args_ptr, args)

    ## Notify CPU of syscall 'syscall'
    unsafe_store!(ptr + 8, n)
    unsafe_store!(ptr, CPU_CALL)

    local try_count = 0
    acquire_lock(llvmptr, CPU_DONE, LOADING, 1)

    ## get return args
    unsafe_store!(ptr, LOADING)
    local ret = unsafe_load(reinterpret(Ptr{R}, ptr_64 + 16))
    unsafe_store!(ptr, IDLE)

    ret
end


macro cpu(ex...)
    # destructure the `@cpu` expression
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the cpu call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    types_kwargs, other_kwargs = split_kwargs(kwargs, [:types])

    if length(types_kwargs) != 1
        throw(ArgumentError("'types' keyword argument is required (for now), with 1 tuple argument"))
    end

    _,val = types_kwargs[1].args

    arg_c = length(args) + 1 # number of arguments + return type
    types = eval(val)::NTuple{arg_c, DataType} # types of arguments

    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # make sure this exists
    # To be safe this should just increment
    # this has multiple problems
    indx = type_to_int!(cpufunctions, (f, val))

    println("hostcall $indx")
    # remember this module
    caller_module = __module__

    # Convert (Int, Int,) -> Tuple{Int, Int} which is the type of the arguments
    types_type_quote = :(Tuple{$(types[2:end]...)})

    # handle_cpucall function that is called from handle_cpucall(ctx::CuContext)
    new_fn = quote
        handle_cpucall(::Val{$indx}, ptr::Ptr{UInt8}) = exec_hostmethode($types_type_quote, $(types[1]), $caller_module.$f, ptr)
    end

    # Put function in julia space
    eval(new_fn)

    # Convert to correct arguments
    args_tuple = Expr(:tuple, args...)

    call_cpu = quote
        CUDA.call_hostcall($(types[1]), $indx, $args_tuple)
    end

    return esc(call_cpu)
end
