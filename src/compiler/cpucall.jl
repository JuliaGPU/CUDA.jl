const CPUCALL_AREA = "hostcall_area"


function wait_and_kill_watcher(e::CuEvent, ctx::CuContext)
    if !has_hostcalls(ctx)
        return
    end

    println("Start the watcher!")

    try
        while !query(e)
            handle_cpucall(ctx)
            yield()
        end
    catch e
        println("Failed $e")
    end
    println("Killed the watcher!")
end

cpucall_area_size() = sizeof(UInt8) * 1024 * 1024

@eval @inline cpucall_area() =
    Base.llvmcall(
        $("""@hostcall_area = weak externally_initialized global i$(WORD_SIZE) 0
             define i64 @entry() #0 {
                 %ptr = load i$(WORD_SIZE), i$(WORD_SIZE)* @hostcall_area, align 8
                 ret i$(WORD_SIZE) %ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), Ptr{Cvoid}, Tuple{})


const cpucall_areas = Dict{CuContext, Mem.HostBuffer}()

dump_memory(ty=UInt8) = dump_memory{UInt8}(ty)
dump_memory(ty::Type{T}, size=cpucall_area_size() ÷ sizeof(ty), ctx::CuContext=context()) where {T} = dump_memory(ty, size, ctx)
function dump_memory(::Type{T}, size::Int64, ctx::CuContext=context()) where {T}
    ptr    = convert(CuPtr{T}, cpucall_areas[ctx])
    cuarray = unsafe_wrap(CuArray{T}, ptr, size)
    println("Dump $cuarray")
end

function reset_cpucall_area!(ctx::CuContext)
    if !has_hostcalls(ctx)
        return
    end

    println("resetting")
    ptr    = convert(Ptr{Int64}, cpucall_areas[ctx])

    unsafe_store!(ptr, 0, 1)
    unsafe_store!(ptr, 0, 2)
end

"Allocate memory for cpu call area"
function create_cpucall_area!(mod::CuModule)
    flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, CPUCALL_AREA)
    cpucall_area = get!(cpucall_areas, mod.ctx,
        Mem.alloc(Mem.Host, cpucall_area_size(), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED))
    flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, cpucall_area))

    ptr    = convert(CuPtr{UInt8}, cpucall_area)
    cuarray = unsafe_wrap(CuArray{UInt8}, ptr, cpucall_area_size())
    fill!(cuarray, 0)

    return
end


has_hostcalls(ctx::CuContext) = haskey(cpucall_areas, ctx)

time_ms() = round(Int64, time() * 1000)

mutable struct Timer
    name :: String
    last_handled :: Int64
end

const timer = Timer("timer 1", time_ms())
const timer2 = Timer("timer 2", time_ms())

function update!(timer::Timer)
    time = time_ms()
    if time > timer.last_handled + 1
        println("$(timer.name): It's been a while $(time - timer.last_handled)ms")
    end
    timer.last_handled = time
end

function handle_cpucall(ctx::CuContext)
    # update!(timer)
    ptr    = convert(Ptr{Int64}, cpucall_areas[ctx])
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, ptr)
    ptr_u8 = convert(Ptr{UInt8}, cpucall_areas[ctx])


    v = atomic_cas!(llvmptr, CPU_CALL, CPU_HANDLING)

    if v != CPU_CALL
        # println("flag $v")
        return
    end

    # t = 0
    while atomic_cas!(llvmptr, CPU_CALL, CPU_HANDLING) == CPU_CALL
        # t += 1
        unsafe_store!(ptr, CPU_HANDLING)
    end
    # println("1: Had to try $t times")


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

    # Notify end
    # t = 0
    while atomic_cas!(llvmptr, CPU_HANDLING, CPU_DONE) == CPU_HANDLING
        # t += 1
        unsafe_store!(ptr, CPU_DONE)
    end
    # println("2: Had to try $t times")

end



struct TypeCache{T, I}
    stash::Dict{T, I}
    vec::Vector{T}
end

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

const cpufunctions = TypeCache{Tuple{Symbol, Expr}, Int64}(Dict(), Vector())

struct HostRef
    index::Int32
end

const host_refs = Vector{WeakRef}()

Base.show(io::IO, t::HostRef) = print(io, "HostRef to $(host_refs[t.index])")
Base.convert(::Type{HostRef}, t::HostRef) = t

function Base.convert(::Type{HostRef}, t::T) where {T}
    push!(host_refs, WeakRef(t))
    return HostRef(length(host_refs))
end

Base.convert(::Type{Any}, t::HostRef) = host_refs[t.index].value
Adapt.adapt(::InvAdaptor, t::HostRef) = host_refs[t.index].value


@inline function acquire_lock(ptr::Core.LLVMPtr{Int64,AS.Global}, expected::Int64, value::Int64, id)
    # try_count = 0
    ## TRY CAPTURE LOCK
    while atomic_cas!(ptr, expected, value) != expected # && try_count < 5000000
        # try_count += 1
    end

    return

    # if try_count == 5000000
    #     @cuprintln("Trycount $id maxed out")
    # end
end


const IDLE = convert(Int64, 0)
const CPU_DONE = convert(Int64, 1)
const LOADING = convert(Int64, 2)
const CPU_CALL = convert(Int64, 3)
const CPU_HANDLING = convert(Int64, 4)


function prettier_string(thing)
    lines = split(string(thing), "\n")
    lines = filter(x -> strip(x)[1] != '#', lines)
    return join(lines, "\n")
end


# Stores a value at a particular address.
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

function warp_serialized(func::Function)
    # Get the current thread's ID.
    thread_id = threadIdx().x - 1

    # Get the size of a warp.
    size = warpsize()

    local result
    i = 0
    while i < size
        if thread_id % size == i
            result = func()
        end
        i += 1
    end
    return result
end


@inline function call_hostcall(::Type{R}, n, args::A) where {R, A}
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, cpucall_area())
    ptr     = reinterpret(Ptr{UInt8}, llvmptr)
    ptr_64  = reinterpret(Ptr{Int64}, llvmptr)

    # warp_serialized() do
        ## STORE ARGS
        acquire_lock(llvmptr, IDLE, LOADING, 1)
        args_ptr =  reinterpret(Ptr{A}, ptr + 16)
        unsafe_store!(args_ptr, args)

        ## Notify CPU of syscall 'syscall'
        unsafe_store!(ptr + 8, n)
        unsafe_store!(ptr, CPU_CALL)

        local try_count = 0
        acquire_lock(llvmptr, CPU_DONE, LOADING, 2)

        ## GET RETURN ARGS
        unsafe_store!(ptr, LOADING)
        local ret = unsafe_load(reinterpret(Ptr{R}, ptr_64 + 16))
        unsafe_store!(ptr, IDLE)

        ret
    # end
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
