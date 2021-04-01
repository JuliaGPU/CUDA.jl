include("hostcall/hostref.jl")
include("hostcall/poller.jl")
include("hostcall/timer.jl")

const HOSTCALLAREA = "hostcall_area"

# Maps CuContext to big hostcall area
const hostcall_areas = Dict{CuContext, Mem.HostBuffer}()
has_hostcalls(ctx::CuContext) = haskey(hostcall_areas, ctx)


# flag states
const IDLE = Int64(0)           # nothing is happening
const HOST_DONE = Int64(1)      # the host has handled hostcall
const LOADING = Int64(2)        # host or device are transfering data
const HOST_CALL = Int64(3)      # host should handle hostcall
const HOST_HANDLING = Int64(4)  # host is handling hostcall

# Variable to indicate how many hostcall areas are/should be created
hostcall_area_count() = 20
# Variable to indicate how large a hostcall area is/should be
hostcall_area_size() = sizeof(UInt8) * 1024

@eval @inline hostcall_area() =
    Base.llvmcall(
        $("""@$(HOSTCALLAREA) = weak externally_initialized global i$(WORD_SIZE) 0
             define i64 @entry() #0 {
                 %ptr = load i$(WORD_SIZE), i$(WORD_SIZE)* @$(HOSTCALLAREA), align 8
                 ret i$(WORD_SIZE) %ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), Ptr{Cvoid}, Tuple{})


function reset_hostcall_area!(ctx::CuContext)
    has_hostcalls(ctx) || return

    ptr = convert(Ptr{Int64}, hostcall_areas[ctx])

    # Unset flag and hostcall id
    unsafe_store!(ptr, 0, 1)
    unsafe_store!(ptr, 0, 2)
end


function create_hostcall_area!(mod::CuModule)
    flag_ptr = CuGlobal{Ptr{Cvoid}}(mod, HOSTCALLAREA)
    hostcall_area = get!(hostcall_areas, mod.ctx,
        Mem.alloc(Mem.Host, hostcall_area_size() * hostcall_area_count(), Mem.HOSTALLOC_DEVICEMAP | Mem.HOSTALLOC_WRITECOMBINED))
    flag_ptr[] = reinterpret(Ptr{Cvoid}, convert(CuPtr{Cvoid}, hostcall_area))

    ptr    = convert(CuPtr{UInt8}, hostcall_area)
    cuarray = unsafe_wrap(CuArray{UInt8}, ptr, hostcall_area_size() * hostcall_area_count())
    fill!(cuarray, 0)

    return
end


"""
    handle_hostcall(llvmptr::Core.LLVMPtr{Int64,AS.Global})

Host side function that checks are executes outstanding hostmethods.
Checking only one hostcall area.
"""
function handle_hostcall(llvmptr::Core.LLVMPtr{Int64,AS.Global})
    start!(timer)
    ptr     = reinterpret(Ptr{Int64}, llvmptr)
    ptr_u8  = reinterpret(Ptr{UInt8}, llvmptr)

    v = atomic_cas!(llvmptr, HOST_CALL, HOST_HANDLING)

    if v != HOST_CALL
        stop!(timer, true)
        return false
    end

    # this code is very fragile
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory
    # 'Note that atomic functions (see Atomic Functions) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices.'
    count = 0
    while atomic_cas!(llvmptr, HOST_CALL, HOST_HANDLING) == HOST_CALL
        unsafe_store!(ptr, HOST_HANDLING)
    end


    # Fetch this hostcall
    hostcall = unsafe_load(ptr+8)
    # update!(timer2)


    try
        ## Handle hostcall
        handle_hostcall(Val(hostcall), ptr_u8+16)
    catch e
        println("ERROR ERROR")
        println(e)
    end

    # this code is very fragile
    while atomic_cas!(llvmptr, HOST_HANDLING, HOST_DONE) == HOST_HANDLING
        unsafe_store!(ptr, HOST_DONE)
    end

    stop!(timer, false)
    return true
end


"""
    handle_hostcall(::Val{N}, ptr::Ptr{UInt8})

Executes hostmethod with id N using hostcall area at `ptr`

This is the catch all function, but specialized functions are generated inside `@cpu`.
Specialized functions are aliased to `exec_hostmethode` with particular types.
"""
function handle_hostcall(::Val{N}, ptr::Ptr{UInt8}) where {N}
    println("Syscall $N not supported")
end


"""
    exec_hostmethode(::Type{A}, ::Type{R}, func::Function, ptr::Ptr{UInt8})

Downloads and invconverts arguments in hostcall area
Actually call the hostmethod
Set converted return argument in hostcall area
"""
function exec_hostmethode(::Type{A}, ::Type{R}, func::Function, ptr::Ptr{UInt8}) where {A, R}
    arg_tuple_ptr = reinterpret(Ptr{A}, ptr)
    arg_tuple = unsafe_load(arg_tuple_ptr)
    args = map(invcudaconvert, arg_tuple)

    # Actually call function
    ret = cudaconvert(func(args...))
    ret_ptr = reinterpret(Ptr{R}, ptr)
    unsafe_store!(ret_ptr, ret)
end


"""
    acquire_lock(ptr::LLVMPtr, expected::Int64, value::Int64, count)

Try to acquire a lock in `ptr + hostcall_area_size() * i` for i in range(count).
Loops over all hostcall areas until an acceptable hostcall area is found.
"""
@inline function acquire_lock(ptr::Core.LLVMPtr{Int64,AS.Global}, expected::Int64, value::Int64, count)
    ## try capture lock

    area_size = hostcall_area_size()
    i = 0

    while atomic_cas!(ptr + (i % count) * area_size, expected, value) != expected
        nanosleep(UInt32(16))
        i += 1
    end

    return ptr + (i % count) * area_size
end

"""
    call_hostcall(::Type{R}, n, args::A)

Device side function to call a hostmethod. This function is invoked with `@cpu`.
`R` is the return argument type.
`n` is the hostcall id, a number to identify the required hostmethod to execute.
"""
@inline function call_hostcall(::Type{R}, n, args::A) where {R, A}
    llvmptr = reinterpret(Core.LLVMPtr{Int64,AS.Global}, hostcall_area())

    ## Store args
    llvmptr = acquire_lock(llvmptr, IDLE, LOADING, hostcall_area_count())


    ptr     = reinterpret(Ptr{UInt8}, llvmptr)
    ptr_64  = reinterpret(Ptr{Int64}, llvmptr)
    args_ptr =  reinterpret(Ptr{A}, ptr + 16)
    unsafe_store!(args_ptr, args)

    ## Notify CPU of syscall 'syscall'
    unsafe_store!(ptr + 8, n)
    unsafe_store!(ptr, HOST_CALL)

    local try_count = 0
    # Just wait for HOST_DONE
    acquire_lock(llvmptr, HOST_DONE, LOADING, 1)

    ## get return args
    unsafe_store!(ptr, LOADING)
    local ret = unsafe_load(reinterpret(Ptr{R}, ptr_64 + 16))
    unsafe_store!(ptr, IDLE)

    ret
end


"""
    TypeCache{T, I}

Struct to relate T values to I index (mostly Int64, Int32)
Same T values are mapped to same I index
"""
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


""" Struct to keep track of all compiled hostmethods """
# This is not really used at it's best
const cpufunctions = TypeCache{Tuple{Symbol, Expr}, Int64}(Dict(), Vector())

"""
    @cpu [kwargs...] func(args...)

High-level interface for call a methode on the host from a device. The `@cpu` macro should prefix a call,
with `func` a callable function or object. Arguments from the device are converted using `invcudaconvert`
on the host before calling the function. Any return arguments are converted using `cudaconvert` before
returning to the device.

The keyword `types` is required and uses a Tuple of types. The first type is the return argument and
the others are the argument types (both as seen on device side).
"""
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

    # handle_hostcall function that is called from handle_hostcall(ctx::CuContext)
    new_fn = quote
        handle_hostcall(::Val{$indx}, ptr::Ptr{UInt8}) = exec_hostmethode($types_type_quote, $(types[1]), $caller_module.$f, ptr)
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


"""
    dump_memory(::Type{T}, ty_count::Int64, ctx::CuContext=context())

Print hostcall area of `ctx` as type `ty`.
"""
dump_memory(ty=UInt8) = dump_memory{UInt8}(ty)
dump_memory(ty::Type{T}, ty_count=hostcall_area_size() ÷ sizeof(ty) * hostcall_area_count(), ctx::CuContext=context()) where {T} = dump_memory(ty, ty_count, ctx)
function dump_memory(::Type{T}, ty_count::Int64, ctx::CuContext=context()) where {T}
    ptr    = convert(CuPtr{T}, hostcall_areas[ctx])
    cuarray = unsafe_wrap(CuArray{T}, ptr, ty_count)
    println("Dump $cuarray")
end

"""
    prettier_string(expr::Expr)::String

Util function to make generated code more readable, can be used for debugging.
"""
function prettier_string(expr)
    lines = split(string(expr), "\n")
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
