include("hostcall/area_manager.jl")
include("hostcall/hostref.jl")
include("hostcall/timer.jl")
include("hostcall/poller.jl")


"""
    handle_hostcall(llvmptr::Core.LLVMPtr{Int64,AS.Global})

Host side function that checks and executes outstanding hostmethods.
Checking only one index.
"""
function handle_hostcall(manager::AreaManager, ctx::CuContext, index::Int64)
    start!(timer)

    checked = check_area(manager, ctx, index)
    if checked === nothing
        stop!(timer, true)
        return false
    end

    (hostcall, ptrs) = checked
    hostcall = Val(hostcall)
    for ptr in ptrs
        try
            ## Handle hostcall
            handle_hostcall(hostcall, convert(Ptr{UInt8}, ptr))
        catch e
            println("ERROR ERROR")
            println(e)
        end
    end

    finish_area(manager, ctx, index)

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
    call_hostcall(::Type{R}, n, args::A)

Device side function to call a hostmethod. This function is invoked with `@cpu`.
`R` is the return argument type.
`n` is the hostcall id, a number to identify the required hostmethod to execute.
"""
@inline function call_hostcall(::Type{R}, n, args::A) where {R, A}
    # Get the area_manager of current execution
    kind_config = get_manager_kind()

    # Acquire lock
    (index, llvmptr) = acquire_lock(kind_config)

    # Store arguments
    args_ptr = reinterpret(Ptr{A}, llvmptr)
    unsafe_store!(args_ptr, args)

    # Call host
    call_host_function(kind_config, index, n)

    # Get return value
    ret_ptr = reinterpret(Ptr{R}, llvmptr)
    local ret = unsafe_load(ret_ptr)

    # Finish method
    finish_function(kind_config, index)

    # Return
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
