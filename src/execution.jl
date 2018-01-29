# Execution control

export
    CuDim, cudacall

"""
    CuDim3(x)

    CuDim3((x,))
    CuDim3((x, y))
    CuDim3((x, y, x))

A type used to specify dimensions, consisting of 3 integers for respectively the `x`, `y`
and `z` dimension. Unspecified dimensions default to `1`.

Often accepted as argument through the `CuDim` type alias, eg. in the case of
[`cudacall`](@ref) or [`launch`](@ref), allowing to pass dimensions as a plain integer or a
tuple without having to construct an explicit `CuDim3` object.
"""
struct CuDim3
    x::Cuint
    y::Cuint
    z::Cuint
end

CuDim3(dims::Integer)             = CuDim3(dims,    Cuint(1), Cuint(1))
CuDim3(dims::NTuple{1,<:Integer}) = CuDim3(dims[1], Cuint(1), Cuint(1))
CuDim3(dims::NTuple{2,<:Integer}) = CuDim3(dims[1], dims[2],  Cuint(1))
CuDim3(dims::NTuple{3,<:Integer}) = CuDim3(dims[1], dims[2],  dims[3])

# Type alias for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `CuDim3((len, 2))`)
const CuDim = Union{Integer,
                    Tuple{Integer},
                    Tuple{Integer, Integer},
                    Tuple{Integer, Integer, Integer}}

"""
    launch(f::CuFunction, blocks::CuDim, threads::CuDim, args...;
           shmem=0, stream=CuDefaultStream())
    launch(f::CuFunction, blocks::CuDim, threads::CuDim, shmem::Int, stream::CuStream, args...)

Low-level call to launch a CUDA function `f` on the GPU, using `blocks` and `threads` as
respectively the grid and block configuration. Dynamic shared memory is allocated according
to `shmem`, and the kernel is launched on stream `stream`.

Arguments to a kernel should either be bitstype, in which case they will be copied to the
internal kernel parameter buffer, or a pointer to device memory.

This is a low-level call, prefer to use [`cudacall`](@ref) instead.
"""
@inline function launch(f::CuFunction, blocks::CuDim, threads::CuDim,
                        shmem::Int, stream::CuStream,
                        args...)
    blocks = CuDim3(blocks)
    threads = CuDim3(threads)
    (blocks.x>0 && blocks.y>0 && blocks.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (threads.x>0 && threads.y>0 && threads.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    _launch(f, blocks, threads, shmem, stream, args...)
end

# we need a generated function to get an args array,
# without having to inspect the types at runtime
@generated function _launch(f::CuFunction, blocks::CuDim3, threads::CuDim3,
                            shmem::Int, stream::CuStream,
                            args::NTuple{N,Any}) where N
    all(isbits, args.parameters) || throw(ArgumentError("Arguments to kernel should be bitstype."))

    ex = Expr(:block)
    push!(ex.args, :(Base.@_inline_meta))

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.

    # put arguments in Ref boxes so that we can get a pointers to them
    arg_refs = Vector{Symbol}(uninitialized, N)
    for i in 1:N
        arg_refs[i] = gensym()
        push!(ex.args, :($(arg_refs[i]) = Base.RefValue(args[$i])))
    end

    # generate an array with pointers
    arg_ptrs = [:(Base.unsafe_convert(Ptr{Cvoid}, $(arg_refs[i]))) for i in 1:N]
    push!(ex.args, :(kernelParams = [$(arg_ptrs...)]))

    # root the argument boxes to the array of pointers,
    # keeping them alive across the call to `cuLaunchKernel`
    if VERSION >= v"0.7.0-DEV.1850"
        push!(ex.args, :(Base.@gc_preserve $(arg_refs...) kernelParams))
    end

    push!(ex.args, :(
        @apicall(:cuLaunchKernel, (
            CuFunction_t,           # function
            Cuint, Cuint, Cuint,    # grid dimensions (x, y, z)
            Cuint, Cuint, Cuint,    # block dimensions (x, y, z)
            Cuint,                  # shared memory bytes,
            CuStream_t,             # stream
            Ptr{Ptr{Cvoid}},        # kernel parameters
            Ptr{Ptr{Cvoid}}),       # extra parameters
            f,
            blocks.x, blocks.y, blocks.z,
            threads.x, threads.y, threads.z,
            shmem, stream, kernelParams, C_NULL)
        )
    )

    return ex
end

"""
    cudacall(f::CuFunction, types, values...;
             blocks::CuDim, threads::CuDim, shmem=0, stream=CuDefaultStream())

`ccall`-like interface for launching a CUDA function `f` on a GPU.

For example:

    vadd = CuFunction(md, "vadd")
    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = CuArray(a)
    bd = CuArray(b)
    c = zeros(Float32, 10)
    cd = CuArray(c)

    cudacall(vadd, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd;
             threads=10)
    c = Array(cd)

The `blocks` and `threads` arguments control the launch configuration, and should both
consist of either an integer, or a tuple of 1 to 3 integers (omitted dimensions default to
1). The `types` argument can contain both a tuple of types, and a tuple type, the latter
being slightly faster.
"""
cudacall

@inline function cudacall(f::CuFunction, types::NTuple{N,DataType}, values::Vararg{Any,N};
                          kwargs...) where N
    # this cannot be inferred properly (because types only contains `DataType`s),
    # which results in the call `@generated _cudacall` getting expanded upon first use
    _cudacall(f, Tuple{types...}, values; kwargs...)
end

@inline function cudacall(f::CuFunction, tt::Type, values::Vararg{Any,N};
                          kwargs...) where N
    # in this case, the type of `tt` is `Tuple{<:DataType,...}`,
    # which means the generated function can be expanded earlier
    _cudacall(f, tt, values; kwargs...)
end

# we need a generated function to get a tuple of converted arguments (using unsafe_convert),
# without having to inspect the types at runtime
@generated function _cudacall(f::CuFunction, tt::Type, args::NTuple{N,Any};
                              blocks::CuDim=1, threads::CuDim=1,
                              shmem::Integer=0, stream::CuStream=CuDefaultStream()) where N
    types = tt.parameters[1].parameters     # the type of `tt` is Type{Tuple{<:DataType...}}

    ex = Expr(:block)
    push!(ex.args, :(Base.@_inline_meta))

    # convert the argument values to match the kernel's signature (specified by the user)
    # (this mimics `lower-ccall` in julia-syntax.scm)

    arg_ptrs = Vector{Symbol}(uninitialized, N)
    for i in 1:N
        converted_arg = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($converted_arg = Base.cconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = Base.unsafe_convert($(types[i]), $converted_arg)))

        # root the cconverted argument to the pointer,
        # keeping them alive across the call to `launch`
        if VERSION >= v"0.7.0-DEV.1850"
            push!(ex.args, :(Base.@gc_preserve $(converted_arg) $(arg_ptrs[i])))
        end
    end

    push!(ex.args, :(launch(f, blocks, threads, shmem, stream, ($(arg_ptrs...),))))

    return ex
end


## attributes

export attributes

@enum(CUfunction_attribute, FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK            = 0x00,
                            FUNC_ATTRIBUTE_SHARED_SIZE_BYTES                = 0x01,
                            FUNC_ATTRIBUTE_CONST_SIZE_BYTES                 = 0x02,
                            FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                 = 0x03,
                            FUNC_ATTRIBUTE_NUM_REGS                         = 0x04,
                            FUNC_ATTRIBUTE_PTX_VERSION                      = 0x05,
                            FUNC_ATTRIBUTE_BINARY_VERSION                   = 0x06,
                            FUNC_ATTRIBUTE_CACHE_MODE_CA                    = 0x07,
                            FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES    = 0x08,
                            FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 0x09)

struct AttributeDict <: AbstractDict{CUfunction_attribute,Cint}
    f::CuFunction
end

attributes(f::CuFunction) = AttributeDict(f)

function Base.getindex(dict::AttributeDict, attr::CUfunction_attribute)
    val = Ref{Cint}()
    @apicall(:cuFuncGetAttribute, (Ptr{Cint}, CUfunction_attribute, CuFunction_t),
             val, attr, dict.f)
    return val[]
end

Base.setindex!(dict::AttributeDict, val::Integer, attr::CUfunction_attribute) =
    @apicall(:cuFuncSetAttribute, (CuFunction_t, CUfunction_attribute, Cint),
             dict.f, attr, val)
