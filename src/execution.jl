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
immutable CuDim3
    x::Cuint
    y::Cuint
    z::Cuint
end

CuDim3{T <: Integer}(g::T) =           CuDim3(g,    Cuint(1), Cuint(1))
CuDim3{T <: Integer}(g::NTuple{1,T}) = CuDim3(g[1], Cuint(1), Cuint(1))
CuDim3{T <: Integer}(g::NTuple{2,T}) = CuDim3(g[1], g[2],     Cuint(1))
CuDim3{T <: Integer}(g::NTuple{3,T}) = CuDim3(g[1], g[2],     g[3])

# Type alias for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `CuDim3((len, 2))`)
const CuDim = Union{CuDim3,
                    Integer,
                    Tuple{Integer},
                    Tuple{Integer, Integer},
                    Tuple{Integer, Integer, Integer}}

"""
    launch(f::CuFunction, griddim::CuDim, blockdim::CuDim, args...;
           shmem=0, stream=CuDefaultStream())
    launch(f::CuFunction, griddim::CuDim, blockdim::CuDim, shmem::Int, stream::CuStream, args...)

Low-level call to launch a CUDA function `f` on the GPU, using `griddim` and `blockdim` as
respectively the grid and block configuration. Dynamic shared memory is allocated according
to `shmem`, and the kernel is launched on stream `stream`.

Arguments to a kernel should either be bitstype, in which case they will be copied to the
internal kernel parameter buffer, or a pointer to device memory.

Both a version with and without keyword arguments is provided, the latter being slightly
faster, but not providing default values for the `shmem` and `stream` arguments.

This is a low-level call, prefer to use [`cudacall`](@ref) instead.
"""
@inline function launch(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                        shmem::Int, stream::CuStream,
                        args...)
    griddim = CuDim3(griddim)
    blockdim = CuDim3(blockdim)
    (griddim.x>0 && griddim.y>0 && griddim.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    _launch(f, griddim, blockdim, shmem, stream, args...)
end

@inline launch(f::CuFunction, griddim::CuDim, blockdim::CuDim, args...;
               shmem::Int=0, stream::CuStream=CuDefaultStream()) =
    launch(f, griddim, blockdim, shmem, stream, args...)

# we need a generated function to get an args array,
# without having to inspect the types at runtime
@generated function _launch{N}(f::CuFunction, griddim::CuDim3, blockdim::CuDim3,
                               shmem::Int, stream::CuStream,
                               args::NTuple{N,Any})
    arg_exprs = [:( args[$i] ) for i in 1:N]
    arg_types = args.parameters

    all(isbits, arg_types) || throw(ArgumentError("Arguments to kernel should be bitstype"))

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.
    # TODO: can this be done with Ref->Ptr instead?
    arg_exprs = [:(Base.pointer_from_objref($ex)) for ex in arg_exprs]

    quote
        Base.@_inline_meta
        @apicall(:cuLaunchKernel, (
            CuFunction_t,           # function
            Cuint, Cuint, Cuint,    # grid dimensions (x, y, z)
            Cuint, Cuint, Cuint,    # block dimensions (x, y, z)
            Cuint,                  # shared memory bytes,
            CuStream_t,             # stream
            Ptr{Ptr{Void}},         # kernel parameters
            Ptr{Ptr{Void}}),        # extra parameters
            f,
            griddim.x, griddim.y, griddim.z,
            blockdim.x, blockdim.y, blockdim.z,
            shmem, stream, [$(arg_exprs...)], C_NULL)
    end
end

"""
    cudacall(f::CuFunction, griddim::CuDim, blockdim::CuDim, types, values;
             shmem=0, stream=CuDefaultStream())
    cudacall(f::CuFunction, griddim::CuDim, blockdim::CuDim,
             shmem::Integer, stream::CuStream,
             types, values)

`ccall`-like interface for launching a CUDA function `f` on a GPU.

For example:

    vadd = CuFunction(md, "vadd")
    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = CuArray(a)
    bd = CuArray(b)
    c = zeros(Float32, 10)
    cd = CuArray(c)

    cudacall(vadd, 10, 1, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), ad, bd, cd)
    c = Array(cd)

The `griddim` and `blockdim` arguments control the launch configuration, and should both
consist of either an integer, or a tuple of 1 to 3 integers (omitted dimensions default to
1).

Both a version with and without keyword arguments is provided, the latter being slightly
faster, but not providing default values for the `shmem` and `stream` arguments. In
addition, the `types` argument can contain both a tuple of types, and a tuple type, again
the former being slightly faster.
"""
@inline cudacall{N}(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                    typespec::Union{NTuple{N,DataType},Type}, values::Vararg{Any,N};
                    shmem::Integer=0, stream::CuStream=CuDefaultStream()) =
    cudacall(f, griddim, blockdim, shmem, stream, typespec, values...)

# kwargs are slow, so we provide versions of cudacall with all arguments specified
#
# in addition, we support both providing a tuple of types, and a tuple type,
# but we always pass a tuple type to the generated function backing `cudacall`
# as that gives it access to the actual types (a tuple of types yields `NTuple{N,Datatype}`)
#
# the most efficient combo is to use `cudacall` without kwargs, specifying a tuple type
# (this is what eg. CUDAnative.jl does)

@inline function cudacall{N}(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                             shmem::Integer, stream::CuStream,
                             types::NTuple{N,DataType}, values::Vararg{Any,N})
    tt = Tuple{types...}
    # this cannot be inferred properly (because types only contains `DataType`s),
    # which results in the call `@generated _cudacall` getting expanded upon first use
    _cudacall(f, griddim, blockdim, shmem, stream, tt, values)
end

@inline function cudacall{N}(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                             shmem::Integer, stream::CuStream,
                             tt::Type, values::Vararg{Any,N})
    # in this case, the type of `tt` is `Tuple{<:DataType,...}`,
    # which means the generated function can be expanded earlier
    _cudacall(f, griddim, blockdim, shmem, stream, tt, values)
end

# we need a generated function to get a tuple of converted arguments (using unsafe_convert),
# without having to inspect the types at runtime
@generated function _cudacall{N}(f::CuFunction, griddim::CuDim, blockdim::CuDim,
                                 shmem::Integer, stream::CuStream,
                                 tt, args::NTuple{N,Any})
    types = tt.parameters[1].parameters     # the type of `tt` is Type{Tuple{<:DataType...}}

    # convert the argument values to match the kernel's signature (specified by the user)
    values = Expr(:tuple)
    for i in 1:N
        push!(values.args, :(Base.unsafe_convert($(types[i]), Base.cconvert($(types[i]), args[$i]))))
    end

    quote
        Base.@_inline_meta
        launch(f, CuDim3(griddim), CuDim3(blockdim), shmem, stream, $values)
    end
end
