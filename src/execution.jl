# Execution control

export
    CuDim, cudacall

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
const CuDim = Union{Integer,
                    Tuple{Integer},
                    Tuple{Integer, Integer},
                    Tuple{Integer, Integer, Integer}}

"""
Launch a CUDA function on a GPU.

This is a low-level call, prefer to use `cudacall` instead.
"""
@inline function launch{N}(f::CuFunction, griddim::CuDim3, blockdim::CuDim3,
                           shmem::Int, stream::CuStream,
                           args::NTuple{N,Any})
    (griddim.x>0 && griddim.y>0 && griddim.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    _launch(f, griddim, blockdim, shmem, stream, args)
end

# we need a generated function to get an args array (DevicePtr->Ptr && pointer_from_objref),
# without having to inspect the types at runtime
@generated function _launch{N}(f::CuFunction, griddim::CuDim3, blockdim::CuDim3,
                               shmem::Int, stream::CuStream,
                               args::NTuple{N,Any})
    arg_exprs = [:( args[$i] ) for i in 1:N]
    arg_types = args.parameters

    all(isbits(t) || t <: DevicePtr for t in arg_types) ||
        throw(ArgumentError("Arguments to kernel should be bitstype or device pointer"))

    # extract inner pointer
    for i in 1:N
        if arg_types[i] <: DevicePtr
            arg_exprs[i] = :($(arg_exprs[i]).ptr)
        end
    end

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.
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
ccall-like interface for launching a CUDA function on a GPU
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