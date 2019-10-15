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

# pack arguments in a buffer that CUDA expects
@generated function pack_arguments(f::Function, args...)
    all(isbitstype, args) || throw(ArgumentError("Arguments to kernel should be bitstype."))

    ex = quote
        Base.@_inline_meta
    end

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.

    # put arguments in Ref boxes so that we can get a pointers to them
    arg_refs = Vector{Symbol}(undef, length(args))
    for i in 1:length(args)
        arg_refs[i] = gensym()
        push!(ex.args, :($(arg_refs[i]) = Base.RefValue(args[$i])))
    end

    # generate an array with pointers
    arg_ptrs = [:(Base.unsafe_convert(Ptr{Cvoid}, $(arg_refs[i]))) for i in 1:length(args)]

    append!(ex.args, (quote
        GC.@preserve $(arg_refs...) begin
            kernelParams = [$(arg_ptrs...)]
            f(kernelParams)
        end
    end).args)
    return ex
end

"""
    launch(f::CuFunction; args...; blocks::CuDim=1, threads::CuDim=1,
           cooperative=false, shmem=0, stream=CuDefaultStream())

Low-level call to launch a CUDA function `f` on the GPU, using `blocks` and `threads` as
respectively the grid and block configuration. Dynamic shared memory is allocated according
to `shmem`, and the kernel is launched on stream `stream`.

Arguments to a kernel should either be bitstype, in which case they will be copied to the
internal kernel parameter buffer, or a pointer to device memory.

This is a low-level call, prefer to use [`cudacall`](@ref) instead.
"""
function launch(f::CuFunction, args...; blocks::CuDim=1, threads::CuDim=1,
                cooperative::Bool=false, shmem::Integer=0,
                stream::CuStream=CuDefaultStream())
    blocks = CuDim3(blocks)
    threads = CuDim3(threads)
    (blocks.x>0 && blocks.y>0 && blocks.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (threads.x>0 && threads.y>0 && threads.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    pack_arguments(args...) do kernelParams
        if cooperative
            @apicall(:cuLaunchCooperativeKernel, (
                CuFunction_t,           # function
                Cuint, Cuint, Cuint,    # grid dimensions (x, y, z)
                Cuint, Cuint, Cuint,    # block dimensions (x, y, z)
                Cuint,                  # shared memory bytes,
                CuStream_t,             # stream
                Ptr{Ptr{Cvoid}}),       # kernel parameters
                f,
                blocks.x, blocks.y, blocks.z,
                threads.x, threads.y, threads.z,
                shmem, stream, kernelParams)
        else
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
        end
    end
end

# convert the argument values to match the kernel's signature (specified by the user)
# (this mimics `lower-ccall` in julia-syntax.scm)
@generated function convert_arguments(f::Function, ::Type{tt}, args...) where {tt}
    types = tt.parameters

    ex = quote
        Base.@_inline_meta
    end

    converted_args = Vector{Symbol}(undef, length(args))
    arg_ptrs = Vector{Symbol}(undef, length(args))
    for i in 1:length(args)
        converted_args[i] = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($(converted_args[i]) = Base.cconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = Base.unsafe_convert($(types[i]), $(converted_args[i]))))
    end

    append!(ex.args, (quote
        GC.@preserve $(converted_args...) begin
            f($(arg_ptrs...))
        end
    end).args)

    return ex
end

"""
    cudacall(f::CuFunction, types, values...; blocks::CuDim, threads::CuDim,
             cooperative=false, shmem=0, stream=CuDefaultStream())

`ccall`-like interface for launching a CUDA function `f` on a GPU.

For example:

    vadd = CuFunction(md, "vadd")
    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = Mem.upload(a)
    bd = Mem.upload(b)
    c = zeros(Float32, 10)
    cd = Mem.alloc(c)

    cudacall(vadd, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd; threads=10)
    Mem.download!(c, cd)

The `blocks` and `threads` arguments control the launch configuration, and should both
consist of either an integer, or a tuple of 1 to 3 integers (omitted dimensions default to
1). The `types` argument can contain both a tuple of types, and a tuple type, the latter
being slightly faster.
"""
cudacall

# FIXME: can we make this infer properly?
cudacall(f::CuFunction, types::Tuple, args...; kwargs...) where {N} =
    cudacall(f, Base.to_tuple_type(types), args...; kwargs...)

function cudacall(f::CuFunction, types::Type, args...; kwargs...)
    convert_arguments(types, args...) do pointers...
        launch(f, pointers...; kwargs...)
    end
end


## attributes

export attributes

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
