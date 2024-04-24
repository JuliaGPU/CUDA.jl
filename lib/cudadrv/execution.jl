# Execution control

export cudacall


## device

# pack arguments in a buffer that CUDA expects
@inline @generated function pack_arguments(f::Function, args...)
    for arg in args
        isbitstype(arg) || throw(ArgumentError("Arguments to kernel should be bitstype."))
    end

    ex = quote end

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
           cooperative=false, shmem=0, stream=stream())

Low-level call to launch a CUDA function `f` on the GPU, using `blocks` and `threads` as
respectively the grid and block configuration. Dynamic shared memory is allocated according
to `shmem`, and the kernel is launched on stream `stream`.

Arguments to a kernel should either be bitstype, in which case they will be copied to the
internal kernel parameter buffer, or a pointer to device memory.

This is a low-level call, prefer to use [`cudacall`](@ref) instead.
"""
function launch(f::CuFunction, args::Vararg{Any,N}; blocks::CuDim=1, threads::CuDim=1,
                cooperative::Bool=false, shmem::Integer=0,
                stream::CuStream=stream()) where {N}
    blockdim = CuDim3(blocks)
    threaddim = CuDim3(threads)

    try
        pack_arguments(args...) do kernelParams
            if cooperative
                cuLaunchCooperativeKernel(f,
                                          blockdim.x, blockdim.y, blockdim.z,
                                          threaddim.x, threaddim.y, threaddim.z,
                                          shmem, stream, kernelParams)
            else
                cuLaunchKernel(f,
                               blockdim.x, blockdim.y, blockdim.z,
                               threaddim.x, threaddim.y, threaddim.z,
                               shmem, stream, kernelParams, C_NULL)
            end
        end
    catch err
        diagnose_launch_failure(f, err; blockdim, threaddim, shmem)
    end
end

@noinline function diagnose_launch_failure(f::CuFunction, err; blockdim, threaddim, shmem)
    if !isa(err, CuError) || !in(err.code, [ERROR_INVALID_VALUE,
                                            ERROR_LAUNCH_OUT_OF_RESOURCES])
        rethrow()
    end

    # essentials
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) ||
        error("Grid dimensions should be non-null")
    (threaddim.x>0 && threaddim.y>0 && threaddim.z>0) ||
        error("Block dimensions should be non-null")

    # check device limits
    dev = device()
    ## block size limit
    threadlim = CuDim3(attribute(dev, DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                       attribute(dev, DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                       attribute(dev, DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
    for dim in (:x, :y, :z)
        if getfield(threaddim, dim) > getfield(threadlim, dim)
            error("Number of threads in $(dim)-dimension exceeds device limit ($(getfield(threaddim, dim)) > $(getfield(threadlim, dim))).")
        end
    end
    ## grid size limit
    blocklim = CuDim3(attribute(dev, DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
                      attribute(dev, DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
                      attribute(dev, DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
    for dim in (:x, :y, :z)
        if getfield(blockdim, dim) > getfield(blocklim, dim)
            error("Number of blocks in $(dim)-dimension exceeds device limit ($(getfield(blockdim, dim)) > $(getfield(blocklim, dim))).")
        end
    end
    ## shared memory limit
    shmem_lim = attribute(dev, DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    if shmem > shmem_lim
        error("Amount of dynamic shared memory exceeds device limit ($(Base.format_bytes(shmem)) > $(Base.format_bytes(shmem_lim))).")
    end

    # check kernel limits
    fattr = attributes(f)
    ## thread limit
    threadlim = fattr[FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
    if threaddim.x * threaddim.y * threaddim.z > threadlim
        error("Number of threads per block exceeds kernel limit ($(threaddim.x * threaddim.y * threaddim.z) > $threadlim).")
    end
    ## shared memory limit
    shmem_lim = fattr[FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]
    if shmem > shmem_lim
        error("Amount of dynamic shared memory exceeds kernel limit ($(Base.format_bytes(shmem)) > $(Base.format_bytes(shmem_lim))).")
    end

    rethrow()
end

# convert the argument values to match the kernel's signature (specified by the user)
# (this mimics `lower-ccall` in julia-syntax.scm)
@inline @generated function convert_arguments(f::Function, ::Type{tt}, args...) where {tt}
    types = tt.parameters

    ex = quote end

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
    cudacall(f, types, values...; blocks::CuDim, threads::CuDim,
             cooperative=false, shmem=0, stream=stream())

`ccall`-like interface for launching a CUDA function `f` on a GPU.

For example:

    vadd = CuFunction(md, "vadd")
    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = alloc(CUDA.DeviceMemory, 10*sizeof(Float32))
    unsafe_copyto!(ad, convert(Ptr{Cvoid}, a), 10*sizeof(Float32)))
    bd = alloc(CUDA.DeviceMemory, 10*sizeof(Float32))
    unsafe_copyto!(bd, convert(Ptr{Cvoid}, b), 10*sizeof(Float32)))
    c = zeros(Float32, 10)
    cd = alloc(CUDA.DeviceMemory, 10*sizeof(Float32))

    cudacall(vadd, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd; threads=10)
    unsafe_copyto!(convert(Ptr{Cvoid}, c), cd, 10*sizeof(Float32)))

The `blocks` and `threads` arguments control the launch configuration, and should both
consist of either an integer, or a tuple of 1 to 3 integers (omitted dimensions default to
1). The `types` argument can contain both a tuple of types, and a tuple type, the latter
being slightly faster.
"""
cudacall

cudacall(f::F, types::Tuple, args::Vararg{Any,N}; kwargs...) where {N,F} =
    cudacall(f, _to_tuple_type(types), args...; kwargs...)

function cudacall(f::F, types::Type{T}, args::Vararg{Any,N}; kwargs...) where {T,N,F}
    launch_closure = function (pointers::Vararg{Any,N})
        launch(f, pointers...; kwargs...)
    end
    convert_arguments(launch_closure, types, args...)
end

# From `julia/base/reflection.jl`, adjusted to add specialization on `t`.
function _to_tuple_type(t)
    if isa(t, Tuple) || isa(t, AbstractArray) || isa(t, SimpleVector)
        t = Tuple{t...}
    end
    if isa(t, Type) && t <: Tuple
        for p in (Base.unwrap_unionall(t)::DataType).parameters
            if isa(p, Core.TypeofVararg)
                p = Base.unwrapva(p)
            end
            if !(isa(p, Type) || isa(p, TypeVar))
                error("argument tuple type must contain only types")
            end
        end
    else
        error("expected tuple type")
    end
    t
end


## host

async_send(data::Ptr{Cvoid}) = ccall(:uv_async_send, Cint, (Ptr{Cvoid},), data)

function launch(f::Base.Callable; stream::CuStream=stream())
    cond = Base.AsyncCondition() do async_cond
        f()
        close(async_cond)
    end

    # the condition object is embedded in a task, so the Julia scheduler keeps it alive

    # callback = @cfunction(async_send, Cint, (Ptr{Cvoid},))
    # See https://github.com/JuliaGPU/CUDA.jl/issues/1314.
    # and https://github.com/JuliaLang/julia/issues/43748
    # TL;DR We are not allowed to cache `async_send` in the sysimage
    # so instead let's just pull out the function pointer and pass it instead.
    callback = cglobal(:uv_async_send)
    cuLaunchHostFunc(stream, callback, cond)
end


## attributes

export attributes

struct AttributeDict <: AbstractDict{CUfunction_attribute,Cint}
    f::CuFunction
end

attributes(f::CuFunction) = AttributeDict(f)

@enum_without_prefix CUfunction_attribute CU_

function Base.getindex(dict::AttributeDict, attr::CUfunction_attribute)
    val = Ref{Cint}()
    cuFuncGetAttribute(val, attr, dict.f)
    return val[]
end

Base.setindex!(dict::AttributeDict, val::Integer, attr::CUfunction_attribute) =
    cuFuncSetAttribute(dict.f, attr, val)
