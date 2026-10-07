# Execution control

# Kernel parameters need the address of the value slot, including for boxed values like Symbol.
mutable struct ArgBox{T}
    const val::T
end

@inline function Base.unsafe_convert(P::Union{Type{Ptr{T}},Type{Ptr{Cvoid}}},
                                     box::ArgBox{T})::P where {T}
    return pointer_from_objref(box)
end


## device

export cudacall

# The launch path passes kernel arguments around as a tuple, and processes them with generated
# functions. Julia does not optimize splatting more than 32 elements (the `max_tuple_splat`
# inference parameter), or `map` over 32 or more elements (`Base.Any32`), which would make
# launching kernels with many arguments slow, and impossible from the device.
#
# That includes methods with both a variable number of arguments and keyword arguments, whose
# lowered keyword-argument wrappers splat the positional arguments. For those, we define the
# positional and `Core.kwcall` methods explicitly, forwarding the arguments as a tuple.

# pack arguments in a buffer that CUDA expects
@inline @generated function pack_arguments(f::F, args::Tuple) where {F}
    n = fieldcount(args)
    quote
        boxes = ($((:(ArgBox(args[$i])) for i in 1:n)...),)
        GC.@preserve args boxes begin
            pointers = ($((:(Base.unsafe_convert(Ptr{Cvoid}, boxes[$i])) for i in 1:n)...),)
            f(Ref(pointers))
        end
    end
end

"""
    launch(f::CuFunction; args...; blocks::CuDim=1, threads::CuDim=1,
           clustersize::CuDim=1, cooperative=false, dependent=false,
           shmem=0, stream=stream())

Low-level call to launch a CUDA function `f` on the GPU, using `blocks` and `threads` as
respectively the grid and block configuration. Dynamic shared memory is allocated according
to `shmem`, and the kernel is launched on stream `stream`. If `clustersize > 1` and compute
capability is `>= 9.0`, [thread block clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
are launched. If `clustersize > 1` and compute capability is `< 9.0`, an error is thrown, as
thread block clusters are not supported.

Setting `dependent=true` marks this as a programmatically dependent launch, allowing it to
overlap with the preceding kernel in `stream`. The preceding kernel should call
[`trigger_programmatic_launch_completion`](@ref), and this kernel must call
[`grid_dependency_synchronize`](@ref) before accessing its results. This feature requires
compute capability 9.0 or higher.

Arguments to a kernel should either be bitstype, in which case they will be copied to the
internal kernel parameter buffer, or a pointer to device memory.

This is a low-level call, prefer to use [`cudacall`](@ref) instead.
"""
launch(f::CuFunction, args::Vararg{Any,N}) where {N} = launch_tuple(f, args)
Core.kwcall(kwargs::NamedTuple, ::typeof(launch), f::CuFunction, args::Vararg{Any,N}) where {N} =
    launch_tuple(f, args; kwargs...)

function launch_tuple(f::CuFunction, args::Tuple; blocks::CuDim=1, threads::CuDim=1,
                      clustersize::CuDim=1, cooperative::Bool=false, dependent::Bool=false,
                      shmem::Integer=0, stream::CuStream=stream())
    blockdim = CuDim3(blocks)
    threaddim = CuDim3(threads)
    clusterdim = CuDim3(clustersize)

    # cover the 32-bit word that 8- and 16-bit atomics operate on (see `pool_alloc`)
    shmem = cld(shmem, 4) * 4

    if dependent
        driver_version() >= v"11.8" ||
            error("Programmatic dependent launch requires CUDA 11.8 or higher")
        capability(device()) >= v"9.0" ||
            error("Programmatic dependent launch requires compute capability 9.0 or higher")
    end

    if driver_version() < v"11.8"
        # cuLaunchKernelEx and its launch attributes require CUDA 11.8.
        if clusterdim.x != 1 || clusterdim.y != 1 || clusterdim.z != 1
            error("Thread block clusters require CUDA 11.8 or higher")
        end
        try
            pack_arguments(args) do kernelParams
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
            diagnose_launch_failure(f, err; blockdim, threaddim, clusterdim, shmem)
        end
        return
    end

    attributes = Ref{NTuple{3,CUlaunchAttribute}}()
    GC.@preserve attributes stream begin
        attributes_ptr = Base.unsafe_convert(Ptr{CUlaunchAttribute}, attributes)
        num_attributes = 0
        if cooperative
            attribute = attributes_ptr + num_attributes * sizeof(CUlaunchAttribute)
            attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            attribute.value.cooperative = 1
            num_attributes += 1
        end
        if clusterdim.x != 1 || clusterdim.y != 1 || clusterdim.z != 1
            attribute = attributes_ptr + num_attributes * sizeof(CUlaunchAttribute)
            attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attribute.value.clusterDim.x = clusterdim.x
            attribute.value.clusterDim.y = clusterdim.y
            attribute.value.clusterDim.z = clusterdim.z
            num_attributes += 1
        end
        if dependent
            attribute = attributes_ptr + num_attributes * sizeof(CUlaunchAttribute)
            attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attribute.value.programmaticStreamSerializationAllowed = 1
            num_attributes += 1
        end

        config_attrs = num_attributes == 0 ? Ptr{CUlaunchAttribute}(C_NULL) : attributes_ptr
        config = CUlaunchConfig(blockdim.x, blockdim.y, blockdim.z,
                                threaddim.x, threaddim.y, threaddim.z,
                                shmem, stream.handle, config_attrs, num_attributes)
        try
            pack_arguments(args) do kernelParams
                cuLaunchKernelEx(config, f, kernelParams, C_NULL)
            end
        catch err
            diagnose_launch_failure(f, err; blockdim, threaddim, clusterdim, shmem)
        end
    end
end

@noinline function diagnose_launch_failure(f::CuFunction, err; blockdim, threaddim,
                                           clusterdim, shmem)
    if !isa(err, CuError) || !in(err.code, [ERROR_INVALID_VALUE,
                                            ERROR_LAUNCH_OUT_OF_RESOURCES])
        rethrow()
    end

    # essentials
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) ||
        error("Grid dimensions $blockdim are not positive")
    (threaddim.x>0 && threaddim.y>0 && threaddim.z>0) ||
        error("Block dimensions $threaddim are not positive")
    (clusterdim.x>0 && clusterdim.y>0 && clusterdim.z>0) ||
        error("Cluster dimensions $clusterdim are not positive")
    (blockdim.x % clusterdim.x == 0 && blockdim.y % clusterdim.y == 0 && blockdim.z % clusterdim.z == 0) ||
        error("Block dimensions $blockdim are not multiples of the cluster dimensions $clusterdim")

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
    ## cluster size limit
    active_clusters = clusterdim.x * clusterdim.y * clusterdim.z
    if capability(dev) >= v"9.0"
        cluster_launch = attribute(dev, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH) != 0
    else
        cluster_launch = false
    end
    if cluster_launch
        # It is difficult to determine the maximum cluster size. There is no attribute to query.
        # If we really want to report this then we should implement a stand-alone function for this
        # and then call this function here.
        #
        # The function to call is `cuOccupancyMaxPotentialClusterSize`,
        # which reports a value that depends on the function's attributes.
    else
        # Thread block clusters are not supported
         if active_clusters > 1
             error("Thread block cluster dimensions exceed device limit ($(clusterdim.x) * $(clusterdim.y) * $(clusterdim.z) > 1). (The device does not support thread block clusters.)")
         end
    end

    # check kernel limits
    fattr = attributes(f)
    nthreads = threaddim.x * threaddim.y * threaddim.z
    ## register pressure (names the resource on the register-binding subset of
    ## thread-limit failures; falls through to the thread-limit check otherwise)
    nregs = fattr[FUNC_ATTRIBUTE_NUM_REGS]
    reg_lim = attribute(dev, DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    if nregs * nthreads > reg_lim
        error("Block register count exceeds device limit ($nregs regs/thread * $nthreads threads/block = $(nregs * nthreads) > $reg_lim regs/block). " *
              "Reduce per-thread register use (e.g. via the `maxregs` compiler kwarg) or launch with fewer threads per block.")
    end
    ## thread limit
    threadlim = fattr[FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
    if nthreads > threadlim
        error("Number of threads per block exceeds kernel limit ($nthreads > $threadlim).")
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
@inline @generated function convert_arguments(f::Function, ::Type{tt}, args::Tuple) where {tt}
    types = tt.parameters
    n = fieldcount(args)

    ex = quote end

    converted_args = Vector{Symbol}(undef, n)
    arg_ptrs = Vector{Symbol}(undef, n)
    for i in 1:n
        converted_args[i] = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($(converted_args[i]) = Base.cconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = Base.unsafe_convert($(types[i]), $(converted_args[i]))))
    end

    append!(ex.args, (quote
        GC.@preserve $(converted_args...) begin
            f(($(arg_ptrs...),))
        end
    end).args)

    return ex
end

"""
    cudacall(f, types, values...; blocks::CuDim, threads::CuDim,
             cooperative=false, dependent=false, shmem=0, stream=stream())

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

# forwards the arguments as a tuple, see `launch_tuple(::CuFunction, ...)`
cudacall(f::F, types::Union{Tuple,Type}, args::Vararg{Any,N}) where {F,N} =
    cudacall_tuple(f, types, args)
Core.kwcall(kwargs::NamedTuple, ::typeof(cudacall), f::F, types::Union{Tuple,Type},
            args::Vararg{Any,N}) where {F,N} =
    cudacall_tuple(f, types, args; kwargs...)

cudacall_tuple(f::F, types::Tuple, args::Tuple; kwargs...) where {F} =
    cudacall_tuple(f, _to_tuple_type(types), args; kwargs...)

function cudacall_tuple(f::F, types::Type{T}, args::Tuple; kwargs...) where {F,T}
    convert_arguments(types, args) do pointers
        launch_tuple(f, pointers; kwargs...)
    end
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

@enum_without_prefix visibility=:public CUfunction_attribute CU_

function Base.getindex(dict::AttributeDict, attr::CUfunction_attribute)
    val = Ref{Cint}()
    cuFuncGetAttribute(val, attr, dict.f)
    return val[]
end

Base.setindex!(dict::AttributeDict, val::Integer, attr::CUfunction_attribute) =
    cuFuncSetAttribute(dict.f, attr, val)
