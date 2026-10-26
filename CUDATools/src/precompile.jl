# @profile infrastructure (GPU-dependent, can't execute during precompilation)
precompile(Tuple{typeof(Profile.detect_cupti)})
precompile(Tuple{typeof(Profile.profile_internally), Function})
precompile(Tuple{typeof(Profile.capture), CUPTI.ActivityConfig, UInt64})
precompile(Tuple{typeof(Profile.profile_metrics), Function})
precompile(Tuple{typeof(Profile.kernel_launches), CUPTI.ActivityConfig, CUDACore.CuContext})

# Reflection depends on a GPU; `func` and `types` are deliberately unspecialized.
for io in (Base.TTY, Base.DevNull)
    precompile(Tuple{typeof(code_native), io, Any, Any})
    precompile(Tuple{typeof(Core.kwcall), @NamedTuple{kernel::Bool}, typeof(code_native),
                     io, Any, Any})
    precompile(Tuple{typeof(code_llvm), io, Any, Any})
    precompile(Tuple{typeof(Core.kwcall), @NamedTuple{kernel::Bool}, typeof(code_llvm),
                     io, Any, Any})
end

using PrecompileTools

@compile_workload begin
    # displaying results takes seconds to compile, and each table's columns depend on the
    # data (e.g. a distribution column for repeated calls, sizes for memory operations),
    # so cover the common variants. the tables need to match what `capture` produces.
    # the profiled region is delimited by the first and last cuCtxSynchronize.
    Opt{T} = Union{Missing,T}
    host = (
        id      = Int[1, 2, 3, 4, 5, 6, 7],
        start   = Float64[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] .* 1e-3,
        stop    = Float64[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1] .* 1e-3,
        name    = String["cuCtxSynchronize", "cuMemAllocFromPoolAsync", "cuMemcpyHtoDAsync_v2",
                         "cuLaunchKernelEx", "cuLaunchKernelEx", "cuMemcpyDtoHAsync_v2",
                         "cuCtxSynchronize"],
        tid     = Int[1, 1, 1, 1, 1, 1, 1],
        details = Opt{String}[missing, "4.000 KiB, device memory", missing, missing,
                              missing, missing, missing],
    )
    device = (
        id      = Int[3, 4, 5, 6],
        start   = Float64[2.2, 3.2, 4.2, 5.2] .* 1e-3,
        stop    = Float64[2.8, 3.8, 4.8, 5.8] .* 1e-3,
        name    = String["[copy pageable to device memory]", "kernel", "kernel",
                         "[copy device to pageable memory]"],
        device  = Int[0, 0, 0, 0],
        context = Int[1, 1, 1, 1],
        stream  = Int[1, 1, 1, 1],
        grid       = Opt{CUDACore.CuDim3}[missing, CUDACore.CuDim3(1,1,1), CUDACore.CuDim3(1,1,1), missing],
        block      = Opt{CUDACore.CuDim3}[missing, CUDACore.CuDim3(1,1,1), CUDACore.CuDim3(1,1,1), missing],
        registers  = Opt{Int64}[missing, 32, 32, missing],
        shared_mem = Opt{@NamedTuple{static::Int64,dynamic::Int64}}[missing, (static=0,dynamic=0), (static=0,dynamic=0), missing],
        local_mem  = Opt{@NamedTuple{thread::Int64,total::Int64}}[missing, (thread=0,total=0), (thread=0,total=0), missing],
        size       = Opt{Int64}[4096, missing, missing, 4096],
        details    = Opt{String}[missing, missing, missing, missing],
    )
    nvtx = (
        id       = Int[1, 1],
        start    = Float64[3.0, 5.0] .* 1e-3,
        type     = Symbol[:start, :end],
        tid      = Int[1, 1],
        name     = Opt{String}["range", missing],
        domain   = Opt{String}["Main", missing],
        payload  = Any[UInt64(0), UInt64(0)],
        color    = Union{Nothing,UInt32}[0x01c318af, 0x01c318af],
        category = Opt{UInt32}[0, 0],
    )
    # a single kernel launch, without allocations, copies or NVTX ranges
    single_kernel = (map(col -> col[[1, 4, 7]], host), map(col -> col[2:2], device),
                     map(col -> col[1:0], nvtx))
    for (host, device, nvtx) in ((host, device, nvtx), single_kernel), trace in (false, true)
        results = Profile.ProfileResults(; host, device, nvtx, trace)
        show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), results)
    end

    # hardware metrics, as collected by `profile_metrics`
    columns = Profile.overview_metrics
    kernels = (; id = Int[4, 5], name = String["kernel", "kernel"],
                 grid = CUDACore.CuDim3[CUDACore.CuDim3(1,1,1), CUDACore.CuDim3(1,1,1)],
                 block = CUDACore.CuDim3[CUDACore.CuDim3(1,1,1), CUDACore.CuDim3(1,1,1)],
                 registers = Int[32, 32], stream = Int[1, 1])
    values = ones(Float64, 2, length(columns))
    for trace in (false, true)
        results = Profile.KernelProfile(; device="GPU", passes=1, kernels,
                                          metrics=[col.metric for col in columns],
                                          values, columns, trace)
        show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), results)
    end
end
