# @profile infrastructure (GPU-dependent, can't execute during precompilation)
precompile(Tuple{typeof(Profile.detect_cupti)})
precompile(Tuple{typeof(Profile.profile_internally), Function})
precompile(Tuple{typeof(Profile.capture), CUPTI.ActivityConfig, UInt64})
precompile(Tuple{typeof(Profile.profile_metrics), Function})
precompile(Tuple{typeof(Profile.kernel_launches), CUPTI.ActivityConfig, CUDACore.CuContext})
precompile(Tuple{typeof(Profile.sample_pcs), Function})
precompile(Tuple{typeof(Profile.sampled_functions), CUPTI.PCSamples, Base.ValueIterator{Dict{UInt32,Vector{UInt8}}}})
precompile(Tuple{typeof(line_tables), Vector{UInt8}})
precompile(Tuple{typeof(Profile.embedded_ptx), Vector{UInt8}})

# displaying results is precompiled by the workload below, which writes to an `IOBuffer`.
# the methods that take the output stream render into a buffer, so precompile them for the
# streams results are typically shown on as well.
for results in (Profile.ProfileResults, Profile.KernelProfile, Profile.PCProfile),
    io in (Base.TTY, Base.PipeEndpoint, IOStream)
    for io in (io, IOContext{io})
        precompile(Tuple{typeof(show), io, MIME"text/plain", results})
        precompile(Tuple{typeof(show), io, MIME"application/json", results})
        precompile(Tuple{typeof(show), io, String, results})
    end
end

for io in (Base.TTY, Base.PipeEndpoint, IOStream), f in (code_sass, code_ptx)
    precompile(Tuple{typeof(f), io, Profile.PCProfile})
    precompile(Tuple{typeof(Core.kwcall), @NamedTuple{debuginfo::Symbol}, typeof(f), io,
                     Profile.PCProfile})
end

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
        shared_mem = Opt{@NamedTuple{static::Int64,dynamic::Int64}}[missing, (static=1024,dynamic=0), (static=0,dynamic=0), missing],
        local_mem  = Opt{@NamedTuple{thread::Int64,total::Int64}}[missing, (thread=16,total=16384), (thread=0,total=0), missing],
        size       = Opt{Int64}[4096, missing, missing, 4096],
        details    = Opt{String}[missing, missing, missing, missing],
    )
    nvtx = (
        id       = Int[1, 2, 1],
        start    = Float64[3.0, 4.0, 5.0] .* 1e-3,
        type     = Symbol[:start, :instant, :end],
        tid      = Int[1, 1, 1],
        name     = Opt{String}["range", "marker", missing],
        domain   = Opt{String}["Main", missing, missing],
        payload  = Any[UInt64(0), missing, UInt64(0)],
        color    = Union{Nothing,UInt32}[0x01c318af, nothing, 0x01c318af],
        category = Opt{UInt32}[0, missing, 0],
    )
    # a single kernel launch, without allocations, copies or NVTX ranges
    single_kernel = (map(col -> col[[1, 4, 7]], host), map(col -> col[2:2], device),
                     map(col -> col[1:0], nvtx))
    # raw traces are left out: they always show the device column, whose names come from
    # the driver. the other columns they add are covered by kernels using shared and local
    # memory.
    for (host, device, nvtx) in ((host, device, nvtx), single_kernel), trace in (false, true)
        results = Profile.ProfileResults(; host, device, nvtx, trace)
        show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), results)
        string(results)
        repr("application/json", results)
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
        string(results)
        repr("application/json", results)
    end

    # PC samples, as collected by `sample_pcs`: a kernel with inlined code, labels, and
    # instructions without source information or PTX code, and one without any samples
    kernel = SourceFrame("", "/tmp/kernel.jl", 3)
    inlined = SourceFrame("getindex", "./array.jl", 10)
    instructions = (; offset=[0, 16, -1, 32, 48], sass=["LDG.E R2, [R2]", "FADD R0, R2, R2",
                                                        ".L_x_0:", "STG.E [R4], R0", "EXIT"],
                      source=[[kernel, inlined], [kernel], SourceFrame[], SourceFrame[], [kernel]])
    samples = [10 0; 2 1; 0 0; 1 0; 0 0]
    text = [".visible .entry _Z6kernel5Int64(", ".param .u64 _Z6kernel5Int64_param_0", ")",
            "{", "\$L__func_begin0:", "", "ld.global.f32 \t%f1, [%rd1];",
            "add.f32 \t%f2, %f1, %f1;", "st.global.f32 \t[%rd1], %f2;", "ret;", "}"]
    ptx = (; text, lines=[7, 8, 0, 9, 0])
    functions = [Profile.SampledFunction("kernel(Int64)", "_Z6kernel5Int64", instructions,
                                         samples, samples .÷ 2, ptx),
                 Profile.SampledFunction("idle()", "_Z4idlev", instructions,
                                         zero(samples), zero(samples), nothing)]
    for format in (:tree, :flat)
        p = Profile.PCProfile(; device="GPU", period=14, total=14, dropped=0,
                                reasons=["long_scoreboard", "wait"], functions, format)
        show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), p)
        show(IOContext(IOBuffer(), :limit => true, :color => true), MIME"text/plain"(), p)
        string(p)
        sprint(show, p; context=:color => true)
        repr("application/json", p)
        for f in (code_sass, code_ptx)
            f(IOBuffer(), p)
            f(IOContext(IOBuffer(), :color => true), p)
        end
    end
    p = Profile.PCProfile(; device="GPU", period=14, total=0, dropped=0, reasons=String[],
                            functions=Profile.SampledFunction[])
    string(p)
end