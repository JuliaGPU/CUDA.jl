@compile_workload begin
    # compile a dummy kernel to PTX to precompile the GPUCompiler pipeline.
    # this doesn't need a GPU — it only uses LLVM.
    let
        function _precompile_vadd(a)
            i = threadIdx().x
            @inbounds a[i] += 1f0
            return nothing
        end

        llvm_support = llvm_compat()
        llvm_cap = maximum(filter(<=(v"7.5"), llvm_support.cap))
        llvm_ptx = maximum(filter(>=(v"6.2"), llvm_support.ptx))

        target = PTXCompilerTarget(; cap=llvm_cap, ptx=llvm_ptx, debuginfo=true)
        params = CUDACompilerParams(; cap=llvm_cap, ptx=llvm_ptx)
        config = CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false)

        tt = Tuple{CuDeviceArray{Float32,1,AS.Global}}
        source = methodinstance(typeof(_precompile_vadd), tt)
        job = CompilerJob(source, config)

        JuliaContext() do ctx
            GPUCompiler.compile(:asm, job)
        end
    end
end

# kernel launch infrastructure
precompile(Tuple{typeof(cufunction), typeof(identity), Type{Tuple{Nothing}}})
precompile(Tuple{typeof(link), CompilerJob, NamedTuple{(:image, :entry), Tuple{Vector{UInt8}, String}}})

# GPUCompiler compilation pipeline (specialized for CUDACore's compile/link)
precompile(Tuple{typeof(GPUCompiler.actual_compilation),
    Dict{Any, CuFunction}, Core.MethodInstance, UInt64,
    CUDACompilerConfig, typeof(compile), typeof(link)})

# scalar reference (used by cuBLAS for alpha/beta parameters)
precompile(Tuple{Type{CuRefValue{Float32}}, Float32})
precompile(Tuple{typeof(pool_free), Managed{DeviceMemory}})
