# `llvm_compat()` requires being able to initialize the NVPTX backend, so we run the
# precompile workload only when that's supported, to be able to load this package also on
# systems where the backend isn't available.
if :NVPTX in LLVM.backends()
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
            # `.sm` is `Set{SMVersion}` (with variants); pick the highest baseline
            # entry <= v"7.5" for a portable precompile artifact.
            llvm_sm = argmax(base_version,
                             filter(sm -> sm.feature_set === :baseline &&
                                          base_version(sm) <= v"7.5",
                                    llvm_support.sm))
            llvm_ptx = maximum(filter(>=(v"6.2"), llvm_support.ptx))

            target = PTXCompilerTarget{Nothing}(; cap=base_version(llvm_sm), ptx=llvm_ptx, debuginfo=true)
            params = CUDACompilerParams(; sm=llvm_sm, ptx=llvm_ptx)
            config = CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false)

            tt = Tuple{CuDeviceArray{Float32,1,AS.Global}}
            source = methodinstance(typeof(_precompile_vadd), tt)
            job = CompilerJob(source, config)

            # On Julia < 1.12, GPU compilation during precompilation leaks foreign
            # MIs into native compilation, causing LLVM errors
            # (e.g. "Cannot select: intrinsic %llvm.nvvm.membar.sys").
            @static if VERSION >= v"1.12-"
                JuliaContext() do ctx
                    GPUCompiler.compile(:asm, job)
                end
            end
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
