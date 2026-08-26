# `llvm_compat()` requires being able to initialize the NVPTX backend, and `ptxas_compat()`
# requires the CUDA compiler, so we run the precompile workload only when both are
# available, to be able to load this package also on systems where the backend isn't
# available or where no CUDA compiler was selected (e.g. no driver and no version
# preference).
if :NVPTX in LLVM.backends() && CUDA_Compiler.is_available()
    @compile_workload begin
        # compile a dummy kernel to precompile the GPUCompiler pipeline.
        # this uses the compiler toolchain, but doesn't need a GPU.
        let
            function _precompile_vadd(a)
                i = threadIdx().x
                @inbounds a[i] += 1f0
                return nothing
            end

            llvm_support = llvm_compat()
            ptxas_support = ptxas_compat()
            # `.sm` is `Set{SMVersion}` (with variants); pick the highest baseline
            # entry <= v"7.5" for a portable precompile artifact.
            llvm_sm = argmax(base_version,
                             filter(sm -> sm.feature_set === :baseline &&
                                          base_version(sm) <= v"7.5",
                                    llvm_support.sm))
            llvm_ptx, ptxas_ptx = default_ptx_versions(llvm_support, ptxas_support)

            target = PTXCompilerTarget(; cap=base_version(llvm_sm), ptx=llvm_ptx, debuginfo=true)
            params = CUDACompilerParams(; sm=llvm_sm, ptx=ptxas_ptx)
            config = CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false)

            tt = Tuple{CuDeviceArray{Float32,1,AS.Global}}
            source = methodinstance(typeof(_precompile_vadd), tt)
            job = CompilerJob(source, config)

            # On Julia < 1.12, GPU compilation during precompilation leaks foreign
            # MIs into native compilation, causing LLVM errors
            # (e.g. "Cannot select: intrinsic %llvm.nvvm.membar.sys").
            @static if VERSION >= v"1.12-"
                # Enroll the foreign CodeInstance in the package image, then populate it
                # through the same cache path used by kernel launches.
                precompile(job)
                compile_or_lookup(job)
            end
        end
    end
end

# kernel launch infrastructure
let CUDACompilerJob = CompilerJob{PTXCompilerTarget, CUDACompilerParams}
    precompile(Tuple{typeof(cufunction), typeof(identity), Type{Tuple{Nothing}}})
    precompile(Tuple{typeof(link_kernel), Vector{UInt8}, String, GPUCompiler.Relocations})

    # GPUCompiler 2.0 caching pipeline (specialized for CUDACore's results struct)
    precompile(Tuple{typeof(compile_or_lookup), CUDACompilerJob})
    precompile(Tuple{typeof(GPUCompiler.cached_results), Type{CUDACompilerResults}, CUDACompilerJob})
end

# GPU-dependent paths cannot be reached by the device-free workload above.
precompile(Tuple{typeof(context), CuDevice})
precompile(Tuple{typeof(create_context), CuDevice, Int})
precompile(Tuple{typeof(active_state)})
precompile(Tuple{typeof(compiler_config), CuDevice})
precompile(Tuple{typeof(Core.kwcall), @NamedTuple{kernel::Bool}, typeof(compiler_config),
                 CuDevice})

# scalar reference (used by cuBLAS for alpha/beta parameters)
precompile(Tuple{Type{CuRefValue{Float32}}, Float32})
precompile(Tuple{typeof(pool_free), Managed{DeviceMemory}})
