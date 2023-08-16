## gpucompiler interface implementation

struct CUDACompilerParams <: AbstractCompilerParams end
const CUDACompilerConfig = CompilerConfig{PTXCompilerTarget, CUDACompilerParams}
const CUDACompilerJob = CompilerJob{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(@nospecialize(job::CUDACompilerJob)) = CUDA

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(@nospecialize(job::CUDACompilerJob), fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

# link libdevice
function GPUCompiler.link_libraries!(@nospecialize(job::CUDACompilerJob), mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    # only link if there's undefined __nv_ functions
    if !any(fn->startswith(fn, "__nv_"), undefined_fns)
        return
    end

    lib = parse(LLVM.Module, read(libdevice))

    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    GPUCompiler.link_library!(mod, lib) # note: destroys lib

    @dispose pm=ModulePassManager() begin
        push!(metadata(mod)["nvvm-reflect-ftz"],
              MDNode([ConstantInt(Int32(1))]))
        run!(pm, mod)
    end

    return
end

GPUCompiler.method_table(@nospecialize(job::CUDACompilerJob)) = method_table

GPUCompiler.kernel_state_type(job::CUDACompilerJob) = KernelState

function GPUCompiler.finish_module!(@nospecialize(job::CUDACompilerJob),
                                    mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{PTXCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if haskey(globals(mod), "global_random_keys")
        f = initialize_rng_state
        ft = typeof(f)
        tt = Tuple{}

        # don't recurse into `initialize_rng_state()` itself
        if job.source.specTypes.parameters[1] == ft
            return entry
        end

        # create a deferred compilation job for `initialize_rng_state()`
        src = methodinstance(ft, tt, GPUCompiler.tls_world_age())
        cfg = CompilerConfig(job.config; kernel=false, name=nothing)
        job = CompilerJob(src, cfg, job.world)
        id = length(GPUCompiler.deferred_codegen_jobs) + 1
        GPUCompiler.deferred_codegen_jobs[id] = job

        # generate IR for calls to `deferred_codegen` and the resulting function pointer
        top_bb = first(blocks(entry))
        bb = BasicBlock(top_bb, "initialize_rng")
        LLVM.@dispose builder=IRBuilder() begin
            position!(builder, bb)

            # call the `deferred_codegen` marker function
            T_ptr = LLVM.Int64Type()
            deferred_codegen_ft = LLVM.FunctionType(T_ptr, [T_ptr])
            deferred_codegen = if haskey(functions(mod), "deferred_codegen")
                functions(mod)["deferred_codegen"]
            else
                LLVM.Function(mod, "deferred_codegen", deferred_codegen_ft)
            end
            fptr = call!(builder, deferred_codegen_ft, deferred_codegen, [ConstantInt(id)])

            # call the `initialize_rng_state` function
            rt = Core.Compiler.return_type(f, tt)
            llvm_rt = convert(LLVMType, rt)
            llvm_ft = LLVM.FunctionType(llvm_rt)
            fptr = inttoptr!(builder, fptr, LLVM.PointerType(llvm_ft))
            call!(builder, llvm_ft, fptr)
            br!(builder, top_bb)
        end

        # XXX: put some of the above behind GPUCompiler abstractions
        #      (e.g., a compile-time version of `deferred_codegen`)
    end
    return entry
end


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per context
const _compiler_caches = Dict{CuContext, Dict{Any, CuFunction}}();
function compiler_cache(ctx::CuContext)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, CuFunction}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, CUDACompilerConfig}()
function compiler_config(dev; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    # determine the toolchain (cached, because this is slow)
    if !isassigned(_toolchain)
        _toolchain[] = supported_toolchain()
    end
    toolchain = _toolchain[]::@NamedTuple{cap::Vector{VersionNumber}, ptx::Vector{VersionNumber}}

    # select the highest capability that is supported by both the entire toolchain, and our
    # device. this is allowed to be lower than the actual device capability (e.g. `sm_89`
    # for a `sm_90` device), because we'll invoke `ptxas` using a higher capability later.
    caps = filter(toolchain_cap -> toolchain_cap <= capability(dev), toolchain.cap)
    isempty(caps) &&
        error("Your $(CUDA.name(dev)) GPU with capability v$(capability(dev)) is not supported anymore")
    cap = maximum(caps)

    # select the PTX ISA we assume to be available
    # (we actually only need 6.2, but NVPTX doesn't support that)
    ptx = v"6.3"

    # NVIDIA bug #3600554: ptxas segfaults with our debug info, fixed in 11.7
    debuginfo = runtime_version() >= v"11.7"

    # create GPUCompiler objects
    target = PTXCompilerTarget(; cap, ptx, debuginfo, kwargs...)
    params = CUDACompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    asm, meta = JuliaContext() do ctx
        GPUCompiler.compile(:asm, job)
    end

    # remove extraneous debug info on lower debug levels
    if Base.JLOptions().debug_level < 2
        # LLVM sets `.target debug` as soon as the debug emission kind isn't NoDebug. this
        # is unwanted, as the flag makes `ptxas` behave as if `--device-debug` were set.
        # ideally, we'd need something like LocTrackingOnly/EmitDebugInfo from D4234, but
        # that got removed in favor of NoDebug in D18808, seemingly breaking the use case of
        # only emitting `.loc` instructions...
        #
        # according to NVIDIA, "it is fine for PTX producers to produce debug info but not
        # set `.target debug` and if `--device-debug` isn't passed, PTXAS will compile in
        # release mode".
        asm = replace(asm, r"(\.target .+), debug" => s"\1")
    end

    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(meta.ir))) do f
        isdeclaration(f) && !LLVM.isintrinsic(f)
    end
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                     "__nvvm_reflect" #= TODO: should have been optimized away =#]
    needs_cudadevrt = !isempty(setdiff(LLVM.name.(undefined_fs), intrinsic_fns))

    # find externally-initialized global variables; we'll access those using CUDA APIs.
    external_gvars = filter(isextinit, collect(globals(meta.ir))) .|> LLVM.name

    # prepare invocations of CUDA compiler tools
    ptxas_opts = String[]
    nvlink_opts = String[]
    ## debug flags
    if Base.JLOptions().debug_level == 1
        push!(ptxas_opts, "--generate-line-info")
    elseif Base.JLOptions().debug_level >= 2
        push!(ptxas_opts, "--device-debug")
        push!(nvlink_opts, "--debug")
    end
    ## relocatable device code
    if needs_cudadevrt
        push!(ptxas_opts, "--compile-only")
    end

    # use the highest device capability that's supported by CUDA. note that we're allowed
    # to query this because the compilation cache is sharded by the device context.
    # XXX: put this in the CompilerTarget to avoid device introspection?
    #      on the other hand, GPUCompiler doesn't care about the actual device capability...
    dev = device()
    caps = filter(toolchain_cap -> toolchain_cap <= capability(dev), cuda_compat().cap)
    cap = maximum(caps)
    # NOTE: we should already have warned about compute compatibility mismatches
    #       during TLS state set-up.
    arch = "sm_$(cap.major)$(cap.minor)"

    # compile to machine code
    # NOTE: we use tempname since mktemp doesn't support suffixes, and mktempdir is slow
    ptx_input = tempname(cleanup=false) * ".ptx"
    ptxas_output = tempname(cleanup=false) * ".cubin"
    write(ptx_input, asm)

    # we could use the driver's embedded JIT compiler, but that has several disadvantages:
    # 1. fixes and improvements are slower to arrive, by using `ptxas` we only need to
    #    upgrade the toolkit to get a newer compiler;
    # 2. version checking is simpler, we otherwise need to use NVML to query the driver
    #    version, which is hard to correlate to PTX JIT improvements;
    # 3. if we want to be able to use newer (minor upgrades) of the CUDA toolkit on an
    #    older driver, we should use the newer compiler to ensure compatibility.
    append!(ptxas_opts, [
        "--verbose",
        "--gpu-name", arch,
        "--output-file", ptxas_output,
        ptx_input
    ])
    proc, log = run_and_collect(`$(ptxas()) $ptxas_opts`)
    log = strip(log)
    if !success(proc)
        reason = proc.termsignal > 0 ? "ptxas received signal $(proc.termsignal)" :
                                       "ptxas exited with code $(proc.exitcode)"
        msg = "Failed to compile PTX code ($reason)"
        msg *= "\nInvocation arguments: $(join(ptxas_opts, ' '))"
        if !isempty(log)
            msg *= "\n" * log
        end
        msg *= "\nIf you think this is a bug, please file an issue and attach $(ptx_input)"
        error(msg)
    elseif !isempty(log)
        @debug "PTX compiler log:\n" * log
    end
    rm(ptx_input)

    # link device libraries, if necessary
    #
    # this requires relocatable device code, which prevents certain optimizations and
    # hurts performance. as such, we only do so when absolutely necessary.
    # TODO: try LTO, `--link-time-opt --nvvmpath /opt/cuda/nvvm`.
    #       fails with `Ignoring -lto option because no LTO objects found`
    if needs_cudadevrt
        nvlink_output = tempname(cleanup=false) * ".cubin"
        append!(nvlink_opts, [
            "--verbose", "--extra-warnings",
            "--arch", arch,
            "--library-path", dirname(libcudadevrt),
            "--library", "cudadevrt",
            "--output-file", nvlink_output,
            ptxas_output
        ])
        proc, log = run_and_collect(`$(nvlink()) $nvlink_opts`)
        log = strip(log)
        if !success(proc)
            reason = proc.termsignal > 0 ? "nvlink received signal $(proc.termsignal)" :
                                           "nvlink exited with code $(proc.exitcode)"
            msg = "Failed to link PTX code ($reason)"
            msg *= "\nInvocation arguments: $(join(nvlink_opts, ' '))"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(ptxas_output)"
            error(msg)
        elseif !isempty(log)
            @debug "PTX linker info log:\n" * log
        end
        rm(ptxas_output)

        image = read(nvlink_output)
        rm(nvlink_output)
    else
        image = read(ptxas_output)
        rm(ptxas_output)
    end

    return (image, entry=LLVM.name(meta.entry), external_gvars)
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    # load as an executable kernel object
    ctx = context()
    mod = CuModule(compiled.image)
    CuFunction(mod, compiled.entry)
end


## helpers

# run a binary and collect all relevant output
function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)

    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))

    return proc, log
end
