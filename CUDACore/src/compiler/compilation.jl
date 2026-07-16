## gpucompiler interface implementation

abstract type AbstractCUDACompilerParams <: AbstractCompilerParams end

Base.@kwdef struct CUDACompilerParams <: AbstractCUDACompilerParams
    sm::SMVersion
    ptx::VersionNumber
end

function Base.hash(params::CUDACompilerParams, h::UInt)
    h = hash(params.sm, h)
    h = hash(params.ptx, h)

    return h
end

const CUDACompilerConfig = CompilerConfig{PTXCompilerTarget, CUDACompilerParams}
const AnyCUDAJob = CompilerJob{PTXCompilerTarget, <:AbstractCUDACompilerParams}

# CUDA 12.0 is the oldest supported toolkit, and Maxwell is the oldest supported GPU.
const minreq = (; ptx=v"8.0", sm=sm"50")

GPUCompiler.runtime_module(@nospecialize(job::AnyCUDAJob)) = CUDACore
function GPUCompiler.lower_relocations!(@nospecialize(job::AnyCUDAJob), mod::LLVM.Module,
                                        relocs::GPUCompiler.Relocations)
    GPUCompiler.emit_patchable_relocations!(mod, relocs)
end

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(@nospecialize(job::AnyCUDAJob), fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

# link libdevice
function GPUCompiler.link_libraries!(@nospecialize(job::AnyCUDAJob), mod::LLVM.Module)
    lib = parse(LLVM.Module, MemoryBufferFile(CUDA_Compiler.libdevice); lazy=true)

    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    # the linker will only materialize libdevice symbols referenced by `mod`
    link!(mod, lib; only_needed=true)  # destroys lib

    @dispose pm=ModulePassManager() begin
        push!(metadata(mod)["nvvm-reflect-ftz"],
              MDNode([ConstantInt(Int32(1))]))
        run!(pm, mod)
    end

    return
end

GPUCompiler.method_table(@nospecialize(job::AnyCUDAJob)) = method_table

GPUCompiler.kernel_state_type(job::AnyCUDAJob) = KernelState

function GPUCompiler.finish_module!(@nospecialize(job::AnyCUDAJob),
                                    mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{PTXCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # Make the compilation target available to device code. GPUCompiler used to provide
    # these globals, but target-specific properties are owned by the back-end now.
    feature_set = job.config.target.feature_set === :arch    ? ArchFeatures :
                  job.config.target.feature_set === :family  ? FamilyFeatures :
                                                               BaselineFeatures
    for (name, value) in ["sm_major"    => job.config.target.cap.major,
                          "sm_minor"    => job.config.target.cap.minor,
                          "sm_features" => UInt32(feature_set),
                          "ptx_major"   => job.config.target.ptx.major,
                          "ptx_minor"   => job.config.target.ptx.minor]
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            initializer!(gv, ConstantInt(LLVM.Int32Type(), value))
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        end
    end

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
        @dispose builder=IRBuilder() begin
            position!(builder, bb)
            subprogram = LLVM.subprogram(entry)
            if subprogram !== nothing
                loc = DILocation(0, 0, subprogram)
                debuglocation!(builder, loc)
            end
            debuglocation!(builder, first(instructions(top_bb)))

            # call the `deferred_codegen` marker function
            T_ptr = if LLVM.version() >= v"17"
                LLVM.PointerType()
            elseif VERSION >= v"1.12.0-DEV.225"
                LLVM.PointerType(LLVM.Int8Type())
            else
                LLVM.Int64Type()
            end
            T_id = convert(LLVMType, Int)
            deferred_codegen_ft = LLVM.FunctionType(T_ptr, [T_id])
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

# stamp `.version` with the ISA we want `ptxas` to validate against
# and `.target` with the arch that `--gpu-name` will use
function rewrite_ptx_header(asm, ptx::VersionNumber, sm::SMVersion)
    return replace(asm,
        r"(\.version .+)"     => ".version $(ptx.major).$(ptx.minor)",
        r"\.target sm_\d+\w*" => ".target $(cpu_name(sm))")
end

function GPUCompiler.mcgen(@nospecialize(job::AnyCUDAJob), mod::LLVM.Module, format)
    @assert format == LLVM.API.LLVMAssemblyFile
    asm = invoke(GPUCompiler.mcgen,
                 Tuple{CompilerJob{PTXCompilerTarget}, LLVM.Module, typeof(format)},
                 job, mod, format)

    # remove extraneous debug info on lower debug levels
    if Base.JLOptions().debug_level < 2
        # LLVM sets `.target debug` as soon as the debug emission kind isn't NoDebug. this
        # is unwanted, as the flag makes `ptxas` behave as if `--device-debug` were set.
        # ideally, we'd need something like LocTrackingOnly/EmitDebugInfo from D4234, but
        # that got removed in favor of NoDebug in D18808, seemingly breaking the use case of
        # only emitting `.loc` instructions...
        #
        # according to NVIDIA, "it is fine for PTX producers to produce debug info but not
        # set `.target debug` and if `--device-debug` isn't passed, ptxas will compile in
        # release mode".
        asm = replace(asm, r"(\.target .+), debug" => s"\1")
    end

    # The rewrite stamps `.target`/`.version` with the *requested* (cuda-side) values.
    # When the GPUCompiler-side target matches, LLVM already emits the right header
    # (including the `a`/`f` suffix, via the CPU name); we only rewrite when they differ,
    # e.g. when we had to clamp the target down for LLVM compatibility.
    sm_param = job.config.params.sm
    ptx_param = job.config.params.ptx
    needs_rewrite = job.config.target.ptx != ptx_param ||
                    job.config.target.cap != base_version(sm_param) ||
                    job.config.target.feature_set !== sm_param.feature_set
    if needs_rewrite
        asm = rewrite_ptx_header(asm, ptx_param, sm_param)
    end

    return asm
end


## compiler implementation (cache, configure, compile, and link)

# GPUCompiler 2.0 caching: back-ends attach a mutable results struct to each cached
# `CodeInstance` (on Julia 1.11+ this is Julia's integrated code cache, which also persists
# artifacts through package precompilation; on 1.10 it's a session-local store). We keep
# conditionally session-portable artifacts (the cubin `image`, entry point, and relocations)
# separate from the
# session-local `CuFunction` handles, which are context-specific and must not be serialized
# into a package image.
mutable struct CUDACompilerResults
    # session-portable artifacts (safe to persist across sessions)
    image::Union{Nothing,Vector{UInt8}}
    entry::Union{Nothing,String}
    relocations::GPUCompiler.Relocations

    # session-local kernel handles, linear-scanned by context; usually holds a single entry
    kernels::Vector{Tuple{CuContext,CuFunction,Vector{Any}}}

    CUDACompilerResults() = new(nothing, nothing, GPUCompiler.Relocations(),
                                Tuple{CuContext,CuFunction,Vector{Any}}[])
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
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

function default_ptx_versions(llvm_support, ptxas_support)
    requested_ptx = minreq.ptx
    llvm_ptxs = filter(>=(requested_ptx), llvm_support.ptx)
    ptxas_ptxs = filter(>=(requested_ptx), ptxas_support.ptx)
    isempty(llvm_ptxs) &&
        error("CUDA.jl requires PTX $requested_ptx, which is not supported by LLVM $(nvptx_llvm_version)")
    isempty(ptxas_ptxs) &&
        error("CUDA.jl requires PTX $requested_ptx, which is not supported by ptxas $(compiler_version())")

    # LLVM must emit an ISA both tools support. Keep ptxas's newest ISA separately for
    # CUDA-side target selection and PTX header rewriting.
    common_ptxs = intersect(llvm_ptxs, ptxas_ptxs)
    isempty(common_ptxs) &&
        error("CUDA.jl requires a PTX ISA supported by both LLVM $(nvptx_llvm_version) " *
              "and ptxas $(compiler_version())")

    return maximum(common_ptxs), maximum(ptxas_ptxs)
end

@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false,
                                         arch=nothing, cap=nothing, ptx=nothing, kwargs...)
    # `cap=` is the deprecated old name for `arch=` (matches nvcc/ptxas `-arch`).
    if cap !== nothing
        arch === nothing ||
            throw(ArgumentError("pass either `arch=` or the deprecated `cap=`, not both"))
        Base.depwarn("the `cap=` kwarg is deprecated; use `arch=` (matching nvcc/ptxas `-arch`) instead.",
                     :cufunction)
        arch = cap
    end
    # `SMVersion` is the universal normalizer: identity for an SMVersion, baseline-promotes
    # a VersionNumber, parses a string. Anything else falls out as a MethodError naturally.
    arch === nothing || (arch = SMVersion(arch))

    # inspect the toolchain
    llvm_support = llvm_compat()
    ptxas_support = ptxas_compat()

    # determine the PTX ISA to use.
    if ptx !== nothing
        # explicit request: take it exactly, validating against the toolchain
        ptx >= minreq.ptx ||
            error("CUDA.jl requires PTX ISA $(minreq.ptx) or higher")
        ptx in llvm_support.ptx ||
            error("Requested PTX ISA $ptx is not supported by LLVM $(nvptx_llvm_version)")
        ptx in ptxas_support.ptx ||
            error("Requested PTX ISA $ptx is not supported by ptxas $(compiler_version())")
        llvm_ptx = ptxas_ptx = ptx
    else
        # default: pick the newest supported PTX ISA at or above our minimum
        llvm_ptx, ptxas_ptx = default_ptx_versions(llvm_support, ptxas_support)
    end

    # when selecting compute capabilities, we prefer the most recent one, as
    # well as prefer to use architecture-accelerated features when available.
    fs_rank(fs::Symbol) = fs === :arch ? 2 : fs === :family ? 1 : 0
    sm_key(sm::SMVersion) = (base_version(sm), fs_rank(sm.feature_set))

    # determine the compute capability to use.
    ## ptxas
    ptx_sms = ptx_sm_support(ptxas_ptx)
    if arch !== nothing
        # explicit request: take it as-is, validating against the PTX ISA
        base_version(arch) >= base_version(minreq.sm) ||
            error("CUDA.jl requires compute capability $(cpu_name(minreq.sm)) or higher")
        arch in ptx_sms ||
            error("$(cpu_name(arch)) is not supported by PTX ISA $(ptxas_ptx)")
        ptxas_sm = arch
    else
        # pick the most specific capability the selected PTX ISA supports whose cubin
        # would actually load on the current device. For baseline that's the onion model;
        # `:arch` requires an exact CC match, `:family` a same-family match.
        ptxas_candidates = filter(ptx_sms) do sm
            base_version(sm) >= base_version(minreq.sm) && runs_on(sm, capability(dev))
        end
        isempty(ptxas_candidates) &&
            error("Compute capability $(capability(dev)) is not supported by ptxas " *
                  "$(compiler_version()) at PTX ISA $(ptxas_ptx)")
        ptxas_sm = argmax(sm_key, ptxas_candidates)
    end
    ## LLVM
    if ptxas_sm in llvm_support.sm
        llvm_sm = ptxas_sm
    else
        # Exact `ptxas_sm` unavailable in LLVM. Fall back to baseline LLVM at a
        # lower base, since arch/family features don't carry across versions.
        baseline_candidates = filter(llvm_support.sm) do sm
            sm.feature_set === :baseline &&
                base_version(minreq.sm) <= base_version(sm) <= base_version(ptxas_sm)
        end
        isempty(baseline_candidates) &&
            error("Compute capability $(cpu_name(ptxas_sm)) is not supported by LLVM $(nvptx_llvm_version)")
        llvm_sm = argmax(sm_key, baseline_candidates)
    end

    # create GPUCompiler objects
    target = PTXCompilerTarget(; cap=base_version(llvm_sm), ptx=llvm_ptx,
                                 feature_set=llvm_sm.feature_set,
                                 debuginfo=true, kwargs...)
    params = CUDACompilerParams(; sm=ptxas_sm, ptx=ptxas_ptx)
    CompilerConfig(target, params; kernel, name, always_inline)
end

# does the host-side layout of an argument type match the device-side one?
#
# the back-end unconditionally aligns 128-bit integers to 16 bytes, whereas Julia only
# started doing so in 1.12, so aggregates with (U)Int128 fields may lay out differently.
# returns the device-side (size, alignment) of `T`, `:opaque` for types whose layout is
# defined by Julia on both sides (e.g. unions, or non-isbits types passed by reference),
# or `:mismatch`.
function device_layout(@nospecialize(T))
    if T === Int128 || T === UInt128
        return (16, 16)
    elseif !(T isa DataType) || !isbitstype(T)
        return :opaque
    elseif fieldcount(T) == 0
        return (sizeof(T), Base.datatype_alignment(T))
    end
    offset = 0
    align = 1
    for i in 1:fieldcount(T)
        field = device_layout(fieldtype(T, i))
        field === :mismatch && return :mismatch
        if field === :opaque || offset < 0
            # we cannot track offsets anymore, but keep verifying nested layouts
            offset = -1
            continue
        end
        field_size, field_align = field
        offset = cld(offset, field_align) * field_align
        offset == fieldoffset(T, i) || return :mismatch
        offset += field_size
        align = max(align, field_align)
    end
    offset < 0 && return :opaque
    size = cld(offset, align) * align
    size == sizeof(T) || return :mismatch
    return (size, align)
end
# walk `T` and every type reachable from it through type parameters and fields, returning
# `true` as soon as `bad(S)` holds for some reached type `S`. we must look through type
# parameters, not just fields: an aggregate with a mismatching layout is typically reached
# through a pointer (e.g. the element type of a `CuDeviceArray`, carried as a type parameter
# and never as a field), so inspecting only the argument's own fields would miss it and the
# kernel would silently read or write garbage.
function layout_reaches(bad, @nospecialize(T), seen=Base.IdSet{Any}())
    (T isa Type && !(T in seen)) || return false
    push!(seen, T)
    bad(T) && return true
    T isa DataType || return false
    any(p -> layout_reaches(bad, p, seen), T.parameters) && return true
    isconcretetype(T) || return false
    any(i -> layout_reaches(bad, fieldtype(T, i), seen), 1:fieldcount(T))
end

device_compatible_layout(@nospecialize(T)) =
    # since Julia 1.12, host and device layouts are identical
    Base.datatype_alignment(Int128) == 16 ||
    !layout_reaches(S -> device_layout(S) === :mismatch, T)

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    asm, meta = JuliaContext() do ctx
        invoke_frozen(GPUCompiler.compile, :asm, job)
    end

    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(meta.ir))) do f
        isdeclaration(f) && !LLVM.isintrinsic(f) &&
        # intrinsics unknown to the in-process LLVM are still lowered by the back-end
        !startswith(LLVM.name(f), "llvm.")
    end
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                     "__nvvm_reflect" #= TODO: should have been optimized away =#]
    needs_cudadevrt = !isempty(setdiff(LLVM.name.(undefined_fs), intrinsic_fns))

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

    sm_param = job.config.params.sm
    ptx_param = job.config.params.ptx
    cap = base_version(sm_param)
    arch = cpu_name(sm_param)

    # validate use of parameter memory
    argtypes = filter([KernelState, job.source.specTypes.parameters...]) do dt
        !isghosttype(dt) && !Core.Compiler.isconstType(dt)
    end
    for dt in argtypes
        if !device_compatible_layout(dt)
            error("Kernel argument of type $dt references 128-bit integer fields. This is only supported on Julia 1.12 or later.")
        end
    end
    param_usage = sum(aligned_sizeof, argtypes)
    param_limit = 4096
    if cap >= v"7.0" && ptx_param >= v"8.1"
        param_limit = 32764
    end
    if param_usage > param_limit
        msg = """Kernel invocation uses too much parameter memory.
                 $(Base.format_bytes(param_usage)) exceeds the $(Base.format_bytes(param_limit)) limit imposed by $(arch) / PTX v$(ptx_param.major).$(ptx_param.minor)."""

        try
            details = "\n\nRelevant parameters:"

            source_types = job.source.specTypes.parameters
            source_argnames = Base.method_argnames(job.source.def)
            while length(source_argnames) < length(source_types)
                # this is probably due to a trailing vararg; repeat its name
                push!(source_argnames, source_argnames[end])
            end

            for (i, typ) in enumerate(source_types)
                if isghosttype(typ) || Core.Compiler.isconstType(typ)
                    continue
                end
                name = source_argnames[i]
                details *= "\n  [$(i-1)] $name::$typ uses $(Base.format_bytes(aligned_sizeof(typ)))"
            end
            details *= "\n"

            if cap >= v"7.0" && ptx_param < v"8.1" && param_usage < 32764
                details *= "\nNote: use a newer CUDA to support more parameters on your device.\n"
            end

            msg *= details
        catch err
            @error "Failed to analyze kernel parameter usage; please file an issue with a reproducer."
        end
        error(msg)
    end

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
    proc, log = run_and_collect(`$(CUDA_Compiler.ptxas()) $ptxas_opts`)
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
        if parse(Bool, get(ENV, "BUILDKITE", "false"))
            run(`buildkite-agent artifact upload $(ptx_input)`)
        end
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
            "--library-path", dirname(CUDA_Compiler.libcudadevrt),
            "--library", "cudadevrt",
            "--output-file", nvlink_output,
            ptxas_output
        ])
        proc, log = run_and_collect(`$(CUDA_Compiler.nvlink()) $nvlink_opts`)
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

    return (image, entry=LLVM.name(meta.entry), relocations=meta.relocations)
end

# link a compiled image into a session-local `CuFunction` on the active context
function link_kernel(@nospecialize(job::CompilerJob), image::Vector{UInt8}, entry::String,
                     relocs::GPUCompiler.Relocations)
    # load as an executable kernel object on the current context
    mod = CuModule(image)
    relocations = GPUCompiler.resolved_relocations(relocs)
    for (name, value) in relocations.slots
        slot = CuGlobal{UInt}(mod, name)
        slot[] = value
    end
    for ((name, offset), value) in relocations.interior
        ptr_ref = Ref{CuPtr{Cvoid}}()
        size_ref = Ref{Csize_t}()
        cuModuleGetGlobal_v2(ptr_ref, size_ref, mod, name)
        offset >= 0 && offset + sizeof(UInt) <= size_ref[] ||
            error("Interior relocation '$name+$offset' is outside its $(size_ref[])-byte global")
        value_ref = Ref(value)
        cuMemcpyHtoD_v2(ptr_ref[] + offset, value_ref, sizeof(UInt))
    end
    return CuFunction(mod, entry), relocations.roots
end

# look up the cached compilation artifacts for `job`, running the compiler on a miss.
#
# Storage is managed by `GPUCompiler.cached_results`: Julia's integrated code cache on 1.11+
# (which also persists artifacts through precompilation), or a session-local store on 1.10.
# `image === nothing` identifies a freshly-created `CUDACompilerResults` that hasn't been
# compiled yet; the `compile_hook` check additionally forces the compile path so that
# reflection consumers (`@device_code_*`) observe the compilation even on a cache hit.
function compile_or_lookup(@nospecialize(job::CompilerJob))::CUDACompilerResults
    res = GPUCompiler.cached_results(CUDACompilerResults, job)
    if res === nothing || res.image === nothing || GPUCompiler.compile_hook[] !== nothing
        compiled = compile(job)
        if GPUCompiler.supports_relocatable_ir()
            res = @something res GPUCompiler.cached_results(CUDACompilerResults, job)
        else
            # The CodeInstance may be serialized, but generated code from a Julia runtime
            # without a reliable GV table is session-bound.
            res = CUDACompilerResults()
        end
        res.image = compiled.image
        res.entry = compiled.entry
        res.relocations = compiled.relocations
    end
    return res
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
