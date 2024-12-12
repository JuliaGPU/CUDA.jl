## gpucompiler interface implementation

Base.@kwdef struct CUDACompilerParams <: AbstractCompilerParams
    cap::VersionNumber
    ptx::VersionNumber
end

function Base.hash(params::CUDACompilerParams, h::UInt)
    h = hash(params.cap, h)
    h = hash(params.ptx, h)

    return h
end

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

function GPUCompiler.mcgen(@nospecialize(job::CUDACompilerJob), mod::LLVM.Module, format)
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
        # set `.target debug` and if `--device-debug` isn't passed, PTXAS will compile in
        # release mode".
        asm = replace(asm, r"(\.target .+), debug" => s"\1")
    end

    # if LLVM couldn't target the requested PTX ISA, bump it in the assembly.
    if job.config.target.ptx != job.config.params.ptx
        ptx = job.config.params.ptx
        asm = replace(asm, r"(\.version .+)" => ".version $(ptx.major).$(ptx.minor)")
    end

    # no need to bump the `.target` directive; we can do that by passing `-arch` to `ptxas`

    asm
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
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false,
                                         cap=nothing, ptx=nothing, kwargs...)
    # determine the toolchain
    llvm_support = llvm_compat()
    cuda_support = cuda_compat()

    # determine the PTX ISA to use. we want at least 6.2, but will use newer if possible.
    requested_ptx = something(ptx, v"6.2")
    llvm_ptxs = filter(>=(requested_ptx), llvm_support.ptx)
    cuda_ptxs = filter(>=(requested_ptx), cuda_support.ptx)
    if ptx !== nothing
        # the user requested a specific PTX ISA
        ## use the highest ISA supported by LLVM
        isempty(llvm_ptxs) &&
            error("Requested PTX ISA $ptx is not supported by LLVM $(LLVM.version())")
        llvm_ptx = maximum(llvm_ptxs)
        ## use the ISA as-is to invoke CUDA
        cuda_ptx = ptx
    else
        # try to do the best thing (i.e., use the newest PTX ISA)
        # XXX: is it safe to just use the latest PTX ISA? isn't it possible for, e.g.,
        #      instructions to get deprecated?
        isempty(llvm_ptxs) &&
            error("CUDA.jl requires PTX $requested_ptx, which is not supported by LLVM $(LLVM.version())")
        llvm_ptx = maximum(llvm_ptxs)
        isempty(cuda_ptxs) &&
            error("CUDA.jl requires PTX $requested_ptx, which is not supported by CUDA driver $(driver_version()) / runtime $(runtime_version())")
        cuda_ptx = maximum(cuda_ptxs)
    end

    # determine the compute capabilities to use. this should match the capability of the
    # current device, but if LLVM doesn't support it, we can target an older capability
    # and pass a different `-arch` to `ptxas`.
    ptx_support = ptx_compat(cuda_ptx)
    requested_cap = @something(cap, min(capability(dev), maximum(ptx_support.cap)))
    llvm_caps = filter(<=(requested_cap), llvm_support.cap)
    if cap !== nothing
        ## use the highest capability supported by LLVM
        isempty(llvm_caps) &&
            error("Requested compute capability $cap is not supported by LLVM $(LLVM.version())")
        llvm_cap = maximum(llvm_caps)
        ## use the capability as-is to invoke CUDA
        cuda_cap = cap
    else
        ## use the highest capability supported by LLVM
        isempty(llvm_caps) &&
            error("Compute capability $(requested_cap) is not supported by LLVM $(LLVM.version())")
        llvm_cap = maximum(llvm_caps)
        ## use the highest capability supported by CUDA
        cuda_caps = filter(<=(capability(dev)), cuda_support.cap)
        isempty(cuda_caps) &&
            error("Compute capability $(requested_cap) is not supported by CUDA driver $(driver_version()) / runtime $(runtime_version())")
        cuda_cap = maximum(cuda_caps)
    end

    # NVIDIA bug #3600554: ptxas segfaults with our debug info, fixed in 11.7
    debuginfo = runtime_version() >= v"11.7"

    # create GPUCompiler objects
    target = PTXCompilerTarget(; cap=llvm_cap, ptx=llvm_ptx, debuginfo, kwargs...)
    params = CUDACompilerParams(; cap=cuda_cap, ptx=cuda_ptx)
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    asm, meta = JuliaContext() do ctx
        GPUCompiler.compile(:asm, job)
    end

    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(meta.ir))) do f
        isdeclaration(f) && !LLVM.isintrinsic(f)
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

    ptx = job.config.params.ptx
    cap = job.config.params.cap
    arch = "sm_$(cap.major)$(cap.minor)"

    # validate use of parameter memory
    argtypes = filter([KernelState, job.source.specTypes.parameters...]) do dt
        !isghosttype(dt) && !Core.Compiler.isconstType(dt)
    end
    param_usage = sum(sizeof, argtypes)
    param_limit = 4096
    if cap >= v"7.0" && ptx >= v"8.1"
        param_limit = 32764
    end
    if param_usage > param_limit
        msg = """Kernel invocation uses too much parameter memory.
                 $(Base.format_bytes(param_usage)) exceeds the $(Base.format_bytes(param_limit)) limit imposed by sm_$(cap.major)$(cap.minor) / PTX v$(ptx.major).$(ptx.minor)."""

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
                details *= "\n  [$(i-1)] $name::$typ uses $(Base.format_bytes(sizeof(typ)))"
            end
            details *= "\n"

            if cap >= v"7.0" && ptx < v"8.1" && param_usage < 32764
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

    return (image, entry=LLVM.name(meta.entry))
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



## opaque closures

# TODO: once stabilised, move bits of this into GPUCompiler.jl

using Core.Compiler: IRCode
using Core: CodeInfo, MethodInstance, CodeInstance, LineNumberNode

# helpers

function compute_ir_rettype(ir::IRCode)
    rt = Union{}
    for i = 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(i)][:stmt]
        if isa(stmt, Core.Compiler.ReturnNode) && isdefined(stmt, :val)
            rt = Core.Compiler.tmerge(Core.Compiler.argextype(stmt.val, ir), rt)
        end
    end
    return Core.Compiler.widenconst(rt)
end

function compute_oc_signature(ir::IRCode, nargs::Int)
    argtypes = Vector{Any}(undef, nargs)
    for i = 1:nargs
        argtypes[i] = Core.Compiler.widenconst(ir.argtypes[i+1])
    end
    return Tuple{argtypes...}
end

function make_oc_codeinfo(ir::IRCode, @nospecialize env...; slotnames=nothing)
    # NOTE: we need ir.argtypes[1] == typeof(env)
    ir = Core.Compiler.copy(ir)
    # if the user didn't specify a definition MethodInstance or filename Symbol to use
    # for the debuginfo, set a filename now
    if ir.debuginfo.def === nothing
        ir.debuginfo.def = Symbol("IR for opaque gpu closure")
    end
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = compute_oc_signature(ir, nargs)
    rt = compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = Base.fill(:none, nargtypes)
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.slotflags = Base.fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    Core.Compiler.ir_to_codeinf!(src, ir)
end

# create a method (like `jl_make_oc_method`)
function make_oc_method(nargs; file=nothing, line=0, world=GPUCompiler.tls_world_age())
    meth = ccall(:jl_new_method_uninit, Ref{Method}, (Any,), Main)
    meth.sig = Tuple
    meth.isva = false
    meth.is_for_opaque_closure = 0
    meth.name = Symbol("opaque gpu closure")
    meth.nargs = nargs + 1
    meth.file = something(file, Symbol())
    meth.line = line
    Base.@atomic meth.primary_world = world
    Base.@atomic meth.deleted_world = typemax(UInt)
    return meth
end

function make_oc_codeinstance(mi::MethodInstance, src::CodeInfo; interp, world, rt)
    owner = Core.Compiler.cache_owner(interp)
    exctype = Any
    inferred_const = C_NULL
    inferred = src
    const_flags = Int32(0)
    min_world = world
    max_world = typemax(UInt)
    ipo_effects = UInt32(0)
    effects = UInt32(0)
    analysis_results = nothing
    relocatability = UInt8(0)
    CodeInstance(mi, owner, rt, exctype, inferred_const, inferred,
                    const_flags, min_world, max_world, ipo_effects, effects,
                    analysis_results, relocatability, src.debuginfo)
end

# generated function `ccall`, working around the restriction that ccall type
# tuples need to be literals. this relies on ccall internals...
@inline @generated function generated_ccall(f::Ptr, _rettyp, _types, vals...)
    ex = quote end

    rettyp = _rettyp.parameters[1]
    types = _types.parameters[1].parameters
    args = [:(vals[$i]) for i in 1:length(vals)]

    # cconvert
    cconverted = [Symbol("cconverted_$i") for i in 1:length(vals)]
    for (dst, typ, src) in zip(cconverted, types, args)
      append!(ex.args, (quote
         $dst = Base.cconvert($typ, $src)
      end).args)
    end

    # unsafe_convert
    unsafe_converted = [Symbol("unsafe_converted_$i") for i in 1:length(vals)]
    for (dst, typ, src) in zip(unsafe_converted, types, cconverted)
      append!(ex.args, (quote
         $dst = Base.unsafe_convert($typ, $src)
      end).args)
    end

    call = Expr(:foreigncall, :f, rettyp, Core.svec(types...), 0,
                QuoteNode(:ccall), unsafe_converted..., cconverted...)
    push!(ex.args, call)
    return ex
end

# static opaque closures

# XXX: because we can't call functions from other CUDA modules, we effectively need to
#      recompile when the target function changes. this, and because of how GPUCompiler's
#      deferred compilation mechanism currently works, is why we have `F` as a type param.

# XXX: because of GPU code requiring specialized signatures, we also need to recompile
#      when the environment or argument types change. together with the above, this
#      negates much of the benefit of opaque closures.

# TODO: support for constructing an opaque closure from source code

# TODO: complete support for passing an environment. this probably requires a split into
#       host and device structures to, e.g., root a CuArray and pass a CuDeviceArray.

struct OpaqueClosure{F, E, A, R}    # func, env, args, ret
    env::E
end

function OpaqueClosure(ir::IRCode, @nospecialize env...;
                       slotnames::Union{Nothing,Vector{Symbol}}=nothing)
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = compute_oc_signature(ir, nargs)
    rt = compute_ir_rettype(ir)
    src = make_oc_codeinfo(ir, env...; slotnames)
    return create_static_oc(src, sig, rt, nargs, env...)
end

function OpaqueClosure(src::CodeInfo, @nospecialize env...; rettype, sig, nargs)
    return create_static_oc(src, sig, rettype, nargs, env...)
end

function create_static_oc(src, @nospecialize(sig), @nospecialize(rt), nargs::Int,
                          @nospecialize env...; file=nothing, line=0)
    config = compiler_config(device(); kernel=false)
    meth = make_oc_method(nargs; file, line)

    # look up a method instance and create a compiler job
    full_sig = Tuple{typeof(env), sig.parameters...}
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), meth, full_sig, Core.svec())
    job = CompilerJob(mi, config, meth.primary_world)

    # create a callable object
    id = length(GPUCompiler.deferred_codegen_jobs) + 1
    GPUCompiler.deferred_codegen_jobs[id] = job
    oc = OpaqueClosure{id, typeof(env), sig, rt}(env)

    opaque_closure_jobs[job] = (; oc, src, rt)
    return oc
end

# device-side call
(oc::OpaqueClosure)(args...) = call(oc, args...)
## NOTE: split into two to make `SciML.isinplace(oc)` work.
##       it also resembles how kernels are called.
@inline function call(oc::OpaqueClosure{F,E,A,R}, args...) where {F,E,A,R}
    ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), F)
    assume(ptr != C_NULL)
    #ccall(ptr, R, (A...), args...)
    generated_ccall(ptr, R, A, args...)
end

# dynamic opaque closures

const jit_opaque_closures = Dict()

struct JITOpaqueClosure{B, T}
    builder::B
    tfunc::T

    function JITOpaqueClosure(builder, tfunc=Returns(nothing); nargs)
        # the device and world are captured at closure construction time, but we only need
        # them when creating the CompilerJob. as we cannot simply encode them in the
        # JITOpaqueClosure object, we store them in a global dictionary instead.
        config = compiler_config(device(); kernel=false)
        meth = make_oc_method(nargs)

        # create a callable object
        oc = new{typeof(builder), typeof(tfunc)}(builder, tfunc)
        jit_opaque_closures[typeof(oc)] = (; env=(), meth, config, oc)

        return oc
    end
end

# device-side call
function (oc::JITOpaqueClosure)(args...)
    rt = oc.tfunc(map(Core.Typeof, args)...)
    call(oc, rt, args...)
end
@inline @generated function call(oct::JITOpaqueClosure{B,T}, ::Type{R}, args...) where {B,T,R}
    rt = R
    (; env, meth, config, oc) = jit_opaque_closures[oct]

    # look up a method instance and create a compiler job
    full_sig = Tuple{typeof(env), args...}
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), meth, full_sig, Core.svec())
    job = CompilerJob(mi, config, meth.primary_world)
    opaque_closure_jobs[job] = (; oc, args, rt)

    # generate a deferred compilation call
    id = length(GPUCompiler.deferred_codegen_jobs) + 1
    GPUCompiler.deferred_codegen_jobs[id] = job
    quote
        ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), $id)
        assume(ptr != C_NULL)
        #ccall(ptr, R, (A...), args...)
        generated_ccall(ptr, $rt, $(Tuple{args...}), args...)
    end
end

# compilation of opaque closures

const opaque_closure_jobs = Dict{CompilerJob,Any}()

function GPUCompiler.prepare_job!(@nospecialize(job::CUDACompilerJob))
    if haskey(opaque_closure_jobs, job)
        rt = opaque_closure_jobs[job].rt
        oc = opaque_closure_jobs[job].oc
        if oc isa JITOpaqueClosure
            args = opaque_closure_jobs[job].args
            nargs = length(args)

            src = oc.builder(args...)
            if src isa IRCode
                nargtypes = length(src.argtypes)
                nargs = nargtypes-1
                sig = compute_oc_signature(src, nargs)
                @assert compute_ir_rettype(src) == rt "Inferred return type does not match the provided return type"
                src = make_oc_codeinfo(src)
            end
        else
            src = opaque_closure_jobs[job].src
        end
        @assert src isa CodeInfo

        # create a code instance and store it in the cache
        interp = GPUCompiler.get_interpreter(job)
        ci = make_oc_codeinstance(job.source, src; interp, job.world, rt)
        Core.Compiler.setindex!(GPUCompiler.ci_cache(job), ci, job.source)
    end

    return
end
