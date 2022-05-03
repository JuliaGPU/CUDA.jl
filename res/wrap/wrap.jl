# script to parse CUDA headers and generate Julia wrappers


#
# Parsing
#

using Clang

function wrap(name, headers...; wrapped_headers=headers, library="lib$name", defines=[], include_dirs=[])
    clang_args = String[]
    append!(clang_args, map(dir->"-I$dir", include_dirs))
    for define in defines
        if isa(define, Pair)
            append!(clang_args, ["-D", "$(first(define))=$(last(define))"])
        else
            append!(clang_args, ["-D", "$define"])
        end
    end

    output_file = "lib$(name).jl"
    common_file = "lib$(name)_common.jl"

    # only wrap headers that are wanted, and make sure we only process each header once
    included_from = Dict()
    function wrap_header(root, current)
        return any(header->endswith(current, header), wrapped_headers) &&
               get!(included_from, current, root) == root
    end

    context = init(;
                    headers = [headers...],
                    output_file = output_file,
                    common_file = common_file,
                    clang_includes = [include_dirs..., CLANG_INCLUDE],
                    clang_args = clang_args,
                    header_wrapped = wrap_header,
                    header_library = x->library,
                    clang_diagnostics = true,
                  )
    run(context)

    return output_file, common_file
end


#
# Fixing-up
#

using CSTParser, Tokenize

## pass infrastructure

struct Edit{T}
    loc::T
    text::String
end

function pass(x, state, f = (x, state)->nothing)
    f(x, state)
    if length(x) > 0
        for a in x
            pass(a, state, f)
        end
    else
        state.offset += x.fullspan
    end
    state
end

function apply(text, edit::Edit{Int})
    string(text[1:edit.loc], edit.text, text[nextind(text, edit.loc):end])
end
function apply(text, edit::Edit{UnitRange{Int}})
    # println("Rewriting '$(text[edit.loc])' to '$(edit.text)'")
    string(text[1:prevind(text, first(edit.loc))], edit.text, text[nextind(text, last(edit.loc)):end])
end


## rewrite passes

mutable struct State
    offset::Int
    edits::Vector{Edit}
end

# insert `@checked` before each function with a `ccall` returning a checked type`
checked_types = [
    "CUresult",
    "CUptiResult",
    "NVPA_Status",
    "nvmlReturn_t",
    "cublasStatus_t",
    "cudnnStatus_t",
    "cufftResult",
    "curandStatus_t",
    "cusolverStatus_t",
    "cusparseStatus_t",
    "cutensorStatus_t",
    "cutensornetStatus_t",
]
function insert_check_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :function
        _, def, body, _ = x
        @assert body isa CSTParser.EXPR && body.head == :block
        @assert length(body) == 1

        # Clang.jl-generated ccalls should be directly part of a function definition
        call = body.args[1]
        @assert call isa CSTParser.EXPR && call.head == :call && call[1].val == "ccall"

        # get the ccall return type
        rv = call[5]

        if rv.val in checked_types
            push!(state.edits, Edit(state.offset, "@checked "))
        end
    end
end

# insert `initialize_context()` before each function with a `ccall` calling non-whitelisted fns
preinit_apicalls = Set{String}([
    # CUDAdrv
    ## error handling
    "cuGetErrorString",
    "cuGetErrorName",
    ## initialization
    "cuInit",
    ## version management
    "cuDriverGetVersion",
    ## device management
    "cuDeviceGet",
    "cuDeviceGetAttribute",
    "cuDeviceGetCount",
    "cuDeviceGetLuid",
    "cuDeviceGetName",
    "cuDeviceGetUuid",
    "cuDeviceTotalMem",
    "cuDeviceGetProperties",     # deprecated
    "cuDeviceComputeCapability", # deprecated
    ## context management
    "cuCtxCreate",
    "cuCtxDestroy",
    "cuCtxSetCurrent",
    "cuCtxGetCurrent",
    "cuCtxPushCurrent",
    "cuCtxPopCurrent",
    "cuCtxGetDevice", # this actually does require a context, but we use it unsafely
    ## primary context management
    "cuDevicePrimaryCtxGetState",
    "cuDevicePrimaryCtxRelease",
    "cuDevicePrimaryCtxReset",
    "cuDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxSetFlags",
    # CUPTI
    "cuptiGetVersion",
    "cuptiGetResultString",
    # NVML
    "nvmlInit",
    "nvmlInitWithFlags",
    "nvmlShutdown",
    "nvmlErrorString",
    # CUBLAS
    "cublasGetVersion",
    "cublasGetProperty",
    "cublasGetCudartVersion",
    "cublasSetLoggerCallback",
    "cublasGetLoggerCallback",
    # CURAND
    "curandGetVersion",
    "curandGetProperty",
    # CUFFT
    "cufftGetVersion",
    "cufftGetProperty",
    # CUSPARSE
    "cusparseGetVersion",
    "cusparseGetProperty",
    "cusparseGetErrorName",
    "cusparseGetErrorString",
    # CUSOLVER
    "cusolverGetVersion",
    "cusolverGetProperty",
    # CUDNN
    "cudnnGetVersion",
    "cudnnGetProperty",
    "cudnnGetCudartVersion",
    "cudnnGetErrorString",
    "cudnnSetCallback",
    "cudnnGetCallback",
    # CUTENSOR
    "cutensorGetVersion",
    "cutensorGetCudartVersion",
    "cutensorGetErrorString",
    # CUSTATEVEC
    "custatevecGetVersion",
    "custatevecGetCudartVersion",
    "custatevecGetErrorString",
    "custatevecLoggerSetCallback",
    "custatevecLoggerGetCallback",
    "custatevecLoggerSetFile",
    "custatevecLoggerOpenFile",
    "custatevecLoggerSetMask",
    "custatevecLoggerForceDisable",
    # CUTENSORNET
    "cutensornetGetVersion",
    "cutensornetGetCudartVersion",
    "cutensornetGetErrorString",
    "cutensornetLoggerSetCallback",
    "cutensornetLoggerGetCallback",
    "cutensornetLoggerSetFile",
    "cutensornetLoggerOpenFile",
    "cutensornetLoggerSetMask",
    "cutensornetLoggerForceDisable",
])
function insert_init_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :call && x.args[1].val == "ccall"
        fun = x[3][2][2].val
        actual_fun = first(split(fun, '_'))    # strip tags

        # call the API initializer
        if !in(actual_fun, preinit_apicalls)
            push!(state.edits, Edit(state.offset, "initialize_context()\n    "))
        end
    end
end

# replace handles from the CUDA runtime library with CUDA driver API equivalents
function rewrite_runtime_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :IDENTIFIER && x.val == "cudaStream_t"
        offset = state.offset
        push!(state.edits, Edit(offset+1:offset+x.span, "CUstream"))
    end
end

# remove aliases that are used to version functions
#
# when NVIDIA changes the behavior of an API, they version the function (`cuFunction_v2`).
# To maintain backwards compatibility, they ship aliases with their headers such that
# compiled binaries will keep using the old version, and newly-compiled ones will use the
# developer's CUDA version. remove those, since we target multiple CUDA versions.
#
# remove this if we ever decide to support a single supported version of CUDA.
cuda_version_alias(lhs, rhs) = occursin(Regex("$(lhs)_v\\d"), rhs)
nvtx_string_alias(lhs, rhs) = occursin(Regex("$(lhs)(A|W|Ex)"), rhs)
function remove_aliases_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :const
        @assert x[2] isa CSTParser.EXPR
        lhs, op, rhs = x[2]
        @assert CSTParser.isoperator(op) && op.val == "="
        if lhs.head == :IDENTIFIER && rhs.head == :IDENTIFIER &&
            (cuda_version_alias(lhs.val, rhs.val) || nvtx_string_alias(lhs.val, rhs.val))
            offset = state.offset
            push!(state.edits, Edit(offset+1:offset+x.span, ""))
        end
    end
end

# collect function definitions in a wrapped file
#
# when updating CUDA, deprecated functions are removed from the headers (but not from the
# binaries). we detect that by looking at the wrapped functions in the generated files.
function collect_function_definitions(file)
    text = read(file, String)

    state = State(0, Edit[])
    ast = CSTParser.parse(text, true)

    definitions = Dict()
    for x in ast
        fn = nothing
        if x isa CSTParser.EXPR && x.head == :function
            _, def, body, _ = x
            fn = def[1].val
        elseif x isa CSTParser.EXPR && x.head === :macrocall
            m, _, y = x
            if y isa CSTParser.EXPR && y.head === :function
                _, def, body, _ = y
                fn = def[1].val
            end
        elseif x isa CSTParser.EXPR && CSTParser.isliteral(x) && x.args === nothing
            # ignore
        else
            error("Unsupported global expression: $x")
        end

        # if we found a function definition, save its range
        if fn !== nothing
            definitions[fn] = (offset=state.offset, span=x.span, fullspan=x.fullspan)
        end

        state.offset += x.fullspan
    end

    return definitions
end


#
# Main application
#

using CUDA_full_jll, CUDNN_jll, CUTENSOR_jll

function process(name, headers...; libname=name, kwargs...)
    new_output_file, new_common_file = wrap(libname, headers...; kwargs...)

    for file in (new_output_file, new_common_file)
        text = read(file, String)


        ## rewriting passes

        state = State(0, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, insert_check_pass)

        state.offset = 0
        pass(ast, state, insert_init_pass)

        state.offset = 0
        pass(ast, state, rewrite_runtime_pass)

        state.offset = 0
        pass(ast, state, remove_aliases_pass)

        # apply
        state.offset = 0
        sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
        for i = 1:length(state.edits)
            text = apply(text, state.edits[i])
        end


        ## header

        squeezed = replace(text, "\n\n\n"=>"\n\n")
        while length(text) != length(squeezed)
            text = squeezed
            squeezed = replace(text, "\n\n\n"=>"\n\n")
        end
        text = squeezed


        write(file, text)
    end


    ## manual patches

    patchdir = joinpath(@__DIR__, "patches", name)
    if isdir(patchdir)
        for entry in readdir(patchdir)
            if endswith(entry, ".patch")
                path = joinpath(patchdir, entry)
                run(`patch -p1 -i $path`)
            end
        end
    end


    ## merge with existing wrappers

    new_output_text = read(new_output_file, String)

    existing_output_file = joinpath(dirname(dirname(@__DIR__)) , "lib", name, basename(new_output_file))
    @show existing_output_file
    @assert isfile(existing_output_file)
    existing_output_text = read(existing_output_file, String)

    new_defs = collect_function_definitions(new_output_file)
    existing_defs = collect_function_definitions(existing_output_file)

    # move removed methods to the 'deprecated' file and remove them from header
    removed = setdiff(keys(existing_defs), keys(new_defs))
    if !isempty(removed)
        deprecated_output_file = joinpath(dirname(dirname(@__DIR__)) , "lib", name, "lib$(libname)_deprecated.jl")
        open(deprecated_output_file, "a") do io
            state = State(0, Edit[])
            for fn in removed
                pos = existing_defs[fn]
                text = existing_output_text[pos.offset+1:pos.offset+pos.span]
                println(io)
                println(io, text)
                push!(state.edits, Edit(pos.offset+1:pos.offset+pos.fullspan, ""))
                @warn "Deprecating definition of $fn"
            end

            # apply removals
            sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
            for i = 1:length(state.edits)
                existing_output_text = apply(existing_output_text, state.edits[i])
            end

            write(existing_output_file, existing_output_text)
        end
    end

    # append new additions and prompt the user to review
    added = setdiff(keys(new_defs), keys(existing_defs))
    if !isempty(added)
        open(existing_output_file, "a") do io
            for fn in added
                pos = new_defs[fn]
                text = new_output_text[pos.offset+1:pos.offset+pos.span]
                println(io)
                println(io, text)
                @warn "Adding definition of $fn, please review!"
            end
        end
    end

    # apply the common file
    cp(new_common_file, joinpath(dirname(dirname(@__DIR__)) , "lib", name, basename(new_common_file)); force=true)

    return
end

function main()
    cuda = joinpath(CUDA_full_jll.artifact_dir, "cuda", "include")
    cupti = joinpath(CUDA_full_jll.artifact_dir, "cuda", "extras", "CUPTI", "include")
    cudnn = joinpath(CUDNN_jll.artifact_dir, "include")
    cutensor = joinpath(CUTENSOR_jll.artifact_dir, "include")
    cuquantum = joinpath(CUQUANTUM_jll.artifact_dir, "include")

    process("cudadrv", "$cuda/cuda.h","$cuda/cudaGL.h", "$cuda/cudaProfiler.h";
            include_dirs=[cuda], libname="cuda")

    process("nvtx", "$cuda/nvtx3/nvToolsExt.h", "$cuda/nvtx3/nvToolsExtCuda.h";
            include_dirs=[cuda])

    process("nvml", "$cuda/nvml.h"; include_dirs=[cuda])

    process("cupti", "$cupti/cupti.h", "$cupti/cupti_profiler_target.h";
            include_dirs=[cuda, cupti],
            wrapped_headers=["cupti_result.h", "cupti_version.h", "cupti_activity.h",
                             "cupti_callbacks.h", "cupti_events.h",
                             "cupti_profiler_target.h",
                             "cupti_metrics.h"], # deprecated, but still required
            defines=["__packed__"=>"", "aligned"=>""])
    # NOTE: libclang (the C API) doesn't support/expose the __packed__/aligned attributes,
    #       so disable them (Julia doesn't support packed structs anyway)

    process("cublas", "$cuda/cublas_v2.h", "$cuda/cublasXt.h";
            wrapped_headers=["cublas_v2.h", "cublas_api.h", "cublasXt.h"],
            include_dirs=[cuda])


    process("cufft", "$cuda/cufft.h"; include_dirs=[cuda])

    process("curand", "$cuda/curand.h"; include_dirs=[cuda])

    process("cusparse", "$cuda/cusparse.h"; include_dirs=[cuda])

    process("cusolver", "$cuda/cusolverDn.h", "$cuda/cusolverSp.h", "$cuda/cusolverMg.h";
            wrapped_headers=["cusolver_common.h", "cusolverDn.h", "cusolverSp.h", "cusolverMg.h"],
            include_dirs=[cuda])

    process("cudnn", "$cudnn/cudnn_version.h", "$cudnn/cudnn_ops_infer.h",
                     "$cudnn/cudnn_ops_train.h", "$cudnn/cudnn_adv_infer.h",
                     "$cudnn/cudnn_adv_train.h", "$cudnn/cudnn_cnn_infer.h",
                     "$cudnn/cudnn_cnn_train.h"; include_dirs=[cuda, cudnn])

    process("cutensor", "$cutensor/cutensor.h";
            wrapped_headers=["cutensor.h", "cutensor/types.h"],
            include_dirs=[cuda, cutensor])
    process("cutensornet", "$cuquantum/cutensornet.h";
            wrapped_headers=["cutensornet.h", "cutensornet/types.h"],
            include_dirs=[cuda, cuquantum])
    process("custatevec", "$cuquantum/custatevec.h";
            wrapped_headers=["custatevec.h", "custatevec/types.h"],
            include_dirs=[cuda, cuquantum])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
