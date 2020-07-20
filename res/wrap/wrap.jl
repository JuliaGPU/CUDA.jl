# script to parse CUDA headers and generate Julia wrappers

using Crayons


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
# Pointer type database
#

using JSON, DataStructures

const db_path = joinpath(@__DIR__, "pointers.json")

function load_db()
    global db
    db = if isfile(db_path)
        JSON.parsefile(db_path; dicttype=DataStructures.OrderedDict)
    else
        Dict{String, Any}()
    end
end

function save_db()
    global db
    open(db_path, "w") do io
        JSON.print(io, db, 4)
    end
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
    if x.args isa Vector
        for a in x.args
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
]
function insert_check_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.FunctionDef
        _, def, body, _ = x.args
        @assert body isa CSTParser.EXPR && body.typ == CSTParser.Block
        @assert length(body.args) == 1

        # Clang.jl-generated ccalls should be directly part of a function definition
        call = body.args[1]
        @assert call isa CSTParser.EXPR && call.typ == CSTParser.Call && call.args[1].val == "ccall"

        # get the ccall return type
        rv = call.args[5]

        if rv.val in checked_types
            push!(state.edits, Edit(state.offset, "@checked "))
        end
    end
end

# rewrite ordinary `ccall`s to `@runtime_ccall`
function rewrite_ccall_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        push!(state.edits, Edit(state.offset, "@runtime_"))

        # rewrite the function handle `libcuda` into a call `libcuda()`
        # FIXME: this used to work by setting `libname="lib$name()"`,
        #        but starting with Julia 1.4 this resulted in `var"libcuda()"`
        handle = x.args[3]
        if handle.typ == CSTParser.TupleH
            fun, lib = handle.args[2], handle.args[4]
            @assert lib.typ == CSTParser.IDENTIFIER

            offset = state.offset + sum(x->x.fullspan, x.args[1:2]) +
                                    sum(x->x.fullspan, handle.args[1:4])
            push!(state.edits, Edit(offset, "()"))
        end
    end
end

# insert `initialize_api()` before each function with a `ccall` calling non-whitelisted fns
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
    # CUTENSOR
    "cutensorGetVersion",
    "cutensorGetCudartVersion",
    "cutensorGetErrorString",
])
function insert_init_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        fun = x.args[3].args[2].args[2].val

        # strip the version tag
        if occursin(r"_v\d$", fun)
            fun = fun[1:end-3]
        end

        # call the API initializer
        if !in(fun, preinit_apicalls)
            push!(state.edits, Edit(state.offset, "initialize_api()\n    "))
        end
    end
end

# change ::Ptr arguments to ::CuPtr / ::PtrOrCuPtr based on user input
function rewrite_pointers_pass(x, state, headers)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        # get the ccall arguments, skipping comma's and parentheses
        handle = x.args[3]
        fn = handle.args[2].args[2].val
        rv = x.args[5]
        tt = x.args[7]
        args = x.args[9:2:end]

        # check for pointer types
        types = tt.args[2:2:end-1]
        is_pointer = Bool[x.typ == CSTParser.Curly && x.args[1].val == "Ptr" for x in types]
        offset = state.offset + sum(x->x.fullspan, x.args[1:6])
        if any(is_pointer)
            if haskey(db, fn)
                replacements = db[fn]

                # regenerate replacements with the new argument names
                changed = false
                old_replacements = collect(replacements)
                replacements = OrderedDict{String,Any}()
                for (i, arg) in enumerate(args)
                    if arg.val != old_replacements[i].first
                        changed = true
                    end
                    replacements[arg.val] = old_replacements[i].second
                end

                if changed
                    db[fn] = replacements
                    save_db()
                end
            else
                # print some context from the header (some mention host/device pointer)
                print(Crayon(foreground=:yellow))
                run(`awk "/\<$fn\>/,/;/" $headers`)
                println(Crayon(reset=true))

                # print pointer arguments and their types
                println(Crayon(foreground = :red), fn, Crayon(reset=true))
                for (i, arg) in enumerate(args)
                    if is_pointer[i]
                        println("- argument $i: $(arg.val)::$(Expr(types[i]))")
                    end
                end
                println()

                # prompt
                run(pipeline(`echo -n $fn`, `xclip -i -selection clipboard`));
                print("GPU pointers> ")
                gpu_pointers = parse.(Int, split(readline(stdin)))
                if gpu_pointers == [0]
                    # 0 is special match for all pointers
                    gpu_pointers = findall(is_pointer)
                elseif !isempty(gpu_pointers) && all(i->i<0, gpu_pointers)
                    # negative indicates all but these
                    gpu_pointers = map(i->-i, gpu_pointers)
                    @assert all(i->is_pointer[i], gpu_pointers) "You selected non-pointer arguments"
                    gpu_pointers = setdiff(findall(is_pointer), gpu_pointers)
                end
                print("Dual GPU/CPU pointers> ")
                dual_pointers = parse.(Int, split(readline(stdin)))
                @assert all(i->is_pointer[i], gpu_pointers ∪ dual_pointers) "You selected non-pointer arguments"

                # generate replacements
                replacements = OrderedDict{String,Any}()
                for (i, arg) in enumerate(args)
                    replacements[arg.val] = if is_pointer[i]
                        if i in gpu_pointers
                            "CuPtr"
                        elseif i in dual_pointers
                            "PtrOrCuPtr"
                        else
                            "Ptr"
                        end
                    else
                        nothing
                    end
                end

                db[fn] = replacements
                save_db()
            end

            # generate edits
            for (i, (_,replacement)) in enumerate(replacements)
                offset += tt.args[2*i-1].fullspan
                if replacement !== nothing && replacement != "Ptr"
                    ptr = types[i].args[1]
                    push!(state.edits, Edit(offset+1:offset+ptr.span, replacement))
                end
                offset += types[i].fullspan
            end

            println()
        end
    end
end

# replace handles from the CUDA runtime library with CUDA driver API equivalents
function rewrite_runtime_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.IDENTIFIER && x.val == "cudaStream_t"
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
    if x isa CSTParser.EXPR && x.typ == CSTParser.Const
        @assert x.args[2].typ == CSTParser.BinaryOpCall
        lhs, op, rhs = x.args[2].args
        @assert op.typ == CSTParser.OPERATOR && op.kind == Tokenize.Tokens.EQ
        if lhs.typ == CSTParser.IDENTIFIER && rhs.typ == CSTParser.IDENTIFIER &&
            (cuda_version_alias(lhs.val, rhs.val) || nvtx_string_alias(lhs.val, rhs.val))
            offset = state.offset
            push!(state.edits, Edit(offset+1:offset+x.span, ""))
        end
    end
end


## indenting passes

mutable struct IndentState
    offset::Int
    lines
    edits::Vector{Edit}
end

function get_lines(text)
    lines = Tuple{Int,Int}[]
    pt = Tokens.EMPTY_TOKEN(Tokens.Token)
    for t in CSTParser.Tokenize.tokenize(text)
        if pt.endpos[1] != t.startpos[1]
            if t.kind == Tokens.WHITESPACE
                nl = findfirst("\n", t.val) != nothing
                if !nl
                    push!(lines, (length(t.val), 0))
                else
                end
            else
                push!(lines, (0, 0))
            end
        elseif t.startpos[1] != t.endpos[1] && t.kind == Tokens.TRIPLE_STRING
            nls = findall(x->x == '\n', t.val)
            for nl in nls
                push!(lines, (t.startpos[2] - 1, nl + t.startbyte))
            end
        elseif t.startpos[1] != t.endpos[1] && t.kind == Tokens.WHITESPACE
            push!(lines, (t.endpos[2], t.endbyte - t.endpos[2] + 1))
        end
        pt = t
    end
    lines
end

function wrap_at_comma(x, state, indent, offset, column)
    comma = nothing
    for y in x
        if column + y.fullspan > 92 && comma !== nothing
            column = indent
            push!(state.edits, Edit(comma, ",\n" * " "^column))
            column += offset - comma[1] - 1 # other stuff might have snuck between the comma and the current expr
            comma = nothing
        elseif y.typ == CSTParser.PUNCTUATION && y.kind == Tokens.COMMA
            comma = (offset+1):(offset+y.fullspan)
        end
        offset += y.fullspan
        column += y.fullspan
    end
end

function indent_ccall_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.MacroCall && x.args[1].args[2].val == "runtime_ccall"
        # figure out how much to indent by looking at where the expr starts
        line = findlast(y -> state.offset >= y[2], state.lines) # index, not the actual number
        line_indent, line_offset = state.lines[line]
        expr_indent = state.offset - line_offset
        indent = expr_indent + sum(x->x.fullspan, x.args[1:2])

        if length(x.args[7]) > 2    # non-empty tuple type
            # break before the tuple type
            offset = state.offset + sum(x->x.fullspan, x.args[1:6])
            push!(state.edits, Edit(offset:offset, "\n" * " "^indent))

            # wrap tuple type
            wrap_at_comma(x.args[7], state, indent+1, offset, indent+1)
        end

        if length(x.args) > 9
            # break before the arguments
            offset = state.offset + sum(x->x.fullspan, x.args[1:8])
            push!(state.edits, Edit(offset:offset, "\n" * " "^indent))

            # wrap arguments
            wrap_at_comma(x.args[9:end], state, indent, offset, indent)
        end
    end
end

function indent_definition_pass(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.FunctionDef
        # figure out how much to indent by looking at where the expr starts
        line = findlast(y -> state.offset >= y[2], state.lines) # index, not the actual number
        line_indent, line_offset = state.lines[line]
        expr_indent = state.offset - line_offset
        indent = expr_indent + x.args[1].fullspan + sum(x->x.fullspan, x.args[2].args[1:2])

        if length(x.args[2]) > 2    # non-empty args
            offset = state.offset + x.args[1].fullspan + sum(x->x.fullspan, x.args[2].args[1:2])
            wrap_at_comma(x.args[2].args[3:end-1], state, indent, offset, indent)
        end
    end
end


#
# Main application
#

using CUDA_full_jll, CUDNN_CUDA102_jll, CUTENSOR_CUDA102_jll

function process(name, headers...; libname=name, rewrite_pointers=true, kwargs...)
    output_file, common_file = wrap(libname, headers...; kwargs...)

    for file in (output_file, common_file)
        text = read(file, String)


        ## rewriting passes

        state = State(0, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, insert_check_pass)

        state.offset = 0
        pass(ast, state, rewrite_ccall_pass)

        state.offset = 0
        pass(ast, state, insert_init_pass)

        if rewrite_pointers
            state.offset = 0
            pass(ast, state, (x,state)->rewrite_pointers_pass(x,state,headers))
        end

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


        ## indentation passes

        lines = get_lines(text)
        state = IndentState(0, lines, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, indent_definition_pass)

        state.offset = 0
        pass(ast, state, indent_ccall_pass)

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

        text = """
            # Automatically-generated headers for $name
            #
            # DO NOT EDIT THIS FILE DIRECTLY.


            """ * squeezed


        write(file, text)
    end


    ## move to destination

    dstdir = joinpath(dirname(dirname(@__DIR__)), "lib", name)
    for src in (output_file, common_file)
        dst = joinpath(dstdir, src)
        cp(src, dst; force=true)
    end


    ## manual patches

    patchdir = joinpath(@__DIR__, "patches", name)
    if isdir(patchdir)
        for entry in readdir(patchdir)
            if endswith(entry, ".patch")
                path = joinpath(patchdir, entry)
                run(`patch -p1 -d $dstdir -i $path`)
            end
        end
    end


    return
end

function main()
    load_db()

    cuda = joinpath(CUDA_full_jll.artifact_dir, "cuda", "include")
    cupti = joinpath(CUDA_full_jll.artifact_dir, "cuda", "extras", "CUPTI", "include")
    cudnn = joinpath(CUDNN_CUDA102_jll.artifact_dir, "include")
    cutensor = joinpath(CUTENSOR_CUDA102_jll.artifact_dir, "include")

    process("cudadrv", "$cuda/cuda.h", "$cuda/cudaProfiler.h";
            include_dirs=[cuda], libname="cuda", rewrite_pointers=false)

    process("nvtx", "$cuda/nvtx3/nvToolsExt.h", "$cuda/nvtx3/nvToolsExtCuda.h";
            include_dirs=[cuda], rewrite_pointers=false)

    process("nvml", "$cuda/nvml.h"; include_dirs=[cuda], rewrite_pointers=false)

    process("cupti", "$cupti/cupti.h", "$cupti/cupti_profiler_target.h";
            include_dirs=[cuda, cupti],
            wrapped_headers=["cupti_result.h", "cupti_version.h", "cupti_activity.h",
                             "cupti_callbacks.h", "cupti_events.h",
                             "cupti_profiler_target.h",
                             "cupti_metrics.h"], # deprecated, but still required
            defines=["__packed__"=>"", "aligned"=>""],
            rewrite_pointers=false)
    # NOTE: libclang (the C API) doesn't support/expose the __packed__/aligned attributes,
    #       so disable them (Julia doesn't support packed structs anyway)

    process("cublas", "$cuda/cublas_v2.h", "$cuda/cublasXt.h";
            wrapped_headers=["cublas_v2.h", "cublas_api.h", "cublasXt.h"],
            defines=["CUBLASAPI"=>""],
            include_dirs=[cuda])
    
    process_if_existing("cudalibmg", "cudalibmg.h";
                        wrapped_headers=["cudalibmg.h", "cudalibmg/types.h"])
    
    incs = [get(ENV, "CUDALIBMG_INCLUDE", cuda_include)]
    process_if_existing("cublasmg", "cublasMg.h";
                        wrapped_headers=["cublasMg.h", "cublasmg/types.h"], includes = incs)


    process("cufft", "$cuda/cufft.h"; include_dirs=[cuda])

    process("curand", "$cuda/curand.h"; include_dirs=[cuda])

    process("cusparse", "$cuda/cusparse.h"; include_dirs=[cuda])

    process("cusolver", "$cuda/cusolverDn.h", "$cuda/cusolverSp.h";
             wrapped_headers=["cusolver_common.h", "cusolverDn.h", "cusolverSp.h"],
             include_dirs=[cuda])

    process("cudnn", "$cudnn/cudnn.h"; include_dirs=[cuda, cudnn])

    process("cutensor", "$cutensor/cutensor.h";
            wrapped_headers=["cutensor.h", "cutensor/types.h"],
            include_dirs=[cuda, cutensor])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
