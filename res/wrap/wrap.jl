# script to parse CUDA headers and generate Julia wrappers

using Crayons


#
# Parsing
#

using Clang

function wrap(name, headers...; library=":lib$name", defines=[])
    include_dirs = map(dir->joinpath(dir, "include"), find_toolkit())
    filter!(isdir, include_dirs)

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

    context = init(;
                    headers = [headers...],
                    output_file = output_file,
                    common_file = common_file,
                    clang_includes = [include_dirs..., CLANG_INCLUDE],
                    clang_args = clang_args,
                    header_wrapped = (root, current)->root == current,
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

const db = if isfile(db_path)
    JSON.parsefile(db_path; dicttype=DataStructures.OrderedDict)
else
    Dict{String, Any}()
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

# insert `@check` before each `ccall` when it returns a checked type
const checked_types = [
    "cublasStatus_t",
    "cudnnStatus_t",
    "cufftResult",
    "curandStatus_t",
    "cusolverStatus_t",
    "cusparseStatus_t",
    "cutensorStatus_t",
]
function insert_check(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        # get the ccall return type
        rv = x.args[5]

        if rv.val in checked_types
            push!(state.edits, Edit(state.offset, "@check "))
        end
    end
end

# change ::Ptr arguments to ::CuPtr / ::PtrOrCuPtr based on user input
function rewrite_pointers(x, state, headers)
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
function rewrite_runtime(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.IDENTIFIER && x.val == "cudaStream_t"
        offset = state.offset
        push!(state.edits, Edit(offset+1:offset+x.span, "CUstream"))
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

function indent_ccall(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
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

function indent_definition(x, state)
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

using CUDAapi

function process(name, headers...; kwargs...)
    output_file, common_file = wrap(name, headers...; kwargs...)

    for file in (output_file, common_file)
        text = read(file, String)


        ## rewriting passes

        state = State(0, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, insert_check)

        state.offset = 0
        pass(ast, state, (x,state)->rewrite_pointers(x,state,headers))

        state.offset = 0
        pass(ast, state, rewrite_runtime)

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
        pass(ast, state, indent_definition)

        state.offset = 0
        pass(ast, state, indent_ccall)

        # apply
        state.offset = 0
        sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
        for i = 1:length(state.edits)
            text = apply(text, state.edits[i])
        end


        write(file, text)
    end


    ## manual patches

    patchdir = joinpath(@__DIR__, "patches")
    for entry in readdir(patchdir)
        if startswith(entry, name) && endswith(entry, ".patch")
            path = joinpath(patchdir, entry)
            run(`patch -p1 -i $path`)
        end
    end


    ## move to destination

    for src in (output_file, common_file)
        dst = joinpath(dirname(dirname(@__DIR__)), "src", name[3:end], src)
        cp(src, dst; force=true)
    end


    return
end

function main()
    # find the CUDA includes directory
    toolkit = find_toolkit()
    cuda_include = nothing
    for dir in toolkit
        cuda_include = joinpath(dir, "include")
        isdir(cuda_include) && break
    end
    isdir(cuda_include) || error("Could not find an 'include' directory in any of the toolkit directories ($(join(toolkit, ", ", " or ")))")
    @info "Found CUDA include directory at $cuda_include"

    function process_if_existing(name, headers...; kwargs...)
        env_var = uppercase(name) * "_INCLUDE"
        include = get(ENV, env_var, cuda_include)
        for header in headers
            if !isfile(joinpath(include, header))
                @error "Cannot wrap $name: could not find $header in $include (override the include directory with the $env_var environment variable)"
                return
            end
        end

        headers = map(header->joinpath(include, header), headers)
        process(name, headers...; kwargs...)
    end

    process_if_existing("cublas",
                        "cublas_v2.h", "cublas_api.h", "cublasXt.h";
                        defines=["CUBLASAPI"=>""])

    process_if_existing("cufft", "cufft.h")

    process_if_existing("curand", "curand.h")

    process_if_existing("cusparse", "cusparse.h")

    process_if_existing("cusolver",
                        "cusolver_common.h", "cusolverDn.h", "cusolverSp.h")

    process_if_existing("cudnn", "cudnn.h")

    process_if_existing("cutensor", "cutensor/types.h", "cutensor.h")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
