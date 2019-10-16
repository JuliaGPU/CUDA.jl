# script to parse CUDA headers and generate Julia wrappers


#
# Parsing
#

using Clang

function wrap(name, headers...; library="lib$name", includes=[], defines=[], undefines=[])
    clang_args = String[]
    append!(clang_args, map(dir->"-I$dir", includes))
    for define in defines
        if isa(define, Pair)
            append!(clang_args, ["-D", "$(first(define))=$(last(define))"])
        else
            append!(clang_args, ["-D", "$define"])
        end
    end
    for undefine in undefines
        append!(clang_args, ["-U", "$undefine"])
    end

    output_file = "lib$(name).jl"
    common_file = "lib$(name)_common.jl"

    context = init(;
                    headers = [headers...],
                    output_file = output_file,
                    common_file = common_file,
                    clang_includes = [includes..., CLANG_INCLUDE],
                    clang_args = clang_args,
                    header_wrapped = (root, current)->root == current,
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
    "CUptiResult",
    "NVPA_Status",
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
        dst = joinpath(dirname(@__DIR__), "src", name, src)
        cp(src, dst; force=true)
    end


    return
end

function main()
    toolkit_dirs = find_toolkit()

    # find the CUDA include directory
    cuda_include = nothing
    for dir in toolkit_dirs
        cuda_include = joinpath(dir, "include")
        isdir(cuda_include) && break
    end
    isdir(cuda_include) || error("Could not find the CUDA include directory in any of the toolkit directories ($(join(toolkit_dirs, ", ", " or ")))")
    @info "Found CUDA include directory directory at $cuda_include"

    # find the CUPTI include directory
    cupti_include = nothing
    for dir in toolkit_dirs
        cupti_include = joinpath(dir, "extras", "CUPTI", "include")
        isdir(cupti_include) && break
    end
    isdir(cupti_include) || error("Could not find the CUPTI include directory in any of the toolkit directories ($(join(toolkit, ", ", " or ")))")
    @info "Found CUPTI include directory directory at $cupti_include"

    function wrap(name, dir, headers...; kwargs...)
        env_var = uppercase(name) * "_INCLUDE"
        include = get(ENV, env_var, dir)
        for header in headers
            if !isfile(joinpath(include, header))
                @error "Cannot wrap $name: could not find $header in $include (override the include directory with the $env_var environment variable)"
                return
            end
        end

        headers = map(header->joinpath(include, header), headers)
        process(name, headers...; kwargs..., includes=[cuda_include, cupti_include])
    end

    wrap("nvtx", cuda_include, "nvtx3/nvToolsExt.h", "nvtx3/nvToolsExtCuda.h")

    wrap("cupti", cupti_include,  "cupti_result.h", "cupti_version.h",
                                  "cupti_activity.h", "cupti_callbacks.h",
                                  "cupti_events.h", "cupti_metrics.h", # deprecated, but still required
                                  "cupti_profiler_target.h";
                                   defines=["__packed__"=>"", "aligned"=>""])
    # NOTE: libclang (the C API) doesn't support/expose the __packed__/aligned attributes,
    #       so disable them (Julia doesn't support packed structs anyway)

    wrap("nvperf", cupti_include, "nvperf_host.h")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
