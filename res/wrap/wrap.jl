# script to parse CUDA headers and generate Julia wrappers

#
# Parsing
#

using Clang

function wrap(name, headers...; library="lib$name", defines=[])
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

    return output_file
end


#
# Fixing-up
#

using CSTParser, Tokenize
using JSON, DataStructures

## rewrite/pass infrastructure

struct Edit{T}
    loc::T
    text::String
end

mutable struct State{T}
    offset::Int
    edits::T
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

## passes

# insert `@check` before each `ccall` when it returns a xxxStatus_t
function insert_check(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        # get the ccall return type
        rv = x.args[5]

        if endswith(rv.val, "Status_t")
            push!(state.edits, Edit(state.offset, "@check "))
        end
    end
end

# change ::Ptr arguments to ::CuPtr / ::PtrOrCuPtr based on user input
function rewrite_pointers(x, state)
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
            println(fn)

            # load the database of pointer argument types
            # NOTE: loading/writing of the db is purposefully done in this inner function,
            #       since realistic headers might have huge amounts of replacements
            #       and you might want to resume the process later on.
            db = if isfile("pointers.json")
                JSON.parsefile("pointers.json"; dicttype=DataStructures.OrderedDict)
            else
                Dict{String, Any}()
            end

            if haskey(db, fn)
                replacements = db[fn]

                # print the cached result
                for (i, (name,replacement)) in enumerate(replacements)
                    if is_pointer[i]
                        println("- argument $i: $name::$(Expr(types[i])) is a $replacement")
                    end
                end
                println()

                # regenerate replacements with the new argument names
                old_replacements = collect(replacements)
                replacements = OrderedDict{String,Any}()
                for (i, arg) in enumerate(args)
                    replacements[arg.val] = old_replacements[i].second
                end
            else
                # print pointer arguments and their types
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
                print("Dual GPU/CPU pointers> ")
                dual_pointers = parse.(Int, split(readline(stdin)))

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
            end

            # always save: the argument names might have changed in the current header
            db[fn] = replacements
            open("pointers.json", "w") do io
                JSON.print(io, db, 4)
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
        push!(state.edits, Edit(offset+1:offset+x.span, "CuStream_t"))
    end
end

# TODO: replace cudaStream_t to CuStream_t


#
# Main application
#

using CUDAapi

function process(args...; kwargs...)
    path = wrap(args...; kwargs...)
    text = read(path, String)

    # passes
    state = State(0, Edit[])
    ast = CSTParser.parse(text, true)

    state.offset = 0
    pass(ast, state, insert_check)

    state.offset = 0
    pass(ast, state, rewrite_pointers)

    state.offset = 0
    pass(ast, state, rewrite_runtime)

    # apply
    state.offset = 0
    sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
    for i = 1:length(state.edits)
        text = apply(text, state.edits[i])
    end

    write(path, text)

    return
end

function main()
    # TODO: use CUDAapi to discover headers
    process("cudnn", "/usr/include/cudnn.h"; library="@libcudnn")
    # wrap("cublas", "/opt/cuda/include/cublas.h",
    #                "/opt/cuda/include/cublas_api.h",
    #                "/opt/cuda/include/cublasXt.h";
    #      defines=["CUBLASAPI"=>""])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
