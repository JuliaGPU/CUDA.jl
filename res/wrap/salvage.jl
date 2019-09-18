# salvage Ptr/CuPtr/PtrOrCuPtr type information from existing library wrappers

using CSTParser, Tokenize
using JSON, DataStructures


## pass infrastructure

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


## passes

mutable struct State
    offset::Int
end

function salvage_ccall(x, state)
    if x isa CSTParser.EXPR && x.typ == CSTParser.Call && x.args[1].val == "ccall"
        # get the ccall arguments, skipping comma's and parentheses
        handle = x.args[3]
        fn = handle.args[2].args[2].val
        rv = x.args[5]
        tt = x.args[7]
        args = x.args[9:2:end]

        # check for pointer types
        types = tt.args[2:2:end-1]
        is_pointer = Bool[x.typ == CSTParser.Curly && x.args[1].val in ("Ptr", "CuPtr", "PtrOrCuPtr") for x in types]
        if any(is_pointer)
            println(fn)

            # load the database of pointer argument types
            db = if isfile("pointers.json")
                JSON.parsefile("pointers.json"; dicttype=DataStructures.OrderedDict)
            else
                Dict{String, Any}()
            end

            # generate replacements
            replacements = OrderedDict{String,Any}()
            for (i, arg) in enumerate(args)
                replacements[arg.val] = if is_pointer[i]
                    println("- argument $i: $(arg.val)::$(Expr(types[i]))")
                    types[i].args[1].val
                else
                    nothing
                end
            end

            # save the database
            db[fn] = replacements
            open("pointers.json", "w") do io
                JSON.print(io, db, 4)
            end

            println()
        end
    end
end


#
# Main application
#

using CUDAapi

function process(path)
    text = read(path, String)

    state = State(0)
    ast = CSTParser.parse(text, true)
    pass(ast, state, salvage_ccall)

    return
end

function main()
    process("../../src/blas/libcublas.jl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
