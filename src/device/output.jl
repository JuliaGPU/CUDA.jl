# print-like functionality

export @cuprint, @cuprintln

const kernel_strings = Union{String,Symbol}[]
struct KernelString
  id::Int
end

function _cuprint(args...)
    actual_args = map(arg -> isa(arg, KernelString) ? kernel_strings[arg.id] : arg, args)
    print(actual_args...)
    return
end

"""
    @cuprint(xs...)
    @cuprintln(xs...)

Print a textual representation of values `xs` to standard output from the GPU.

Limited string interpolation is also possible:

```julia
    @cuprint("Hello, World ", 42, "\\n")
    @cuprint "Hello, World \$(42)\\n"
```
"""
macro cuprint(parts...)
    args = []

    parts = [parts...]
    while true
        isempty(parts) && break

        part = popfirst!(parts)

        # handle string interpolation
        if isa(part, Expr) && part.head == :string
            parts = vcat(part.args, parts)
            continue
        end

        if isa(part, String)
            push!(kernel_strings, part)
            id = length(kernel_strings)
            push!(args, :(CUDA.KernelString($id)))
        elseif isa(part, QuoteNode)
            push!(kernel_strings, part.value)
            id = length(kernel_strings)
            push!(args, :(CUDA.KernelString($id)))
        else
            push!(args, part)
        end
    end

    esc(quote
        CUDA.@hostcall CUDA._cuprint($(args...))::Nothing
    end)
end

@doc (@doc @cuprint) ->
macro cuprintln(parts...)
    esc(quote
        CUDA.@cuprint($(parts...), "\n")
    end)
end

export @cushow

"""
    @cushow(ex)

GPU analog of `Base.@show`. It comes with the same type restrictions as [`@cuprintf`](@ref).

```julia
@cushow threadIdx().x
```
"""
macro cushow(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(CUDA.@cuprintln($(sprint(Base.show_unquoted,ex)*" = "),
                                          begin local value = $(esc(ex)) end)))
    end
    isempty(exs) || push!(blk.args, :value)
    blk
end
