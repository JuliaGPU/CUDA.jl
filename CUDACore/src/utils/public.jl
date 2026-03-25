# public keyword support for Julia < 1.11

_public_symbol(s::Symbol) = s
_public_symbol(e::Expr) = e.args[1]  # macrocall: @foo → :@foo
_public_symbols(s::Symbol) = [s]
function _public_symbols(e::Expr)
    if e.head == :tuple
        Symbol[_public_symbol(a) for a in e.args]
    else
        # single macrocall like @foo
        [_public_symbol(e)]
    end
end

# Track public names so they can be discovered on Julia < 1.11
const PUBLIC_NAMES = Symbol[]

macro public(symbols_expr)
    syms = _public_symbols(symbols_expr)
    append!(PUBLIC_NAMES, syms)
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, syms...))
    else
        nothing
    end
end
export @public
