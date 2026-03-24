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
macro public(symbols_expr)
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, _public_symbols(symbols_expr)...))
    end
end
export @public
