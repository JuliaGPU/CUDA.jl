export
    mkstemps

# Generate a temporary file with specific suffix
function mkstemps(suffix::AbstractString)
    b = joinpath(tempdir(), "tmpXXXXXX$suffix")
    # NOTE: mkstemps modifies b, which should be a NULL-terminated string
    p = ccall(:mkstemps, Int32, (Cstring, Cint), b, length(suffix))
    systemerror(:mktemp, p == -1)
    return (b, fdio(p, true))
end