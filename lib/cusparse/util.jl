# utility functions for the CUSPARSE wrappers

"""
check that the dimensions of matrix `X` and vector `Y` make sense for a multiplication
"""
function chkmvdims(X, n, Y, m)
    if length(X) != n
        throw(DimensionMismatch("X must have length $n, but has length $(length(X))"))
    elseif length(Y) != m
        throw(DimensionMismatch("Y must have length $m, but has length $(length(Y))"))
    end
end

"""
check that the dimensions of matrices `X` and `Y` make sense for a multiplication
"""
function chkmmdims( B, C, k, l, m, n )
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end
