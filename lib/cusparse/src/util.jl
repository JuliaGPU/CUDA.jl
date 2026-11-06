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
check that the dimensions of matrices `B` and `C` make sense for a multiplication
"""
function chkmmdims(B, C, k, l, m, n)
    if size(B) != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C) != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

"""
check that the dimensions of arrays `B` and `C` make sense for a batched matrix-matrix multiplication
"""
function chkbmmdims(B, C, k, l, m, n)
    if size(B)[1:2] != (k,l)
        throw(DimensionMismatch("B has dimensions $(size(B)) but needs ($k,$l)"))
    elseif size(C)[1:2] != (m,n)
        throw(DimensionMismatch("C has dimensions $(size(C)) but needs ($m,$n)"))
    end
end

"""
    cint_view(A)

The legacy routines of cuSPARSE and cuSOLVER only take 32-bit indices. Return `A` with
`Cint` indices, sharing its values (which in-place routines update), or `A` itself if its
indices are `Cint` already.
"""
cint_view(A::Union{CuSparseMatrixCSR{<:Any,Cint},CuSparseMatrixCSC{<:Any,Cint},
                   CuSparseMatrixCOO{<:Any,Cint},CuSparseVector{<:Any,Cint}}) = A
cint_view(A::CuSparseMatrixBSR{<:Any,Cint}) = A
function cint_view(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO,CuSparseVector})
    max(size(A)..., nnz(A) + 1) <= typemax(Cint) ||
        throw(ArgumentError("this cuSPARSE routine takes 32-bit indices, which cannot represent a $(join(size(A), '×')) array with $(nnz(A)) stored entries"))
    buffers = GPUArrays.sparse_buffers(A)
    return typeof(A).name.wrapper(map(b -> convert(CuVector{Cint}, b), Base.front(buffers))...,
                                  last(buffers), A isa CuSparseVector ? length(A) : size(A))
end
cint_view(A::CuSparseMatrixBSR) =
    CuSparseMatrixBSR{eltype(A),Cint}(A.rowPtr, A.colVal, nonzeros(A), size(A), A.blockDim,
                                      A.dir, A.nnzb)

# the forwarding methods for other index types use this, to avoid recursing forever when the
# element type is not supported either
function check_forwarding(f, A)
    A === cint_view(A) && throw(MethodError(f, (A,)))
    return
end
