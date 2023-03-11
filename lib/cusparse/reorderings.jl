# routines that implement reorderings

export color

"""
    color(A::CuSparseMatrixCSC, index::SparseChar='O'; percentage::Number=1.0)
    color(A::CuSparseMatrixCSR, index::SparseChar='O'; percentage::Number=1.0)

This function performs the coloring of the adjacency graph associated with the matrix A.
The coloring is an assignment of colors (integer numbers) to nodes, such that neighboring nodes have distinct colors.
An approximate coloring algorithm is used in this routine, and is stopped when a certain percentage of nodes has been colored.
The rest of the nodes are assigned distinct colors (an increasing sequence of integers numbers, starting from the last integer used previously).
The reordering is such that nodes that have been assigned the same color are reordered to be next to each other.

The matrix A passed to this routine, must be stored as a general matrix and have a symmetric sparsity pattern.
If the matrix is non-symmetric the user should pass A + Aᵀ as a parameter to this routine.
"""
function color end

for (fname, subty, elty) in ((:cusparseScsrcolor, :Float32, :Float32),
                             (:cusparseDcsrcolor, :Float64, :Float64),
                             (:cusparseCcsrcolor, :Float32, :ComplexF32),
                             (:cusparseZcsrcolor, :Float64, :ComplexF64))
    @eval begin
        function color(A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}}, index::SparseChar='O'; percentage::Number=1.0)
            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m, n = size(A)
            (m != n) && throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))

            info = cusparseColorInfo_t[0]
            cusparseCreateColorInfo(info)

            ncolors = Ref{Cint}(-1)
            coloring = CuVector{Cint}(undef, m)
            reordering = CuVector{Cint}(undef, m)

            if isa(A, CuSparseMatrixCSR)
                $fname(handle(), m, nnz(A), desc, nonzeros(A), A.rowPtr, A.colVal, Ref{$subty}(percentage), ncolors, coloring, reordering, info[1])
            else
                $fname(handle(), m, nnz(A), desc, nonzeros(A), A.colPtr, A.rowVal, Ref{$subty}(percentage), ncolors, coloring, reordering, info[1])
            end
            cusparseDestroyColorInfo(info[1])
            ncolors[], coloring, reordering
        end
    end
end
