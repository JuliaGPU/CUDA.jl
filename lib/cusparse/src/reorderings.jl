# routines that implement reorderings

export color

"""
    color(A; index='O', percentage=1.0) -> ncolors, coloring, reordering

Color the adjacency graph of the square sparse CSR or CSC matrix `A`, which must have a
symmetric sparsity pattern: pass `A + transpose(A)` otherwise. Neighbouring nodes get
different colors.

The coloring is approximate, and stops when `percentage` of the nodes are colored; the
other nodes get distinct colors after the last one used. `coloring[i]` is the color of node
`i`, and `reordering` is a permutation that groups nodes of the same color together.

Wraps `cusparse<t>csrcolor`, which NVIDIA has deprecated, for the `Float32`, `Float64`,
`ComplexF32` and `ComplexF64` element types.
"""
function color end

for (fname, subty, elty) in ((:cusparseScsrcolor, :Float32, :Float32),
                             (:cusparseDcsrcolor, :Float64, :Float64),
                             (:cusparseCcsrcolor, :Float32, :ComplexF32),
                             (:cusparseZcsrcolor, :Float64, :ComplexF64))
    @eval begin
        function color(A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}}; index::SparseChar='O', percentage::Number=1.0)
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
