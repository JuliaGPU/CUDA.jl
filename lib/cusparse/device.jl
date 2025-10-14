# on-device sparse array functionality
# should be excluded from coverage counts
# COV_EXCL_START
using SparseArrays


const CuSparseDeviceColumnView{Tv, Ti} = SubArray{Tv, 1, <:GPUArrays.GPUSparseDeviceMatrixCSC{Tv, Ti}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
function SparseArrays.nonzeros(x::CuSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    @inbounds y = view(SparseArrays.nonzeros(A), SparseArrays.nzrange(A, colidx))
    return y
end

function SparseArrays.nonzeroinds(x::CuSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    @inbounds y = view(SparseArrays.rowvals(A), SparseArrays.nzrange(A, colidx))
    return y
end
SparseArrays.rowvals(x::CuSparseDeviceColumnView) = SparseArrays.nonzeroinds(x)

function SparseArrays.nnz(x::CuSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    return length(SparseArrays.nzrange(A, colidx))
end

# COV_EXCL_STOP
