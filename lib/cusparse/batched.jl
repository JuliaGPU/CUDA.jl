# concat
function batchcat(A::CuSparseMatrixCSR...)
    b = length(A)
    CuSparseArrayCSR(cat([A[i].rowPtr for i=1:b]...; dims=2),
                     cat([A[i].colVal for i=1:b]...; dims=2),
                     cat([A[i].nzVal for i=1:b]...; dims=2),
                     (size(A[1])..., b))
end

# mm! (C dense = A sparse * B dense) (generic.jl)

# scalar addition/subtraction, scalar mul/div (see interfaces.jl +412)

# chkmmdims (see util.jl)
