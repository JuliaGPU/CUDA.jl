function Base.cat(A::CuSparseMatrixCSR, As::CuSparseMatrixCSR...; dims=3)
    @assert dims ≥ 3 "only batch-dimension cat supported"
    As = (A, As...)
    newsize = (size(As[1])..., ones(Int, dims-3)..., length(As))
    CuSparseArrayCSR(cat([A.rowPtr for A in As]...; dims=dims-1),
                     cat([A.colVal for A in As]...; dims=dims-1),
                     cat([A.nzVal  for A in As]...; dims=dims-1),
                     newsize)
end

function Base.cat(A::CuSparseArrayCSR, As::CuSparseArrayCSR...; dims=3)
    @assert dims ≥ 3 "only batch-dimension cat supported"
    As = (A, As...)
    rowPtr = cat([A.rowPtr for A in As]...; dims=dims-1)
    CuSparseArrayCSR(rowPtr,
                     cat([A.colVal for A in As]...; dims=dims-1),
                     cat([A.nzVal  for A in As]...; dims=dims-1),
                     (size(As[1])[1:2]..., size(rowPtr)[2:end]...))
end

# we can't reshape the first two dimensions
function Base.reshape(A::Union{CuSparseArrayCSR, CuSparseMatrixCSR}, ::Colon, ::Colon, bshape::Int64...) 
    CuSparseArrayCSR(reshape(A.rowPtr, :, bshape...),
                     reshape(A.colVal, :, bshape...),
                     reshape(A.nzVal,  :, bshape...),
                     (size(A)[1:2]..., bshape...))
end

function Base.reshape(A::CuSparseArrayCSR, dims::Int64...)
    s1, s2, bshape = dims[1], dims[2], dims[3:end]
    @assert s1 == size(A, 1) && s2 == size(A, 2)
    CuSparseArrayCSR(reshape(A.rowPtr, :, bshape...),
                     reshape(A.colVal, :, bshape...),
                     reshape(A.nzVal,  :, bshape...),
                     (size(A)[1:2]..., bshape...))
end

# reshape to have a single batch dimension
function Base.reshape(A::CuSparseArrayCSR, ::Colon, ::Colon, ::Colon)
    b = prod(size(A)[3:end])
    CuSparseArrayCSR(reshape(A.rowPtr, :, b),
                     reshape(A.colVal, :, b),
                     reshape(A.nzVal,  :, b),
                     (size(A)[1:2]..., b))
end

# repeat non-matrix dimensions
function Base.repeat(A::Union{CuSparseArrayCSR, CuSparseMatrixCSR}, r1::Int64, r2::Int64, rs::Int64...)
    @assert r1 == 1 && r2 == 1 "Cannot repeat matrix dimensions of CuSparseCSR"
    CuSparseArrayCSR(repeat(A.rowPtr, 1, rs...),
                     repeat(A.colVal, 1, rs...),
                     repeat(A.nzVal,  1, rs...),
                     (size(A)[1:2]..., [size(A,i+2)*rs[i] for i=1:length(rs)]...))
end
