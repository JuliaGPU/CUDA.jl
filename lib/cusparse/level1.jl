# sparse linear algebra functions that perform operations between dense and sparse vectors

export axpyi!, axpyi, sctr!, sctr, gthr!, gthr, gthrz!, roti!, roti

"""
    axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

Computes `alpha * X + Y` for sparse `X` and dense `Y`.
"""
axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

for (fname,elty) in ((:cusparseSaxpyi, :Float32),
                     (:cusparseDaxpyi, :Float64),
                     (:cusparseCaxpyi, :ComplexF32),
                     (:cusparseZaxpyi, :ComplexF64))
    @eval begin
        function axpyi!(alpha::Number,
                        X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            $fname(handle(), nnz(X), alpha, nonzeros(X), nonzeroinds(X), Y, index)
            Y
        end
    end
end
function axpyi(alpha::Number, X::CuSparseVector, Y::CuVector, index::SparseChar)
    axpyi!(alpha, X, copy(Y), index)
end
function axpyi(X::CuSparseVector{T}, Y::CuVector{T}, index::SparseChar) where T
    axpyi!(one(T), X, copy(Y), index)
end

"""
    gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices.
"""
gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSgthr, :Float32),
                     (:cusparseDgthr, :Float64),
                     (:cusparseCgthr, :ComplexF32),
                     (:cusparseZgthr, :ComplexF64))
    @eval begin
        function gthr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            $fname(handle(), nnz(X), Y, nonzeros(X), nonzeroinds(X), index)
            X
        end
    end
end
function gthr(X::CuSparseVector, Y::CuVector, index::SparseChar)
    gthr!(copy(X), Y, index)
end

"""
    gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices, and zeros out those elements of `Y`.
"""
gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSgthrz, :Float32),
                     (:cusparseDgthrz, :Float64),
                     (:cusparseCgthrz, :ComplexF32),
                     (:cusparseZgthrz, :ComplexF64))
    @eval begin
        function gthrz!(X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            $fname(handle(), nnz(X), Y, nonzeros(X), nonzeroinds(X), index)
            X,Y
        end
    end
end
function gthrz(X::CuSparseVector, Y::CuVector, index::SparseChar)
    gthrz!(copy(X), copy(Y), index)
end

"""
    roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar)

Performs the Givens rotation specified by `c` and `s` to sparse `X` and dense `Y`.
"""
roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar)
for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       c::Number,
                       s::Number,
                       index::SparseChar)
            $fname(handle(), nnz(X), nonzeros(X), nonzeroinds(X), Y, c, s, index)
            X,Y
        end
    end
end
function roti(X::CuSparseVector,
              Y::CuVector,
              c::Number,
              s::Number,
              index::SparseChar)
    roti!(copy(X),copy(Y),c,s,index)
end

"""
    sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Set `Y[:] = X[:]` for dense `Y` and sparse `X`.
"""
sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSsctr, :Float32),
                     (:cusparseDsctr, :Float64),
                     (:cusparseCsctr, :ComplexF32),
                     (:cusparseZsctr, :ComplexF64))
    @eval begin
        function sctr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            $fname(handle(), nnz(X), nonzeros(X), nonzeroinds(X), Y, index)
            Y
        end
    end
end
function sctr(X::CuSparseVector{T}, index::SparseChar) where T
    sctr!(X, CUDA.zeros(T, size(X)[1]),index)
end
