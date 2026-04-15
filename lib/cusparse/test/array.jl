@testset "array" begin
    x = sprand(m,0.2)
    d_x = CuSparseVector(x)
    @test length(d_x) == m
    @test size(d_x)   == (m,)
    @test size(d_x,1) == m
    @test size(d_x,2) == 1
    @test ndims(d_x)  == 1
    dense_d_x = CuVector(x)
    dense_d_x2 = CuVector(d_x)
    CUDACore.@allowscalar begin
        @test sprint(show, d_x) == replace(sprint(show, x), "SparseVector{Float64, Int64}"=>"cuSPARSE.CuSparseVector{Float64, Int32}", "sparsevec(["=>"sparsevec(Int32[")
        @test sprint(show, MIME"text/plain"(), d_x) == replace(sprint(show, MIME"text/plain"(), x), "SparseVector{Float64, Int64}"=>"CuSparseVector{Float64, Int32}", "sparsevec(["=>"sparsevec(Int32[")
        @test Array(d_x[:])        == x[:]
        @test d_x[firstindex(d_x)] == x[firstindex(x)]
        @test d_x[div(end, 2)]     == x[div(end, 2)]
        @test d_x[end]             == x[end]
        @test Array(d_x[firstindex(d_x):end]) == x[firstindex(x):end]
        @test Array(dense_d_x[firstindex(d_x):end]) == x[firstindex(x):end]
        @test Array(dense_d_x2[firstindex(d_x):end]) == x[firstindex(x):end]
    end
    @test_throws BoundsError d_x[firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1]
    @test nnz(d_x)    == nnz(x)
    @test Array(nonzeros(d_x)) == nonzeros(x)
    @test Array(nonzeroinds(d_x)) == nonzeroinds(x)
    @test Array(rowvals(d_x)) == nonzeroinds(x)
    @test nnz(d_x)    == length(nonzeros(d_x))
    d_y = copy(d_x)
    CUDACore.unsafe_free!(d_y)
    x = sprand(m,0.2)
    d_x = CuSparseMatrixCSC{Float64}(x)
    @test size(d_x) == (m, 1)
    x = sprand(m,n,0.2)
    d_x = CuSparseMatrixCSC(x)
    d_tx = CuSparseMatrixCSC(transpose(x))
    d_ax = CuSparseMatrixCSC(adjoint(x))
    @test size(d_tx) == (n,m)
    @test size(d_ax) == (n,m)
    @test CuSparseMatrixCSC(d_x) === d_x
    @test length(d_x) == m*n
    @test size(d_x)   == (m,n)
    @test size(d_x,1) == m
    @test size(d_x,2) == n
    @test size(d_x,3) == 1
    @test ndims(d_x)  == 2
    CUDACore.@allowscalar begin
        @test sprint(show, d_x) == sprint(show, SparseMatrixCSC(d_x))
        @test sprint(show, MIME"text/plain"(), d_x) == replace(sprint(show, MIME"text/plain"(), x), "SparseMatrixCSC{Float64, Int64}"=>"CuSparseMatrixCSC{Float64, Int32}")
        @test Array(d_x[:])        == x[:]
        @test d_x[:, :]            == x[:, :]
        @test d_tx[:, :]           == transpose(x)[:, :]
        @test d_ax[:, :]           == adjoint(x)[:, :]
        @test d_x[(1, 1)]          == x[1, 1]
        @test d_x[firstindex(d_x)] == x[firstindex(x)]
        @test d_x[div(end, 2)]     == x[div(end, 2)]
        @test d_x[end]             == x[end]
        @test d_x[firstindex(d_x), firstindex(d_x)] == x[firstindex(x), firstindex(x)]
        @test d_x[div(end, 2), div(end, 2)]         == x[div(end, 2), div(end, 2)]
        @test d_x[end, end]        == x[end, end]
        @test Array(d_x[firstindex(d_x):end]) == x[:]
        for i in 1:size(x, 2)
            @test Array(d_x[:, i]) == x[:, i]
        end
    end
    @test_throws BoundsError d_x[firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1]
    @test_throws BoundsError d_x[firstindex(d_x) - 1, firstindex(d_x) - 1]
    @test_throws BoundsError d_x[end + 1, end + 1]
    @test_throws BoundsError d_x[firstindex(d_x) - 1:end + 1, :]
    @test_throws BoundsError d_x[firstindex(d_x) - 1, :]
    @test_throws BoundsError d_x[end + 1, :]
    @test_throws BoundsError d_x[:, firstindex(d_x) - 1:end + 1]
    @test_throws BoundsError d_x[:, firstindex(d_x) - 1]
    @test_throws BoundsError d_x[:, end + 1]
    @test nnz(d_x)    == nnz(x)
    @test Array(nonzeros(d_x)) == nonzeros(x)
    @test nnz(d_x)    == length(nonzeros(d_x))
    @test !issymmetric(d_x)
    @test !ishermitian(d_x)
    @test_throws ArgumentError size(d_x,0)
    @test_throws ArgumentError cuSPARSE.CuSparseVector(x)
    d_y = copy(d_x)
    CUDACore.unsafe_free!(d_y)
    y = sprand(k,n,0.2)
    d_y = CuSparseMatrixCSC(y)
    @test_throws ArgumentError copyto!(d_y,d_x)
    x = sprand(m,n,0.2)
    d_x = CuSparseMatrixCOO(x)
    d_tx = CuSparseMatrixCOO(transpose(x))
    d_ax = CuSparseMatrixCOO(adjoint(x))
    d_tcx = CuSparseMatrixCOO(transpose(CuSparseMatrixCSC(x)))
    d_acx = CuSparseMatrixCOO(adjoint(CuSparseMatrixCSC(x)))
    # reordered I, J to test other indexing path
    d_rx = CuSparseMatrixCOO{eltype(d_x), Int32}(copy(d_x.colInd), copy(d_x.rowInd), copy(d_x.nzVal))
    @test CuSparseMatrixCOO(d_x) === d_x
    @test length(d_x) == m*n
    @test size(d_x)   == (m,n)
    @test size(d_rx)  == (n,m)
    @test size(d_x,1) == m
    @test size(d_x,2) == n
    @test size(d_x,3) == 1
    @test ndims(d_x)  == 2
    d_x2 = copy(d_x)
    @test d_x2 isa CuSparseMatrixCOO
    @test size(d_x2) == size(d_x)
    @test length(d_x) == length(x)
    CUDACore.@allowscalar begin
        @test Array(d_x[:])        == x[:]
        @test d_x[firstindex(d_x)] == x[firstindex(x)]
        @test d_x[div(end, 2)]     == x[div(end, 2)]
        @test d_x[end]             == x[end]
        @test d_tx[:, 1]           == transpose(x)[:, 1]
        @test d_ax[1, :]           == adjoint(x)[1, :]
        @test d_tcx[:, 1]          == transpose(x)[:, 1]
        @test d_acx[1, :]          == adjoint(x)[1, :]
        @test d_rx[:, 1]           == transpose(x)[:, 1]
        @test d_rx[1, :]           == transpose(x)[1, :]
        @test d_x[firstindex(d_x), firstindex(d_x)] == x[firstindex(x), firstindex(x)]
        @test d_x[div(end, 2), div(end, 2)]         == x[div(end, 2), div(end, 2)]
        @test d_x[end, end]        == x[end, end]
        @test Array(d_x[firstindex(d_x):end]) == x[:]
        for i in 1:size(x, 2)
            @test Array(d_x[:, i]) == x[:, i]
        end
        for i in 1:size(x, 1)
            @test Array(d_x[i, :]) == x[i, :]
        end
    end
    # scalar getindex at every (i, j) — regression test for #3100,
    # where COO column search crossed into the next row's entries.
    # Construct the COO directly from row-sorted arrays to avoid depending
    # on whether a given CUDA version's CSC→CSR→COO path sorts by row.
    let dense = sparse(reshape(1:16, 4, 4)),
        d_dense = CuSparseMatrixCOO(
            CuArray(Int32[1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]),
            CuArray(Int32[1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4]),
            CuArray([1,5,9,13, 2,6,10,14, 3,7,11,15, 4,8,12,16]),
            (4, 4))
        CUDACore.@allowscalar begin
            for j in axes(dense, 2), i in axes(dense, 1)
                @test d_dense[i, j] == dense[i, j]
            end
        end
    end
    # sparse case with empty rows and missing entries
    let s = sparse([1, 1, 3, 4], [1, 3, 2, 4], [10, 20, 30, 40], 4, 4),
        d_s = CuSparseMatrixCOO(
            CuArray(Int32[1, 1, 3, 4]),
            CuArray(Int32[1, 3, 2, 4]),
            CuArray([10, 20, 30, 40]),
            (4, 4))
        CUDACore.@allowscalar begin
            for j in axes(s, 2), i in axes(s, 1)
                @test d_s[i, j] == s[i, j]
            end
        end
    end
    y = sprand(k,n,0.2)
    d_y = CuSparseMatrixCOO(y)
    @test_throws ArgumentError copyto!(d_y,d_x)
    d_y = CuSparseMatrixCSR(d_y)
    d_x = CuSparseMatrixCSR(d_x)
    d_z = copy(d_x)
    CUDACore.unsafe_free!(d_z)
    @test CuSparseMatrixCSR(d_x) === d_x
    @test reshape(d_x, :, :, 1, 1, 1) isa CuSparseArrayCSR
    @test_throws ArgumentError("Cannot repeat matrix dimensions of CuSparseCSR") repeat(d_x, 2, 1, 3)
    @test repeat(d_x, 1, 1, 3) isa CuSparseArrayCSR
    @test reshape(repeat(d_x, 1, 1, 3), size(d_x, 1), size(d_x, 2), 3, 1, 1) isa CuSparseArrayCSR
    # to hit the CuSparseArrayCSR path
    CUDACore.unsafe_free!(repeat(d_x, 1, 1, 3))
    CUDACore.@allowscalar begin
        @test startswith(sprint(show, MIME"text/plain"(), repeat(d_x, 1, 1, 2)), "$m×$n×2 CuSparseArrayCSR{Float64, Int32, 3} with $(2*nnz(d_x)) stored entries:\n")
    end
    @test length(d_x) == m*n
    @test_throws ArgumentError copyto!(d_y,d_x)
    CUDACore.@allowscalar begin
        for i in 1:size(y, 1)
          @test d_y[i, :] ≈ y[i, :]
        end
        for j in 1:size(y, 2)
          @test d_y[:, j] ≈ y[:, j]
        end
        @test d_y[1, 1] ≈ y[1, 1]
    end
    d_y = CuSparseMatrixBSR(d_y, blockdim)
    d_x = CuSparseMatrixBSR(d_x, blockdim)
    @test CuSparseMatrixBSR(d_x) === d_x
    d_z = copy(d_x)
    CUDACore.unsafe_free!(d_z)
    @test_throws ArgumentError copyto!(d_y,d_x)
    d_y_mat = CuMatrix{eltype(d_y)}(d_y)
    CUDACore.@allowscalar begin
        @test d_y[1, 1] ≈ y[1, 1]
        @test d_y_mat[1, 1] ≈ y[1, 1]
    end
    x = sprand(m,0.2)
    d_x = CuSparseVector(x)
    @test size(d_x, 1) == m
    @test size(d_x, 2) == 1
    @test_throws ArgumentError size(d_x, 0)
    y = sprand(n,0.2)
    d_y = CuSparseVector(y)
    @test_throws ArgumentError copyto!(d_y,d_x)
    x = sprand(m,m,0.2)
    d_x = Symmetric(CuSparseMatrixCSC(x + transpose(x)))
    @test issymmetric(d_x)
    x = sprand(ComplexF64, m, m, 0.2)
    d_x = Hermitian(CuSparseMatrixCSC(x + x'))
    @test ishermitian(d_x)
    d_x = Hermitian{ComplexF64}(CuSparseMatrixCSC(x + x'))
    @test ishermitian(d_x)
    x = sprand(m,m,0.2)
    d_x = UpperTriangular(CuSparseMatrixCSC(x))
    @test istriu(d_x)
    @test !istril(d_x)
    d_x = LowerTriangular(CuSparseMatrixCSC(x))
    @test !istriu(d_x)
    @test istril(d_x)
    d_x = UpperTriangular(transpose(CuSparseMatrixCSC(x)))
    @test istriu(d_x)
    @test !istril(d_x)
    d_x = LowerTriangular(transpose(CuSparseMatrixCSC(x)))
    @test !istriu(d_x)
    @test istril(d_x)
    d_x = UpperTriangular(triu(transpose(CuSparseMatrixCSC(x)), 1))
    @test istriu(d_x)
    @test !istril(d_x)
    d_x = UpperTriangular(triu(adjoint(CuSparseMatrixCSC(x)), 1))
    @test istriu(d_x)
    @test !istril(d_x)
    d_x = LowerTriangular(tril(transpose(CuSparseMatrixCSC(x)), -1))
    @test !istriu(d_x)
    @test istril(d_x)
    d_x = LowerTriangular(tril(adjoint(CuSparseMatrixCSC(x)), -1))
    @test !istriu(d_x)
    @test istril(d_x)

    A = sprand(n, n, 0.2)
    d_A = CuSparseMatrixCSC(A)
    @test Array(getcolptr(d_A)) == getcolptr(A)
    i, j, v = findnz(A)
    d_i, d_j, d_v = findnz(d_A)
    @test Array(d_i) == i && Array(d_j) == j && Array(d_v) == v
    i = unique(sort(rand(1:n, 10)))
    vals = rand(length(i))
    d_i = CuArray(i)
    d_vals = CuArray(vals)
    v = sparsevec(i, vals, n)
    d_v = sparsevec(d_i, d_vals, n)
    @test Array(d_v.iPtr) == v.nzind
    @test Array(d_v.nzVal) == v.nzval
    @test d_v.len == v.n

    # `cu` should propagate the requested memory type to inner CuArrays (#2974),
    # and apply the same opinionated eltype conversion as the dense path
    let A = sprand(ComplexF64, m, n, 0.2)
        d_A = cu(A; unified=true)
        @test d_A isa CuSparseMatrixCSC{ComplexF32}
        @test d_A.colPtr isa CuArray{Int32, 1, CUDACore.UnifiedMemory}
        @test d_A.rowVal isa CuArray{Int32, 1, CUDACore.UnifiedMemory}
        @test d_A.nzVal  isa CuArray{ComplexF32, 1, CUDACore.UnifiedMemory}

        d_A = cu(A)
        @test d_A isa CuSparseMatrixCSC{ComplexF32}
        @test d_A.colPtr isa CuArray{Int32, 1, CUDACore.DeviceMemory}
        @test d_A.rowVal isa CuArray{Int32, 1, CUDACore.DeviceMemory}
        @test d_A.nzVal  isa CuArray{ComplexF32, 1, CUDACore.DeviceMemory}
    end
    let B = sprand(Float64, m, n, 0.2)
        d_B = cu(B; unified=true)
        @test d_B isa CuSparseMatrixCSC{Float32}
        @test d_B.nzVal isa CuArray{Float32, 1, CUDACore.UnifiedMemory}
    end
    let v = sprand(Float64, m, 0.2)
        d_v = cu(v; unified=true)
        @test d_v isa CuSparseVector{Float32}
        @test d_v.iPtr  isa CuArray{Int32, 1, CUDACore.UnifiedMemory}
        @test d_v.nzVal isa CuArray{Float32, 1, CUDACore.UnifiedMemory}
    end
    # Float16 should be preserved (matching dense `cu` semantics)
    let A = SparseMatrixCSC{Float16}(sprand(Float32, m, n, 0.2))
        d_A = cu(A; unified=true)
        @test d_A isa CuSparseMatrixCSC{Float16}
        @test d_A.nzVal isa CuArray{Float16, 1, CUDACore.UnifiedMemory}
    end
    # the plain constructor must keep its non-opinionated semantics
    let A = sprand(Float64, m, n, 0.2)
        @test CuSparseMatrixCSC(A) isa CuSparseMatrixCSC{Float64}
    end
end
