using cuBLAS
using LinearAlgebra

# cuBLAS addresses matrices through a leading dimension, so it can't handle views that skip
# rows. every wrapper that takes a matrix has to reject those rather than read the wrong
# elements. the GEMM family, symm, hemm and trmm are covered in their own test files.

# a copy of X as a view that skips every other row
function skip_rows(X)
    v = view(similar(X, 2size(X, 1), size(X, 2)), 1:2:2size(X, 1), :)
    copyto!(v, X)
end

# call f with each of its matrix operands in turn replaced by such a view; for a batch,
# only its last matrix is replaced
function test_rejects(f, args...)
    for i in eachindex(args)
        arg = args[i]
        bad = if arg isa AbstractMatrix{<:Union{AbstractFloat, Complex}}
            skip_rows(arg)
        elseif arg isa Vector{<:AbstractMatrix}
            StridedCuMatrix{eltype(arg[end])}[arg[1:end-1]..., skip_rows(arg[end])]
        else
            continue
        end
        @test_throws ArgumentError f(Base.setindex(args, bad, i)...)
    end
end

m = 4

@testset "level 2" begin
    T = Float32
    A = CuArray(rand(T, m, m))
    Ab = CuArray(rand(T, 3, m))  # band storage
    x = CuArray(rand(T, m))
    y = CuArray(rand(T, m))
    test_rejects(cuBLAS.gemv!, 'N', one(T), A, x, zero(T), y)
    test_rejects(cuBLAS.gbmv!, 'N', m, 1, 1, one(T), Ab, x, zero(T), y)
    test_rejects(cuBLAS.symv!, 'U', one(T), A, x, zero(T), y)
    test_rejects(cuBLAS.sbmv!, 'U', 1, one(T), Ab, x, zero(T), y)
    test_rejects(cuBLAS.tbmv!, 'U', 'N', 'N', 1, Ab, x)
    test_rejects(cuBLAS.tbsv!, 'U', 'N', 'N', 1, Ab, x)
    test_rejects(cuBLAS.trmv!, 'U', 'N', 'N', A, x)
    test_rejects(cuBLAS.trsv!, 'U', 'N', 'N', A, x)
    test_rejects(cuBLAS.ger!, one(T), x, y, A)
    test_rejects(cuBLAS.syr!, 'U', one(T), x, A)

    T = ComplexF32
    A = CuArray(rand(T, m, m))
    Ab = CuArray(rand(T, 3, m))
    x = CuArray(rand(T, m))
    y = CuArray(rand(T, m))
    test_rejects(cuBLAS.hemv!, 'U', one(T), A, x, zero(T), y)
    test_rejects(cuBLAS.hbmv!, 'U', 1, one(T), Ab, x, zero(T), y)
    test_rejects(cuBLAS.her!, 'U', one(real(T)), x, A)
    test_rejects(cuBLAS.her2!, 'U', one(T), x, y, A)

    As = [CuArray(rand(T, m, m)) for _ in 1:2]
    xs = [CuArray(rand(T, m)) for _ in 1:2]
    ys = [CuArray(rand(T, m)) for _ in 1:2]
    test_rejects(cuBLAS.gemv_batched!, 'N', one(T), As, xs, zero(T), ys)
end

@testset "level 3" begin
    T = Float32
    A = CuArray(rand(T, m, m))
    B = CuArray(rand(T, m, m))
    C = CuArray(rand(T, m, m))
    test_rejects(cuBLAS.syrk!, 'U', 'N', one(T), A, zero(T), C)
    test_rejects(cuBLAS.syrkx!, 'U', 'N', one(T), A, B, zero(T), C)
    test_rejects(cuBLAS.syr2k!, 'U', 'N', one(T), A, B, zero(T), C)
    test_rejects(cuBLAS.trsm!, 'L', 'U', 'N', 'N', one(T), A, B)
    test_rejects(cuBLAS.trsm_batched!, 'L', 'U', 'N', 'N', one(T), [A, A], [B, B])

    T = ComplexF32
    A = CuArray(rand(T, m, m))
    B = CuArray(rand(T, m, m))
    C = CuArray(rand(T, m, m))
    test_rejects(cuBLAS.herk!, 'U', 'N', one(real(T)), A, zero(real(T)), C)
    test_rejects(cuBLAS.her2k!, 'U', 'N', one(T), A, B, zero(real(T)), C)
end

@testset "extensions" begin
    T = Float32
    A = CuArray(rand(T, m, m) + m*I)
    B = CuArray(rand(T, m, m))
    C = CuArray(rand(T, m, m))
    x = CuArray(rand(T, m))
    test_rejects(cuBLAS.geam!, 'N', 'N', one(T), A, one(T), B, C)
    test_rejects(cuBLAS.dgmm!, 'L', A, x, C)

    As = [copy(A), copy(A)]
    Bs = [copy(B), copy(B)]
    Cs = [copy(C), copy(C)]
    pivot = CuArray(ones(Cint, m, 2))
    test_rejects(cuBLAS.getrf_batched!, As, true)
    test_rejects(cuBLAS.getrs_batched!, 'N', As, Bs, pivot)
    test_rejects(cuBLAS.getri_batched!, As, Cs, pivot)
    test_rejects(cuBLAS.matinv_batched, As)
    test_rejects(cuBLAS.geqrf_batched!, As)

    # all matrices in a batch share the leading dimension of the first one
    padded = view(CuArray(rand(T, m + 1, m)), 1:m, :)
    @test_throws DimensionMismatch cuBLAS.getrf_batched!(StridedCuMatrix{T}[A, padded], true)
end

@testset "cublasXt" begin
    T = ComplexF32
    for X in (CuArray, Array)
        A = X(rand(T, m, m))
        B = X(rand(T, m, m))
        C = X(rand(T, m, m))
        test_rejects(cuBLAS.xt_syrk!, 'U', 'N', one(T), A, zero(T), C)
        test_rejects(cuBLAS.xt_syrkx!, 'U', 'N', one(T), A, B, zero(T), C)
        test_rejects(cuBLAS.xt_herk!, 'U', 'N', one(real(T)), A, zero(real(T)), C)
        test_rejects(cuBLAS.xt_her2k!, 'U', 'N', one(T), A, B, zero(real(T)), C)
        test_rejects(cuBLAS.xt_trsm!, 'L', 'U', 'N', 'N', one(T), A, B)
    end
end

@testset "one-row matrices" begin
    x = CuArray(rand(Float32, 3))
    y = CuArray(zeros(Float32, 1))
    # a single row has no row stride to honor
    A = view(CuArray(rand(Float32, 2, 3)), 1:2:1, :)
    @test Array(cuBLAS.gemv!('N', 1f0, A, x, 0f0, y)) ≈ Array(A) * Array(x)
    # but reversed columns still have no leading dimension
    R = view(CuArray(rand(Float32, 1, 3)), :, 3:-1:1)
    @test_throws ArgumentError cuBLAS.gemv!('N', 1f0, R, x, 0f0, y)
end
