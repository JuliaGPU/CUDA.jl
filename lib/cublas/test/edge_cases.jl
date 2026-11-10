using LinearAlgebra

@testset "triangular operations on strided views" begin
    n = 4
    # the first n rows can be addressed by BLAS, every other row can't
    @testset "$T, rows $rows" for T in (Float32, Float64, ComplexF32), rows in (1:n, 1:2:2n)
        v(X) = view(X, rows, 1:n)

        A = rand(T, 2n, 2n)
        for i in 1:n
            A[rows[i], i] += 4
        end
        B = rand(T, 2n, n)

        @test testf((A, B) -> UpperTriangular(v(A)) \ v(B), A, B)
        @test testf((A, B) -> LowerTriangular(v(A))' \ v(B), A, B)
        @test testf((A, B) -> UnitUpperTriangular(v(A)) \ v(B)', A, B)
        @test testf((A, B) -> v(B) / UpperTriangular(v(A)), A, B)
        @test testf((A, B) -> v(B)' / transpose(LowerTriangular(v(A))), A, B)

        @test testf((A, B) -> UpperTriangular(v(A)) * v(B), A, B)
        @test testf((A, B) -> transpose(LowerTriangular(v(A))) * v(B), A, B)
        @test testf((A, B) -> v(B) * UnitUpperTriangular(v(A)), A, B)
        @test testf((A, B) -> v(B) * LowerTriangular(v(A))', A, B)

        @test testf((A, B) -> copy(ldiv!(LowerTriangular(v(A)), v(B))), A, B)
        @test testf((A, B) -> copy(lmul!(UpperTriangular(v(A)), v(B))), A, B)
        @test testf((A, B) -> copy(rdiv!(v(B), UnitLowerTriangular(v(A)))), A, B)

        @test testf(A -> triu(parent(inv(UpperTriangular(v(A))))), A)
        @test testf(A -> tril(parent(inv(LowerTriangular(v(A))))), A)

        # LinearAlgebra's BLAS calls only take views with contiguous columns
        if step(rows) == 1
            @test testf((A, B) -> Symmetric(v(A)) * v(B), A, B)
            @test testf((A, B) -> v(B)' * Hermitian(v(A), :L), A, B)
        end

        @testset "vector $vrows" for vrows in (1:n, 1:2:2n, n:-1:1)
            w(x) = view(x, vrows)
            b = rand(T, 2n)
            @test testf((A, b) -> UpperTriangular(v(A)) \ w(b), A, b)
            @test testf((A, b) -> LowerTriangular(v(A))' \ w(b), A, b)
            @test testf((A, b) -> copy(ldiv!(UpperTriangular(v(A)), w(b))), A, b)
            @test testf((A, b) -> v(A) * w(b), A, b)
            if step(rows) == 1
                @test testf((A, b) -> UnitLowerTriangular(v(A)) * w(b), A, b)
                @test testf((A, b) -> Symmetric(v(A)) * w(b), A, b)
                @test testf((A, b) -> Hermitian(v(A))' * w(b), A, b)
            end
        end
    end
end

@testset "false is a strong zero, $T" for T in (Float32, ComplexF32)
    n = 4
    nan(d...) = fill(T(NaN), d...)
    A, B, x = rand(T, n, n), rand(T, n, n), rand(T, n)

    @test testf((C, A, B) -> mul!(C, A, B, false, true), rand(T, n, n), nan(n, n), B)
    @test testf((C, A, B) -> mul!(C, A, B, true, false), nan(n, n), A, B)
    @test testf((y, A, x) -> mul!(y, A, x, false, true), rand(T, n), nan(n, n), x)
    @test testf((y, A, x) -> mul!(y, A, x, true, false), nan(n), A, x)
    @test testf((C, A, B) -> mul!(C, Symmetric(A), B, true, false), nan(n, n), A, B)
    @test testf((y, A, x) -> mul!(y, Symmetric(A), x, true, false), nan(n), A, x)
    @test testf((C, A) -> mul!(C, A', A, true, false), nan(n, n), A)
    @test testf((C, A) -> transpose!(C, A), nan(n, n), A)
    @test testf((C, x, A) -> mul!(C, Diagonal(x), A), nan(n, n), x, A)
    @test testf((C, A, B) -> mul!(C, UpperTriangular(A), B), nan(n, n), A, B)

    @testset "$(isnothing(rows) ? "dense" : "rows $rows")" for rows in (nothing, 1:2:12)
        v(x) = isnothing(rows) ? x : view(x, rows)
        m = isnothing(rows) ? 6 : 12
        x, y = rand(T, m), rand(T, m)
        @test testf(x -> rmul!(v(x), false), nan(m))
        @test testf((x, y) -> axpby!(T(2), v(x), false, v(y)), x, nan(m))
        @test testf((x, y) -> axpby!(false, v(x), T(3), v(y)), nan(m), y)
        @test testf((x, y) -> copy(first(rotate!(v(x), v(y), false, true))), nan(m), y)
    end
end

@testset "Diagonal" begin
    @testset "norm(::Diagonal, $p)" for p in (2, 1, Inf, -Inf, 0, -1)
        @test testf(d -> norm(Diagonal(d), p), rand(Float32, 4))
    end
    @test_throws SingularException inv(Diagonal(CuArray(Float32[1, 0, 2])))
end

@testset "dot of wrapped arrays, $T" for T in (Float32, ComplexF32)
    A, B = rand(T, 4, 6), rand(T, 4, 6)
    @test testf((A, B) -> dot(A', B'), A, B)
    @test testf((A, B) -> dot(transpose(A), transpose(B)), A, B)
    @test testf((A, B) -> dot(Symmetric(A[:, 1:4]), Symmetric(B[:, 1:4])), A, B)
    @test testf((A, B) -> dot(Hermitian(A[:, 1:4], :L), Hermitian(B[:, 1:4], :L)), A, B)
    @test testf((x, y) -> dot(view(x, 1:4), view(y, 1:4)), vec(A), vec(B))
end

@testset "dot of strided views, $T" for T in (Float32, ComplexF32)
    x, y = rand(T, 12), rand(T, 12)
    @test testf((x, y) -> dot(view(x, 1:2:12), view(y, 1:2:12)), x, y)
    @test testf((x, y) -> dot(view(x, 12:-2:1), view(y, 1:2:12)), x, y)
    T <: Complex && @test testf((x, y) -> transpose(view(x, 1:2:12)) * view(y, 1:2:12), x, y)
end
