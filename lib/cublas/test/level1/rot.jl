using cuBLAS
using LinearAlgebra

@testset for T in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20

    @testset "rotate!" begin
        @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
        @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
    end

    @testset "reflect!" begin
        @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
        @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
    end

    @testset "rot! with device-side scalars" begin
        x, y = rand(T, m), rand(T, m)
        c = rand(real(T))
        @testset for S in unique((real(T), T))
            s = rand(S)
            ex, ey = rotate!(copy(x), copy(y), c, s)
            # a view selects the element it starts at, not the first of its parent
            offset_view(v) = view(CuArray([zero(v), v, zero(v)]), 2:3)
            wraps = (identity, CuRef, v -> CuArray([v]), offset_view)
            for cwrap in wraps, swrap in wraps
                dx, dy = CuArray(x), CuArray(y)
                cuBLAS.rot!(m, dx, dy, cwrap(c), swrap(s))
                @test Array(dx) ≈ ex
                @test Array(dy) ≈ ey
            end
        end
    end

    @testset "negative strides" begin # JuliaGPU/CUDA.jl#3410
        rev(a) = view(a, lastindex(a):-2:1)
        fwd(a) = view(a, 1:2:lastindex(a))
        x, y = rand(T, m), rand(T, m)
        c, s = rand(real(T)), rand(T)
        for f! in (rotate!, reflect!)
            a, b, da, db = copy(x), copy(y), CuArray(x), CuArray(y)
            f!(rev(a), fwd(b), c, s)
            f!(rev(da), fwd(db), c, s)
            @test Array(da) ≈ a
            @test Array(db) ≈ b
        end
    end

    @testset "Givens" begin # JuliaGPU/CUDA.jl#115
        n = 7
        A = rand(T, m, n)
        @testset for GT in unique((real(T), T))
            G = LinearAlgebra.Givens{GT}(2, 5, rand(real(T)), rand(GT))
            @test testf(A -> lmul!(G, A), A)
            @test testf(A -> lmul!(G', A), A)
            @test testf(A -> rmul!(A, G), copy(A'))
            @test testf(A -> G * A, A)
            @test testf(A -> A * G', copy(A'))
            @test testf(x -> lmul!(G, x), rand(T, m))
            @test testf(A -> (G * G') * A, A)
            @test testf(A -> lmul!(G, A), zeros(T, m, 0))
        end

        # complex Givens rotations store a complex cosine with zero imaginary part
        G, _ = givens(A[1, 1], A[3, 1], 1, 3)
        @test testf(A -> lmul!(G, A), A)

        # only the rotated rows or columns need to be strided
        G = LinearAlgebra.Givens(1, 3, rand(real(T)), rand(T))
        dA = CuArray(rand(T, 2m, 2n))
        rows, cols = view(dA, 2:2:2m, 1:2:2n), view(dA, 1:2:2m, 2:2:2n)
        A, B = Array(rows), Array(cols)
        @test lmul!(G, rows) === rows
        @test Array(rows) ≈ lmul!(G, A)
        @test Array(rmul!(cols, G)) ≈ rmul!(B, G)

        @test_throws DimensionMismatch lmul!(LinearAlgebra.Givens(1, m+1, one(T), zero(T)),
                                             CuArray(A))
        @test_throws DimensionMismatch rmul!(CuArray(A), LinearAlgebra.Givens(1, n+1, one(T), zero(T)))
    end

    @testset "rotg!" begin
        a = rand(T)
        b = rand(T)
        a_copy = copy(a)
        b_copy = copy(b)
        a, b, c, s = cuBLAS.rotg!(a, b)
        rot = [c s; -conj(s) c] * [a_copy; b_copy]
        @test rot ≈ [a; 0]
        if T <: Real
            @test a^2 ≈ a_copy^2 + b_copy^2
        end
        @test c^2 + abs2(s) ≈ one(T)
    end

    @testset "rotg! with device-side scalars" begin
        a, b = rand(T), rand(T)
        ha, hb, hc, hs = cuBLAS.rotg!(a, b)
        for wrap in (CuRef, v -> CuArray([v]))
            da, db = wrap(a), wrap(b)
            ra, rb, c, s = cuBLAS.rotg!(da, db)
            @test ra === da && rb === db
            @test c isa CUDACore.AbstractCuRef{real(T)}
            @test s isa CUDACore.AbstractCuRef{T}
            @test CUDACore.@allowscalar(da[]) ≈ ha
            @test CUDACore.@allowscalar(db[]) ≈ hb
            @test c[] ≈ hc
            @test s[] ≈ hs

            dc, ds = wrap(zero(real(T))), wrap(zero(T))
            _, _, rc, rs = cuBLAS.rotg!(wrap(a), wrap(b), dc, ds)
            @test rc === dc && rs === ds
            @test CUDACore.@allowscalar(dc[]) ≈ hc
            @test CUDACore.@allowscalar(ds[]) ≈ hs
            x, y = rand(T, m), rand(T, m)
            dx, dy = CuArray(x), CuArray(y)
            cuBLAS.rot!(m, dx, dy, dc, ds)
            ex, ey = rotate!(copy(x), copy(y), hc, hs)
            @test Array(dx) ≈ ex
            @test Array(dy) ≈ ey
        end
    end

    if T <: Real
        H = rand(T, 2, 2)
        @testset "rotm! flag=$flag" for (flag, flag_H) in ((T(-2), [one(T) zero(T); zero(T) one(T)]),
                                                           (-one(T), H),
                                                           (zero(T), [one(T) H[1,2]; H[2, 1] one(T)]),
                                                           (one(T), [H[1,1] one(T); -one(T) H[2, 2]]))
            rot_n = 2
            x = rand(T, rot_n)
            y = rand(T, rot_n)
            dx = CuArray(x)
            dy = CuArray(y)
            dx, dy = cuBLAS.rotm!(rot_n, dx, dy, CuArray(vcat(flag, H...)))
            h_x = collect(dx)
            h_y = collect(dy)
            @test h_x ≈ [x[1] * flag_H[1,1] + y[1] * flag_H[1,2]; x[2] * flag_H[1, 1] + y[2] * flag_H[1, 2]]
            @test h_y ≈ [x[1] * flag_H[2,1] + y[1] * flag_H[2,2]; x[2] * flag_H[2, 1] + y[2] * flag_H[2, 2]]
        end

        @testset "rotmg!" begin
            gpu_param = CuArray{T}(undef, 5)
            x1 = rand(T)
            y1 = rand(T)
            d1 = zero(T)
            d2 = zero(T)
            x1_copy = copy(x1)
            y1_copy = copy(y1)
            d1, d2, x1, y1 = cuBLAS.rotmg!(d1, d2, x1, y1, gpu_param)
            cpu_param = Array(gpu_param)
            flag = cpu_param[1]
            H = zeros(T, 2, 2)
            if flag == -2
                H[1, 1] = one(T)
                H[1, 2] = zero(T)
                H[2, 1] = zero(T)
                H[2, 2] = one(T)
            elseif flag == -1
                H[1, 1] = cpu_param[2]
                H[1, 2] = cpu_param[3]
                H[2, 1] = cpu_param[4]
                H[2, 2] = cpu_param[5]
            elseif iszero(flag)
                H[1, 1] = one(T)
                H[1, 2] = cpu_param[3]
                H[2, 1] = cpu_param[4]
                H[2, 2] = one(T)
            elseif flag == 1
                H[1, 1] = cpu_param[2]
                H[1, 2] = one(T)
                H[2, 1] = -one(T)
                H[2, 2] = cpu_param[5]
            end
            out = H * [(√d1) * x1_copy; (√d2) * y1_copy]
            @test out[2] ≈ zero(T)
        end

        @testset "rotmg! with device-side scalars" begin
            d1, d2, x1, y1 = rand(T), rand(T), rand(T), rand(T)
            host_param = CuArray(zeros(T, 5))
            hd1, hd2, hx1, _ = cuBLAS.rotmg!(d1, d2, x1, y1, host_param)
            for wrap in (CuRef, v -> CuArray([v])), y in (y1, wrap(y1))
                param = CuArray(zeros(T, 5))
                dd1, dd2, dx1 = wrap(d1), wrap(d2), wrap(x1)
                r = cuBLAS.rotmg!(dd1, dd2, dx1, y, param)
                @test r[1] === dd1 && r[2] === dd2 && r[3] === dx1 && r[5] === param
                @test CUDACore.@allowscalar(dd1[]) ≈ hd1
                @test CUDACore.@allowscalar(dd2[]) ≈ hd2
                @test CUDACore.@allowscalar(dx1[]) ≈ hx1
                @test Array(param) ≈ Array(host_param)
            end
            @test_throws DimensionMismatch cuBLAS.rotmg!(d1, d2, x1, y1, CuArray{T}(undef, 4))
            @test_throws DimensionMismatch cuBLAS.rotm!(2, CuArray(rand(T, 2)), CuArray(rand(T, 2)),
                                                        CuArray{T}(undef, 4))
        end
    end
end
