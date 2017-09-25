using CuArrays.BLAS: band, bandex

@testset "BLAS" begin

m = 20
n = 35
k = 13

#################
# level 1 tests #
#################

@testset "Level 1" begin

@testset for T in [Float32, Float64, Complex64, Complex128]
  A = CuArray(rand(T, m))
  B = CuArray{T}(m)
  CuArrays.BLAS.blascopy!(m,A,1,B,1)
  @test collect(A) == collect(B)

  A = rand(T, m)
  @test CuArrays.BLAS.asum(cu(A)) ≈ BLAS.asum(A)

  testf(scale!, rand(T, 6, 9, 3), rand())
  testf(dot, rand(T, m), rand(T, m))
  testf(norm, rand(T, m))
end

@testset "axpy!" begin
    function test_axpy!_1(alpha,A,B)
        n1 = length(A)
        d_A = CuArray(A)
        d_B1 = CuArray(B)
        CuArrays.BLAS.axpy!(n1,alpha,d_A,1,d_B1,1)
        B1 = collect(d_B1)
        host_axpy = alpha*A + B
        @test host_axpy ≈ B1
    end
    test_axpy!_1(2.0f0,rand(Float32,m),rand(Float32,m))
    test_axpy!_1(2.0,rand(Float64,m),rand(Float64,m))
    test_axpy!_1(2.0f0+im*2.0f0,rand(Complex64,m),rand(Complex64,m))
    test_axpy!_1(2.0+im*2.0,rand(Complex128,m),rand(Complex128,m))

    function test_axpy!_2(alpha,A,B)
        n1 = length(A)
        d_A = CuArray(A)
        d_B1 = CuArray(B)
        CuArrays.BLAS.axpy!(alpha,d_A,d_B1)
        B1 = collect(d_B1)
        host_axpy = alpha*A + B
        @test host_axpy ≈ B1
    end
    test_axpy!_2(2.0f0,rand(Float32,m),rand(Float32,m))
    test_axpy!_2(2.0,rand(Float64,m),rand(Float64,m))
    test_axpy!_2(2.0f0+im*2.0f0,rand(Complex64,m),rand(Complex64,m))
    test_axpy!_2(2.0+im*2.0,rand(Complex128,m),rand(Complex128,m))
end

@testset "iamax and iamin" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        A = rand(elty, m)
        n1 = length(A)
        d_A = CuArray(A)
        Aabs = abs.(A)
        imin1 = CuArrays.BLAS.iamin(n1,d_A,1)
        imax1 = CuArrays.BLAS.iamax(n1,d_A,1)
        imin2 = CuArrays.BLAS.iamin(d_A)
        imax2 = CuArrays.BLAS.iamax(d_A)
        host_imin = indmin(Aabs)
        host_imax = indmax(Aabs)
        @test imin1 == imin2 == host_imin
        @test imin1 == imin2 == host_imin
    end
end

end # level 1 testset

@testset "Level 2" begin

@testset "gemv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        alpha = convert(elty,1)
        beta = convert(elty,1)
        A = rand(elty,m,n)
        d_A = CuArray(A)

        # test y = A*x + y
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        y = A*x + y
        CuArrays.BLAS.gemv!('N',alpha,d_A,d_x,beta,d_y)
        h_y = collect(d_y)
        @test y ≈ h_y
        A_mul_B!(d_y,d_A,d_x)
        h_y = collect(d_y)
        @test h_y ≈ A*x

        # test x = A.'*y + x
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        x = A.'*y + x
        CuArrays.BLAS.gemv!('T',alpha,d_A,d_y,beta,d_x)
        h_x = collect(d_x)
        @test x ≈ h_x
        At_mul_B!(d_x,d_A,d_y)
        h_x = collect(d_x)
        @test h_x ≈ A.'*y

        # test x = A'*y + x
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        x = A'*y + x
        CuArrays.BLAS.gemv!('C',alpha,d_A,d_y,beta,d_x)
        h_x = collect(d_x)
        @test x ≈ h_x
        Ac_mul_B!(d_x,d_A,d_y)
        h_x = collect(d_x)
        @test h_x ≈ A'*y
    end
end

@testset "gemv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        alpha = convert(elty,2)
        A = rand(elty,m,n)
        d_A = CuArray(A)
        # test y = alpha*(A*x)
        x = rand(elty,n)
        d_x = CuArray(x)
        y1 = alpha*(A*x)
        y2 = A*x
        d_y1 = CuArrays.BLAS.gemv('N',alpha,d_A,d_x)
        d_y2 = CuArrays.BLAS.gemv('N',d_A,d_x)
        h_y1 = collect(d_y1)
        h_y2 = collect(d_y2)
        @test y1 ≈ h_y1
        @test y2 ≈ h_y2
        @test y2 ≈ collect(d_A * d_x)

        # test x = alpha*(A.'*y)
        y = rand(elty,m)
        d_y = CuArray(y)
        x1 = alpha*(A.'*y)
        x2 = A.'*y
        d_x1 = CuArrays.BLAS.gemv('T',alpha,d_A,d_y)
        d_x2 = CuArrays.BLAS.gemv('T',d_A,d_y)
        h_x1 = collect(d_x1)
        h_x2 = collect(d_x2)
        @test x1 ≈ h_x1
        @test x2 ≈ h_x2
        @test x2 ≈ collect(d_A.' * d_y)

        # test x = alpha*(A'*y)
        y = rand(elty,m)
        d_y = CuArray(y)
        x1 = alpha*(A'*y)
        x2 = A'*y
        d_x1 = CuArrays.BLAS.gemv('C',alpha,d_A,d_y)
        d_x2 = CuArrays.BLAS.gemv('C',d_A,d_y)
        h_x1 = collect(d_x1)
        h_x2 = collect(d_x2)
        @test y1 ≈ h_y1
        @test y2 ≈ h_y2
        @test x2 ≈ collect(d_A' * d_y)
    end
end

@testset "gbmv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = convert(elty,2)
        beta = convert(elty,3)
        # bands
        ku = 2
        kl = 3
        # generate banded matrix
        A = rand(elty,m,n)
        A = bandex(A,kl,ku)
        # get packed format
        Ab = band(A,kl,ku)
        d_Ab = CuArray(Ab)
        # test y = alpha*A*x + beta*y
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        CuArrays.BLAS.gbmv!('N',m,kl,ku,alpha,d_Ab,d_x,beta,d_y)
        BLAS.gbmv!('N',m,kl,ku,alpha,Ab,x,beta,y)
        h_y = collect(d_y)
        @test y ≈ h_y
        # test y = alpha*A.'*x + beta*y
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        CuArrays.BLAS.gbmv!('T',m,kl,ku,alpha,d_Ab,d_y,beta,d_x)
        BLAS.gbmv!('T',m,kl,ku,alpha,Ab,y,beta,x)
        h_x = collect(d_x)
        @test x ≈ h_x
        # test y = alpha*A'*x + beta*y
        x = rand(elty,n)
        d_x = CuArray(x)
        y = rand(elty,m)
        d_y = CuArray(y)
        CuArrays.BLAS.gbmv!('C',m,kl,ku,alpha,d_Ab,d_y,beta,d_x)
        BLAS.gbmv!('C',m,kl,ku,alpha,Ab,y,beta,x)
        h_x = collect(d_x)
        @test x ≈ h_x
    end
end

@testset "gbmv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = convert(elty,2)
        # bands
        ku = 2
        kl = 3
        # generate banded matrix
        A = rand(elty,m,n)
        A = bandex(A,kl,ku)
        # get packed format
        Ab = band(A,kl,ku)
        d_Ab = CuArray(Ab)
        # test y = alpha*A*x
        x = rand(elty,n)
        d_x = CuArray(x)
        d_y = CuArrays.BLAS.gbmv('N',m,kl,ku,alpha,d_Ab,d_x)
        y = zeros(elty,m)
        y = BLAS.gbmv('N',m,kl,ku,alpha,Ab,x)
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "symv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = convert(elty,2)
        beta = convert(elty,3)
        # generate symmetric matrix
        A = rand(elty,m,m)
        A = A + A.'
        # generate vectors
        x = rand(elty,m)
        y = rand(elty,m)
        # copy to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # execute on host
        BLAS.symv!('U',alpha,A,x,beta,y)
        # execute on device
        CuArrays.BLAS.symv!('U',alpha,d_A,d_x,beta,d_y)
        # compare results
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "symv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate symmetric matrix
        A = rand(elty,m,m)
        A = A + A.'
        # generate vectors
        x = rand(elty,m)
        # copy to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute on host
        y = BLAS.symv('U',A,x)
        # execute on device
        d_y = CuArrays.BLAS.symv('U',d_A,d_x)
        # compare results
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "hemv!" begin
    @testset for elty in [Complex64, Complex128]
        # parameters
        alpha = convert(elty,2)
        beta = convert(elty,3)
        # generate hermitian matrix
        A = rand(elty,m,m)
        A = A + A'
        # generate vectors
        x = rand(elty,m)
        y = rand(elty,m)
        # copy to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # execute on host
        BLAS.hemv!('U',alpha,A,x,beta,y)
        # execute on device
        CuArrays.BLAS.hemv!('U',alpha,d_A,d_x,beta,d_y)
        # compare results
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "hemv" begin
    @testset for elty in [Complex64, Complex128]
        # generate hermitian matrix
        A = rand(elty,m,m)
        A = A + A.'
        # generate vectors
        x = rand(elty,m)
        # copy to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute on host
        y = BLAS.hemv('U',A,x)
        # execute on device
        d_y = CuArrays.BLAS.hemv('U',d_A,d_x)
        # compare results
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "sbmv!" begin
    @testset for elty in [Float32, Float64]
        # parameters
        alpha = convert(elty,3)
        beta = convert(elty,2.5)
        # generate symmetric matrix
        A = rand(elty,m,m)
        A = A + A'
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,nbands,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x and y
        x = rand(elty,m)
        y = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # sbmv!
        CuArrays.BLAS.sbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
        y = alpha*(A*x) + beta*y
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "sbmv" begin
    @testset for elty in [Float32, Float64]
        # parameters
        alpha = convert(elty,3)
        beta = convert(elty,2.5)
        # generate symmetric matrix
        A = rand(elty,m,m)
        A = A + A'
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,nbands,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x and y
        x = rand(elty,m)
        y = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        # sbmv!
        d_y = CuArrays.BLAS.sbmv('U',nbands,d_AB,d_x)
        y = A*x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "hbmv!" begin
    @testset for elty in [Complex64, Complex128]
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate Hermitian matrix
        A = rand(elty,m,m)
        A = A + ctranspose(A)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,nbands,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x and y
        x = rand(elty,m)
        y = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # hbmv!
        CuArrays.BLAS.hbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
        y = alpha*(A*x) + beta*y
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "hbmv" begin
    @testset for elty in [Complex64, Complex128]
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate Hermitian matrix
        A = rand(elty,m,m)
        A = A + ctranspose(A)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,nbands,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x and y
        x = rand(elty,m)
        y = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        # hbmv
        d_y = CuArrays.BLAS.hbmv('U',nbands,d_AB,d_x)
        y = A*x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "tbmv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,0,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x and y
        x = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        # tbmv!
        CuArrays.BLAS.tbmv!('U','N','N',nbands,d_AB,d_x)
        x = A*x
        # compare
        h_x = collect(d_x)
        @test x ≈ h_x
    end
end

@testset "tbmv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,0,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x
        x = rand(elty,m)
        # move to host
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        # tbmv!
        d_y = CuArrays.BLAS.tbmv!('U','N','N',nbands,d_AB,d_x)
        y = A*x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "tbsv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,0,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        #tbsv!
        CuArrays.BLAS.tbsv!('U','N','N',nbands,d_AB,d_x)
        x = A\x
        # compare
        h_x = collect(d_x)
        @test x ≈ h_x
    end
end

@testset "tbsv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        # restrict to 3 bands
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,0,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_AB = CuArray(AB)
        d_x = CuArray(x)
        #tbsv
        d_y = CuArrays.BLAS.tbsv('U','N','N',nbands,d_AB,d_x)
        y = A\x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "trmv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        A = triu(A)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute trmv!
        CuArrays.BLAS.trmv!('U','N','N',d_A,d_x)
        x = A*x
        # compare
        h_x = collect(d_x)
        @test x ≈ h_x
    end
end

@testset "trmv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        A = triu(A)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute trmv!
        d_y = CuArrays.BLAS.trmv('U','N','N',d_A,d_x)
        y = A*x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "trsv!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        A = triu(A)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute trsv!
        CuArrays.BLAS.trsv!('U','N','N',d_A,d_x)
        x = A\x
        # compare
        h_x = collect(d_x)
        @test x ≈ h_x
    end
end

@testset "trsv" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate triangular matrix
        A = rand(elty,m,m)
        A = triu(A)
        # generate vector
        x = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # execute trsv!
        d_y = CuArrays.BLAS.trsv('U','N','N',d_A,d_x)
        y = A\x
        # compare
        h_y = collect(d_y)
        @test y ≈ h_y
    end
end

@testset "ger!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # construct matrix and vectors
        A = rand(elty,m,n)
        x = rand(elty,m)
        y = rand(elty,n)
        alpha = convert(elty,2)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # perform rank one update
        CuArrays.BLAS.ger!(alpha,d_x,d_y,d_A)
        A = (alpha*x)*y' + A
        # move to host and compare
        h_A = collect(d_A)
        @test A ≈ h_A
    end
end

@testset "syr!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # construct matrix and vector
        A = rand(elty,m,m)
        A = A + A.'
        x = rand(elty,m)
        alpha = convert(elty,2)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # perform rank one update
        CuArrays.BLAS.syr!('U',alpha,d_x,d_A)
        A = (alpha*x)*x.' + A
        # move to host and compare upper triangles
        h_A = collect(d_A)
        A = triu(A)
        h_A = triu(h_A)
        @test A ≈ h_A
    end
end

@testset "her!" begin
    @testset for elty in [Complex64, Complex128]
        local m = 2
        # construct matrix and vector
        A = rand(elty,m,m)
        A = A + A'
        x = rand(elty,m)
        alpha = convert(elty,2)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        # perform rank one update
        CuArrays.BLAS.her!('U',alpha,d_x,d_A)
        A = (alpha*x)*x' + A
        # move to host and compare upper triangles
        h_A = collect(d_A)
        A = triu(A)
        h_A = triu(h_A)
        @test A ≈ h_A
    end
end

@testset "her2!" begin
    @testset for elty in [Complex64, Complex128]
        local m = 2
        # construct matrix and vector
        A = rand(elty,m,m)
        A = A + A'
        x = rand(elty,m)
        y = rand(elty,m)
        alpha = convert(elty,2)
        # move to device
        d_A = CuArray(A)
        d_x = CuArray(x)
        d_y = CuArray(y)
        # perform rank one update
        CuArrays.BLAS.her2!('U',alpha,d_x,d_y,d_A)
        A = (alpha*x)*y' + y*(alpha*x)' + A
        # move to host and compare upper triangles
        h_A = collect(d_A)
        A = triu(A)
        h_A = triu(h_A)
        @test A ≈ h_A
    end
end

@testset "gemm!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate matrices
        A = rand(elty,m,k)
        B = rand(elty,k,n)
        C1 = rand(elty,m,n)
        C2 = copy(C1)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C1 = CuArray(C1)
        d_C2 = CuArray(C2)
        # C = (alpha*A)*B + beta*C
        CuArrays.BLAS.gemm!('N','N',alpha,d_A,d_B,beta,d_C1)
        A_mul_B!(d_C2, d_A, d_B)
        h_C1 = collect(d_C1)
        h_C2 = collect(d_C2)
        C1 = (alpha*A)*B + beta*C1
        C2 = A*B
        # compare
        @test C1 ≈ h_C1
        @test C2 ≈ h_C2
    end
end

@testset "gemm" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        B = rand(elty,k,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # C = (alpha*A)*B + beta*C
        d_C = CuArrays.BLAS.gemm('N','N',d_A,d_B)
        C = A*B
        C2 = d_A * d_B
        # compare
        h_C = collect(d_C)
        h_C2 = collect(C2)
        @test C ≈ h_C
        @test C ≈ h_C2
    end
end

@testset "gemm_batched!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate matrices
        A = [rand(elty,m,k) for i in 1:10]
        B = [rand(elty,k,n) for i in 1:10]
        C = [rand(elty,m,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_B = CuArray{elty, 2}[]
        d_C = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
            push!(d_B,CuArray(B[i]))
            push!(d_C,CuArray(C[i]))
        end
        # C = (alpha*A)*B + beta*C
        CuArrays.BLAS.gemm_batched!('N','N',alpha,d_A,d_B,beta,d_C)
        for i in 1:length(d_C)
            C[i] = (alpha*A[i])*B[i] + beta*C[i]
            h_C = collect(d_C[i])
            #compare
            @test C[i] ≈ h_C
        end
    end
end

@testset "gemm_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,m,k) for i in 1:10]
        B = [rand(elty,k,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_B = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A, CuArray(A[i]))
            push!(d_B, CuArray(B[i]))
        end
        # C = A*B
        d_C = CuArrays.BLAS.gemm_batched('N','N',d_A,d_B)
        for i in 1:length(A)
            C = A[i]*B[i]
            h_C = collect(d_C[i])
            @test C ≈ h_C
        end
    end
end

@testset "symm!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = A + A.'
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # C = (alpha*A)*B + beta*C
        CuArrays.BLAS.symm!('L','U',alpha,d_A,d_B,beta,d_C)
        C = (alpha*A)*B + beta*C
        # compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "symm" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,m)
        A = A + A.'
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # C = (alpha*A)*B + beta*C
        d_C = CuArrays.BLAS.symm('L','U',d_A,d_B)
        C = A*B
        # compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "syrk!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        C = rand(elty,m,m)
        C = C + C.'
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # move to device
        d_A = CuArray(A)
        d_C = CuArray(C)
        # C = (alpha*A)*A.' + beta*C
        CuArrays.BLAS.syrk!('U','N',alpha,d_A,beta,d_C)
        C = (alpha*A)*A.' + beta*C
        C = triu(C)
        # move to host and compare
        h_C = collect(d_C)
        h_C = triu(C)
        @test C ≈ h_C
    end
end

@testset "syrk" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        # move to device
        d_A = CuArray(A)
        # C = A*A.'
        d_C = CuArrays.BLAS.syrk('U','N',d_A)
        C = A*A.'
        C = triu(C)
        # move to host and compare
        h_C = collect(d_C)
        h_C = triu(C)
        @test C ≈ h_C
    end
end

@testset "herk!" begin
    @testset for elty in [Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        C = rand(elty,m,m)
        C = C + C'
        # parameters
        alpha = rand(elty)
        beta = rand(elty)
        # move to device
        d_A = CuArray(A)
        d_C = CuArray(C)
        CuArrays.BLAS.herk!('U','N',alpha,d_A,beta,d_C)
        C = alpha*(A*A') + beta*C
        C = triu(C)
        # move to host and compare
        h_C = collect(d_C)
        h_C = triu(C)
        @test C ≈ h_C
    end
end

@testset "herk" begin
    @testset for elty in [Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        # move to device
        d_A = CuArray(A)
        # C = A*A'
        d_C = CuArrays.BLAS.herk('U','N',d_A)
        C = A*A'
        C = triu(C)
        # move to host and compare
        h_C = collect(d_C)
        h_C = triu(C)
        @test C ≈ h_C
    end
end

@testset "syr2k!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        #local m = 3
        #local k = 1
        # generate parameters
        alpha = rand(elty)
        beta = rand(elty)
        # generate matrices
        A = rand(elty,m,k)
        B = rand(elty,m,k)
        C = rand(elty,m,m)
        C = C + C.'
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # compute
        #C = alpha*(A*B.') + conj(alpha)*(B*A.') + beta*C
        C = alpha*(A*B.' + B*A.') + beta*C
        CuArrays.BLAS.syr2k!('U','N',alpha,d_A,d_B,beta,d_C)
        # move back to host and compare
        C = triu(C)
        h_C = collect(d_C)
        h_C = triu(h_C)

        @test C ≈ h_C
    end
end

@testset "syr2k" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameters
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,k)
        B = rand(elty,m,k)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        #C = alpha*(A*B.') + conj(alpha)*(B*A.') + beta*C
        C = alpha*(A*B.' + B*A.')
        d_C = CuArrays.BLAS.syr2k('U','N',alpha,d_A,d_B)
        # move back to host and compare
        C = triu(C)
        h_C = collect(d_C)
        h_C = triu(h_C)
        @test C ≈ h_C
    end
end

@testset "her2k!" begin
    @testset for (elty1, elty2) in [(Complex64, Float32), (Complex128, Float64)]
        # generate parameters
        alpha = rand(elty1)
        beta = rand(elty2)
        # generate matrices
        A = rand(elty1,m,k)
        B = rand(elty1,m,k)
        C = rand(elty1,m,m)
        C = C + C'
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # compute
        #C = alpha*(A*B') + conj(alpha)*(B*A') + beta*C
        C = alpha*(A*B') + conj(alpha)*(B*A') + beta*C
        CuArrays.BLAS.her2k!('U','N',alpha,d_A,d_B,beta,d_C)
        # move back to host and compare
        C = triu(C)
        h_C = collect(d_C)
        h_C = triu(h_C)
        @test C ≈ h_C
    end
end

@testset "her2k" begin
    @testset for elty in [Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,k)
        B = rand(elty,m,k)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        C = A*B' + B*A'
        d_C = CuArrays.BLAS.her2k('U','N',d_A,d_B)
        # move back to host and compare
        C = triu(C)
        h_C = collect(d_C)
        h_C = triu(h_C)
        @test C ≈ h_C
    end
end

@testset "trmm!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = triu(A)
        B = rand(elty,m,n)
        C = zeros(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # compute
        C = alpha*A*B
        CuArrays.BLAS.trmm!('L','U','N','N',alpha,d_A,d_B,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "trmm" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = triu(A)
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        C = alpha*A*B
        d_C = CuArrays.BLAS.trmm('L','U','N','N',alpha,d_A,d_B)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "trsm!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = triu(A)
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        C = alpha*(A\B)
        CuArrays.BLAS.trsm!('L','U','N','N',alpha,d_A,d_B)
        # move to host and compare
        h_C = collect(d_B)
        @test C ≈ h_C
    end
end

@testset "trsm" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = triu(A)
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        C = alpha*(A\B)
        d_C = CuArrays.BLAS.trsm('L','U','N','N',alpha,d_A,d_B)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "trsm_batched!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        map!((x) -> triu(x), A, A)
        B = [rand(elty,m,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_B = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
            push!(d_B,CuArray(B[i]))
        end
        # compute
        CuArrays.BLAS.trsm_batched!('L','U','N','N',alpha,d_A,d_B)
        # move to host and compare
        for i in 1:length(d_B)
            C = alpha*(A[i]\B[i])
            h_C = collect(d_B[i])
            #compare
            @test C ≈ h_C
        end
    end
end

@testset "trsm_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        map!((x) -> triu(x), A, A)
        B = [rand(elty,m,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_B = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
            push!(d_B,CuArray(B[i]))
        end
        # compute
        d_C = CuArrays.BLAS.trsm_batched('L','U','N','N',alpha,d_A,d_B)
        # move to host and compare
        for i in 1:length(d_C)
            C = alpha*(A[i]\B[i])
            h_C = collect(d_C[i])
            @test C ≈ h_C
        end
    end
end

@testset "hemm!" begin
    @testset for elty in [Complex64, Complex128]
        # generate parameters
        alpha = rand(elty)
        beta  = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = A + ctranspose(A)
        @test ishermitian(A)
        B = rand(elty,m,n)
        C = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # compute
        C = alpha*(A*B) + beta*C
        CuArrays.BLAS.hemm!('L','L',alpha,d_A,d_B,beta,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "hemm" begin
    @testset for elty in [Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        # generate matrices
        A = rand(elty,m,m)
        A = A + ctranspose(A)
        @test ishermitian(A)
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        # compute
        C = alpha*(A*B)
        d_C = CuArrays.BLAS.hemm('L','U',alpha,d_A,d_B)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "geam!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameters
        alpha = rand(elty)
        beta  = rand(elty)
        # generate matrices
        A = rand(elty,m,n)
        B = rand(elty,m,n)
        C = zeros(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_C = CuArray(C)
        # compute
        C = alpha*A + beta*B
        CuArrays.BLAS.geam!('N','N',alpha,d_A,beta,d_B,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C

        #test in place versions too
        C = rand(elty,m,n)
        d_C = CuArray(C)
        C = alpha*C + beta*B
        CuArrays.BLAS.geam!('N','N',alpha,d_C,beta,d_B,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
        C = rand(elty,m,n)
        d_C = CuArray(C)
        C = alpha*A + beta*C
        CuArrays.BLAS.geam!('N','N',alpha,d_A,beta,d_C,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C

        #test setting C to zero
        C = rand(elty,m,n)
        d_C = CuArray(C)
        alpha = zero(elty)
        beta  = zero(elty)
        CuArrays.BLAS.geam!('N','N',alpha,d_A,beta,d_B,d_C)
        h_C = collect(d_C)
        @test h_C ≈ zeros(elty,m,n)

        # bounds checking
        @test_throws DimensionMismatch CuArrays.BLAS.geam!('N','T',alpha,d_A,beta,d_B,d_C)
        @test_throws DimensionMismatch CuArrays.BLAS.geam!('T','T',alpha,d_A,beta,d_B,d_C)
        @test_throws DimensionMismatch CuArrays.BLAS.geam!('T','N',alpha,d_A,beta,d_B,d_C)
    end
end

@testset "geam" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate parameter
        alpha = rand(elty)
        beta  = rand(elty)
        # generate matrices
        A = rand(elty,m,n)
        B = rand(elty,m,n)
        # move to device
        d_A = CuArray(A)
        d_B = CuArray(B)
        C = zeros(elty,m,n)
        # compute
        C = alpha*A + beta*B
        d_C = CuArrays.BLAS.geam('N','N',alpha,d_A,beta,d_B)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

@testset "getrf_batched!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        local k
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        pivot, info = CuArrays.BLAS.getrf_batched!(d_A, false)
        h_info = collect(info)
        for As in 1:length(d_A)
            C   = lufact!(copy(A[As]), Val{false}) # lufact(A[As],pivot=false)
            h_A = collect(d_A[As])
            #reconstruct L,U
            dL = eye(elty,m)
            dU = zeros(elty,(m,m))
            k = h_info[As]
            if( k >= 0 )
                dL += tril(h_A,-k-1)
                dU += triu(h_A,k)
            end
            #compare
            @test isapprox(C[:L], dL, rtol=1e-2)
            @test isapprox(C[:U], dU, rtol=1e-2)
        end
        for i in 1:length(A)
            d_A[ i ] = CuArray(A[i])
        end
        pivot, info = CuArrays.BLAS.getrf_batched!(d_A, true)
        h_info = collect(info)
        h_pivot = collect(pivot)
        for As in 1:length(d_A)
            C   = lufact(A[As])
            h_A = collect(d_A[As])
            #reconstruct L,U
            dL = eye(elty,m)
            dU = zeros(elty,(m,m))
            k = h_info[As]
            if( k >= 0 )
                dL += tril(h_A,-k-1)
                dU += triu(h_A,k)
            end
            #compare pivots
            @test length(setdiff(h_pivot[:,As],C[:p])) == 0
            #make device pivot matrix
            P = eye(m)
            for row in 1:m
                temp = copy(P[row,:])
                P[row,:] = P[h_pivot[row,As],:]
                P[h_pivot[row,As],:] = temp
            end
            @test inv(P)*dL*dU ≈ inv(C[:P]) * C[:L] * C[:U]
        end
    end
end

@testset "getrf_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        local k
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        pivot, info, d_B = CuArrays.BLAS.getrf_batched(d_A, false)
        h_info = collect(info)
        for Bs in 1:length(d_B)
            C   = lufact!(copy(A[Bs]),Val{false}) # lufact(A[Bs],pivot=false)
            h_B = collect(d_B[Bs])
            #reconstruct L,U
            dL = eye(elty,m)
            dU = zeros(elty,(m,m))
            k = h_info[Bs]
            if( h_info[Bs] >= 0 )
                dU += triu(h_B,k)
                dL += tril(h_B,-k-1)
            end
            #compare
            @test isapprox(C[:L], dL, rtol=1e-2)
            @test isapprox(C[:U], dU, rtol=1e-2)
        end
    end
end

@testset "getri_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        pivot, info = CuArrays.BLAS.getrf_batched!(d_A, true)
        h_info = collect(info)
        for Cs in 1:length(h_info)
            @test h_info[Cs] == 0
        end
        pivot, info, d_C = CuArrays.BLAS.getri_batched(d_A, pivot)
        h_info = collect(info)
        for Cs in 1:length(d_C)
            C   = inv(A[Cs])
            h_C = collect(d_C[Cs])
            @test h_info[Cs] == 0
            @test C ≈ h_C
        end
    end
end

@testset "matinv_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,m,m) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        info, d_C = CuArrays.BLAS.matinv_batched(d_A)
        for Cs in 1:length(d_C)
            C   = inv(A[Cs])
            h_C = collect(d_C[Cs])
            @test C ≈ h_C
        end
    end
end

@testset "geqrf_batched!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,m,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        tau, d_A = CuArrays.BLAS.geqrf_batched!(d_A)
        for As in 1:length(d_A)
            C   = qrfact(A[As])
            h_A = collect(d_A[As])
            h_tau = collect(tau[As])
            # build up Q
            Q = eye(elty,min(m,n))
            for i in 1:min(m,n)
                v = zeros(elty,m)
                v[i] = one(elty)
                v[i+1:m] = h_A[i+1:m,i]
                Q *= eye(elty,m) - h_tau[i] * v * v'
            end
            @test Q≈full(C[:Q])
        end
    end
end

@testset "geqrf_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,m,n) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
        end
        tau, d_B = CuArrays.BLAS.geqrf_batched!(d_A)
        for Bs in 1:length(d_B)
            C   = qrfact(A[Bs])
            h_B = collect(d_B[Bs])
            h_tau = collect(tau[Bs])
            # build up Q
            Q = eye(elty,min(m,n))
            for i in 1:min(m,n)
                v = zeros(elty,m)
                v[i] = one(elty)
                v[i+1:m] = h_B[i+1:m,i]
                Q *= eye(elty,m) - h_tau[i] * v * v'
            end
            @test Q≈full(C[:Q])
        end
    end
end

@testset "gels_batched!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,n,n) for i in 1:10]
        C = [rand(elty,n,k) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_C = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
            push!(d_C,CuArray(C[i]))
        end
        d_A, d_C, info = CuArrays.BLAS.gels_batched!('N',d_A, d_C)
        for Cs in 1:length(d_C)
            X = A[Cs]\C[Cs]
            h_C = collect(d_C[Cs])
            @test X≈h_C
        end
    end
end

@testset "gels_batched" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = [rand(elty,n,n) for i in 1:10]
        C = [rand(elty,n,k) for i in 1:10]
        # move to device
        d_A = CuArray{elty, 2}[]
        d_C = CuArray{elty, 2}[]
        for i in 1:length(A)
            push!(d_A,CuArray(A[i]))
            push!(d_C,CuArray(C[i]))
        end
        d_B, d_D, info = CuArrays.BLAS.gels_batched('N',d_A, d_C)
        for Ds in 1:length(d_D)
            X = A[Ds]\C[Ds]
            h_D = collect(d_D[Ds])
            @test X ≈ h_D
        end
    end
end

@testset "dgmm!" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,n)
        C = rand(elty,m,n)
        X = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_C = CuArray(C)
        d_X = CuArray(X)
        # compute
        C = diagm(X) * A
        CuArrays.BLAS.dgmm!('L',d_A,d_X,d_C)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
        # bounds checking
        @test_throws DimensionMismatch CuArrays.BLAS.dgmm!('R',d_A,d_X,d_C)
        A = rand(elty,m,m)
        d_A = CuArray(A)
        @test_throws DimensionMismatch CuArrays.BLAS.dgmm!('L',d_A,d_X,d_C)
    end
end

@testset "dgmm" begin
    @testset for elty in [Float32, Float64, Complex64, Complex128]
        # generate matrices
        A = rand(elty,m,n)
        X = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_X = CuArray(X)
        # compute
        C = diagm(X) * A
        d_C = CuArrays.BLAS.dgmm('L',d_A,d_X)
        # move to host and compare
        h_C = collect(d_C)
        @test C ≈ h_C
    end
end

end # level 2 testset

end
