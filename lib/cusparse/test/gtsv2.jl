@testset "gtsv2" begin
    dl1 = [0; 1; 3]
    d1 = [1; 1; 4]
    du1 = [1; 2; 0]
    B1 = [1 0 0; 0 1 0; 0 0 1]
    X1 = [1/3 2/3 -1/3; 2/3 -2/3 1/3; -1/2 1/2 0]

    dl2 = [0; 1; 1; 1; 1; 1; 0]
    d2 = [6; 4; 4; 4; 4; 4; 6]
    du2 = [0; 1; 1; 1; 1; 1; 0]
    B2 = [0; 1; 2; -6; 2; 1; 0]
    X2 = [0; 0; 1; -2; 1; 0; 0]

    dl3 = [0; 1; 1; 7; 6; 3; 8; 6; 5; 4]
    d3 = [2; 3; 3; 2; 2; 4; 1; 2; 4; 5]
    du3 = [1; 2; 1; 6; 1; 3; 5; 7; 3; 0]
    B3 = [1; 2; 6; 34; 10; 1; 4; 22; 25; 3]
    X3 = [1; -1; 2; 1; 3; -2; 0; 4; 2; -1]
    for pivoting ∈ (false, true)
        @testset "gtsv2 with pivoting=$pivoting -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            @testset "example 1" begin
                dl1_d = CuVector{elty}(dl1)
                d1_d = CuVector{elty}(d1)
                du1_d = CuVector{elty}(du1)
                B1_d = CuArray{elty}(B1)
                X1_d = gtsv2(dl1_d, d1_d, du1_d, B1_d; pivoting)
                @test collect(X1_d) ≈ X1
                gtsv2!(dl1_d, d1_d, du1_d, B1_d; pivoting)
                @test collect(B1_d) ≈ X1
            end
            @testset "example 2" begin
                dl2_d = CuVector{elty}(dl2)
                d2_d = CuVector{elty}(d2)
                du2_d = CuVector{elty}(du2)
                B2_d = CuArray{elty}(B2)
                X2_d = gtsv2(dl2_d, d2_d, du2_d, B2_d; pivoting)
                @test collect(X2_d) ≈ X2
                gtsv2!(dl2_d, d2_d, du2_d, B2_d; pivoting)
                @test collect(B2_d) ≈ X2
            end
            @testset "example 3" begin
                dl3_d = CuVector{elty}(dl3)
                d3_d = CuVector{elty}(d3)
                du3_d = CuVector{elty}(du3)
                B3_d = CuArray{elty}(B3)
                X3_d = gtsv2(dl3_d, d3_d, du3_d, B3_d; pivoting)
                @test collect(X3_d) ≈ X3
                gtsv2!(dl3_d, d3_d, du3_d, B3_d; pivoting)
                @test collect(B3_d) ≈ X3
            end
        end
    end
end
