using CUDA.CUSPARSE
using SparseArrays
using Test
typeSet = [Float32, Float64, ComplexF32, ComplexF64]
n = 10
@testset "similar" begin
    @testset "similar(A::$h{$elty},$newElty)" for elty in typeSet,
                                     newElty in typeSet,
                                     (h,dimPtr,dimVal) in ((CuSparseMatrixCSR,:rowPtr,:colVal), (CuSparseMatrixCSC,:colPtr,:rowVal))
        
        A = sprand(elty, n, n, rand())
        dA = h(A)
        
        C_simple = similar(dA)
        C_dims = similar(dA,(n,n+1))
        C_eltype = similar(dA,newElty)
        C_full = similar(dA,newElty,(n,n+1))
        @test typeof(C_simple) ≈ typeof(dA)
        @test typeof(C_dims) ≈ typeof(dA)
        @test typeof(C_eltype) ≈ typeof(dA)
        @test typeof(C_full) ≈ typeof(dA)

        properties = Set([propertynames(dA)]);

        conserved_simple = Set([dimPtr,dimVal,dims,nnz])
        structure_conserved_simple = setdiff(properties,conserved_simple);
        for propertyname in conserved_simple
            @test getproperty(C_simple,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_simple
            @test length(getproperty(C_simple,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_simple,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        conserved_dims = Set([nnz])
        structure_conserved_dims = setdiff(properties,union(conserved_dims,dimVal,:dims))
        @test getproperty(C_dims,:dims) ≈ (n,n+1)
        for propertyname in conserved_dims
            @test getproperty(C_dims,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_dims
            @test length(getproperty(C_dims,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_dims,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        conserved_eltype = Set([nnz,dims,dimPtr,dimVal])
        structure_conserved_eltype = setdiff(properties,union(conserved_eltype,:nzVal))
        @test eltype(getproperty(C_eltype,:nzVal)) ≈ newElty
        @test length(getproperty(C_eltype,:nzVal)) ≈ length(getproperty(dA,:nzVal))
        for propertyname in conserved_eltype
            @test getproperty(C_dims,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_eltype
            @test length(getproperty(C_dims,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_dims,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        @test eltype(getproperty(C_full,:nzVal)) ≈ newElty
        @test length(getproperty(C_full,:nzVal)) ≈ length(getproperty(dA,:nzVal))
        @test length(getproperty(C_full,:dimPtr)) ≈ length(getproperty(dA,:dimPtr))
        @test getproperty(C_dims,:nnz) ≈ getproperty(dA,:nnz)
        @test getproperty(C_full,:dims) ≈ (n,n+1)
    end
        
    @testset "similar(A::$f($h{$elty}),$newElty)" for elty in typeSet,
        newElty in typeSet,
        f in (transpose, x->reshape(x,n,n)),
        (h,dimPtr,dimVal) in ((CuSparseMatrixCSR,:rowPtr,:colVal), (CuSparseMatrixCSC,:colPtr,:rowVal))

        A = sprand(elty, n, n, rand())
        dA = f(h(A))
        
        C_simple = similar(dA)
        C_dims = similar(dA,(n,n+1))
        C_eltype = similar(dA,newElty)
        C_full = similar(dA,newElty,(n,n+1))
        @test typeof(C_simple) ≈ typeof(dA)
        @test typeof(C_dims) ≈ typeof(dA)
        @test typeof(C_eltype) ≈ typeof(dA)
        @test typeof(C_full) ≈ typeof(dA)

        properties = Set([propertynames(dA)]);

        conserved_simple = Set([nnz])
        structure_conserved_simple = setdiff(properties,conserved_simple);
        for propertyname in conserved_simple
            @test getproperty(C_simple,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_simple
            @test length(getproperty(C_simple,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_simple,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        conserved_dims = Set([nnz])
        structure_conserved_dims = setdiff(properties,union(conserved_dims,dimVal,:dims))
        @test getproperty(C_dims,:dims) ≈ (n,n+1)
        for propertyname in conserved_dims
            @test getproperty(C_dims,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_dims
            @test length(getproperty(C_dims,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_dims,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        conserved_eltype = Set([nnz,dims,dimPtr,dimVal])
        structure_conserved_eltype = setdiff(properties,union(conserved_eltype,:nzVal))
        @test eltype(getproperty(C_eltype,:nzVal)) ≈ newElty
        @test length(getproperty(C_eltype,:nzVal)) ≈ length(getproperty(dA,:nzVal))
        for propertyname in conserved_eltype
            @test getproperty(C_dims,propertyname) ≈ getproperty(dA,propertyname)
        end
        for propertyname in structure_conserved_eltype
            @test length(getproperty(C_dims,propertyname)) ≈ length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_dims,propertyname)) ≈ eltype(getproperty(dA,propertyname))
        end

        @test eltype(getproperty(C_full,:nzVal)) ≈ newElty
        @test length(getproperty(C_full,:nzVal)) ≈ length(getproperty(dA,:nzVal))
        @test length(getproperty(C_full,:dimPtr)) ≈ length(getproperty(dA,:dimPtr))
        @test getproperty(C_dims,:nnz) ≈ getproperty(dA,:nnz)
        @test getproperty(C_full,:dims) ≈ (n,n+1)
    end

    
end
