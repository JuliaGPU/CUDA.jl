using CUDA.CUSPARSE
using SparseArrays
using Test
typeSet = [Float32, Float64, ComplexF32, ComplexF64]
n = 10
@testset "similar" begin
    @testset "similar(A::$h{$elty},Tv::$newElty)" for elty in typeSet,
                                     newElty in typeSet,
                                     (h,dimPtr,dimVal) in ((CuSparseMatrixCSR,:rowPtr,:colVal), (CuSparseMatrixCSC,:colPtr,:rowVal))
        
        A = sprand(elty, n, n, rand())
        dA = h(A)
        
        C_simple = similar(dA)
        C_dims = similar(dA,(n,n+1))
        C_eltype = similar(dA,newElty)
        C_full = similar(dA,newElty,(n,n+1))
        @test typeof(C_simple) == typeof(dA)
        @test typeof(C_dims) == typeof(dA)
        @test (typeof(C_eltype) == typeof(dA) && elty==newElty) || ((typeof(C_eltype) != typeof(dA) && elty!=newElty))
        @test (typeof(C_full) == typeof(dA) && elty==newElty) || ((typeof(C_full) != typeof(dA) && elty!=newElty))

        properties = Set(propertynames(dA));

        conserved_simple = Set([dimPtr,dimVal,:dims,:nnz])
        structure_conserved_simple = setdiff(properties,conserved_simple);
        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname simple conserved" for propertyname in conserved_simple
            @test getproperty(C_simple,propertyname) == getproperty(dA,propertyname)
        end

        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname simple structure conserved" for propertyname in structure_conserved_simple
            @test length(getproperty(C_simple,propertyname)) == length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_simple,propertyname)) == eltype(getproperty(dA,propertyname))
        end

        conserved_dims = Set([:nnz])
        if h==CuSparseMatrixCSR # Making the array one column longer increases colPtr length but not rowPtr length
            structure_conserved_dims = setdiff(properties,union(conserved_dims,Set([dimVal,:dims])))
        else #CSC
            structure_conserved_dims = setdiff(properties,union(conserved_dims,Set([dimVal,:dims,dimPtr])))
            @test length(getproperty(C_dims,dimPtr)) == length(getproperty(dA,dimPtr)) + 1
            @test eltype(getproperty(C_dims,dimPtr)) == eltype(getproperty(dA,dimPtr))
        end
        @test getproperty(C_dims,:dims) == (n,n+1)
        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname dims conserved" for propertyname in conserved_dims
            @test getproperty(C_dims,propertyname) == getproperty(dA,propertyname)
        end

        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname dims structure conserved" for propertyname in structure_conserved_dims
            @test length(getproperty(C_dims,propertyname)) == length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_dims,propertyname)) == eltype(getproperty(dA,propertyname))
        end

        conserved_eltype = Set([:nnz,:dims,dimPtr,dimVal])
        structure_conserved_eltype = setdiff(properties,union(conserved_eltype,[:nzVal]))
        @test eltype(getproperty(C_eltype,:nzVal)) == newElty
        @test length(getproperty(C_eltype,:nzVal)) == length(getproperty(dA,:nzVal))
        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname elty conserved" for propertyname in conserved_eltype
            @test getproperty(C_eltype,propertyname) == getproperty(dA,propertyname)
        end

        @testset "similar(A::$h{$elty},Tv::$newElty) $propertyname elty structure conserved" for propertyname in structure_conserved_eltype
            @test length(getproperty(C_eltype,propertyname)) == length(getproperty(dA,propertyname))
            @test eltype(getproperty(C_eltype,propertyname)) == eltype(getproperty(dA,propertyname))
        end

        @testset "similar(A::$h{$elty},Tv::$newElty) full" begin
            @test eltype(getproperty(C_full,:nzVal)) == newElty
            @test length(getproperty(C_full,:nzVal)) == length(getproperty(dA,:nzVal))
            @test h==CuSparseMatrixCSR ? length(getproperty(C_full,dimPtr)) == length(getproperty(dA,dimPtr)) : length(getproperty(C_full,dimPtr)) == length(getproperty(dA,dimPtr))+1
            @test getproperty(C_dims,:nnz) == getproperty(dA,:nnz)
            @test getproperty(C_full,:dims) == (n,n+1)
        end
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
        @test typeof(C_simple) == typeof(parent(dA))
        @test typeof(C_dims) == typeof(parent(dA))
        @test (typeof(C_eltype) == typeof(parent(dA)) && elty==newElty) || ((typeof(C_eltype) != typeof(parent(dA)) && elty!=newElty))
        @test (typeof(C_full) == typeof(parent(dA)) && elty==newElty) || ((typeof(C_full) != typeof(parent(dA)) && elty!=newElty))

        properties = Set(propertynames(parent(dA)));

        conserved_simple = Set([:nnz])
        structure_conserved_simple = setdiff(properties,conserved_simple);
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname simple conserved" for propertyname in conserved_simple
            @test getproperty(C_simple,propertyname) == getproperty(parent(dA),propertyname)
        end
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname simple structure conserved" for propertyname in structure_conserved_simple
            @test length(getproperty(C_simple,propertyname)) == length(getproperty(parent(dA),propertyname))
            @test eltype(getproperty(C_simple,propertyname)) == eltype(getproperty(parent(dA),propertyname))
        end

        conserved_dims = Set([:nnz])
        if h==CuSparseMatrixCSR
            structure_conserved_dims = setdiff(properties,union(conserved_dims,Set([dimVal,:dims])))
        else #CSC
            structure_conserved_dims = setdiff(properties,union(conserved_dims,Set([dimVal,:dims,dimPtr])))
            @test length(getproperty(C_dims,dimPtr)) == length(getproperty(parent(dA),dimPtr)) + 1
            @test eltype(getproperty(C_dims,dimPtr)) == eltype(getproperty(parent(dA),dimPtr))
        end
        @test getproperty(C_dims,:dims) == (n,n+1)
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname dims conserved" for propertyname in conserved_dims
            @test getproperty(C_dims,propertyname) == getproperty(parent(dA),propertyname)
        end
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname dims structure conserved" for propertyname in structure_conserved_dims
            @test length(getproperty(C_dims,propertyname)) == length(getproperty(parent(dA),propertyname))
            @test eltype(getproperty(C_dims,propertyname)) == eltype(getproperty(parent(dA),propertyname))
        end

        conserved_eltype = Set([:nnz,:dims])
        structure_conserved_eltype = setdiff(properties,union(conserved_eltype,[:nzVal]))
        @test eltype(getproperty(C_eltype,:nzVal)) == newElty
        @test length(getproperty(C_eltype,:nzVal)) == length(getproperty(parent(dA),:nzVal))
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname elty conserved" for propertyname in conserved_eltype
            @test getproperty(C_eltype,propertyname) == getproperty(parent(dA),propertyname)
        end
        @testset "similar(A::$f($h{$elty}),$newElty) $propertyname elty structure conserved" for propertyname in structure_conserved_eltype
            @test length(getproperty(C_eltype,propertyname)) == length(getproperty(parent(dA),propertyname))
            @test eltype(getproperty(C_eltype,propertyname)) == eltype(getproperty(parent(dA),propertyname))
        end
        @testset "similar(A::$f($h{$elty}),$newElty) full" begin
            @test eltype(getproperty(C_full,:nzVal)) == newElty
            @test length(getproperty(C_full,:nzVal)) == length(getproperty(parent(dA),:nzVal))
            @test h==CuSparseMatrixCSR ? length(getproperty(C_full,dimPtr)) == length(getproperty(parent(dA),dimPtr)) : length(getproperty(C_full,dimPtr)) == (length(getproperty(parent(dA),dimPtr))+1)
            @test getproperty(C_dims,:nnz) == getproperty(parent(dA),:nnz)
            @test getproperty(C_full,:dims) == (n,n+1)
        end
    end

    
end
