@testset "Logical indexing" begin

    bools = [false, true, false, true]

    v = [1, 2, 3, 4]
    v2 = CuArray(v)
    @test getindex(v2, bools) == getindex(v, bools)

    v = [1:10;]
    v2 = CuArray(v)

    @test v2[v2 .> 3] == v[v .> 3]
    @test filter(x->x^2 - 2 > 10, v2) == filter(x->x^2 - 2 > 10, v)

end
