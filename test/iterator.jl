@testset "CuIterator" begin
    batch_count = 10
    max_batch_items = 3
    max_ndims = 3
    sizes = 20:50
    rand_shape = () -> rand(sizes, rand(1:max_ndims))
    batches = [[rand(Float32, rand_shape()...) for _ in 1:rand(1:max_batch_items)] for _ in 1:batch_count]
    cubatches = CuIterator(batch for batch in batches) # ensure generators are accepted
    previous_cubatch = missing
    for (batch, cubatch) in zip(batches, cubatches)
        @test ismissing(previous_cubatch) || all(x -> x.freed, previous_cubatch)
        @test batch == Array.(cubatch)
        @test all(x -> x isa CuArray, cubatch)
        previous_cubatch = cubatch
    end
end
