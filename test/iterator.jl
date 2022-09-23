batch_count = 10
max_batch_items = 3
max_ndims = 3
sizes = 20:50

rand_shape = () -> rand(sizes, rand(1:max_ndims))
batches = [[rand(Float32, rand_shape()...) for _ in 1:rand(1:max_batch_items)]
                                           for _ in 1:batch_count]
cubatches = CuIterator(batch for batch in batches) # ensure generators are accepted

previous_cubatch = missing
for (batch, cubatch) in zip(batches, cubatches)
    global previous_cubatch
    @test ismissing(previous_cubatch) || all(x -> x.storage === nothing, previous_cubatch)
    @test batch == Array.(cubatch)
    @test all(x -> x isa CuArray, cubatch)
    previous_cubatch = cubatch
end

@test Base.IteratorSize(typeof(cubatches)) isa Base.HasShape{1}
@test length(cubatches) == length(batch for batch in batches)
@test axes(cubatches) == axes(batch for batch in batches)

@test Base.IteratorEltype(typeof(cubatches)) isa Base.EltypeUnknown
@test eltype(cubatches) == eltype(batch for batch in batches) == Any
@test Base.IteratorEltype(typeof(CuIterator(batches))) isa Base.HasEltype
@test eltype(CuIterator(batches)) == eltype(batches)  # Vector
