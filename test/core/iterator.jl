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
    @test ismissing(previous_cubatch) || all(x -> x.data.freed, previous_cubatch)
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

it_nt = CuIterator((x=Float32[i,i/2], y=i) for i in 1:4)
@test first(it_nt).x isa CuArray{Float32}
batch1, state = iterate(it_nt)
@test batch1.x == cu([1,1/2])
batch2, _ = iterate(it_nt, state)
@test batch2.x == cu([2,2/2])
@test batch1.x.data.freed  # unsafe_free! has worked inside

it_vec = CuIterator([[i,i/2], [i/3, i/4]] for i in 1:4)
@test first(it_vec)[1] isa CuArray{Float64}

# test element type conversion using a custom adaptor
it_float64 = CuIterator([[1.0]])
@test first(it_float64) isa CuArray{Float64}
it_float32 = CuIterator(CuArray{Float32}, [[1.0]])
@test first(it_float32) isa CuArray{Float32}

using StaticArrays: SVector, SA
it_static = CuIterator([SA[i,i/2], SA[i/3, i/4]] for i in 1:4)
@test first(it_static) isa CuArray{<:SVector}
