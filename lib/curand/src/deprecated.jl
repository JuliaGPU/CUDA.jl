# Deprecated in v6.6: the offset into the random sequence is a keyword argument.

function Random.seed!(rng::LibraryRNG, seed, offset)
    Base.depwarn("`Random.seed!(rng::LibraryRNG, seed, offset)` is deprecated, use `Random.seed!(rng, seed; offset)` instead.", :seed!)
    Random.seed!(rng, seed; offset)
end
