# Native kernel-based RNG — deprecated alias for GPUArrays.RNG

using CUDACore.GPUArrays

Base.@deprecate_binding NativeRNG GPUArrays.RNG false

native_make_seed() = Base.rand(Random.RandomDevice(), UInt32)
