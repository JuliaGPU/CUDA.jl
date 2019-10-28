module CUDA


using Reexport


@reexport using CUDAdrv

@eval $(Symbol("@elapsed")) = $(getfield(CUDAdrv, Symbol("@elapsed")))
@eval $(Symbol("@profile")) = $(getfield(CUDAdrv, Symbol("@profile")))


@reexport using CUDAnative


@reexport using CuArrays

## array constructors
const zeros = CuArrays.zeros
const ones  = CuArrays.ones
const fill  = CuArrays.fill

## random numbers
const fill          = CuArrays.fill
const seed!         = CuArrays.seed!
const rand          = CuArrays.rand
const randn         = CuArrays.randn
const rand_logn     = CuArrays.rand_logn
const rand_poisson  = CuArrays.rand_poisson

@eval $(Symbol("@sync"))        = $(getfield(CuArrays, Symbol("@sync")))
@eval $(Symbol("@time"))        = $(getfield(CuArrays, Symbol("@time")))
@eval $(Symbol("@allocated"))   = $(getfield(CuArrays, Symbol("@allocated")))


end
