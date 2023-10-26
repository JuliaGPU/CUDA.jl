using Enzyme

function square_kernel!(x)
    i = threadIdx().x
    x[i] *= x[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x)
    @cuda blocks = 1 threads = length(x) square_kernel!(x)
    return nothing
end

A = CUDA.rand(64)
dA = CUDA.ones(64)
A .= (1:1:64)
dA .= 1
Enzyme.autodiff(Forward, square!, Duplicated(A, dA))
@test all(dA .≈ (2:2:128))

A = CUDA.rand(32)
dA = CUDA.ones(32)
dA2 = CUDA.ones(32)
A .= (1:1:32)
dA .= 1
dA2 .= 3
Enzyme.autodiff(Forward, square!, BatchDuplicated(A, (dA, dA2)))
@test all(dA .≈ (2:2:64))
@test all(dA2 .≈ 3*(2:2:64))
