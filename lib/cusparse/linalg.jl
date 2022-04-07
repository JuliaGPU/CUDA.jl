using LinearAlgebra

function sum_dim1(A::CuSparseMatrixCSR)
    function kernel(out, dA)
        idx = threadIdx().x
        idx < length(dA.rowPtr) || return
        s = zero(eltype(dA))
        for k in dA.rowPtr[idx]:dA.rowPtr[idx+1]-1
            s += dA.nzVal[k]
        end
        out[idx] = s
        return
    end

    m, n = size(A)
    rowsum = CUDA.CuArray{eltype(A)}(undef, m)
    kernel_f = @cuda launch=false kernel(rowsum, A)
    
    threads = min(n, CUDA.maxthreads(kernel_f))
    blocks = cld(n, threads)
    kernel_f(rowsum, A; threads, blocks)
    return rowsum
end

function LinearAlgebra.opnorm(A::CuSparseMatrixCSR, p::Real=2)
    if p == Inf
        return maximum(sum_dim1(A))
    else
        error("p=$p is not supported")
    end
end

LinearAlgebra.opnorm(A::CuSparseMatrixCSC, p::Real=2) = opnorm(CuSparseMatrixCSR(A), p)
