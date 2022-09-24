using LinearAlgebra

function sum_dim1(A::CuSparseMatrixCSR)
    function kernel(Tnorm, out, dA)
        idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
        idx < length(dA.rowPtr) || return
        s = zero(Tnorm)
        for k in dA.rowPtr[idx]:dA.rowPtr[idx+1]-1
            s += abs(dA.nzVal[k])
        end
        out[idx] = s
        return
    end

    m, n = size(A)
    Tnorm = typeof(float(real(zero(eltype(A)))))
    Tsum = promote_type(Float64,Tnorm)
    rowsum = CUDA.CuArray{Tsum}(undef, m)
    kernel_f = @cuda launch=false kernel(Tnorm, rowsum, A)
    
    config = launch_configuration(kernel_f.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel_f(Tnorm, rowsum, A; threads, blocks)
    return rowsum
end

function sum_dim2(A::CuSparseMatrixCSR)
    function kernel(Tnorm, out, dA)
        idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
        idx < length(dA.colPtr) || return
        s = zero(Tnorm)
        for k in dA.colPtr[idx]:dA.colPtr[idx+1]-1
            s += abs(dA.nzVal[k])
        end
        out[idx] = s
        return
    end

    A = CuSparseMatrixCSC(A)
    m, n = size(A)
    Tnorm = typeof(float(real(zero(eltype(A)))))
    Tsum = promote_type(Float64,Tnorm)
    colsum = CUDA.CuArray{Tsum}(undef, n)
    kernel_f = @cuda launch=false kernel(Tnorm, colsum, A)
    
    config = launch_configuration(kernel_f.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)
    kernel_f(Tnorm, colsum, A; threads, blocks)
    return colsum
end

function LinearAlgebra.opnorm(A::CuSparseMatrixCSR, p::Real=2)
    if p == Inf
        return maximum(sum_dim1(A))
    elseif p == 1
        return maximum(sum_dim2(A))
    else
        error("p=$p is not supported")
    end
end

LinearAlgebra.opnorm(A::CuSparseMatrixCSC, p::Real=2) = opnorm(CuSparseMatrixCSR(A), p)
