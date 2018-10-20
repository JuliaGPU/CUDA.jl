using CuArrays: @cuindex, cudims

function mapreducedim_kernel_serial(f, op, R, A, range)
    I = @cuindex R
    newrange = map((r, i) -> r === nothing ? i : r, range, I)
    for I′ in CartesianIndices(newrange)
        @inbounds R[I...] = op(R[I...], f(A[I′]))
    end
    return
end

@inline function reduce_block(arr::CuDeviceArray, op)
    sync_threads()
    len = blockDim().x
    while len != 1
        sync_threads()
        skip = (len + 1) >> 1
        reduce_to = threadIdx().x - skip
        if 0 < reduce_to <= (len >> 1)
            arr[reduce_to] = op(arr[reduce_to], arr[threadIdx().x])
        end
        len = skip
    end
    sync_threads()
end

function mapreducedim_kernel_parallel(f, op, R::CuDeviceArray{T}, A::CuDeviceArray{T},
                             CIS, Rlength, Slength) where {T}
    for Ri_base in 0:(gridDim().x * blockDim().y):(Rlength-1)
        Ri = Ri_base + (blockIdx().x - 1) * blockDim().y + threadIdx().y
        Ri > Rlength && return
        RI = Tuple(CartesianIndices(R)[Ri])
        S = @cuStaticSharedMem(T, 512)
        Si_folded_base = (threadIdx().y - 1) * blockDim().x
        Si_folded = Si_folded_base + threadIdx().x
        # serial reduction of A into S by Slength ÷ xthreads
        for Si_base in 0:blockDim().x:(Slength-1)
            Si = Si_base + threadIdx().x
            Si > Slength && break
            SI = Tuple(CIS[Si])
            AI = ifelse.(size(R) .== 1, SI, RI)
            if Si_base == 0
                S[Si_folded] = f(A[AI...])
            else
                S[Si_folded] = op(S[Si_folded], f(A[AI...]))
            end
        end
        # block-parallel reduction of S to S[1] by xthreads
        reduce_block(view(S, (Si_folded_base + 1):512), op)
        # reduce S[1] into R
        threadIdx().x == 1 && (R[Ri] = op(R[Ri], S[Si_folded]))
    end
    return
end

function Base._mapreducedim!(f, op, R::CuArray{T}, A::CuArray{T}) where {T}
    # the kernel as generated from `f` and `op` can require lots of registers (eg. #160),
    # so we need to be careful about how many threads we launch not to run out of them.
    Rlength = length(R)
    Ssize = ifelse.(size(R) .== 1, size(A), 1)
    Slength = prod(Ssize)
    CIS = CartesianIndices(Ssize)

    parallel_args = (f, op, R, A, CIS, Rlength, Slength)
    GC.@preserve parallel_args begin
        parallel_kargs = Tuple(adapt(CUDAnative.Adaptor(), arg) for arg in parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(mapreducedim_kernel_parallel, parallel_tt)
        max_thr = CUDAnative.maxthreads(parallel_kernel)

        outer_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, max_thr)
        inner_thr = min(512 ÷ outer_thr, Slength, max_thr)
        if inner_thr < 8
            # we can saturate the GPU with serial reduction
            range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
            blk, thr = cudims(R)
            @cuda(blocks=blk, threads=thr, mapreducedim_kernel_serial(f, op, R, A, range))
        else
            blk, thr = (Rlength - 1) ÷ outer_thr + 1, (inner_thr, outer_thr, 1)
            parallel_kernel(parallel_kargs...; threads=thr, blocks=blk)
        end
    end

    return R
end
