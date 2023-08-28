# TODO: implement mapreducedim!

function Base.mapreduce(f, op, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC};
                        dims=:, init=nothing)
    # figure out the destination container type by looking at the initializer element,
    # or by relying on inference to reason through the map and reduce functions
    if init === nothing
        ET = Broadcast.combine_eltypes(f, (A,))
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("mapreduce cannot figure the output element type, please pass an explicit init value")

        init = GPUArrays.neutral_element(op, ET)
    else
        ET = typeof(init)
    end

    # we only handle reducing along one of the two dimensions,
    # or a complete reduction (requiring an additional pass)
    in(dims, [Colon(), 1, 2]) || error("only dims=:, dims=1 or dims=2 is supported")

    if A isa CuSparseMatrixCSR && dims == 1
        A = CuSparseMatrixCSC(A)
    elseif A isa CuSparseMatrixCSC && dims == 2
        A = CuSparseMatrixCSR(A)
    end

    m, n = size(A)
    if A isa CuSparseMatrixCSR
        output = CuArray{ET}(undef, m)

        kernel = @cuda launch=false csr_reduce_kernel(f, op, init, output, A)
        config = launch_configuration(kernel.fun)
        threads = min(m, config.threads)
        blocks = cld(m, threads)
    elseif A isa CuSparseMatrixCSC
        output = CuArray{ET}(undef, (1, n))

        kernel = @cuda launch=false csc_reduce_kernel(f, op, init, output, A)
        config = launch_configuration(kernel.fun)
        threads = min(n, config.threads)
        blocks = cld(n, threads)
    end
    kernel(f, op, init, output, A; threads, blocks)

    if dims == Colon()
        mapreduce(f, op, output; init)
    else
        output
    end
end

function csr_reduce_kernel(f::F, op::OP, neutral, output::CuDeviceArray, args...) where {F, OP}
    # every thread processes an entire row
    row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row > size(output, 1) && return
    iter = @inbounds CSRIterator{Int}(row, args...)

    val = op(neutral, neutral)

    # reduce the values for this row
    for (col, ptrs) in iter
        I = CartesianIndex(row, col)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end
        val = op(val, f(vals...))
    end

    @inbounds output[row] = val
    return
end

function csc_reduce_kernel(f::F, op::OP, neutral, output::CuDeviceArray, args...) where {F, OP}
    # every thread processes an entire column
    col = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    col > size(output, 2) && return
    iter = @inbounds CSCIterator{Int}(col, args...)

    val = op(neutral, neutral)

    # reduce the values for this col
    for (row, ptrs) in iter
        I = CartesianIndex(row, col)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end
        val = op(val, f(vals...))
    end

    @inbounds output[col] = val
    return
end
