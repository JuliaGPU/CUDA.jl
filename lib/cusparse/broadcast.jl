using Base.Broadcast: Broadcasted

using CUDA: CuArrayStyle

# TODO: support more types (SparseVector, SparseMatrixCSC, COO, BSR)


## sparse broadcast style

# broadcast container type promotion for combinations of sparse arrays and other types
struct CuSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
struct CuSparseMatStyle <: Broadcast.AbstractArrayStyle{2} end
Broadcast.BroadcastStyle(::Type{<:CuSparseVector}) = CuSparseVecStyle()
Broadcast.BroadcastStyle(::Type{<:CuSparseMatrix}) = CuSparseMatStyle()
const SPVM = Union{CuSparseVecStyle,CuSparseMatStyle}

# CuSparseVecStyle handles 0-1 dimensions, CuSparseMatStyle 0-2 dimensions.
# CuSparseVecStyle promotes to CuSparseMatStyle for 2 dimensions.
# Fall back to DefaultArrayStyle for higher dimensionality.
CuSparseVecStyle(::Val{0}) = CuSparseVecStyle()
CuSparseVecStyle(::Val{1}) = CuSparseVecStyle()
CuSparseVecStyle(::Val{2}) = CuSparseMatStyle()
CuSparseVecStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()
CuSparseMatStyle(::Val{0}) = CuSparseMatStyle()
CuSparseMatStyle(::Val{1}) = CuSparseMatStyle()
CuSparseMatStyle(::Val{2}) = CuSparseMatStyle()
CuSparseMatStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()

Broadcast.BroadcastStyle(::CuSparseVecStyle, ::CuArrayStyle{1}) = CuSparseVecStyle()
Broadcast.BroadcastStyle(::CuSparseVecStyle, ::CuArrayStyle{2}) = CuSparseMatStyle()
Broadcast.BroadcastStyle(::CuSparseMatStyle, ::CuArrayStyle{2}) = CuSparseMatStyle()

# don't wrap sparse arrays with Extruded
Broadcast.extrude(x::CuSparseVecOrMat) = x


## detection of zero-preserving functions

# modified from SparseArrays.jl

# capturescalars takes a function (f) and a tuple of broadcast arguments, and returns a
# partially-evaluated function and a reduced argument tuple where all scalar operations have
# been applied already.
@inline function capturescalars(f, mixedargs)
    let (passedsrcargstup, makeargs) = _capturescalars(mixedargs...)
        parevalf = (passed...) -> f(makeargs(passed...)...)
        return (parevalf, passedsrcargstup)
    end
end
# Work around losing Type{T}s as DataTypes within the tuple that makeargs creates
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Ref{Type{S}}, Vararg{Any}}) where {T, S} =
    # This definition is identical to the one above and necessary only for
    # avoiding method ambiguity.
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{CuSparseVecOrMat, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((a1, args...)->f(a1, T, args...), (mixedargs[1], Base.tail(Base.tail(mixedargs))...))
@inline capturescalars(f, mixedargs::Tuple{Union{Ref,AbstractArray{<:Any,0}}, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(mixedargs[1], T, args...), Base.tail(Base.tail(mixedargs)))

scalararg(::Number) = true
scalararg(::Any) = false
scalarwrappedarg(::Union{AbstractArray{<:Any,0},Ref}) = true
scalarwrappedarg(::Any) = false

@inline function _capturescalars()
    return (), () -> ()
end
@inline function _capturescalars(arg, mixedargs...)
    let (rest, f) = _capturescalars(mixedargs...)
        if scalararg(arg)
            return rest, @inline function(tail...)
                (arg, f(tail...)...)
            end # add back scalararg after (in makeargs)
        elseif scalarwrappedarg(arg)
            return rest, @inline function(tail...)
                (arg[], f(tail...)...) # TODO: This can put a Type{T} in a tuple
            end # unwrap and add back scalararg after (in makeargs)
        else
            return (arg, rest...), @inline function(head, tail...)
                (head, f(tail...)...)
            end # pass-through to broadcast
        end
    end
end
@inline function _capturescalars(arg) # this definition is just an optimization (to bottom out the recursion slightly sooner)
    if scalararg(arg)
        return (), () -> (arg,) # add scalararg
    elseif scalarwrappedarg(arg)
        return (), () -> (arg[],) # unwrap
    else
        return (arg,), (head,) -> (head,) # pass-through
    end
end

@inline _iszero(x) = x == 0
@inline _iszero(x::Number) = Base.iszero(x)
@inline _iszero(x::AbstractArray) = Base.iszero(x)
@inline _zeros_eltypes(A) = (zero(eltype(A)),)
@inline _zeros_eltypes(A, Bs...) = (zero(eltype(A)), _zeros_eltypes(Bs...)...)


## iteration helpers

"""
    CSRIterator{Ti}(row, args...)

A GPU-compatible iterator for accessing the elements of a single row `row` of several CSR
matrices `args` in one go. The row should be in-bounds for every sparse argument. Each
iteration returns a 2-element tuple: The current column, and each arguments' pointer index
(or 0 if that input didn't have an element at that column). The pointers can then be used to
access the elements themselves.

For convenience, this iterator can be passed non-sparse arguments as well, which will be
ignored (with the returned `col`/`ptr` values set to 0).
"""
struct CSRIterator{Ti,N,ATs}
    row::Ti
    col_ends::NTuple{N, Ti}
    args::ATs
end

function CSRIterator{Ti}(row, args::Vararg{<:Any, N}) where {Ti,N}
    # check that `row` is valid for all arguments
    @boundscheck begin
        ntuple(Val(N)) do i
            arg = @inbounds args[i]
            arg isa CuSparseDeviceMatrixCSR && checkbounds(arg, row, 1)
        end
    end

    col_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        if arg isa CuSparseDeviceMatrixCSR
            @inbounds(arg.rowPtr[row+1i32])
        else
            zero(Ti)
        end
    end

    CSRIterator{Ti, N, typeof(args)}(row, col_ends, args)
end

@inline function Base.iterate(iter::CSRIterator{Ti,N}, state=nothing) where {Ti,N}
    # helper function to get the column of a sparse array at a specific pointer
    @inline function get_col(i, ptr)
        arg = @inbounds iter.args[i]
        if arg isa CuSparseDeviceMatrixCSR
            col_end = @inbounds iter.col_ends[i]
            if ptr < col_end
                return @inbounds arg.colVal[ptr] % Ti
            end
        end
        typemax(Ti)
    end

    # initialize the state
    # - ptr: the current index into the colVal/nzVal arrays
    # - col: the current column index (cached so that we don't have to re-read each time)
    state = something(state,
        ntuple(Val(N)) do i
            arg = @inbounds iter.args[i]
            if arg isa CuSparseDeviceMatrixCSR
                ptr = @inbounds iter.args[i].rowPtr[iter.row] % Ti
                col = @inbounds get_col(i, ptr)
            else
                ptr = typemax(Ti)
                col = typemax(Ti)
            end
            (; ptr, col)
        end
    )

    # determine the column we're currently processing
    cols = ntuple(i -> @inbounds(state[i].col), Val(N))
    cur_col = min(cols...)
    cur_col == typemax(Ti) && return

    # fetch the pointers (we don't look up the values, as the caller might want to index
    # the sparse array directly, e.g., to mutate it). we don't return `ptrs` from the state
    # directly, but first convert the `typemax(Ti)` to a more convenient zero value.
    # NOTE: these values may end up unused by the caller (e.g. in the count_nnzs kernels),
    #       but LLVM appears smart enough to filter them away.
    ptrs = ntuple(Val(N)) do i
        ptr, col = @inbounds state[i]
        col == cur_col ? ptr : zero(Ti)
    end

    # advance the state
    new_state = ntuple(Val(N)) do i
        ptr, col = @inbounds state[i]
        if col == cur_col
            ptr += one(Ti)
            col = get_col(i, ptr)
        end
        (; ptr, col)
    end

    return (cur_col, ptrs), new_state
end

struct CSCIterator{Ti,N,ATs}
    col::Ti
    row_ends::NTuple{N, Ti}
    args::ATs
end

function CSCIterator{Ti}(col, args::Vararg{<:Any, N}) where {Ti,N}
    # check that `col` is valid for all arguments
    @boundscheck begin
        ntuple(Val(N)) do i
            arg = @inbounds args[i]
            arg isa CuSparseDeviceMatrixCSR && checkbounds(arg, 1, col)
        end
    end

    row_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        x = if arg isa CuSparseDeviceMatrixCSC
            @inbounds(arg.colPtr[col+1i32])
        else
            zero(Ti)
        end
        x
    end

    CSCIterator{Ti, N, typeof(args)}(col, row_ends, args)
end

@inline function Base.iterate(iter::CSCIterator{Ti,N}, state=nothing) where {Ti,N}
    # helper function to get the column of a sparse array at a specific pointer
    @inline function get_col(i, ptr)
        arg = @inbounds iter.args[i]
        if arg isa CuSparseDeviceMatrixCSC
            col_end = @inbounds iter.row_ends[i]
            if ptr < col_end
                return @inbounds arg.rowVal[ptr] % Ti
            end
        end
        typemax(Ti)
    end

    # initialize the state
    # - ptr: the current index into the rowVal/nzVal arrays
    # - row: the current row index (cached so that we don't have to re-read each time)
    state = something(state,
        ntuple(Val(N)) do i
            arg = @inbounds iter.args[i]
            if arg isa CuSparseDeviceMatrixCSC
                ptr = @inbounds iter.args[i].colPtr[iter.col] % Ti
                row = @inbounds get_col(i, ptr)
            else
                ptr = typemax(Ti)
                row = typemax(Ti)
            end
            (; ptr, row)
        end
    )

    # determine the row we're currently processing
    rows = ntuple(i -> @inbounds(state[i].row), Val(N))
    cur_row = min(rows...)
    cur_row == typemax(Ti) && return

    # fetch the pointers (we don't look up the values, as the caller might want to index
    # the sparse array directly, e.g., to mutate it). we don't return `ptrs` from the state
    # directly, but first convert the `typemax(Ti)` to a more convenient zero value.
    # NOTE: these values may end up unused by the caller (e.g. in the count_nnzs kernels),
    #       but LLVM appears smart enough to filter them away.
    ptrs = ntuple(Val(N)) do i
        ptr, row = @inbounds state[i]
        row == cur_row ? ptr : zero(Ti)
    end

    # advance the state
    new_state = ntuple(Val(N)) do i
        ptr, row = @inbounds state[i]
        if row == cur_row
            ptr += one(Ti)
            row = get_col(i, ptr)
        end
        (; ptr, row)
    end

    return (cur_row, ptrs), new_state
end

# helpers to index a sparse or dense array
function _getindex(arg::Union{CuSparseDeviceMatrixCSR,CuSparseDeviceMatrixCSC}, I, ptr)
    if ptr == 0
        zero(eltype(arg))
    else
        @inbounds arg.nzVal[ptr]
    end
end
_getindex(arg, I, ptr) = Broadcast._broadcast_getindex(arg, I)


## sparse broadcast implementation

# TODO: unify CSC/CSR kernels

# kernel to count the number of non-zeros in a row, to determine the row offsets
function compute_offsets_kernel(::Type{<:CuSparseMatrixCSR}, offsets::AbstractVector{Ti},
                                args...) where Ti
    # every thread processes an entire row
    row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row > length(offsets)-1 && return
    iter = @inbounds CSRIterator{Ti}(row, args...)

    # count the nonzero columns of all inputs
    accum = zero(Ti)
    for (col, vals) in iter
        accum += one(Ti)
    end

    # the way we write the nnz counts is a bit strange, but done so that the result
    # after accumulation can be directly used as the rowPtr array of a CSR matrix.
    @inbounds begin
        if row == 1
            offsets[1] = 1
        end
        offsets[row+1] = accum
    end

    return
end
function compute_offsets_kernel(::Type{<:CuSparseMatrixCSC}, offsets::AbstractVector{Ti},
                                args...) where Ti
    # every thread processes an entire columm
    col = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    col > length(offsets)-1 && return
    iter = @inbounds CSCIterator{Ti}(col, args...)

    # count the nonzero columns of all inputs
    accum = zero(Ti)
    for (col, vals) in iter
        accum += one(Ti)
    end

    # the way we write the nnz counts is a bit strange, but done so that the result
    # after accumulation can be directly used as the colPtr array of a CSC matrix.
    @inbounds begin
        if col == 1
            offsets[1] = 1
        end
        offsets[col+1] = accum
    end

    return
end

# broadcast kernels that iterate the elements of sparse arrays
function sparse_to_sparse_broadcast_kernel(f, output::CuSparseDeviceMatrixCSR{<:Any,Ti},
                                           offsets::Union{AbstractVector,Nothing},
                                           args...) where {Ti}
    # every thread processes an entire row
    row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row > size(output, 1) && return
    iter = @inbounds CSRIterator{Ti}(row, args...)

    # fetch the row offset, and write it to the output
    @inbounds begin
        output_ptr = output.rowPtr[row] = offsets[row]
        if row == size(output, 1)
            output.rowPtr[row+1i32] = offsets[row+1i32]
        end
    end

    # set the values for this row
    for (col, ptrs) in iter
        I = CartesianIndex(row, col)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output.colVal[output_ptr] = col
        @inbounds output.nzVal[output_ptr] = f(vals...)
        output_ptr += one(Ti)
    end

    return
end
function sparse_to_sparse_broadcast_kernel(f, output::CuSparseDeviceMatrixCSC{<:Any,Ti},
                                           offsets::Union{AbstractVector,Nothing},
                                           args...) where {Ti}
    # every thread processes an entire column
    col = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    col > size(output, 2) && return
    iter = @inbounds CSCIterator{Ti}(col, args...)

    # fetch the column offset, and write it to the output
    @inbounds begin
        output_ptr = output.colPtr[col] = offsets[col]
        if col == size(output, 2)
            output.colPtr[col+1i32] = offsets[col+1i32]
        end
    end

    # set the values for this col
    for (row, ptrs) in iter
        I = CartesianIndex(col, row)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output.rowVal[output_ptr] = row
        @inbounds output.nzVal[output_ptr] = f(vals...)
        output_ptr += one(Ti)
    end

    return
end
function sparse_to_dense_broadcast_kernel(::Type{<:CuSparseMatrixCSR}, f,
                                          output::CuDeviceArray, args...)
    # every thread processes an entire row
    row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row > size(output, 1) && return
    iter = @inbounds CSRIterator{Int}(row, args...)

    # set the values for this row
    for (col, ptrs) in iter
        I = CartesianIndex(row, col)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output[I] = f(vals...)
    end

    return
end
function sparse_to_dense_broadcast_kernel(::Type{<:CuSparseMatrixCSC}, f,
                                          output::CuDeviceArray, args...)
    # every thread processes an entire column
    col = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    col > size(output, 2) && return
    iter = @inbounds CSCIterator{Int}(col, args...)

    # set the values for this col
    for (row, ptrs) in iter
        I = CartesianIndex(row, col)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output[I] = f(vals...)
    end

    return
end

function Broadcast.copy(bc::Broadcasted{<:Union{CuSparseVecStyle,CuSparseMatStyle}})
    # on 1.6, ntuple closures often fail to infer
    VERSION < v"1.7" && @warn "Sparse broadcast is only supported on Julia 1.7 or higher" maxlog=1

    # find the sparse inputs
    bc = Broadcast.flatten(bc)
    sparse_args = findall(bc.args) do arg
        arg isa AbstractCuSparseArray
    end
    sparse_types = unique(map(i->nameof(typeof(bc.args[i])), sparse_args))
    if length(sparse_types) > 1
        error("broadcast with multiple types of sparse arrays ($(join(sparse_types, ", "))) is not supported")
    end
    sparse_typ = typeof(bc.args[first(sparse_args)])
    sparse_typ <: Union{CuSparseMatrixCSR,CuSparseMatrixCSC} ||
        error("broadcast with sparse arrays is currently only implemented for CSR and CSC matrices")
    Ti = if sparse_typ <: CuSparseMatrixCSR
        reduce(promote_type, map(i->eltype(bc.args[i].rowPtr), sparse_args))
    elseif sparse_typ <: CuSparseMatrixCSC
        reduce(promote_type, map(i->eltype(bc.args[i].colPtr), sparse_args))
    end

    # determine the output type
    Tv = Broadcast.combine_eltypes(bc.f, eltype.(bc.args))
    if !Base.isconcretetype(Tv)
        error("""GPU sparse broadcast resulted in non-concrete element type $Tv.
                 This probably means that the function you are broadcasting contains an error or type instability.""")
    end

    # partially-evaluate the function, removing scalars.
    parevalf, passedsrcargstup = capturescalars(bc.f, bc.args)
    # check if the partially-evaluated function preserves zeros. if so, we'll only need to
    # apply it to the sparse input arguments, preserving the sparse structure.
    if all(arg->isa(arg, AbstractSparseArray), passedsrcargstup)
        fofzeros = parevalf(_zeros_eltypes(passedsrcargstup...)...)
        fpreszeros = _iszero(fofzeros)
    else
        fpreszeros = false
    end

    # the kernels below parallelize across rows or cols, not elements, so it's unlikely
    # we'll launch many threads. to maximize utilization, parallelize across blocks first.
    rows, cols = size(bc)
    function compute_launch_config(kernel)
        config = launch_configuration(kernel.fun)
        if sparse_typ <: CuSparseMatrixCSR
            threads = min(rows, config.threads)
            blocks = max(cld(rows, threads), config.blocks)
            threads = cld(rows, blocks)
        elseif sparse_typ <: CuSparseMatrixCSC
            threads = min(cols, config.threads)
            blocks = max(cld(cols, threads), config.blocks)
            threads = cld(cols, blocks)
        end
        (; threads, blocks)
    end

    # allocate the output container
    if !fpreszeros
        # either we have dense inputs, or the function isn't preserving zeros,
        # so use a dense output to broadcast into.
        output = CuArray{Tv}(undef, size(bc))

        # since we'll be iterating the sparse inputs, we need to pre-fill the dense output
        # with appropriate values (while setting the sparse inputs to zero). we do this by
        # re-using the dense broadcast implementation.
        nonsparse_args = map(bc.args) do arg
            # NOTE: this assumes the broadcst is flattened, but not yet preprocessed
            if arg isa AbstractCuSparseArray
                zero(eltype(arg))
            else
                arg
            end
        end
        broadcast!(bc.f, output, nonsparse_args...)
    elseif length(sparse_args) == 1
        # we only have a single sparse input, so we can reuse its structure for the output.
        # this avoids a kernel launch and costly synchronization.
        sparse_arg = bc.args[first(sparse_args)]
        if sparse_typ <: CuSparseMatrixCSR
            offsets = rowPtr = sparse_arg.rowPtr
            colVal = similar(sparse_arg.colVal)
            nzVal = similar(sparse_arg.nzVal)
            output = CuSparseMatrixCSR(rowPtr, colVal, nzVal, size(bc))
        elseif sparse_typ <: CuSparseMatrixCSC
            offsets = colPtr = sparse_arg.colPtr
            rowVal = similar(sparse_arg.rowVal)
            nzVal = similar(sparse_arg.nzVal)
            output = CuSparseMatrixCSC(colPtr, rowVal, nzVal, size(bc))
        end
        # NOTE: we don't use CUSPARSE's similar, because that copies the structure arrays,
        #       while we do that in our kernel (for consistency with other code paths)
    else
        # determine the number of non-zero elements per row so that we can create an
        # appropriately-structured output container
        offsets = if sparse_typ <: CuSparseMatrixCSR
            CuArray{Ti}(undef, rows+1)
        elseif sparse_typ <: CuSparseMatrixCSC
            CuArray{Ti}(undef, cols+1)
        end
        let
            args = (sparse_typ, offsets, bc.args...)
            kernel = @cuda launch=false compute_offsets_kernel(args...)
            threads, blocks = compute_launch_config(kernel)
            kernel(args...; threads, blocks)
        end

        # accumulate these values so that we can use them directly as row pointer offsets,
        # as well as to get the total nnz count to allocate the sparse output array.
        # cusparseXcsrgeam2Nnz computes this in one go, but it doesn't seem worth the effort
        accumulate!(Base.add_sum, offsets, offsets)
        total_nnz = @allowscalar last(offsets[end]) - 1

        output = if sparse_typ <: CuSparseMatrixCSR
            colVal = CuArray{Ti}(undef, total_nnz)
            nzVal = CuArray{Tv}(undef, total_nnz)
            CuSparseMatrixCSR(offsets, colVal, nzVal, size(bc))
        elseif sparse_typ <: CuSparseMatrixCSC
            rowVal = CuArray{Ti}(undef, total_nnz)
            nzVal = CuArray{Tv}(undef, total_nnz)
            CuSparseMatrixCSC(offsets, rowVal, nzVal, size(bc))
        end
    end

    # perform the actual broadcast
    if output isa AbstractCuSparseArray
        args = (bc.f, output, offsets, bc.args...)
        kernel = @cuda launch=false sparse_to_sparse_broadcast_kernel(args...)
        threads, blocks = compute_launch_config(kernel)
        kernel(args...; threads, blocks)
    else
        args = (sparse_typ, bc.f, output, bc.args...)
        kernel = @cuda launch=false sparse_to_dense_broadcast_kernel(args...)
        threads, blocks = compute_launch_config(kernel)
        kernel(args...; threads, blocks)
    end

    return output
end
