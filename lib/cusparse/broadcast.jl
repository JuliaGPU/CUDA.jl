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
@inline function _capturescalars(arg)
    # this definition is just an optimization (to bottom out the recursion slightly sooner)
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


## COV_EXCL_START
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

function CSRIterator{Ti}(row, args::Vararg{Any, N}) where {Ti,N}
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

function CSCIterator{Ti}(col, args::Vararg{Any, N}) where {Ti,N}
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
@inline function _getindex(arg::Union{CuSparseDeviceMatrixCSR{Tv},
                                      CuSparseDeviceMatrixCSC{Tv},
                                      CuSparseDeviceVector{Tv}}, I, ptr)::Tv where {Tv}
    if ptr == 0
        return zero(Tv)
    else
        return @inbounds arg.nzVal[ptr]::Tv
    end
end

@inline function _getindex(arg::CuDeviceArray{Tv}, I, ptr)::Tv where {Tv}
    return @inbounds arg[I]::Tv
end
@inline _getindex(arg, I, ptr) = Broadcast._broadcast_getindex(arg, I)

## sparse broadcast implementation

iter_type(::Type{<:CuSparseMatrixCSC}, ::Type{Ti}) where {Ti} = CSCIterator{Ti}
iter_type(::Type{<:CuSparseMatrixCSR}, ::Type{Ti}) where {Ti} = CSRIterator{Ti}
iter_type(::Type{<:CuSparseDeviceMatrixCSC}, ::Type{Ti}) where {Ti} = CSCIterator{Ti}
iter_type(::Type{<:CuSparseDeviceMatrixCSR}, ::Type{Ti}) where {Ti} = CSRIterator{Ti}

_has_row(A, offsets, row::Int32, fpreszeros::Bool) = fpreszeros ? 0i32 : row
_has_row(A::CuDeviceArray, offsets, row::Int32, ::Bool) = row
function _has_row(A::CuSparseDeviceVector, offsets, row::Int32, ::Bool)::Int32
    for row_ix in 1i32:length(A.iPtr)
        arg_row = @inbounds A.iPtr[row_ix]
        arg_row == row && return row_ix
        arg_row > row && break
    end
    return 0i32
end

function _get_my_row(first_row)::Int32
    row_ix = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    return row_ix + first_row - 1i32
end

function compute_offsets_kernel(::Type{<:CuSparseVector}, first_row::Ti, last_row::Ti,
                                fpreszeros::Bool, offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                args...) where {Ti, N}
    row = _get_my_row(first_row)
    row > last_row && return

    # TODO load arg.iPtr slices into shared memory
    row_is_nnz = 0i32
    arg_row_is_nnz = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        _has_row(arg, offsets, row, fpreszeros)::Int32
    end
    row_is_nnz = 0i32
    for i in 1:N
        row_is_nnz |= @inbounds arg_row_is_nnz[i]
    end
    key = (row_is_nnz == 0i32) ? typemax(Ti) : row
    @inbounds offsets[row - first_row + 1i32] = key => arg_row_is_nnz
    return
end

# kernel to count the number of non-zeros in a row, to determine the row offsets
function compute_offsets_kernel(T::Type{<:Union{CuSparseMatrixCSR, CuSparseMatrixCSC}},
                                offsets::AbstractVector{Ti},
                                args...) where Ti
    # every thread processes an entire row
    leading_dim = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    leading_dim > length(offsets)-1 && return
    iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

    # count the nonzero leading_dims of all inputs
    accum = zero(Ti)
    for (leading_dim, vals) in iter
        accum += one(Ti)
    end

    # the way we write the nnz counts is a bit strange, but done so that the result
    # after accumulation can be directly used as the rowPtr/colPtr array of a CSR/CSC matrix.
    @inbounds begin
        if leading_dim == 1
            offsets[1] = 1
        end
        offsets[leading_dim+1] = accum
    end

    return
end

function sparse_to_sparse_broadcast_kernel(f::F, output::CuSparseDeviceVector{Tv,Ti},
                                           offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                           args...) where {Tv, Ti, N, F}
    row_ix = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row_ix > output.nnz && return
    row_and_ptrs = @inbounds offsets[row_ix]
    row          = @inbounds row_and_ptrs[1]
    arg_ptrs     = @inbounds row_and_ptrs[2]
    vals = ntuple(Val(N)) do i
        @inline
        arg = @inbounds args[i]
        # ptr is 0 if the sparse vector doesn't have an element at this row
        # ptr is 0 if the arg is a scalar AND f preserves zeros
        ptr = @inbounds arg_ptrs[i]
        _getindex(arg, row, ptr)
    end
    output_val = f(vals...)
    @inbounds output.iPtr[row_ix]  = row
    @inbounds output.nzVal[row_ix] = output_val
    return
end

function sparse_to_sparse_broadcast_kernel(f, output::T, offsets::Union{AbstractVector,Nothing},
                                           args...) where {Ti, T<:Union{CuSparseDeviceMatrixCSR{<:Any,Ti},
                                                                        CuSparseDeviceMatrixCSC{<:Any,Ti}}}
    # every thread processes an entire row
    leading_dim = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    leading_dim_size = output isa CuSparseDeviceMatrixCSR ? size(output, 1) : size(output, 2)
    leading_dim > leading_dim_size && return
    iter = @inbounds iter_type(T, Ti)(leading_dim, args...)


    output_ptrs  = output isa CuSparseDeviceMatrixCSR ? output.rowPtr : output.colPtr
    output_ivals = output isa CuSparseDeviceMatrixCSR ? output.colVal : output.rowVal
    # fetch the row offset, and write it to the output
    @inbounds begin
        output_ptr = output_ptrs[leading_dim] = offsets[leading_dim]
        if leading_dim == leading_dim_size
            output_ptrs[leading_dim+1i32] = offsets[leading_dim+1i32]
        end
    end

    # set the values for this row
    for (sub_leading_dim, ptrs) in iter
        index_first  = output isa CuSparseDeviceMatrixCSR ? leading_dim : sub_leading_dim
        index_second = output isa CuSparseDeviceMatrixCSR ? sub_leading_dim : leading_dim
        I = CartesianIndex(index_first, index_second)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output_ivals[output_ptr] = sub_leading_dim
        @inbounds output.nzVal[output_ptr] = f(vals...)
        output_ptr += one(Ti)
    end

    return
end
function sparse_to_dense_broadcast_kernel(T::Type{<:Union{CuSparseMatrixCSR{Tv, Ti},
                                                          CuSparseMatrixCSC{Tv, Ti}}}, f,
                                          output::CuDeviceArray, args...) where {Tv, Ti}
    # every thread processes an entire row
    leading_dim = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    leading_dim_size = T <: CuSparseMatrixCSR ? size(output, 1) : size(output, 2)
    leading_dim > leading_dim_size && return
    iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

    # set the values for this row
    for (sub_leading_dim, ptrs) in iter
        index_first  = T <: CuSparseMatrixCSR ? leading_dim : sub_leading_dim
        index_second = T <: CuSparseMatrixCSR ? sub_leading_dim : leading_dim
        I = CartesianIndex(index_first, index_second)
        vals = ntuple(Val(length(args))) do i
            arg = @inbounds args[i]
            ptr = @inbounds ptrs[i]
            _getindex(arg, I, ptr)
        end

        @inbounds output[I] = f(vals...)
    end

    return
end

function sparse_to_dense_broadcast_kernel(::Type{<:CuSparseVector}, f::F,
                                          output::CuDeviceArray{Tv},
                                          offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                          args...) where {Tv, F, N, Ti}
    # every thread processes an entire row
    row_ix = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    row_ix > length(output) && return
    row_and_ptrs = @inbounds offsets[row_ix]
    row          = @inbounds row_and_ptrs[1]
    arg_ptrs     = @inbounds row_and_ptrs[2]
    vals = ntuple(Val(length(args))) do i
        @inline
        arg = @inbounds args[i]
        # ptr is 0 if the sparse vector doesn't have an element at this row
        # ptr is row if the arg is dense OR a scalar with non-zero-preserving f
        # ptr is 0 if the arg is a scalar AND f preserves zeros
        ptr = @inbounds arg_ptrs[i]
        _getindex(arg, row, ptr)
    end
    @inbounds output[row] = f(vals...)
    return
end
## COV_EXCL_STOP

function Broadcast.copy(bc::Broadcasted{<:Union{CuSparseVecStyle,CuSparseMatStyle}})
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
    sparse_typ <: Union{CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseVector} ||
        error("broadcast with sparse arrays is currently only implemented for vectors and CSR and CSC matrices")
    Ti = if sparse_typ <: CuSparseMatrixCSR
        reduce(promote_type, map(i->eltype(bc.args[i].rowPtr), sparse_args))
    elseif sparse_typ <: CuSparseMatrixCSC
        reduce(promote_type, map(i->eltype(bc.args[i].colPtr), sparse_args))
    elseif sparse_typ <: CuSparseVector
        reduce(promote_type, map(i->eltype(bc.args[i].iPtr), sparse_args))
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
    rows, cols = get(size(bc), 1, 1), get(size(bc), 2, 1)   # `size(bc, ::Int)` is missing
    function compute_launch_config(kernel)
        config = launch_configuration(kernel.fun)
        if sparse_typ <: CuSparseMatrixCSR
            threads = min(rows, config.threads)
            blocks  = max(cld(rows, threads), config.blocks)
            threads = cld(rows, blocks)
        elseif sparse_typ <: CuSparseMatrixCSC
            threads = min(cols, config.threads)
            blocks  = max(cld(cols, threads), config.blocks)
            threads = cld(cols, blocks)
        elseif sparse_typ <: CuSparseVector
            threads = 512
            blocks  = max(cld(rows, threads), config.blocks)
        end
        (; threads, blocks)
    end
    # for CuSparseVec, figure out the actual row range we need to address, e.g. if m = 2^20
    # but the only rows present in any sparse vector input are between 2 and 128, no need to
    # launch massive threads.
    # TODO: use the difference here to set the thread count
    overall_first_row = one(Ti)
    overall_last_row = Ti(rows)
    offsets = nothing
    # allocate the output container
    if !fpreszeros && sparse_typ <: Union{CuSparseMatrixCSR, CuSparseMatrixCSC}
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
    elseif length(sparse_args) == 1 && sparse_typ <: Union{CuSparseMatrixCSR, CuSparseMatrixCSC}
        # we only have a single sparse input, so we can reuse its structure for the output.
        # this avoids a kernel launch and costly synchronization.
        sparse_arg = bc.args[first(sparse_args)]
        if sparse_typ <: CuSparseMatrixCSR
            offsets = rowPtr = sparse_arg.rowPtr
            colVal  = similar(sparse_arg.colVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = CuSparseMatrixCSR(rowPtr, colVal, nzVal, size(bc))
        elseif sparse_typ <: CuSparseMatrixCSC
            offsets = colPtr = sparse_arg.colPtr
            rowVal  = similar(sparse_arg.rowVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = CuSparseMatrixCSC(colPtr, rowVal, nzVal, size(bc))
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
        elseif sparse_typ <: CuSparseVector
            CUDA.@allowscalar begin
                arg_first_rows = ntuple(Val(length(bc.args))) do i
                    bc.args[i] isa CuSparseVector && return bc.args[i].iPtr[1]
                    return one(Ti)
                end
                arg_last_rows = ntuple(Val(length(bc.args))) do i
                    bc.args[i] isa CuSparseVector && return bc.args[i].iPtr[end]
                    return Ti(rows)
                end
            end
            overall_first_row = min(arg_first_rows...)
            overall_last_row  = max(arg_last_rows...)
            CuVector{Pair{Ti, NTuple{length(bc.args), Ti}}}(undef, overall_last_row - overall_first_row + 1)
        end
        let
            args = if sparse_typ <: CuSparseVector
                (sparse_typ, overall_first_row, overall_last_row, fpreszeros, offsets, bc.args...)
            else
                (sparse_typ, offsets, bc.args...)
            end
            kernel = @cuda launch=false compute_offsets_kernel(args...)
            threads, blocks = compute_launch_config(kernel)
            kernel(args...; threads, blocks)
        end
        # accumulate these values so that we can use them directly as row pointer offsets,
        # as well as to get the total nnz count to allocate the sparse output array.
        # cusparseXcsrgeam2Nnz computes this in one go, but it doesn't seem worth the effort
        if !(sparse_typ <: CuSparseVector)
            accumulate!(Base.add_sum, offsets, offsets)
            total_nnz = @allowscalar last(offsets[end]) - 1
        else
            sort!(offsets; by=first)
            total_nnz = mapreduce(x->first(x) != typemax(first(x)), +, offsets)
        end
        output = if sparse_typ <: CuSparseMatrixCSR
            colVal = CuArray{Ti}(undef, total_nnz)
            nzVal  = CuArray{Tv}(undef, total_nnz)
            CuSparseMatrixCSR(offsets, colVal, nzVal, size(bc))
        elseif sparse_typ <: CuSparseMatrixCSC
            rowVal = CuArray{Ti}(undef, total_nnz)
            nzVal  = CuArray{Tv}(undef, total_nnz)
            CuSparseMatrixCSC(offsets, rowVal, nzVal, size(bc))
        elseif sparse_typ <: CuSparseVector && !fpreszeros
            CuArray{Tv}(undef, size(bc))
        elseif sparse_typ <: CuSparseVector && fpreszeros
            iPtr   = CUDA.zeros(Ti, total_nnz)
            nzVal  = CUDA.zeros(Tv, total_nnz)
            CuSparseVector(iPtr, nzVal, rows)
        end
        if sparse_typ <: CuSparseVector && !fpreszeros
            nonsparse_args = map(bc.args) do arg
                # NOTE: this assumes the broadcst is flattened, but not yet preprocessed
                if arg isa AbstractCuSparseArray
                    zero(eltype(arg))
                else
                    arg
                end
            end
            broadcast!(bc.f, output, nonsparse_args...)
        end
    end
    # perform the actual broadcast
    if output isa AbstractCuSparseArray
        args   = (bc.f, output, offsets, bc.args...)
        kernel = @cuda launch=false sparse_to_sparse_broadcast_kernel(args...)
    else
        args   = sparse_typ <: CuSparseVector ? (sparse_typ, bc.f, output, offsets, bc.args...) :
                                                (sparse_typ, bc.f, output, bc.args...)
        kernel = @cuda launch=false sparse_to_dense_broadcast_kernel(args...)
    end
    threads, blocks = compute_launch_config(kernel)
    kernel(args...; threads, blocks)

    return output
end
