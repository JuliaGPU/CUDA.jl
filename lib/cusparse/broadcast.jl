using Base.Broadcast: Broadcasted

using CUDA: CuArrayStyle


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


## sparse broadcast implementation

function Broadcast.copy(bc::Broadcasted{<:Union{CuSparseVecStyle,CuSparseMatStyle}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if !Base.isconcretetype(ElType)
        error("""GPU sparse broadcast resulted in non-concrete element type $ElType.
                 This probably means that the function you are broadcasting contains an error or type instability.""")
    end

    # we only support broadcast involving a single sparse array
    bc = Broadcast.flatten(bc)
    sparse_args = findall(bc.args) do arg
        arg isa CUSPARSE.AbstractCuSparseArray
    end
    if length(sparse_args) != 1
        error("broadcast with multiple sparse arguments not supported")
    end
    sparse_arg = sparse_args[1]
    # XXX: can we handle multiple sparse arguments? problem is that a kernel then can't
    # simply index one of those and use the same indices for the other sparse inputs. we
    # could equalize the sparse inputs first, making sure they have values at the maximal
    # set of indices, but I'm not sure how to implement that operation either.

    # partially-evaluate the function, removing scalars
    parevalf, passedsrcargstup = capturescalars(bc.f, bc.args)
    # if all we have left is sparse arrays, we can check if the partially-evaluated function
    # preserves zeros. if so, we'll only need to apply it to the sparse input arguments.
    if all(arg->isa(arg, AbstractSparseArray), passedsrcargstup)
        fofzeros = parevalf(_zeros_eltypes(passedsrcargstup...)...)
        fpreszeros = _iszero(fofzeros)
    else
        fpreszeros = false
    end
    dest = if fpreszeros
        similar(bc.args[sparse_arg], ElType)
    else
        # either we have dense inputs, or the function isn't preserving zeros,
        # so use a dense output to broadcast into.
        CuArray{ElType}(undef, size(bc.args[sparse_arg]))
    end

    _copyto!(dest, bc, sparse_arg)

    # TODO: if we had dense inputs, but the function was preserving zeros,
    #       try to re-sparsify the output?
end

# TODO
# function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:CuSparseMatStyle})

# version of copyto! that deals with a single sparse argument
function _copyto!(dest, bc, idx)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    # the _copyto! implementations below only process elements of our sparse input array.
    if isa(dest, AbstractSparseArray)
        # if we're writing to a sparse output, we'll be using the exact same indices
        # as from the input. this requires the layout of the sparse array to be identical,
        # which is too costly to check at run time, so just compare the nnz elements.
        nnz(dest) == nnz(bc.args[idx]) ||
          error("Destination of sparse broadcast should have identical layout")
    else
        # if we're broadcasting to a dense output -- likely because the function isn't
        # zero-preserving -- we first broadcast to fill the elements not in our sparse array
        # by setting those to zero and re-using the dense broadcast implementation
        nonsparse_args = map(bc.args) do arg
            # NOTE: this assumes the broadcst is flattened, but not yet preprocessed
            if arg isa CUSPARSE.AbstractCuSparseArray
                zero(eltype(arg))
            else
                arg
            end
        end
        broadcast!(bc.f, dest, nonsparse_args...)
    end

    bc′ = Broadcast.preprocess(dest, bc)
    sparse_arg = bc′.args[idx]
    other_args = [bc′.args[begin:idx-1]..., bc′.args[idx+1:end]...]
    _copyto!(typeof(bc.args[idx]), dest, bc′.f, sparse_arg, idx, other_args)
end
_copyto!(T::Type, dest, f, sparse_arg, idx, other_args) =
    error("Broadcast with sparse array of type $T not implemented")

# TODO: broadcast sparse vector with something 2d?

function _copyto!(::Type{<:CuSparseVector}, dest, f, sparse_arg, idx, other_args)
    function kernel(dest, f, arg, ::Val{idx}, other_args...) where idx
        # every thread processes a single element
        i = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
        if i > nnz(arg)
            return
        end

        # @inbounds begin
            row = arg.iPtr[i]

            # get the argument values at this index
            val = arg.nzVal[i]
            other_vals = Broadcast._getindex(other_args, row)

            # apply the broadcast function
            output = Broadcast._broadcast_getindex_evalf(f, other_vals[begin:idx-1]...,
                                                         val, other_vals[idx:end]...)

            # store the result
            if dest isa AbstractSparseVector
              dest.nzVal[i] = output
            else
              dest[row] = output
            end
        # end
        return
    end

    cols = nnz(sparse_arg)
    kernel = @cuda launch=false kernel(dest, f, sparse_arg, Val(idx), other_args...)
    config = launch_configuration(kernel.fun)
    threads = min(cols, config.threads)
    blocks = cld(cols, threads)
    kernel(dest, f, sparse_arg, Val(idx), other_args...; threads, blocks)

    return dest
end

function _copyto!(::Type{<:CuSparseMatrixCSC}, dest, f, sparse_arg, idx, other_args)
    function kernel(dest, f, arg, ::Val{idx}, other_args...) where idx
        # every thread processes an entire column
        col = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
        if col >= length(arg.colPtr)
            return
        end

        # TODO: parallelize across these as well? uncertain, but may yield improvements
        @inbounds for i in arg.colPtr[col]:arg.colPtr[col+1]-1
            row = arg.rowVal[i]
            I = CartesianIndex(row, col)

            # get the argument values at this index
            val = arg.nzVal[i]
            other_vals = Broadcast._getindex(other_args, I)

            # apply the broadcast function
            output = Broadcast._broadcast_getindex_evalf(f, other_vals[begin:idx-1]...,
                                                         val, other_vals[idx:end]...)

            # store the result
            if dest isa AbstractSparseMatrix
              dest.nzVal[i] = output
            else
              dest[I] = output
            end
        end
        return
    end

    cols = length(sparse_arg.colPtr)-1
    kernel = @cuda launch=false kernel(dest, f, sparse_arg, Val(idx), other_args...)
    config = launch_configuration(kernel.fun)
    threads = min(cols, config.threads)
    blocks = cld(cols, threads)
    kernel(dest, f, sparse_arg, Val(idx), other_args...; threads, blocks)

    return dest
end

function _copyto!(::Type{<:CuSparseMatrixCSR}, dest, f, sparse_arg, idx, other_args)
    function kernel(dest, f, arg, ::Val{idx}, other_args...) where idx
        # every thread processes an entire row
        row = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
        if row >= length(arg.rowPtr)
            return
        end

        # TODO: parallelize across these as well? uncertain, but may yield improvements
        @inbounds for i in arg.rowPtr[row]:arg.rowPtr[row+1]-1
            col = arg.colVal[i]
            I = CartesianIndex(row, col)

            # get the argument values at this index
            val = arg.nzVal[i]
            other_vals = Broadcast._getindex(other_args, I)

            # apply the broadcast function
            output = Broadcast._broadcast_getindex_evalf(f, other_vals[begin:idx-1]...,
                                                         val, other_vals[idx:end]...)

            # store the result
            if dest isa AbstractSparseMatrix
              dest.nzVal[i] = output
            else
              dest[I] = output
            end
        end
        return
    end

    rows = length(sparse_arg.rowPtr)-1
    kernel = @cuda launch=false kernel(dest, f, sparse_arg, Val(idx), other_args...)
    config = launch_configuration(kernel.fun)
    threads = min(rows, config.threads)
    blocks = cld(rows, threads)
    kernel(dest, f, sparse_arg, Val(idx), other_args...; threads, blocks)

    return dest
end

function _copyto!(::Type{<:CuSparseMatrixCOO}, dest, f, sparse_arg, idx, other_args)
    function kernel(dest, f, arg, ::Val{idx}, other_args...) where idx
        # every thread processes a single element
        i = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
        if i > nnz(arg)
            return
        end

        @inbounds begin
            row = arg.rowInd[i]
            col = arg.colInd[i]
            I = CartesianIndex(row, col)

            # get the argument values at this index
            val = arg.nzVal[i]
            other_vals = Broadcast._getindex(other_args, I)

            # apply the broadcast function
            output = Broadcast._broadcast_getindex_evalf(f, other_vals[begin:idx-1]...,
                                                         val, other_vals[idx:end]...)

            # store the result
            if dest isa AbstractSparseMatrix
              dest.nzVal[i] = output
            else
              dest[I] = output
            end
        end
        return
    end

    nels = nnz(sparse_arg)
    kernel = @cuda launch=false kernel(dest, f, sparse_arg, Val(idx), other_args...)
    config = launch_configuration(kernel.fun)
    threads = min(nels, config.threads)
    blocks = cld(nels, threads)
    kernel(dest, f, sparse_arg, Val(idx), other_args...; threads, blocks)

    return dest
end
