using Base.Cartesian

function cudims(n::Integer)
  threads = min(n, 256)
  ceil(Int, n / threads), threads
end

cudims(a::AbstractArray) = cudims(length(a))

@inline ind2sub_(a::AbstractArray{T,0}, i) where T = ()
@inline ind2sub_(a, i) = Tuple(CartesianIndices(a)[i])

macro cuindex(A)
  quote
    A = $(esc(A))
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(A) && return
    ind2sub_(A, i)
  end
end


@generated function nindex(i::T, ls::NTuple{N,T}) where {N,T}
  na = one(i)
  quote
    Base.@_inline_meta
    $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :($na, $na), one(i):i(N)))
  end
end

@inline function catindex(dim, I::NTuple{N}, shapes) where N
  @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
end

function growdims(dim, x)
  if ndims(x) >= dim
    x
  else
    reshape(x, size.((x,), 1:dim)...)
  end
end

function _cat(dim, dest, xs...)
  function kernel(dim, dest, xs)
    I = @cuindex dest
    @inbounds n, I′ = catindex(dim, Int.(I), size.(xs))
    @inbounds dest[I...] = xs[n][I′...]
    return
  end
  xs = growdims.(dim, xs)
  blk, thr = cudims(dest)
  @cuda blocks=blk threads=thr kernel(dim, dest, xs)
  return dest
end

function Base.cat_t(dims::Integer, T::Type, x::CuArray, xs::CuArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(xs..., dims=1)
Base.hcat(xs::CuArray...) = cat(xs..., dims=2)


"""
    @sync ex

Run expression `ex` and synchronize the GPU afterwards. This is a CPU-friendly
synchronization, i.e. it performs a blocking synchronization without increasing CPU load. As
such, this operation is preferred over implicit synchronization (e.g. when performing a
memory copy) for high-performance applications.

It is also useful for timing code that executes asynchronously.
"""
macro sync(ex)
    quote
        local e = CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC | CUDAdrv.EVENT_DISABLE_TIMING)
        local ret = $(esc(ex))
        CUDAdrv.record(e)
        CUDAdrv.synchronize(e)
        ret
    end
end

"""
    @argout call(..., out(output_arg), ...)

Change the behavior of a function call, returning the value of an argument instead.
A common use case is to pass a newly-created Ref and immediately dereference that output:

    @argout(some_getter(Ref{Int}()))[]

"""
macro argout(ex)
    Meta.isexpr(ex, :call) || throw(ArgumentError("@argout macro should be applied to a function call"))

    # look for an output argument (`out(...)`)
    output_arg = 0
    args = ex.args[2:end]
    for (i,arg) in enumerate(args)
        if Meta.isexpr(arg, :call) && arg.args[1] == :out
            output_arg == 0 || throw(ArgumentError("There can only be one output argument (both argument $output_arg and $i are marked out)"))
            output_arg = i
        end
    end
    output_arg == 0 && throw(ArgumentError("No output argument found"))

    # get the arguments
    Largs = ex.args[2:output_arg]
    ret_arg = ex.args[output_arg+1].args[2] # strip the call to `out`
    Rargs = ex.args[output_arg+2:end]

    @gensym ret_val
    ex.args[output_arg+1] = ret_val
    esc(quote
        $ret_val = $ret_arg
        $ex
        $ret_val
    end)
end

"""
    @workspace size=getWorkspaceSize(args...) [eltyp=UInt8] buffer -> begin
      useWorkspace(workspace, sizeof(workspace))
    end

Create a GPU workspace vector with element type `eltyp` and size in number of elements (in
the default case of an UInt8 element type this equals to the amount of bytes) determined by
calling `getWorkspaceSize`, and pass it to the  closure for use in calling `useWorkspace`.
Afterwards, the buffer is put back into the memory pool for reuse.

This helper protects against the rare but real issue of `getWorkspaceSize` returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).

Use of this macro should be as physically close as possible to the function that actually
uses the workspace, to minimize the risk of GC interventions between the allocation and use
of the workspace.

"""
macro workspace(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    sz = nothing
    eltyp = :UInt8
    for kwarg in kwargs
        key,val = kwarg.args
        if key == :size
            sz = val
        elseif key == :eltyp
            eltyp = val
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    # TODO: support a fallback size, in the case the workspace can't be allocated (for CUTENSOR)
    if sz === nothing
        throw(ArgumentError("@workspace macro needs a size argument"))
    end

    return quote
        sz = $(esc(sz))
        workspace = nothing
        while workspace === nothing || length(workspace) < sz
            workspace = CuArray{$(esc(eltyp))}(undef, sz)
            sz = $(esc(sz))
        end

        $(esc(code))(workspace)
    end
end
