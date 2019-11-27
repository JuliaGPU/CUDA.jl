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
    @workspace getWorkspaceSize(args...) do workspace
      useWorkspace(workspace, sizeof(workspace))
    end

Create GPU workspace of type `CuVector{UInt8}` with size in bytes determined by calling
`getWorkspaceSize`, and pass it to the do-block closure for use in calling `useWorkspace`.
Afterwards, the buffer is put back into the memory pool for reuse.

This helper protects against the rare but real issue of `getWorkspaceSize` returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).

Use of this macro should be as physically close as possible to the function that actually
uses the workspace, to minimize the risk of GC interventions between the allocation and use
of the workspace.

If one of the `args` passed to `getWorkspaceSize` is wrapped with the (nonexistent) `output`
function, e.g. `getWorkspaceSize(a, output(b), c)`, this is interpreted as an
dereferenceable output argument that returns the workspace size in stead of the function.
The most common use case is for this to be a `Ref`, for example, `output(Ref{Cint}())`.
There can only be one such argument.

"""
macro workspace(ex)
    # TODO: make the workspacesize a kwarg
    # TODO: support a fallback size, in the case the workspace can't be allocated (for CUTENSOR)
    Meta.isexpr(ex, :do)
    sz = ex.args[1]
    code = ex.args[2]

    # if the workspace getter is a function call that passes a newly constructed Ref{T}()[]
    # this indicates an output argument that returns the worksize.
    if Meta.isexpr(sz, :call)
        # look for an output argument (`output(...)`)
        output_arg = 0
        args = sz.args[2:end]
        for (i,arg) in enumerate(args)
            if Meta.isexpr(arg, :call) && arg.args[1] == :output
                @assert output_arg==0 "@workspace: multiple output arguments detected"
                output_arg = i
            end
        end

        # if we have an output argument, replace the size getter
        if output_arg != 0
            Largs = sz.args[2:output_arg]
            ref_arg = sz.args[output_arg+1].args[2] # strip the call to `output`
            Rargs = sz.args[output_arg+2:end]

            @gensym ref_val
            sz.args[output_arg+1] = ref_val
            sz = quote
                $ref_val = $ref_arg
                $sz
                $ref_val[]
            end
        end
    end

    return quote
        sz = $(esc(sz))
        workspace = nothing
        while workspace === nothing || sizeof(workspace) < sz
            workspace = CuArray{UInt8}(undef, sz)
            sz = $(esc(sz))
        end

        try
          $(esc(code))(workspace)
        finally
          unsafe_free!(workspace)
        end
    end
end
