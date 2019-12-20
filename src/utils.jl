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

If no output argument is specified, `nothing` will be returned.
Multiple output arguments return a tuple.

"""
macro argout(ex)
    Meta.isexpr(ex, :call) || throw(ArgumentError("@argout macro should be applied to a function call"))

    block = quote end

    # look for output arguments (`out(...)`)
    output_vars = []
    args = ex.args[2:end]
    for (i,arg) in enumerate(args)
        if Meta.isexpr(arg, :call) && arg.args[1] == :out
            # allocate a variable
            @gensym output_val
            push!(block.args, :($output_val = $(ex.args[i+1].args[2]))) # strip `output(...)`
            push!(output_vars, output_val)

            # replace the argument
            ex.args[i+1] = output_val
        end
    end

    # generate a return
    push!(block.args, ex)
    if isempty(output_vars)
        push!(block.args, :(nothing))
    elseif length(output_vars) == 1
        push!(block.args, :($(output_vars[1])))
    else
        push!(block.args, :(tuple($(output_vars...))))
    end

    esc(block)
end

"""
    @workspace size=getWorkspaceSize(args...) [eltyp=UInt8] [fallback=nothing] buffer->begin
      useWorkspace(workspace, sizeof(workspace))
    end

Create a GPU workspace vector with element type `eltyp` and size in number of elements (in
the default case of an UInt8 element type this equals to the amount of bytes) determined by
calling `getWorkspaceSize`, and pass it to the closure for use in calling `useWorkspace`.
A fallback workspace size `fallback` can be specified if the regular size would lead to OOM.
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
    fallback = nothing
    for kwarg in kwargs
        key,val = kwarg.args
        if key == :size
            sz = val
        elseif key == :eltyp
            eltyp = val
        elseif key == :fallback
            fallback = val
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    if sz === nothing
        throw(ArgumentError("@workspace macro needs a size argument"))
    end

    # break down the closure to a let block to prevent JuliaLang/julia#15276
    Meta.isexpr(code, :(->)) || throw(ArgumentError("@workspace macro should be applied to a closure"))
    length(code.args) == 2 || throw(ArgumentError("@workspace closure should take exactly one argument"))
    code_arg = code.args[1]
    code = code.args[2]

    return quote
        sz = $(esc(sz))
        workspace = nothing
        try
          while workspace === nothing || length(workspace) < sz
              workspace = CuArray{$(esc(eltyp))}(undef, sz)
              sz = $(esc(sz))
          end
        catch ex
            $fallback === nothing && rethrow()
            isa(ex, OutOfGPUMemoryError) || rethrow()
            workspace = CuArray{UInt8}(undef, $fallback)
        end

        let $(esc(code_arg)) = workspace
            $(esc(code))
        end
    end
end

if VERSION <= v"1.1"
  # JuliaLang/julia#30187-like functionality, but only for CuContext dicts to avoid clashes
  function Base.get!(default::Base.Callable, d::IdDict{CuContext,V}, @nospecialize(key)) where {V}
    if !haskey(d, key)
      d[key] = default()
    end
    return d[key]
  end
end
