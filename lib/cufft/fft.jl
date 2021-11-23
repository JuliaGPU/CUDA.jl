# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft,
    Plan, ScaledPlan

using LinearAlgebra

Base.:(*)(p::Plan{T}, x::StridedCuArray) where {T} = p * copy1(T, x)
Base.:(*)(p::ScaledPlan, x::StridedCuArray) = rmul!(p.p * x, p.scale)

# FakeCuArray is used to pass the strides and size information into `create_plan`,
# and avoid having to allocate an actual CuArray in `plan_brfft`, `plan_inv`, etc.
struct FakeCuArray{T,N} <: AbstractArray{T,N}
    sz::NTuple{N,Int}
    st::NTuple{N,Int}
    FakeCuArray{T}(sz::NTuple{N,Int}) where {T,N} =
        new{T,N}(sz, cumprod((1,sz[1:end-1]...)))
end
Base.size(a::FakeCuArray) = a.sz
Base.strides(a::FakeCuArray) = a.st
FakeCuArray{T}(sz...) where T = FakeCuArray{T}(sz)

const CuPlanArray{T,N} = Union{FakeCuArray{T,N}, StridedCuArray{T,N}}


## plan structure

# K is a flag for forward/backward
# also used as an alias for r2c/c2r

abstract type CuFFTPlan{T<:cufftNumber, K, inplace} <: Plan{T} end

# for some reason, cufftHandle is an integer and not a pointer...
Base.convert(::Type{cufftHandle}, p::CuFFTPlan) = p.handle

function CUDA.unsafe_free!(plan::CuFFTPlan, stream::CuStream=stream())
    @context! skip_destroyed=true plan.ctx cufftDestroy(plan)
    unsafe_free!(plan.workarea, stream)
end

unsafe_finalize!(plan::CuFFTPlan) = unsafe_free!(plan, default_stream())

mutable struct cCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    workarea::CuVector{Int8}
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    istride::NTuple{N,Int} # Julia strides of input array
    ostride::NTuple{N,Int} # Julia strides of output array
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, workarea::CuVector{Int8},
                                       X::CuPlanArray{T,N}, Y::CuPlanArray{T,N}, region, xtype;
                                       stream::CuStream=stream()) where {T<:cufftComplexes,K,inplace,N}
        # TODO: enforce consistency of sizey?
        p = new(handle, context(), stream, workarea, size(X), size(Y), strides(X), strides(Y), xtype, region)
        finalizer(unsafe_finalize!, p)
        p
    end
end

mutable struct rCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    workarea::CuVector{Int8}
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    istride::NTuple{N,Int} # Julia strides of input array
    ostride::NTuple{N,Int} # Julia strides of output array
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function rCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, workarea::CuVector{Int8},
                                       X::CuPlanArray{T,N}, Y::CuPlanArray{T2,N}, region, xtype;
                                       stream::CuStream=stream()) where {T<:cufftNumber,T2<:cufftNumber,K,inplace,N}
        # TODO: enforce consistency of sizey?
        p = new(handle, context(), stream, workarea, size(X), size(Y), strides(X), strides(Y), xtype, region)
        finalizer(unsafe_finalize!, p)
        p
    end
end

const xtypenames = Dict{cufftType,String}(CUFFT_R2C => "real-to-complex",
                                          CUFFT_C2R => "complex-to-real",
                                          CUFFT_C2C => "complex",
                                          CUFFT_D2Z => "d.p. real-to-complex",
                                          CUFFT_Z2D => "d.p. complex-to-real",
                                          CUFFT_Z2Z => "d.p. complex")

function showfftdims(io, sz, T)
    if isempty(sz)
        print(io,"0-dimensional")
    elseif length(sz) == 1
        print(io, sz[1], "-element")
    else
        print(io, join(sz, "×"))
    end
    print(io, " CuArray of ", T)
end

function Base.show(io::IO, p::CuFFTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "CUFFT in-place " : "CUFFT ",
          xtypenames[p.xtype],
          K == CUFFT_FORWARD ? " forward" : " backward",
          " plan for ")
    showfftdims(io, p.sz, T)
end

Base.size(p::CuFFTPlan) = p.sz

# FFT plans can be user-created on a different task, whose stream might be different from
# the one used in the current task. call this function before every API call that performs
# operations on a stream to ensure the plan is using the correct task-local stream.
@inline function update_stream(plan::CuFFTPlan)
    new_stream = stream()
    if plan.stream != new_stream
        plan.stream = new_stream
        cufftSetStream(plan, new_stream)

        # replace the workarea by one (asynchronously) allocated on the current stream
        new_workarea = similar(plan.workarea)
        cufftSetWorkArea(plan, new_workarea)
        CUDA.unsafe_free!(plan.workarea)
        plan.workarea = new_workarea
    end
    return
end


## plan methods

# TODO: implement padded storage dimensions

# Only 1D batch is supported.
# For some strided layout, we can "reshape" to reduce the batch dimension
function reduce_howmany!(sz, ist, ost)
    # We found a processable setting
    cion(x) = x |> only |> Cint
    length(sz) == 0 && return (batch = Cint(1), dists = (typemax(Cint), typemax(Cint)))
    length(sz) == 1 && return (batch = cion(sz), dists = (cion(ist), cion(ost)))

    # move the last element to the excluded location, and resize! the vector.
    _reduce!(a,i) = begin
        @inbounds a[i] = a[end]
        resize!(a, length(a) - 1)
    end

    # The following check seems redundant, but i think this won't be the bottleneck
    @inbounds for i in eachindex(sz), j in eachindex(sz)
        if sz[i] .* (ist[i], ost[i]) == (ist[j], ost[j])
            sz[i] *= sz[j]
            _reduce!.(tuple(sz, ist, ost), j)
            return reduce_howmany!(sz, ist, ost)
        end
    end

    throw(ArgumentError("Unreducable batch setting"))
end

# The strides information is represented by:
#   i/ostrides = [i/ostride, i/onembed[end:2]...] |> cumprod
# Thus not all strided layout is supported.
# This function translate the stride information and checks it.
function check_dims(sz, ist, ost)
    # CUFFT only support 1, 2, and 3D tranforms
    0 < length(sz) < 4 || throw(ArgumentError("Length of region must be 1,2,3!"))
    # region = (3,1) is now supported, so reverse is replaced with sort.
    ind = sortperm(ist; rev = true)
    sz, ist, ost = sz[ind], ist[ind], ost[ind]

    # calculate the nembed and check
    _div(x,y) = begin
        x ÷ y * y == x || throw(ArgumentError("Unsupport Layout"))
        x ÷ y
    end
    sts = ist[end], ost[end] # pick the i/ostride

    # i/onembeds[0] is useless, we can fix it to 2147483647
    nembeds = Cint[typemax(Cint), _div.(ist[1:end-1],ist[2:end])...],
              Cint[typemax(Cint), _div.(ost[1:end-1],ost[2:end])...]

    (sz = Cint[sz...], sts = Cint.(sts), nembeds = nembeds)
end

# dims_howmany is named after FFTW’s guru interface.
# The size and i/ostrides of fft region are stored in dims.
# And the batch informations are stored in howmany.
# Dimension with size 1 will be excluded at the first stage.
function dims_howmany(X::CuPlanArray, Y::CuPlanArray, sz, region)
    reg = unique!(Int[region...])
    length(reg) < length(region) && throw(ArgumentError("each dimension can be transformed at most once"))
    ist, ost = [strides(X)...], [strides(Y)...]
    reg = filter!(i -> sz[i] > 1, reg)
    oreg = filter(i -> !in(i, reg) && sz[i] > 1, 1:ndims(X))
    # translate the layout information
    dims = check_dims(sz[reg], ist[reg], ost[reg])
    howmany = reduce_howmany!(sz[oreg], ist[oreg], ost[oreg])
    return dims, howmany
end

# XXX: the strided 2D fft with batch number >= 256 gives wrong result sunder CUDA 10.x.
@inline allowstrided(x) = version() >= v"10.2.1" || throw(ArgumentError("StridedCuArray is not supported for CUDA 10.x"))
@inline allowstrided(::FakeCuArray)  = nothing
@inline allowstrided(::DenseCuArray) = nothing

function create_plan(xtype, X::CuPlanArray, Y::CuPlanArray, region)
    # do the version check at the very beginning
    allowstrided(X)
    allowstrided(Y)
    if xtype in (CUFFT_C2R, CUFFT_Z2D)
        dims, howmany = dims_howmany(X, Y, [size(Y)...], region) # brfft use outputs' size
    else
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
    end
    nrank = length(dims.sz)
    batch, (idist, odist) = howmany.batch, howmany.dists
    sz, (inembed, onembed), (istride, ostride)  = dims.sz, dims.nembeds, dims.sts

    # initialize the plan handle
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    # take control over the workarea
    cufftSetAutoAllocation(handle, 0)
    cufftSetStream(handle, stream())

    # make the plan
    # NOTE: we're only using the advanced API, like FFTW.jl
    worksize_ref = Ref{Csize_t}()
    cufftMakePlanMany(handle, nrank, sz,
                    inembed, istride, idist, onembed, ostride, odist,
                    xtype, batch, worksize_ref)

    # assign the workarea
    workarea = CuArray{Int8}(undef, worksize_ref[])
    cufftSetWorkArea(handle, workarea)

    handle, workarea
end

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::StridedCuArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::StridedCuArray{<:Real}, region) = $pf(complexfloat(x), region)
        $f(x::StridedCuArray{<:Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::StridedCuArray{<:Complex{<:Union{Integer,Rational}}}, region) = $pf(complexfloat(x), region)
    end
end
rfft(x::StridedCuArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(realfloat(x), region)
plan_rfft(x::StridedCuArray{<:Real}, region) = plan_rfft(realfloat(x), region)

# region is an iterable subset of dimensions
# spec. an integer, range, tuple, or array

# inplace complex
function plan_fft!(X::StridedCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = true
    xtype = (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle, workarea = create_plan(xtype, X, X, region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, X, region, xtype)
end

function plan_bfft!(X::StridedCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = true
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle, workarea = create_plan(xtype, X, X, region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, X, region, xtype)
end

# out-of-place complex
function plan_fft(X::StridedCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z
    inplace = false

    Y = FakeCuArray{T}(size(X))
    handle, workarea = create_plan(xtype, X, Y, region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, Y, region, xtype)
end

function plan_bfft(X::StridedCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    Y = FakeCuArray{T}(size(X))
    handle, workarea = create_plan(xtype, X, Y, region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, Y, region, xtype)
end

# out-of-place real-to-complex
function plan_rfft(X::StridedCuArray{T,N}, region) where {T<:cufftReals,N}
    K = CUFFT_FORWARD
    inplace = false
    xtype =  (T == cufftReal) ? CUFFT_R2C : CUFFT_D2Z

    halfdim = minimum(region)
    ydims = collect(size(X))
    ydims[halfdim] = div(ydims[halfdim],2)+1
    Y = FakeCuArray{complex(T)}(ydims...)
    handle, workarea = create_plan(xtype, X, Y, region)

    rCuFFTPlan{T,K,inplace,N}(handle, workarea, X, Y, region, xtype)
end

function plan_brfft(X::StridedCuArray{T,N}, d::Integer, region::Any) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2R : CUFFT_Z2D

    halfdim = minimum(region)
    ydims = collect(size(X))
    ydims[halfdim] = d
    Y = FakeCuArray{real(T)}(ydims...)

    # brfft will break input, a copy is performed before execution,
    # so the istrides should follow the copy's layout
    X′ = FakeCuArray{T}(size(X))

    handle, workarea = create_plan(xtype, X′, Y, region)

    rCuFFTPlan{T,K,inplace,N}(handle, workarea, X′, Y, region, xtype) # istride is useless
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}) where {T,N,inplace}
    X = FakeCuArray{T}(p.sz)
    handle, workarea = create_plan(p.xtype, X, X, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}(handle, workarea, X, X, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}) where {T,N,inplace}
    X = FakeCuArray{T}(p.sz)
    handle, workarea = create_plan(p.xtype, X, X, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}(handle, workarea, X, X, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_INVERSE,inplace,N}
                  ) where {T<:cufftComplexes,N,inplace}
    X = FakeCuArray{real(T)}(p.osz)
    Y = FakeCuArray{T}(p.sz)
    xtype = p.xtype == CUFFT_C2R ? CUFFT_R2C : CUFFT_D2Z
    handle, workarea = create_plan(xtype, X, Y, p.region)
    ScaledPlan(rCuFFTPlan{real(T),CUFFT_FORWARD,inplace,N}(handle, workarea, X, Y, p.region, xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_FORWARD,inplace,N}
                  ) where {T<:cufftReals,N,inplace}
    X = FakeCuArray{complex(T)}(p.osz)
    Y = FakeCuArray{T}(p.sz)
    xtype = p.xtype == CUFFT_R2C ? CUFFT_C2R : CUFFT_Z2D
    handle, workarea = create_plan(xtype, X, Y, p.region)
    ScaledPlan(rCuFFTPlan{complex(T),CUFFT_INVERSE,inplace,N}(handle, workarea, X, Y,
                                                              p.region, xtype),
               normalization(Y, p.region))
end


## plan execution

function assert_applicable(p::CuFFTPlan{T,K}, X::StridedCuArray{T}) where {T,K}
    (size(X) == p.sz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
    p.xtype in (CUFFT_C2R,CUFFT_Z2D) || (strides(X) == p.istride) || #brfft don't need to check the istride
        throw(ArgumentError("CuFFT plan applied to wrong-stride input"))
end

function assert_applicable(p::CuFFTPlan{T,K}, X::StridedCuArray{T}, Y::StridedCuArray{Ty}) where {T,K,Ty}
    assert_applicable(p, X)
    (size(Y) == p.osz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size output"))
    (strides(Y) == p.ostride) ||
        throw(ArgumentError("CuFFT plan applied to wrong-stride output"))
    # type errors should be impossible by dispatch, but just in case:
    if p.xtype ∈ [CUFFT_C2R, CUFFT_Z2D]
        (Ty == real(T)) ||
            throw(ArgumentError("Type mismatch for argument Y"))
    elseif p.xtype ∈ [CUFFT_R2C, CUFFT_D2Z]
        (Ty == complex(T)) ||
            throw(ArgumentError("Type mismatch for argument Y"))
    else
        (Ty == T) ||
            throw(ArgumentError("Type mismatch for argument Y"))
    end
end

function unsafe_execute!(plan::cCuFFTPlan{cufftComplex,K,true,N},
                         x::StridedCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    update_stream(plan)
    cufftExecC2C(plan, x, x, K)
end

# XXX: inplace brfft will destroy the input for strided layout; even though `plan_brfft!`
#      is currently not implemented, we force the input to be dense.
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,true,N},
                         x::DenseCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    update_stream(plan)
    cufftExecC2R(plan, x, x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftComplex,K,false,N},
                         x::StridedCuArray{cufftComplex,N}, y::StridedCuArray{cufftComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    update_stream(plan)
    cufftExecC2C(plan, x, y, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
                         x::StridedCuArray{cufftComplex,N}, y::StridedCuArray{cufftReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecC2R(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,false,N},
                         x::StridedCuArray{cufftReal,N}, y::StridedCuArray{cufftComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_R2C
    update_stream(plan)
    cufftExecR2C(plan, x, y)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::StridedCuArray{cufftDoubleComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    cufftExecZ2Z(plan, x, x, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::DenseCuArray{cufftDoubleComplex,N}) where {K,N}
    update_stream(plan)
    @assert plan.xtype == CUFFT_Z2D
    cufftExecZ2D(plan, x, x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::StridedCuArray{cufftDoubleComplex,N}, y::StridedCuArray{cufftDoubleComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    update_stream(plan)
    cufftExecZ2Z(plan, x, y, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::StridedCuArray{cufftDoubleComplex,N}, y::StridedCuArray{cufftDoubleReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2D
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecZ2D(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleReal,K,false,N},
                         x::StridedCuArray{cufftDoubleReal,N}, y::StridedCuArray{cufftDoubleComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_D2Z
    update_stream(plan)
    cufftExecD2Z(plan, x, y)
end

function LinearAlgebra.mul!(y::StridedCuArray{Ty}, p::CuFFTPlan{T,K,false}, x::StridedCuArray{T}
                           ) where {Ty,T,K}
    assert_applicable(p,x,y)
    unsafe_execute!(p,x,y)
    return y
end

function Base.:(*)(p::cCuFFTPlan{T,K,true,N}, x::StridedCuArray{T,N}) where {T,K,N}
    assert_applicable(p,x)
    unsafe_execute!(p,x)
    x
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_FORWARD,false,N}, x::StridedCuArray{T,N}
           ) where {T<:cufftReals,N}
    @assert p.xtype ∈ [CUFFT_R2C,CUFFT_D2Z]
    y = CuArray{complex(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_INVERSE,false,N}, x::StridedCuArray{T,N}
           ) where {T<:cufftComplexes,N}
    @assert p.xtype ∈ [CUFFT_C2R,CUFFT_Z2D]
    y = CuArray{real(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function Base.:(*)(p::cCuFFTPlan{T,K,false,N}, x::StridedCuArray{T,N}) where {T,K,N}
    y = CuArray{T,N}(undef, p.osz)
    mul!(y,p,x)
    y
end
