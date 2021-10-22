# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft,
    Plan, ScaledPlan

using LinearAlgebra

Base.:(*)(p::Plan{T}, x::DenseCuArray) where {T} = p * copy1(T, x)
Base.:(*)(p::ScaledPlan, x::DenseCuArray) = rmul!(p.p * x, p.scale)


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
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, workarea::CuVector{Int8},
                                       X::DenseCuArray{T,N}, sizey::Tuple, region, xtype;
                                       stream::CuStream=stream()) where {T<:cufftNumber,K,inplace,N}
        # maybe enforce consistency of sizey
        p = new(handle, context(), stream, workarea, size(X), sizey, xtype, region)
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
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function rCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, workarea::CuVector{Int8},
                                       X::DenseCuArray{T,N}, sizey::Tuple, region, xtype;
                                       stream::CuStream=stream()) where {T<:cufftNumber,K,inplace,N}
        # maybe enforce consistency of sizey
        p = new(handle, context(), stream, workarea, size(X), sizey, xtype, region)
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

# Note: we don't implement padded storage dimensions
function create_plan(xtype, xdims, region)
    nrank = length(region)
    sz = [xdims[i] for i in region]
    csz = copy(sz)
    csz[1] = div(sz[1],2) + 1
    batch = prod(xdims) ÷ prod(sz)

    # initialize the plan handle
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    # take control over the workarea
    cufftSetAutoAllocation(handle, 0)
    cufftSetStream(handle, stream())

    # make the plan
    worksize_ref = Ref{Csize_t}()
    if (nrank == 1) && (batch == 1)
        cufftMakePlan1d(handle, sz[1], xtype, 1, worksize_ref)
    elseif (nrank == 2) && (batch == 1)
        cufftMakePlan2d(handle, sz[2], sz[1], xtype, worksize_ref)
    elseif (nrank == 3) && (batch == 1)
        cufftMakePlan3d(handle, sz[3], sz[2], sz[1], xtype, worksize_ref)
    else
        rsz = (length(sz) > 1) ? rsz = reverse(sz) : sz
        if ((region...,) == ((1:nrank)...,))
            # handle simple case ... simply! (for robustness)
           cufftMakePlanMany(handle, nrank, Cint[rsz...], C_NULL, 1, 1, C_NULL, 1, 1,
                             xtype, batch, worksize_ref)
        else
            if nrank==1 || all(diff(collect(region)) .== 1)
                # _stride: successive elements in innermost dimension
                # _dist: distance between first elements of batches
                if region[1] == 1
                    istride = 1
                    idist = prod(sz)
                    cdist = prod(csz)
                else
                    if region[end] != length(xdims)
                        throw(ArgumentError("batching dims must be sequential"))
                    end
                    istride = prod(xdims[1:region[1]-1])
                    idist = 1
                    cdist = 1
                end
                inembed = Cint[rsz...]
                cnembed = (length(csz) > 1) ? Cint[reverse(csz)...] : Cint[csz[1]]
                ostride = istride
                if xtype == CUFFT_R2C || xtype == CUFFT_D2Z
                    odist = cdist
                    onembed = cnembed
                else
                    odist = idist
                    onembed = inembed
                end
                if xtype == CUFFT_C2R || xtype == CUFFT_Z2D
                    idist = cdist
                    inembed = cnembed
                end
            else
                if any(diff(collect(region)) .< 1)
                    throw(ArgumentError("region must be an increasing sequence"))
                end
                cdims = collect(xdims)
                cdims[region[1]] = div(cdims[region[1]],2)+1

                if region[1] == 1
                    istride = 1
                    ii=1
                    while (ii < nrank) && (region[ii] == region[ii+1]-1)
                        ii += 1
                    end
                    idist = prod(xdims[1:ii])
                    cdist = prod(cdims[1:ii])
                    ngaps = 0
                else
                    istride = prod(xdims[1:region[1]-1])
                    idist = 1
                    cdist = 1
                    ngaps = 1
                end
                nem = ones(Int,nrank)
                cem = ones(Int,nrank)
                id = 1
                for ii=1:nrank-1
                    if region[ii+1] > region[ii]+1
                        ngaps += 1
                    end
                    while id < region[ii+1]
                        nem[ii] *= xdims[id]
                        cem[ii] *= cdims[id]
                        id += 1
                    end
                    @assert nem[ii] >= sz[ii]
                end
                if region[end] < length(xdims)
                    ngaps += 1
                end
                # CUFFT represents batches by a single stride (_dist)
                # so we must verify that region is consistent with this:
                if ngaps > 1
                    throw(ArgumentError("batch regions must be sequential"))
                end

                inembed = Cint[reverse(nem)...]
                cnembed = Cint[reverse(cem)...]
                ostride = istride
                if xtype == CUFFT_R2C || xtype == CUFFT_D2Z
                    odist = cdist
                    onembed = cnembed
                else
                    odist = idist
                    onembed = inembed
                end
                if xtype == CUFFT_C2R || xtype == CUFFT_Z2D
                    idist = cdist
                    inembed = cnembed
                end
            end
            cufftMakePlanMany(handle, nrank, Cint[rsz...],
                              inembed, istride, idist, onembed, ostride, odist,
                              xtype, batch, worksize_ref)
        end
    end

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
        $f(x::DenseCuArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::DenseCuArray{<:Real}, region) = $pf(complexfloat(x), region)
        $f(x::DenseCuArray{<:Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::DenseCuArray{<:Complex{<:Union{Integer,Rational}}}, region) = $pf(complexfloat(x), region)
    end
end
rfft(x::DenseCuArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(realfloat(x), region)
plan_rfft(x::DenseCuArray{<:Real}, region) = plan_rfft(realfloat(x), region)

# region is an iterable subset of dimensions
# spec. an integer, range, tuple, or array

# inplace complex
function plan_fft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = true
    xtype = (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle, workarea = create_plan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, size(X), region, xtype)
end

function plan_bfft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = true
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle, workarea = create_plan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, size(X), region, xtype)
end

# out-of-place complex
function plan_fft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z
    inplace = false

    handle, workarea = create_plan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, size(X), region, xtype)
end

function plan_bfft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle, workarea = create_plan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, workarea, X, size(X), region, xtype)
end

# out-of-place real-to-complex
function plan_rfft(X::DenseCuArray{T,N}, region) where {T<:cufftReals,N}
    K = CUFFT_FORWARD
    inplace = false
    xtype =  (T == cufftReal) ? CUFFT_R2C : CUFFT_D2Z

    handle, workarea = create_plan(xtype, size(X), region)

    ydims = collect(size(X))
    ydims[region[1]] = div(ydims[region[1]],2)+1

    rCuFFTPlan{T,K,inplace,N}(handle, workarea, X, (ydims...,), region, xtype)
end

function plan_brfft(X::DenseCuArray{T,N}, d::Integer, region::Any) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2R : CUFFT_Z2D
    ydims = collect(size(X))
    ydims[region[1]] = d

    handle, workarea = create_plan(xtype, (ydims...,), region)

    rCuFFTPlan{T,K,inplace,N}(handle, workarea, X, (ydims...,), region, xtype)
end

# FIXME: plan_inv methods allocate needlessly (to provide type parameters)
# Perhaps use FakeArray types to avoid this.

function plan_inv(p::cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    handle, workarea = create_plan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}(handle, workarea, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    handle, workarea = create_plan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}(handle, workarea, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_INVERSE,inplace,N}
                  ) where {T<:cufftComplexes,N,inplace}
    X = CuArray{real(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_C2R ? CUFFT_R2C : CUFFT_D2Z
    handle, workarea = create_plan(xtype, p.osz, p.region)
    ScaledPlan(rCuFFTPlan{real(T),CUFFT_FORWARD,inplace,N}(handle, workarea, X, p.sz, p.region, xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_FORWARD,inplace,N}
                  ) where {T<:cufftReals,N,inplace}
    X = CuArray{complex(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_R2C ? CUFFT_C2R : CUFFT_Z2D
    handle, workarea = create_plan(xtype, p.sz, p.region)
    ScaledPlan(rCuFFTPlan{complex(T),CUFFT_INVERSE,inplace,N}(handle, workarea, X, p.sz,
                                                              p.region, xtype),
               normalization(Y, p.region))
end


## plan execution

function assert_applicable(p::CuFFTPlan{T,K}, X::DenseCuArray{T}) where {T,K}
    (size(X) == p.sz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,K}, X::DenseCuArray{T}, Y::DenseCuArray{Ty}) where {T,K,Ty}
    assert_applicable(p, X)
    (size(Y) == p.osz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size output"))
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
                         x::DenseCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    update_stream(plan)
    cufftExecC2C(plan, x, x, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,true,N},
                         x::DenseCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    update_stream(plan)
    cufftExecC2R(plan, x, x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftComplex,K,false,N},
                         x::DenseCuArray{cufftComplex,N}, y::DenseCuArray{cufftComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecC2C(plan, x, y, K)
    unsafe_free!(x)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
                         x::DenseCuArray{cufftComplex,N}, y::DenseCuArray{cufftReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecC2R(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,false,N},
                         x::DenseCuArray{cufftReal,N}, y::DenseCuArray{cufftComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_R2C
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecR2C(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::DenseCuArray{cufftDoubleComplex,N}) where {K,N}
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
                         x::DenseCuArray{cufftDoubleComplex,N}, y::DenseCuArray{cufftDoubleComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecZ2Z(plan, x, y, K)
    unsafe_free!(x)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::DenseCuArray{cufftDoubleComplex,N}, y::DenseCuArray{cufftDoubleReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2D
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecZ2D(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleReal,K,false,N},
                         x::DenseCuArray{cufftDoubleReal,N}, y::DenseCuArray{cufftDoubleComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_D2Z
    x = copy(x)  # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055
    update_stream(plan)
    cufftExecD2Z(plan, x, y)
    unsafe_free!(x)
end

function LinearAlgebra.mul!(y::DenseCuArray{Ty}, p::CuFFTPlan{T,K,false}, x::DenseCuArray{T}
                           ) where {Ty,T,K}
    assert_applicable(p,x,y)
    unsafe_execute!(p,x,y)
    return y
end

function Base.:(*)(p::cCuFFTPlan{T,K,true,N}, x::DenseCuArray{T,N}) where {T,K,N}
    assert_applicable(p,x)
    unsafe_execute!(p,x)
    x
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_FORWARD,false,N}, x::DenseCuArray{T,N}
           ) where {T<:cufftReals,N}
    @assert p.xtype ∈ [CUFFT_R2C,CUFFT_D2Z]
    y = CuArray{complex(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_INVERSE,false,N}, x::DenseCuArray{T,N}
           ) where {T<:cufftComplexes,N}
    @assert p.xtype ∈ [CUFFT_C2R,CUFFT_Z2D]
    y = CuArray{real(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function Base.:(*)(p::cCuFFTPlan{T,K,false,N}, x::DenseCuArray{T,N}) where {T,K,N}
    y = CuArray{T,N}(undef, p.osz)
    mul!(y,p,x)
    y
end
