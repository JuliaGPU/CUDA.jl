# Note: we don't implement padded storage dimensions
function _mkplan(xtype, xdims, region)
    nrank = length(region)
    sz = [xdims[i] for i in region]
    csz = copy(sz)
    csz[1] = div(sz[1],2) + 1
    batch = prod(xdims) ÷ prod(sz)

    pp = Ref{cufftHandle_t}()
    if (nrank == 1) && (batch == 1)
        @check ccall((:cufftPlan1d,libcufft),cufftStatus_t,
                     (Ptr{cufftHandle_t}, Cint, cufftType, Cint),
                     pp, sz[1], xtype, 1)
    elseif (nrank == 2) && (batch == 1)
        @check ccall((:cufftPlan2d,libcufft),cufftStatus_t,
                     (Ptr{cufftHandle_t}, Cint, Cint, cufftType),
                     pp, sz[2], sz[1], xtype)
    elseif (nrank == 3) && (batch == 1)
        @check ccall((:cufftPlan3d,libcufft),cufftStatus_t,
                     (Ptr{cufftHandle_t}, Cint, Cint, Cint, cufftType),
                     pp, sz[3], sz[2], sz[1], xtype)

    else
        rsz = (length(sz) > 1) ? rsz = reverse(sz) : sz
        if ((region...,) == ((1:nrank)...,))
            # handle simple case ... simply! (for robustness)
            @check ccall((:cufftPlanMany,libcufft),cufftStatus_t,
                         (Ptr{cufftHandle_t}, Cint, Ptr{Cint}, # rank, dims
                          Ptr{Cint}, Cint, Cint, # nembed,stride,dist (input)
                          Ptr{Cint}, Cint, Cint, # nembed,stride,dist (output)
                          cufftType, Cint),
                         pp, nrank, Cint[rsz...], C_NULL, 1, 1, C_NULL, 1, 1,
                         xtype, batch)
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
            @check ccall((:cufftPlanMany,libcufft),cufftStatus_t,
                         (Ptr{cufftHandle_t}, Cint, Ptr{Cint}, # rank, dims
                          Ptr{Cint}, Cint, Cint, # nembed,stride,dist (input)
                          Ptr{Cint}, Cint, Cint, # nembed,stride,dist (output)
                          cufftType, Cint),
                         pp, nrank, Cint[rsz...],
                         inembed, istride, idist, onembed, ostride, odist,
                         xtype, batch)
        end
    end
    pp[]
end

# this is used implicitly in the unsafe_execute methods below:
unsafe_convert(::Type{cufftHandle_t}, p::CuFFTPlan) = p.plan

convert(::Type{cufftHandle_t}, p::CuFFTPlan) = p.plan

destroy_plan(plan::CuFFTPlan) =
    ccall((:cufftDestroy,libcufft), Nothing, (cufftHandle_t,), plan.plan)

function assert_applicable(p::CuFFTPlan{T,K}, X::CuArray{T}) where {T,K}
    (size(X) == p.sz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,K}, X::CuArray{T}, Y::CuArray{Ty}) where {T,K,Ty}
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
                         x::CuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    @check ccall((:cufftExecC2C,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftComplex},
                  Cint),
                 plan, x, x, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,true,N},
                         x::CuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    @check ccall((:cufftExecC2R,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftComplex}),
                 plan, x, x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftComplex,K,false,N},
                         x::CuArray{cufftComplex,N}, y::CuArray{cufftComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    @check ccall((:cufftExecC2C,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftComplex}, Cint),
                 plan, x, y, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
                         x::CuArray{cufftComplex,N}, y::CuArray{cufftReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    @check ccall((:cufftExecC2R,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftComplex}, CuPtr{cufftReal}),
                 plan, x, y)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,false,N},
                         x::CuArray{cufftReal,N}, y::CuArray{cufftComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_R2C
    @check ccall((:cufftExecR2C,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftReal}, CuPtr{cufftComplex}),
                 plan, x, y)
end

# double prec.
function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::CuArray{cufftDoubleComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    @check ccall((:cufftExecZ2Z,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex},
                  Cint),
                 plan, x, x, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::CuArray{cufftDoubleComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_Z2D
    @check ccall((:cufftExecZ2D,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}),
                 plan, x, x)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::CuArray{cufftDoubleComplex,N}, y::CuArray{cufftDoubleComplex}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    @check ccall((:cufftExecZ2Z,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleComplex}, Cint),
                 plan, x, y, K)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::CuArray{cufftDoubleComplex,N}, y::CuArray{cufftDoubleReal}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_Z2D
    @check ccall((:cufftExecZ2D,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleComplex}, CuPtr{cufftDoubleReal}),
                 plan, x, y)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleReal,K,false,N},
                         x::CuArray{cufftDoubleReal,N}, y::CuArray{cufftDoubleComplex,N}
                         ) where {K,N}
    @assert plan.xtype == CUFFT_D2Z
    @check ccall((:cufftExecD2Z,libcufft), cufftStatus_t,
                 (cufftHandle_t, CuPtr{cufftDoubleReal}, CuPtr{cufftDoubleComplex}),
                 plan, x, y)
end
