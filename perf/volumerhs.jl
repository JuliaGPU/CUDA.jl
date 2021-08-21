module VolumeRHS

using BenchmarkTools
using CUDA
using StableRNGs
using StaticArrays

function loopinfo(name, expr, nodes...)
    if expr.head != :for
        error("Syntax error: pragma $name needs a for loop")
    end
    push!(expr.args[2].args, Expr(:loopinfo, nodes...))
    return expr
end

macro unroll(expr)
    expr = loopinfo("@unroll", expr, (Symbol("llvm.loop.unroll.full"),))
    return esc(expr)
end

# HACK: module-local versions of core arithmetic; needed to get FMA
for (jlf, f) in zip((:+, :*, :-), (:add, :mul, :sub))
    for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
        ir = """
            %x = f$f contract nsz $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.@pure function $jlf(a::$T, b::$T)
                Base.@_inline_meta
                Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
            end
        end
    end
    @eval function $jlf(args...)
        Base.$jlf(args...)
    end
end

let (jlf, f) = (:div_arcp, :div)
    for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
        ir = """
            %x = f$f fast $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.@pure function $jlf(a::$T, b::$T)
                @Base._inline_meta
                Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
            end
        end
    end
    @eval function $jlf(args...)
        Base.$jlf(args...)
    end
end
rcp(x) = div_arcp(one(x), x) # still leads to rcp.rn which is also a function call

# div_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_fdividef", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
# rcp(x) = div_fast(one(x), x)

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
_x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                x = _x,   y = _y,   z = _z)

const N = 4
const nmoist = 0
const ntrace = 0

Base.@irrational grav 9.81 BigFloat(9.81)
Base.@irrational gdm1 0.4 BigFloat(0.4)

function volumerhs!(rhs, Q, vgeo, gravity, D, nelem)
    Q = Base.Experimental.Const(Q)
    vgeo = Base.Experimental.Const(vgeo)
    D = Base.Experimental.Const(D)

    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    s_D = @cuStaticSharedMem eltype(D) (Nq, Nq)
    s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)
    s_G = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)

    r_rhsρ = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsU = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsV = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsW = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsE = MArray{Tuple{Nq}, eltype(rhs)}(undef)

    e = blockIdx().x
    j = threadIdx().y
    i = threadIdx().x

    @inbounds begin
        for k in 1:Nq
            r_rhsρ[k] = zero(eltype(rhs))
            r_rhsU[k] = zero(eltype(rhs))
            r_rhsV[k] = zero(eltype(rhs))
            r_rhsW[k] = zero(eltype(rhs))
            r_rhsE[k] = zero(eltype(rhs))
        end

        # fetch D into shared
        s_D[i, j] = D[i, j]
        @unroll for k in 1:Nq
            sync_threads()

            # Load values will need into registers
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

            # GPU performance trick
            # Allow optimizations to use the reciprocal of an argument rather than perform division.
            # IEEE floating-point division is implemented as a function call
            ρinv = rcp(ρ)
            ρ2inv = rcp(2ρ)
            # ρ2inv = 0.5f0 * pinv

            P = gdm1*(E - (U^2 + V^2 + W^2)*ρ2inv - ρ*gravity*z)

            fluxρ_x = U
            fluxU_x = ρinv * U * U + P
            fluxV_x = ρinv * U * V
            fluxW_x = ρinv * U * W
            fluxE_x = ρinv * U * (E + P)

            fluxρ_y = V
            fluxU_y = ρinv * V * U
            fluxV_y = ρinv * V * V + P
            fluxW_y = ρinv * V * W
            fluxE_y = ρinv * V * (E + P)

            fluxρ_z = W
            fluxU_z = ρinv * W * U
            fluxV_z = ρinv * W * V
            fluxW_z = ρinv * W * W + P
            fluxE_z = ρinv * W * (E + P)

            s_F[i, j,  _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
            s_F[i, j,  _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
            s_F[i, j,  _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
            s_F[i, j,  _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
            s_F[i, j,  _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

            s_G[i, j,  _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
            s_G[i, j,  _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
            s_G[i, j,  _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
            s_G[i, j,  _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
            s_G[i, j,  _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

            r_Hρ = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
            r_HU = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
            r_HV = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
            r_HW = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
            r_HE = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

            # one shared access per 10 flops
            for n = 1:Nq
                Dkn = s_D[k, n]

                r_rhsρ[n] += Dkn * r_Hρ
                r_rhsU[n] += Dkn * r_HU
                r_rhsV[n] += Dkn * r_HV
                r_rhsW[n] += Dkn * r_HW
                r_rhsE[n] += Dkn * r_HE
            end

            r_rhsW[k] -= MJ * ρ * gravity

            sync_threads()

            # loop of ξ-grid lines
            @unroll for n = 1:Nq
                Dni = s_D[n, i]
                Dnj = s_D[n, j]

                r_rhsρ[k] += Dni * s_F[n, j, _ρ]
                r_rhsρ[k] += Dnj * s_G[i, n, _ρ]

                r_rhsU[k] += Dni * s_F[n, j, _U]
                r_rhsU[k] += Dnj * s_G[i, n, _U]

                r_rhsV[k] += Dni * s_F[n, j, _V]
                r_rhsV[k] += Dnj * s_G[i, n, _V]

                r_rhsW[k] += Dni * s_F[n, j, _W]
                r_rhsW[k] += Dnj * s_G[i, n, _W]

                r_rhsE[k] += Dni * s_F[n, j, _E]
                r_rhsE[k] += Dnj * s_G[i, n, _E]
            end
        end # k

        @unroll for k in 1:Nq
            MJI = vgeo[i, j, k, _MJI, e]

            # Updates are a performance bottleneck
            # primary source of stall_long_sb
            rhs[i, j, k, _U, e] += MJI*r_rhsU[k]
            rhs[i, j, k, _V, e] += MJI*r_rhsV[k]
            rhs[i, j, k, _W, e] += MJI*r_rhsW[k]
            rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k]
            rhs[i, j, k, _E, e] += MJI*r_rhsE[k]
        end
    end
    return
end

function main()
    DFloat = Float32
    nelem = 240_000

    rng = StableRNG(123)

    Nq = N + 1
    nvar = _nstate + nmoist + ntrace

    Q = 1 .+ CuArray(rand(rng, DFloat, Nq, Nq, Nq, nvar, nelem))
    Q[:, :, :, _E, :] .+= 20
    vgeo = CuArray(rand(rng, DFloat, Nq, Nq, Nq, _nvgeo, nelem))

    # make sure the entries of the mass matrix satisfy the inverse relation
    vgeo[:, :, :, _MJ, :] .+= 3
    vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

    D = CuArray(rand(rng, DFloat, Nq, Nq))

    rhs = CuArray(zeros(DFloat, Nq, Nq, Nq, nvar, nelem))

    threads=(N+1, N+1)

    kernel = @cuda launch=false volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    # XXX: should we print these for all kernels? maybe upload them to Codespeed?
    @info """volumerhs! details:
              - $(CUDA.registers(kernel)) registers, max $(CUDA.maxthreads(kernel)) threads
              - $(Base.format_bytes(CUDA.memory(kernel).local)) local memory,
                $(Base.format_bytes(CUDA.memory(kernel).shared)) shared memory,
                $(Base.format_bytes(CUDA.memory(kernel).constant)) constant memory"""
    results = @benchmark begin
        CUDA.@sync $kernel($rhs, $Q, $vgeo, $(DFloat(grav)), $D, $nelem;
                           threads=$threads, blocks=$nelem)
    end

    # BenchmarkTools captures inputs, JuliaCI/BenchmarkTools.jl#127, so forcibly free them
    CUDA.unsafe_free!(rhs)
    CUDA.unsafe_free!(Q)
    CUDA.unsafe_free!(vgeo)
    CUDA.unsafe_free!(D)

    results
end

end

VolumeRHS.main()
