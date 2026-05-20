using SpecialFunctions

@testset "math" begin
    @testset "log10" begin
        for T in (Float32, Float64)
            @test testf(a->log10.(a), T[100])
        end
    end

    @testset "pow" begin
        for T in (Float16, Float32, Float64, ComplexF32, ComplexF64)
            range = (T<:Integer) ? (T(5):T(10)) : T
            @test testf((x,y)->x.^y, rand(Float32, 1), rand(range, 1))
            @test testf((x,y)->x.^y, rand(Float32, 1), -rand(range, 1))
        end
    end
    
    @testset "min/max" begin
        for T in (Float32, Float64)
            @test testf((x,y)->max.(x, y), rand(Float32, 1), rand(T, 1))
            @test testf((x,y)->min.(x, y), rand(Float32, 1), rand(T, 1))
        end
    end

    @testset "isinf" begin
      for x in (Inf32, Inf, NaN16, NaN32, NaN)
        @test testf(x->isinf.(x), [x])
      end
    end

    @testset "isnan" begin
      for x in (Inf32, Inf, NaN16, NaN32, NaN)
        @test testf(x->isnan.(x), [x])
      end
    end

    for op in (exp, angle, exp2, exp10,)
        @testset "$op" begin
            for T in (Float32, Float64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end
        end
    end
    for op in (expm1,)
        @testset "$op" begin
            # FIXME: add expm1(::Float16) to Base
            for T in (Float32, Float64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end
        end
    end

    for op in (exp, abs, abs2, angle, log)
        @testset "Complex - $op" begin
            for T in (ComplexF16, ComplexF32, ComplexF64)
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end
        end
    end
    @testset "mod and rem" begin
        for T in (Float16, Float32, Float64)
            @test testf(a->rem.(a, T(2)), T[0, 1, 1.5, 2, -1])
            @test testf(a->rem.(a, T(2), RoundNearest), T[0, 1, 1.5, 2, -1])
            @test testf(a->mod.(a, T(2)), T[0, 1, 1.5, 2, -1])
        end
    end

    @testset "rsqrt" begin
        # GPUCompiler.jl#173: a CUDA-only device function fails to validate
        function kernel(a)
            a[] = CUDA.rsqrt(a[])
            return
        end

        # make sure this test uses an actual device function
        @test_throws ErrorException kernel(ones(1))

        for T in (Float16, Float32)
            a = CuArray{T}([4])
            @cuda kernel(a)
            @test Array(a) == [0.5]
        end
    end

    @testset "fma" begin
        for T in (Float16, Float32, Float64)
            @test testf((x,y,z)->fma.(x,y,z), rand(T, 1), rand(T, 1), rand(T, 1))
            @test testf((x,y,z)->fma.(x,y,z), rand(T, 1), -rand(T, 1), -rand(T, 1))
        end
    end

    @testset "directed rounding" begin
        # Verify each NVPTX directed-rounding intrinsic lowers to the expected
        # PTX instruction. The full PTX listing also catches accidental
        # round-trips through fp32 or libdevice helpers.
        #
        # `.rn` is the default PTX rounding mode, so on older LLVM the suffix
        # is elided (`add.f32` ≡ `add.rn.f32`). For the same reason LLVM may
        # fold `sub_rn(x,y) = add_rn(x,-y)` back to `sub.f32`. The non-rn modes
        # always emit the explicit suffix since the default doesn't apply.
        for op in (:add, :mul, :div, :fma), rnd in (:rn, :rz, :rm, :rp)
            fname = Symbol(op, :_, rnd)
            f = getfield(CUDA, fname)
            for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
                args = ntuple(_ -> T(1), op === :fma ? 3 : 2)
                kernel = if op === :fma
                    (out, x, y, z) -> (out[] = f(x, y, z); nothing)
                else
                    (out, x, y) -> (out[] = f(x, y); nothing)
                end
                buf = CuArray{T}(undef, 1)
                ptx = sprint(io->(@device_code_ptx io=io @cuda launch=false kernel(buf, args...)))
                accepted = rnd === :rn ?
                    ("$(op).rn.$(suffix)", "$(op).$(suffix)") :
                    ("$(op).$(rnd).$(suffix)",)
                @test any(s -> occursin(s, ptx), accepted)
            end
        end

        # NVPTX has no sub.<rnd> intrinsic; sub_<rnd>(x,y) reuses add_<rnd>(x,-y).
        # For non-rn modes LLVM keeps the rounded add; for rn (the default) it
        # may fold back to a plain `sub`.
        for rnd in (:rn, :rz, :rm, :rp)
            f = getfield(CUDA, Symbol(:sub_, rnd))
            for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
                kernel = (out, x, y) -> (out[] = f(x, y); nothing)
                buf = CuArray{T}(undef, 1)
                ptx = sprint(io->(@device_code_ptx io=io @cuda launch=false kernel(buf, T(1), T(1))))
                accepted = rnd === :rn ?
                    ("add.rn.$(suffix)", "add.$(suffix)", "sub.$(suffix)") :
                    ("add.$(rnd).$(suffix)",)
                @test any(s -> occursin(s, ptx), accepted)
            end
        end

        # Numerical checks: pick inputs whose true result is not exactly
        # representable, so the four rounding modes give distinct outputs.

        @testset "fma_($T)" for T in (Float32, Float64)
            # The true value of fma(1, 1, 2^-prec) is 1.0 + half-ulp, an exact
            # tie that distinguishes all four rounding modes for positive values.
            prec = T === Float32 ? 24 : 53
            eps_step = T(2)^-prec
            function fma_kernel(out, x, y, z)
                out[1] = CUDA.fma_rn(x, y, z)
                out[2] = CUDA.fma_rz(x, y, z)
                out[3] = CUDA.fma_rp(x, y, z)
                out[4] = CUDA.fma_rm(x, y, z)
                return
            end
            out_d = CUDA.zeros(T, 4)
            @cuda threads=1 fma_kernel(out_d, one(T), one(T), eps_step)
            out = Array(out_d)
            @test out[1] == one(T)                # RN ties to even (mantissa LSB 0)
            @test out[2] == one(T)                # RZ
            @test out[3] == nextfloat(one(T))     # RP
            @test out[4] == one(T)                # RM
        end

        @testset "add_/mul_/div_($T)" for T in (Float32, Float64)
            # 0.1*3 ≠ 0.3 in any binary float, so mul_rn and mul_rp differ from
            # mul_rz and mul_rm by one ulp.
            x, y = T(1)/T(10), T(3)
            function mul_kernel(out, x, y)
                out[1] = CUDA.mul_rn(x, y)
                out[2] = CUDA.mul_rz(x, y)
                out[3] = CUDA.mul_rp(x, y)
                out[4] = CUDA.mul_rm(x, y)
                return
            end
            out_d = CUDA.zeros(T, 4)
            @cuda threads=1 mul_kernel(out_d, x, y)
            mul_out = Array(out_d)
            @test mul_out[4] <= mul_out[1] <= mul_out[3]   # RM ≤ RN ≤ RP
            @test mul_out[2] == mul_out[4]                 # RZ == RM for positive
            @test mul_out[3] == nextfloat(mul_out[4])      # 1 ulp gap

            # div 1/3 is inexact and positive.
            function div_kernel(out, x, y)
                out[1] = CUDA.div_rn(x, y)
                out[2] = CUDA.div_rz(x, y)
                out[3] = CUDA.div_rp(x, y)
                out[4] = CUDA.div_rm(x, y)
                return
            end
            out_d = CUDA.zeros(T, 4)
            @cuda threads=1 div_kernel(out_d, T(1), T(3))
            div_out = Array(out_d)
            @test div_out[4] <= div_out[1] <= div_out[3]
            @test div_out[2] == div_out[4]
            @test div_out[3] == nextfloat(div_out[4])

            # add 1 + 2^-(prec+1) lies strictly between 1.0 and nextfloat(1.0).
            prec = T === Float32 ? 24 : 53
            eps_half = T(2)^-(prec+1)
            function add_kernel(out, x, y)
                out[1] = CUDA.add_rn(x, y)
                out[2] = CUDA.add_rz(x, y)
                out[3] = CUDA.add_rp(x, y)
                out[4] = CUDA.add_rm(x, y)
                return
            end
            out_d = CUDA.zeros(T, 4)
            @cuda threads=1 add_kernel(out_d, one(T), eps_half)
            add_out = Array(out_d)
            @test add_out[1] == one(T)              # RN ties to even
            @test add_out[2] == one(T)              # RZ
            @test add_out[3] == nextfloat(one(T))   # RP
            @test add_out[4] == one(T)              # RM
        end

        @testset "sub_($T)" for T in (Float32, Float64)
            # Pick a y whose addition to x is inexact so we can see directed
            # rounding. 1.0 - 2^-(prec+1) is exactly halfway between
            # prevfloat(1.0) and 1.0.
            prec = T === Float32 ? 24 : 53
            eps_half = T(2)^-(prec+1)
            function sub_kernel(out, x, y)
                out[1] = CUDA.sub_rn(x, y)
                out[2] = CUDA.sub_rz(x, y)
                out[3] = CUDA.sub_rp(x, y)
                out[4] = CUDA.sub_rm(x, y)
                return
            end
            out_d = CUDA.zeros(T, 4)
            @cuda threads=1 sub_kernel(out_d, one(T), eps_half)
            sub_out = Array(out_d)
            @test sub_out[1] == one(T)              # RN ties to even
            @test sub_out[2] == prevfloat(one(T))   # RZ
            @test sub_out[3] == one(T)              # RP
            @test sub_out[4] == prevfloat(one(T))   # RM
        end
    end

    @testset "muladd" begin
        for T in (Float16, Float32, Float64)
            @test testf((x,y,z)->muladd.(x,y,z), rand(T, 1), rand(T, 1), rand(T, 1))
            @test testf((x,y,z)->muladd.(x,y,z), rand(T, 1), -rand(T, 1), -rand(T, 1))
        end
    end

    # something from SpecialFunctions.jl
    @testset "erf" begin
        @test testf(a->SpecialFunctions.erf.(a), Float32[1.0])
    end
    @testset "loggamma" begin
        @test testf(a->SpecialFunctions.loggamma.(a), Float32[1.0])
    end

    @testset "exp" begin
        # JuliaGPU/CUDA.jl#1085: exp uses Base.sincos performing a global CPU load
        @test testf(x->exp.(x), [1e7im])
    end
    
    @testset "Real - $op" for op in (abs, abs2, exp, exp10, log, log10)
        @testset "$T" for T in (Float16, Float32, Float64)
            @test testf(x->op.(x), rand(T, 1))
        end
    end
    
    @testset "Float16 - $op" for op in (exp,exp2,exp10,log,log2,log10)
        all_float_16 = collect(reinterpret(Float16, pattern) for pattern in  UInt16(0):UInt16(1):typemax(UInt16))
        all_float_16 = filter(!isnan, all_float_16)
        if op in (log, log2, log10)
            all_float_16 = filter(>=(0), all_float_16)
        end
        @test testf(x->map(op, x), all_float_16)
    end

    @testset "fastmath" begin
        # libdevice provides some fast math functions
        a(x) = cos(x)
        b(x) = @fastmath cos(x)
        @test Array(map(a, cu([0.1,0.2]))) ≈ Array(map(b, cu([0.1,0.2])))

        # JuliaGPU/CUDA.jl#1352: some functions used to fall back to libm
        f(x) = log1p(x)
        g(x) = @fastmath log1p(x)
        @test Array(map(f, cu([0.1,0.2]))) ≈ Array(map(g, cu([0.1,0.2])))

        # JuliaGPU/CUDA.jl#2886: LLVM below v18 emits non-existing min.NaN.f64/max.NaN.f64
        f(a, b) = @fastmath max(a, b)
        @test Array(map(f, CuArray([1.0, 2.0]), CuArray([4.0, 3.0]))) == [4.0, 3.0]

        # JuliaGPU/CUDA.jl#3065: pow_fast with integer exponent used unsupported llvm.powi
        function fastpow_kernel(A, y)
            i = threadIdx().x
            @inbounds @fastmath A[i] = A[i]^y
            return nothing
        end
        for T in (Float32, Float64)
            A = CUDA.ones(T, 4)
            @cuda threads=4 fastpow_kernel(A, Int32(3))
            @test Array(A) == ones(T, 4)
            @cuda threads=4 fastpow_kernel(A, 3)
            @test Array(A) == ones(T, 4)
        end

        # Float16 hardware approximations: tanh.approx.f16 / ex2.approx.f16 on sm_75+
        if capability(device()) >= v"7.5"
            tanh_fast_kernel(x) = @fastmath tanh(x)
            exp2_fast_kernel(x) = @fastmath exp2(x)
            xs = Float16[-1, -0.5, 0, 0.5, 1]
            @test Array(map(tanh_fast_kernel, cu(xs))) ≈ tanh.(xs) atol = Float16(1e-3)
            @test Array(map(exp2_fast_kernel, cu(xs))) ≈ exp2.(xs) atol = Float16(1e-3)
        end
    end

    @testset "byte_perm" begin
        bytes = UInt32[i for i in 1:8]
        x = bytes[4]<<24 | bytes[3]<<16 | bytes[2]<<8 | bytes[1]<<0
        y = bytes[8]<<24 | bytes[7]<<16 | bytes[6]<<8 | bytes[5]<<0
        sel = UInt32[4, 2, 4, 2]
        code = sel[4]<<12 | sel[3]<<8 | sel[2]<<4 | sel[1]<<0
        r = bytes[sel[4]+1]<<24 | bytes[sel[3]+1]<<16 | bytes[sel[2]+1]<<8 | bytes[sel[1]+1]<<0

        function kernel1(a)
            a[3] = CUDA.byte_perm(a[1], a[2], code % Int32)
            return
        end
        function kernel2(a)
            a[3] = CUDA.byte_perm(a[1], a[2], code % UInt16)
            return
        end

        for T in [UInt32, Int32]
            a = CuArray{T}([x, y, 0])
            @cuda kernel1(a)
            @test Array(a)[3] == r
            a = CuArray{T}([x, y, 0])
            @cuda kernel2(a)
            @test Array(a)[3] == r
        end
    end

    @testset "@fastmath sincos" begin
        # JuliaGPU/CUDA.jl#1606: FastMath.sincos fell back to regular sin/cos
        @test @filecheck CUDA.code_ptx(NTuple{3,CuDeviceArray{Float32,1,AS.Global}}) do a, b, c
            @check "sin.approx.f32"
            @check "cos.approx.f32"
            @check_not "__nv"  # from libdevice
            @inbounds b[], c[] = @fastmath sincos(a[])
            return
        end
    end

    @testset "inv" begin
        # Base.inv should use accurate rcp instructions (rcp.rn).
        # PTX-level patterns for inv / inv_fast / div / div_fast live in
        # `test/core/math.jl`; here we only sanity-check correctness on GPU.
        for T in (Float32, Float64)
            @test testf(x -> inv.(x), rand(T, 10) .+ T(0.1))
            @test testf(x -> inv.(x), T[0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
        end
    end

    @testset "inv_fast" begin
        fast_inv(x) = @fastmath inv(x)
        xs32 = Float32[0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        @test Array(map(fast_inv, cu(xs32))) ≈ inv.(xs32) rtol = 1.0f-4
        xs64 = Float64[0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        @test Array(map(fast_inv, CuArray(xs64))) ≈ inv.(xs64) rtol = 1.0e-10
    end

    @testset "div_fast Float64" begin
        fast_div(x, y) = @fastmath x / y
        xs = rand(Float64, 10) .+ 0.1
        ys = rand(Float64, 10) .+ 0.1
        @test Array(map(fast_div, CuArray(xs), CuArray(ys))) ≈ xs ./ ys rtol = 1.0e-10
    end

    @testset "JuliaGPU/CUDA.jl#2111: min/max should return NaN" begin
        for T in [Float32, Float64]
            AT = CuArray{T}
            @test isequal(Array(min.(AT([NaN]), AT([Inf]))), [NaN])
            @test isequal(minimum(AT([NaN])), NaN)

            @test isequal(Array(max.(AT([NaN]), AT([-Inf]))), [NaN])
            @test isequal(maximum(AT([NaN])), NaN)
        end
    end

    # PTX lowering pins for the standard math ops. Most of these used to
    # require `@device_override`s pointing at libdevice; now they're handled
    # by Julia + the NVPTX backend + GPUCompiler's `apply_fastmath!`,
    # `PTXFDivFastPass`, and `PTXFSqrtFastPass`. Each testset pins the actual
    # PTX so the wiring stays put across {f32, f64} × {plain, `@fastmath`} ×
    # {default, job-wide `fastmath=true`}.

    @testset "abs PTX" begin
        for fastmath in (false, true)
            # f32: job-wide fastmath flips to the `.ftz` variant.
            @test @filecheck CUDA.code_ptx(Tuple{Float32}; fastmath) do x
                @check cond=fastmath  "abs.ftz.f32"
                @check cond=!fastmath "abs.f32"
                @check_not "__nv_"
                abs(x)
            end
            # f64: no FTZ on PTX for f64.
            @test @filecheck CUDA.code_ptx(Tuple{Float64}; fastmath) do x
                @check "abs.f64"
                @check_not "__nv_"
                abs(x)
            end
        end
        @test @filecheck CUDA.code_ptx(Tuple{Int32}) do x
            @check "abs.s32"
            @check_not "__nv_"
            abs(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Int64}) do x
            @check "abs.s64"
            @check_not "__nv_"
            abs(x)
        end
    end

    @testset "floor/ceil/trunc PTX" begin
        for (op, rnd) in ((floor, "rmi"), (ceil, "rpi"), (trunc, "rzi"))
            for fastmath in (false, true)
                @test @filecheck CUDA.code_ptx(Tuple{Float32}; fastmath) do x
                    @check cond=fastmath  "cvt.$rnd.ftz.f32.f32"
                    @check cond=!fastmath "cvt.$rnd.f32.f32"
                    @check_not "__nv_"
                    op(x)
                end
                @test @filecheck CUDA.code_ptx(Tuple{Float64}; fastmath) do x
                    @check "cvt.$rnd.f64.f64"
                    @check_not "__nv_"
                    op(x)
                end
            end
        end
    end

    @testset "isnan/isinf/isfinite PTX" begin
        # All three should be pure FP compares / bit-tests, no libdevice.
        for T in (Float32, Float64), op in (isnan, isinf, isfinite)
            @test @filecheck CUDA.code_ptx(Tuple{T}) do x
                @check_not "__nv_"
                op(x)
            end
        end
        # `isnan(x) = x != x` is the cleanest: a single `setp.nan.fXX`.
        @test @filecheck CUDA.code_ptx(Tuple{Float32}) do x
            @check "setp.nan.f32"
            isnan(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64}) do x
            @check "setp.nan.f64"
            isnan(x)
        end
    end

    @testset "signbit PTX" begin
        for T in (Float32, Float64)
            @test @filecheck CUDA.code_ptx(Tuple{T}) do x
                @check_not "__nv_"
                signbit(x)
            end
        end
    end

    @testset "copysign PTX" begin
        # NVPTX has no single copysign instruction (custom-lowered to bit ops);
        # we just verify libdevice isn't on the path.
        for T in (Float32, Float64)
            @test @filecheck CUDA.code_ptx(Tuple{T, T}) do x, y
                @check_not "__nv_"
                copysign(x, y)
            end
        end
    end

    @testset "min/max PTX" begin
        # Plain `min`/`max` propagate NaN (Julia semantics). f32 with sm_80+
        # + LLVM 14+ gets `min.NaN.f32`/`max.NaN.f32` directly; f64 has to
        # emulate since PTX has no `.NaN` variant for f64.
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}) do x, y
            @check "min.NaN.f32"
            min(x, y)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}) do x, y
            @check "max.NaN.f32"
            max(x, y)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}) do x, y
            @check_not "__nv_"
            min(x, y)
        end

        # `@fastmath min/max` = `ifelse(y > x, x, y)`, a plain compare + select.
        for (T, s) in ((Float32, "f32"), (Float64, "f64"))
            @test @filecheck CUDA.code_ptx(Tuple{T, T}) do x, y
                @check "setp.lt.$s"
                @check "selp.$s"
                @fastmath min(x, y)
            end
            @test @filecheck CUDA.code_ptx(Tuple{T, T}) do x, y
                @check "setp.lt.$s"
                @check "selp.$s"
                @fastmath max(x, y)
            end
        end
    end

    @testset "fma/muladd PTX" begin
        # `Base.fma` lowers to `llvm.fma.fXX` (have_fma branch folded for
        # f32/f64 by GPUCompiler; for f16 we keep an explicit override).
        # `Base.muladd` lowers to `fmul contract + fadd contract`, which the
        # backend fuses. Either way: a single `fma.rn` per type.
        for (T, s) in ((Float16, "f16"), (Float32, "f32"), (Float64, "f64"))
            @test @filecheck CUDA.code_ptx(Tuple{T, T, T}) do x, y, z
                @check "fma.rn.$s"
                @check_not "__nv_fma"
                fma(x, y, z)
            end
            @test @filecheck CUDA.code_ptx(Tuple{T, T, T}) do x, y, z
                @check "fma.rn.$s"
                muladd(x, y, z)
            end
        end
    end

    @testset "sqrt PTX" begin
        # Inherits from Julia (`llvm.sqrt.fXX`). Plain → `sqrt.rn.fXX`;
        # per-call `@fastmath` → `sqrt.approx.fXX` (via `PTXFSqrtFastPass`);
        # job-wide `fastmath=true` → the FTZ variant via `apply_fastmath!`.
        for (T, s) in ((Float32, "f32"), (Float64, "f64"))
            @test @filecheck CUDA.code_ptx(Tuple{T}) do x
                @check "sqrt.rn.$s"
                @check_not "sqrt.approx"
                sqrt(x)
            end
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}) do x
            @check "sqrt.approx.f32"
            @check_not "sqrt.approx.ftz"
            @fastmath sqrt(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}; fastmath=true) do x
            @check "sqrt.approx.ftz.f32"
            sqrt(x)
        end
        # NVPTX has no native fast f64 sqrt; backend builds it from rsqrt + rcp.
        @test @filecheck CUDA.code_ptx(Tuple{Float64}) do x
            @check "rsqrt.approx.f64"
            @fastmath sqrt(x)
        end
    end

    @testset "div/inv PTX" begin
        # `Base.{/, inv}` and their fast variants are handled by GPUCompiler's
        # `PTXFDivFastPass`. `inv(x) = 1/x`; NVPTX pattern-matches
        # `fdiv 1.0, x` to `rcp.rn`.
        for (T, s) in ((Float32, "f32"), (Float64, "f64"))
            @test @filecheck CUDA.code_ptx(Tuple{T, T}) do x, y
                @check "div.rn.$s"
                x / y
            end
            @test @filecheck CUDA.code_ptx(Tuple{T}) do x
                @check "rcp.rn.$s"
                inv(x)
            end
        end

        # `@fastmath` on f32: pass picks the non-FTZ `div.approx.f32` since
        # the job isn't fast; f64 always uses rcp+Newton.
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}) do x, y
            @check "div.approx.f32"
            @check_not "div.approx.ftz"
            @fastmath x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}) do x
            @check "div.approx.f32"
            @check_not "div.approx.ftz"
            @fastmath inv(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}) do x, y
            @check "rcp.approx.ftz.f64"
            @fastmath x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64}) do x
            @check "rcp.approx.ftz.f64"
            @fastmath inv(x)
        end

        # Job-wide `fastmath=true` stamps `afn` on every fdiv → same as
        # `@fastmath`, and f32 additionally picks up FTZ.
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}; fastmath=true) do x, y
            @check "div.approx.ftz.f32"
            x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}; fastmath=true) do x
            @check "div.approx.ftz.f32"
            inv(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}; fastmath=true) do x, y
            @check "rcp.approx.ftz.f64"
            x / y
        end
    end
end
