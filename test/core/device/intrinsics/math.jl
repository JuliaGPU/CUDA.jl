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

        # Integer exponents: accuracy must not depend on the exponent's width.
        # __nv_powi/__nv_powif used to back the Int32 methods and drifted by
        # ~1500 ulp on the cases below, while Int64 went via __nv_pow.
        @testset "integer exponent ($T, $I)" for T in (Float32, Float64),
                                                 I in (Int32, Int64)
            x = T[1.001, 0.9, 1.1, 1.0001, 2, 0.5, -2]
            n = I[500, 200, 30, 5000, 7, -3, 5]
            @test Array(CuArray(x) .^ CuArray(n)) ≈ x .^ n rtol=8*eps(T)
            # the sign comes from the integer exponent: Float32(16_777_217) is even
            # (not compared with the CPU, which gets this wrong on Julia 1.13.0)
            @test Array(CuArray(T[-1]) .^ CuArray(I[16_777_217])) == T[-1]
        end

        # Both widths have to agree with each other, not just with the CPU
        @testset "integer exponent width agreement ($T)" for T in (Float32, Float64)
            x = CuArray(T[1.001, 0.9, 1.1, 1.0001])
            n = Int64[500, 200, 30, 5000]
            @test Array(x .^ CuArray(Int32.(n))) == Array(x .^ CuArray(n))
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
        
    @testset "acosh" begin
        for T in (Float32, Float64)
            @test testf(x->acosh.(x), rand(T, 1) .+ 1)
        end
    end

    for op in (exp, angle, exp2, exp10, asinh, atanh, asin, atan, acos)
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

        # NVPTX has no `llvm.nvvm.sub.<rnd>` intrinsic, so sub_<rnd>(x,y) is
        # implemented as add_<rnd>(x,-y). PTX itself does accept rounding
        # modifiers on `sub`, so the backend may fold back to a real
        # `sub.<rnd>.<suffix>` (LLVM 22) or keep the rounded add (older LLVM).
        # For `rn` the suffix may be elided entirely.
        for rnd in (:rn, :rz, :rm, :rp)
            f = getfield(CUDA, Symbol(:sub_, rnd))
            for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
                kernel = (out, x, y) -> (out[] = f(x, y); nothing)
                buf = CuArray{T}(undef, 1)
                ptx = sprint(io->(@device_code_ptx io=io @cuda launch=false kernel(buf, T(1), T(1))))
                accepted = rnd === :rn ?
                    ("add.rn.$(suffix)", "add.$(suffix)",
                     "sub.rn.$(suffix)", "sub.$(suffix)") :
                    ("add.$(rnd).$(suffix)", "sub.$(rnd).$(suffix)")
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

        @testset "no contraction ($T)" for T in (Float32, Float64)
            # (1+ε)(1-ε) = 1-ε² rounds to 1, so a separately rounded product cancels
            # exactly, while an fma would keep the -ε²
            a, b = 1 + eps(T), 1 - eps(T)
            function kernel(out, a, b, c)
                out[1] = CUDA.add_rn(a * b, c)
                out[2] = CUDA.mul_rn(a, b) + c
                return
            end
            out = CuArray{T}(undef, 2)
            @cuda kernel(out, a, b, -one(T))
            @test Array(out) == zeros(T, 2)
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
    
    @testset "sincospi" begin
        @testset "$T" for T in (Float32, Float64)
            @test testf(x->sincospi.(x), rand(T, 1))
        end
    end

    @testset "Real - $op" for op in (abs, abs2, exp, exp10, log, log10)
        @testset "$T" for T in (Float16, Float32, Float64)
            @test testf(x->op.(x), rand(T, 1))
        end
    end

    @testset "ldexp" begin
        @testset "$T" for T in (Float32, Float64)
            @test testf(x->ldexp.(x, 4), rand(T, 1))
        end
    end

    @testset "log2" begin
        @testset "$T" for T in (Float32, Float64)
            @test testf(x->log2.(x), rand(T, 1))
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
        # NOTE: every helper needs a distinct name; redefining a local function
        #       replaces the earlier method throughout the scope.
        cos_kernel(x) = cos(x)
        cos_fast_kernel(x) = @fastmath cos(x)
        @test Array(map(cos_kernel, cu([0.1,0.2]))) ≈ Array(map(cos_fast_kernel, cu([0.1,0.2])))

        # JuliaGPU/CUDA.jl#1352: some functions used to fall back to libm
        log1p_kernel(x) = log1p(x)
        log1p_fast_kernel(x) = @fastmath log1p(x)
        @test Array(map(log1p_kernel, cu([0.1,0.2]))) ≈ Array(map(log1p_fast_kernel, cu([0.1,0.2])))

        # JuliaGPU/CUDA.jl#2886: LLVM below v18 emits non-existing min.NaN.f64/max.NaN.f64
        max_fast_kernel(a, b) = @fastmath max(a, b)
        @test Array(map(max_fast_kernel, CuArray([1.0, 2.0]), CuArray([4.0, 3.0]))) == [4.0, 3.0]

        # JuliaGPU/CUDA.jl#3065: pow_fast with integer exponent used unsupported llvm.powi
        function fastpow_kernel(A, y)
            i = threadIdx().x
            @inbounds @fastmath A[i] = A[i]^y
            return nothing
        end
        for T in (Float16, Float32, Float64)
            A = CUDA.ones(T, 4)
            @cuda threads=4 fastpow_kernel(A, Int32(3))
            @test Array(A) == ones(T, 4)
            @cuda threads=4 fastpow_kernel(A, 3)
            @test Array(A) == ones(T, 4)
        end
        # larger exponents use `__nv_fast_powf`, which is NaN for negative bases
        for T in (Float16, Float32, Float64), y in (4, 5, 7, -2, -3, Int32(5), Int32(-5))
            x = T[1.5, -0.5, 3, -2]
            A = CuArray(x)
            @cuda threads=4 fastpow_kernel(A, y)
            @test all(isapprox.(Array(A), x .^ y; rtol=16*eps(T)))
        end
        # literal exponents take the same path, through `pow_fast(x, ::Val)`
        fastpow7(x) = @fastmath x^7
        x = Float32[-0.5, -2]
        @test all(isapprox.(Array(map(fastpow7, cu(x))), x .^ 7; rtol=16*eps(Float32)))
        # the sign comes from the integer exponent: Float32(16_777_217) is even
        A = CuArray(Float32[-1])
        @cuda threads=1 fastpow_kernel(A, 16_777_217)
        @test Array(A) == Float32[-1]

        # Test fast tan on Float32
        tan_kernel(x) = tan(x)
        tan_fast_kernel(x) = @fastmath tan(x)
        @test Array(map(tan_kernel, cu([0.1,0.2]))) ≈ Array(map(tan_fast_kernel, cu([0.1,0.2])))

        # exp, exp2 and exp10 for Float32 and Float64
        exp_fast_kernel(x) = @fastmath exp(x)
        exp2_fast_kernel(x) = @fastmath exp2(x)
        exp10_fast_kernel(x) = @fastmath exp10(x)
        for T in (Float32, Float64)
            x = rand(T, 4)
            A = CuArray(x)
            @test Array(map(exp_fast_kernel, A)) ≈ exp.(x) rtol=16*eps(T)
            @test Array(map(exp2_fast_kernel, A)) ≈ exp2.(x) rtol=16*eps(T)
            @test Array(map(exp10_fast_kernel, A)) ≈ exp10.(x) rtol=16*eps(T)
        end

        # the remaining libdevice __nv_fast_* functions
        log_fast_kernel(x) = @fastmath log(x)
        log2_fast_kernel(x) = @fastmath log2(x)
        log10_fast_kernel(x) = @fastmath log10(x)
        sin_fast_kernel(x) = @fastmath sin(x)
        pow_fast_kernel(x, y) = @fastmath x^y
        x = Float32[0.1, 0.5, 2, 10]
        @test Array(map(log_fast_kernel, cu(x))) ≈ log.(x) rtol=1f-5
        @test Array(map(log2_fast_kernel, cu(x))) ≈ log2.(x) rtol=1f-5
        @test Array(map(log10_fast_kernel, cu(x))) ≈ log10.(x) rtol=1f-5
        @test Array(map(sin_fast_kernel, cu(x))) ≈ sin.(x) atol=1f-5
        @test Array(map(pow_fast_kernel, cu(x), cu(x))) ≈ x .^ x rtol=1f-5

        # Float16 hardware approximations: tanh.approx.f16 / ex2.approx.f16 on sm_75+
        if capability(device()) >= v"7.5"
            tanh_fast_kernel(x) = @fastmath tanh(x)
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

    # dp4a requires sm_61+
    if capability(device()) >= v"6.1"
    @testset "dp4a" begin
        # Pure-Julia reference: unpack four bytes from a packed Int32/UInt32,
        # dot-product them (with the respective signed/unsigned semantics), and
        # add the accumulator.
        function ref_dp4a_ss(a::Int32, b::Int32, c::Int32)
            ba = reinterpret(NTuple{4,Int8}, a)
            bb = reinterpret(NTuple{4,Int8}, b)
            c + Int32(ba[1])*Int32(bb[1]) + Int32(ba[2])*Int32(bb[2]) +
                Int32(ba[3])*Int32(bb[3]) + Int32(ba[4])*Int32(bb[4])
        end
        function ref_dp4a_su(a::Int32, b::UInt32, c::Int32)
            ba = reinterpret(NTuple{4,Int8},  a)
            bb = reinterpret(NTuple{4,UInt8}, b)
            c + Int32(ba[1])*Int32(bb[1]) + Int32(ba[2])*Int32(bb[2]) +
                Int32(ba[3])*Int32(bb[3]) + Int32(ba[4])*Int32(bb[4])
        end
        function ref_dp4a_us(a::UInt32, b::Int32, c::Int32)
            ba = reinterpret(NTuple{4,UInt8}, a)
            bb = reinterpret(NTuple{4,Int8},  b)
            c + Int32(ba[1])*Int32(bb[1]) + Int32(ba[2])*Int32(bb[2]) +
                Int32(ba[3])*Int32(bb[3]) + Int32(ba[4])*Int32(bb[4])
        end
        function ref_dp4a_uu(a::UInt32, b::UInt32, c::UInt32)
            ba = reinterpret(NTuple{4,UInt8}, a)
            bb = reinterpret(NTuple{4,UInt8}, b)
            c + UInt32(ba[1])*UInt32(bb[1]) + UInt32(ba[2])*UInt32(bb[2]) +
                UInt32(ba[3])*UInt32(bb[3]) + UInt32(ba[4])*UInt32(bb[4])
        end

        # Kernels: each writes one result per thread (we launch 1 thread, one
        # case per test to keep the kernel signatures simple).
        function kernel_ss(out, a, b, c)
            out[] = CUDA.dp4a(a, b, c)
            return
        end
        function kernel_su(out, a, b, c)
            out[] = CUDA.dp4a(a, b, c)
            return
        end
        function kernel_us(out, a, b, c)
            out[] = CUDA.dp4a(a, b, c)
            return
        end
        function kernel_uu(out, a, b, c)
            out[] = CUDA.dp4a(a, b, c)
            return
        end

        # Helper: pack four Int8/UInt8 values (little-endian: b0 in bits 7:0).
        # Use reinterpret(Int32/UInt32, NTuple{4,Int8/UInt8}) — portable and avoids
        # integer-width pitfalls in the shift+or approach.
        pack_s(b0, b1, b2, b3) = reinterpret(Int32,  (b0%Int8,  b1%Int8,  b2%Int8,  b3%Int8))
        pack_u(b0, b1, b2, b3) = reinterpret(UInt32, (b0%UInt8, b1%UInt8, b2%UInt8, b3%UInt8))

        @testset "ss — signed × signed" begin
            cases = [
                # (a_bytes…, b_bytes…, c, label)
                (Int32(0), Int32(0), Int32(0)),                       # all zeros
                (pack_s(1,2,3,4), pack_s(5,6,7,8), Int32(10)),       # 1*5+2*6+3*7+4*8+10 = 80
                (pack_s(127,127,127,127), pack_s(1,1,1,1), Int32(0)), # max positive bytes
                (pack_s(-128,-128,-128,-128), pack_s(1,1,1,1), Int32(0)), # most-negative bytes
                (pack_s(-1,-1,-1,-1), pack_s(-1,-1,-1,-1), Int32(0)), # neg*neg
                (Int32(-1), Int32(-1), Int32(100)),                   # 0xFF packing
            ]
            for (a, b, c) in cases
                expected = ref_dp4a_ss(a, b, c)
                buf = CuArray{Int32}(undef, 1)
                @cuda threads=1 kernel_ss(buf, a, b, c)
                @test Array(buf)[] == expected
            end
        end

        @testset "su — signed × unsigned" begin
            cases = [
                (Int32(0), UInt32(0), Int32(0)),
                (pack_s(1,2,3,4), pack_u(5,6,7,8), Int32(10)),        # 1*5+…+10 = 80
                (pack_s(127,0,-128,1), pack_u(255,128,1,0), Int32(5)),
                (pack_s(-1,-1,-1,-1), pack_u(255,255,255,255), Int32(0)), # -1 * 255 * 4 = -1020
            ]
            for (a, b, c) in cases
                expected = ref_dp4a_su(a, b, c)
                buf = CuArray{Int32}(undef, 1)
                @cuda threads=1 kernel_su(buf, a, b, c)
                @test Array(buf)[] == expected
            end
        end

        @testset "us — unsigned × signed" begin
            cases = [
                (UInt32(0), Int32(0), Int32(0)),
                (pack_u(1,2,3,4), pack_s(5,6,7,8), Int32(10)),
                (pack_u(255,128,0,1), pack_s(-1,1,-128,127), Int32(0)),
            ]
            for (a, b, c) in cases
                expected = ref_dp4a_us(a, b, c)
                buf = CuArray{Int32}(undef, 1)
                @cuda threads=1 kernel_us(buf, a, b, c)
                @test Array(buf)[] == expected
            end
        end

        @testset "uu — unsigned × unsigned" begin
            cases = [
                (UInt32(0), UInt32(0), UInt32(0)),
                (pack_u(1,2,3,4), pack_u(5,6,7,8), UInt32(10)),       # 80
                (pack_u(255,255,255,255), pack_u(1,1,1,1), UInt32(0)), # 4*255 = 1020
                (pack_u(255,255,255,255), pack_u(255,255,255,255), UInt32(0)), # 4*255^2 = 260100
            ]
            for (a, b, c) in cases
                expected = ref_dp4a_uu(a, b, c)
                buf = CuArray{UInt32}(undef, 1)
                @cuda threads=1 kernel_uu(buf, a, b, c)
                @test Array(buf)[] == expected
            end
        end

        @testset "PTX instruction selection" begin
            # The backend must emit the dp4a instruction, not a software
            # emulation sequence.
            @test @filecheck CUDA.code_ptx(Tuple{CuDeviceArray{Int32,1,AS.Global},Int32,Int32,Int32}) do out, a, b, c
                @check "dp4a"
                @inbounds out[] = CUDA.dp4a(a, b, c)
                return
            end
        end
    end
    end # capability >= v"6.1"

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

    @testset "min/max should order signed zeros" begin
        for T in [Float32, Float64]
            AT = CuArray{T}
            z, mz = T(0.0), T(-0.0)
            @test isequal(Array(min.(AT([z]), AT([mz]))), [mz])
            @test isequal(Array(min.(AT([mz]), AT([z]))), [mz])
            @test isequal(Array(max.(AT([z]), AT([mz]))), [z])
            @test isequal(Array(max.(AT([mz]), AT([z]))), [z])
            @test isequal(Array(minmax.(AT([z]), AT([mz]))), [(mz, z)])
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
        # PTX `abs.s` is undefined for INT_MIN, so since LLVM 23 only `llvm.abs` with
        # the poison flag lowers to it; Julia's `abs` becomes `neg` + `max` instead.
        @test @filecheck CUDA.code_ptx(Tuple{Int32}) do x
            @check "{{abs|max}}.s32"
            @check_not "__nv_"
            abs(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Int64}) do x
            @check "{{abs|max}}.s64"
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
        # Plain `min`/`max` follow IEEE 754-2019 minimum/maximum (Julia
        # semantics: NaN-propagating, -0.0 < +0.0). The back-end uses the
        # native `min.NaN`/`max.NaN` instructions for f32 on sm_80+, and
        # expands to plain min/max plus NaN/signed-zero fix-ups elsewhere.
        # sm_80 needs a toolchain that can target it (CUDA 11.0+)
        sm80 = v"8.0" in CUDACore.ptxas_compat().cap
        if sm80
            @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}; arch=sm"80") do x, y
                @check "min.NaN.f32"
                min(x, y)
            end
            @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}; arch=sm"80") do x, y
                @check "max.NaN.f32"
                max(x, y)
            end
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}; arch=sm"60") do x, y
            @check "min.f32"
            @check_not "__nv_"
            min(x, y)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}) do x, y
            @check "min.f64"
            @check_not "__nv_"
            min(x, y)
        end

        # `@fastmath min/max` are fast `llvm.minnum`/`llvm.minimum` calls
        # (depending on the Julia version), which the back-end selects to a
        # single min/max instruction. With fast math NaN inputs are excluded,
        # so it doesn't matter whether the NaN-propagating variant gets picked
        # (fast `minimum` on f32 + sm_80; pin the arch for determinism).
        if sm80
            for (T, s) in ((Float32, "f32"), (Float64, "f64"))
                @test @filecheck CUDA.code_ptx(Tuple{T, T}; arch=sm"80") do x, y
                    @check "{{min.(NaN.)?$s}}"
                    @fastmath min(x, y)
                end
                @test @filecheck CUDA.code_ptx(Tuple{T, T}; arch=sm"80") do x, y
                    @check "{{max.(NaN.)?$s}}"
                    @fastmath max(x, y)
                end
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

    @testset "rsqrt PTX" begin
        # `CUDA.rsqrt(x)` directly calls the NVPTX `rsqrt.approx.{f,d}`
        # intrinsic — no libdevice, and no `@fastmath` so caller-side NaN/Inf
        # checks aren't DCE'd by `nnan`/`ninf` propagation. f16 computes in
        # f32, so it still hits the f32 instruction.
        for (T, s) in ((Float32, "f32"), (Float64, "f64"))
            @test @filecheck CUDA.code_ptx(Tuple{T}) do x
                @check "rsqrt.approx.$s"
                @check_not "sqrt.approx"
                @check_not "__nv_"
                CUDA.rsqrt(x)
            end
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float16}) do x
            @check "rsqrt.approx.f32"
            @check_not "__nv_"
            CUDA.rsqrt(x)
        end
    end

    @testset "div/inv PTX" begin
        # `Base.{/, inv}` lower to plain `fdiv`; NVPTX pattern-matches
        # `fdiv 1.0, x` (i.e. `inv`) to the dedicated `rcp` instructions.
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

        # `@fastmath` on f32: the back-end honors `afn`, picking the non-FTZ
        # variants since the job isn't fast, and the dedicated `rcp` for
        # reciprocals. f64 is rewritten by GPUCompiler to rcp+Newton, so also
        # check for the refinement fmas (a raw rcp would be too inaccurate).
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}) do x, y
            @check "div.approx.f32"
            @check_not "div.approx.ftz"
            @fastmath x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}) do x
            @check "rcp.approx.f32"
            @check_not "rcp.approx.ftz"
            @fastmath inv(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}) do x, y
            @check "rcp.approx.ftz.f64"
            @check "fma.rn.f64"
            @fastmath x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64}) do x
            @check "rcp.approx.ftz.f64"
            @check "fma.rn.f64"
            @fastmath inv(x)
        end

        # Job-wide `fastmath=true` stamps `afn` on every fdiv → same as
        # `@fastmath`, and f32 additionally picks up FTZ.
        @test @filecheck CUDA.code_ptx(Tuple{Float32, Float32}; fastmath=true) do x, y
            @check "div.approx.ftz.f32"
            x / y
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float32}; fastmath=true) do x
            @check "rcp.approx.ftz.f32"
            inv(x)
        end
        @test @filecheck CUDA.code_ptx(Tuple{Float64, Float64}; fastmath=true) do x, y
            @check "rcp.approx.ftz.f64"
            @check "fma.rn.f64"
            x / y
        end
    end

    @testset "expm1 Float16" begin
        x = Float16[-1, -0.5, 0, 0.001, 0.5, 1]
        @test Array(map(expm1, cu(x))) == Float16.(expm1.(Float32.(x)))
    end

    @testset "ldexp Int32" begin
        # integer literals hit Base's generic method; the override takes Int32
        for T in (Float32, Float64)
            x = T[0.75, -1.5, 3]
            @test Array(map(x -> ldexp(x, Int32(4)), CuArray(x))) == ldexp.(x, 4)
            @test Array(map(x -> ldexp(x, Int32(-2)), CuArray(x))) == ldexp.(x, -2)
        end
    end

    # CUDA-only device functions error on the host, so they can't go through
    # `map`/`broadcast` (which infer the host return type); use a plain kernel.
    function gpu_map(f, ::Type{R}, xs...) where {R}
        function kernel(out, f, xs...)
            i = threadIdx().x
            @inbounds out[i] = f(map(x -> x[i], xs)...)
            return
        end
        out = CuArray{R}(undef, length(first(xs)))
        @cuda threads=length(out) kernel(out, f, map(CuArray, xs)...)
        return Array(out)
    end

    @testset "logb/ilogb" begin
        for T in (Float32, Float64)
            x = T[0.25, 0.5, 1, 3, 1000, -0.75]
            @test gpu_map(CUDACore.logb, T, x) == T.(exponent.(x))
            @test gpu_map(CUDACore.ilogb, Int32, x) == Int32.(exponent.(x))
        end
    end

    @testset "bit twiddling" begin
        for T in (Int32, UInt32, Int64, UInt64)
            U = unsigned(T)
            x = T[0, 1, 2, 0x70, typemax(T), typemax(T) >> 7, typemin(T)]
            @test gpu_map(CUDACore.brev, U, x) == bitreverse.(x .% U)
            @test gpu_map(CUDACore.clz, Int32, x) == Int32.(leading_zeros.(x))
            @test gpu_map(CUDACore.ffs, Int32, x) ==
                  Int32[iszero(v) ? 0 : trailing_zeros(v) + 1 for v in x]
            @test gpu_map(CUDACore.popc, Int32, x) == Int32.(count_ones.(x))
        end
    end

    @testset "nearbyint/nextafter" begin
        for T in (Float32, Float64)
            x = T[0.5, 1.5, 2.5, -0.5, -2.5, 2.4, -2.6]
            # rounds to nearest, ties to even
            @test gpu_map(CUDACore.nearbyint, T, x) == round.(x)

            y = T[1, -1, 0, 1000]
            @test gpu_map(CUDACore.nextafter, T, y, fill(T(Inf), 4)) == nextfloat.(y)
            @test gpu_map(CUDACore.nextafter, T, y, fill(T(-Inf), 4)) == prevfloat.(y)
            @test gpu_map(CUDACore.nextafter, T, y, y) == y
        end
    end

    @testset "rcbrt" begin
        for T in (Float32, Float64)
            x = T[0.125, 1, 8, 27, -64, 1000]
            @test gpu_map(CUDACore.rcbrt, T, x) ≈ inv.(cbrt.(x)) rtol=4*eps(T)
        end
    end

    @testset "powi" begin
        for T in (Float32, Float64)
            x = T[1.5, -2, 0.5, 3]
            n = Int32[2, 3, -4, 7]
            # libdevice squares at the working precision, so allow some drift
            @test gpu_map(CUDACore.powi, T, x, n) ≈ x .^ n rtol=16*eps(T)
        end
    end

    @testset "saturate" begin
        x = Float32[-1, 0.25, 1, 2, Inf, -Inf, NaN]
        @test isequal(gpu_map(CUDACore.saturate, Float32, x),
                      Float32[0, 0.25, 1, 1, 1, 0, 0])
    end

    @testset "normcdf/normcdfinv" begin
        for T in (Float32, Float64)
            x = T[-3, -1, 0, 0.5, 2]
            @test gpu_map(CUDACore.normcdf, T, x) ≈
                  SpecialFunctions.erfc.(-x ./ sqrt(T(2))) ./ 2 rtol=8*eps(T)
            p = T[0.01, 0.25, 0.5, 0.75, 0.99]
            @test gpu_map(CUDACore.normcdfinv, T, p) ≈
                  -sqrt(T(2)) .* SpecialFunctions.erfcinv.(2 .* p) rtol=16*eps(T)
        end
    end

    @testset "integer arithmetic" begin
        x = Int32[7, -5, 100, typemax(Int32), typemin(Int32), -1]
        y = Int32[3, 9, -100, typemax(Int32), typemin(Int32), 1]
        z = Int32[1, 2, 3, 0, 0, 4]

        @test gpu_map(CUDACore.sad, Int32, x[1:3], y[1:3], z[1:3]) ==
              abs.(x[1:3] .- y[1:3]) .+ z[1:3]
        # unsigned variants used to round-trip through Int32, throwing for large values
        ux, uy, uz = UInt32[7, 5, 100, typemax(UInt32), 0], UInt32[3, 9, 1, 0, typemax(UInt32)], UInt32[1, 2, 3, 0, 0]
        @test gpu_map(CUDACore.sad, UInt32, ux, uy, uz) ==
              UInt32.(abs.(Int64.(ux) .- Int64.(uy))) .+ uz

        # only the low 24 bits of each operand participate
        a = Int32[3, -1000, 4095, 2^24 + 3]
        b = Int32[7, 2000, -4095, 5]
        @test gpu_map(CUDACore.mul24, Int32, a, b) == Int32[21, -2_000_000, -4095^2, 15]
        ua, ub = UInt32[3, 4095, 2^24 + 3, 0xffffff], UInt32[7, 4095, 5, 0xffffff]
        @test gpu_map(CUDACore.mul24, UInt32, ua, ub) ==
              (UInt64.(ua .& 0xffffff) .* UInt64.(ub .& 0xffffff)) .% UInt32

        @test gpu_map(CUDACore.mulhi, Int32, x, y) == Int32.((Int64.(x) .* Int64.(y)) .>> 32)
        ux = UInt32[7, typemax(UInt32), 0x8000_0000]
        uy = UInt32[3, typemax(UInt32), 4]
        @test gpu_map(CUDACore.mulhi, UInt32, ux, uy) ==
              UInt32.((UInt64.(ux) .* UInt64.(uy)) .>> 32)

        lx = Int64[7, -5, typemax(Int64), typemin(Int64)]
        ly = Int64[3, typemax(Int64), typemax(Int64), 2]
        @test gpu_map(CUDACore.mul64hi, Int64, lx, ly) ==
              Int64.((Int128.(lx) .* Int128.(ly)) .>> 64)
        ulx = UInt64[7, typemax(UInt64), 0x8000_0000_0000_0000]
        uly = UInt64[3, typemax(UInt64), 4]
        @test gpu_map(CUDACore.mul64hi, UInt64, ulx, uly) ==
              UInt64.((UInt128.(ulx) .* UInt128.(uly)) .>> 64)

        # halving adds don't overflow the intermediate sum
        @test gpu_map(CUDACore.hadd, Int32, x, y) == Int32.((Int64.(x) .+ Int64.(y)) .>> 1)
        @test gpu_map(CUDACore.rhadd, Int32, x, y) == Int32.((Int64.(x) .+ Int64.(y) .+ 1) .>> 1)
        @test gpu_map(CUDACore.hadd, UInt32, ux, uy) == UInt32.((UInt64.(ux) .+ UInt64.(uy)) .>> 1)
        @test gpu_map(CUDACore.rhadd, UInt32, ux, uy) == UInt32.((UInt64.(ux) .+ UInt64.(uy) .+ 1) .>> 1)
    end

    @testset "dim/scalbn" begin
        for T in (Float32, Float64)
            x = T[3, 1, -2, 5]
            y = T[1, 3, -5, 5]
            @test gpu_map(CUDACore.dim, T, x, y) == max.(x .- y, zero(T))
            n = Int32[0, 3, -2, 10]
            @test gpu_map(CUDACore.scalbn, T, x, n) == ldexp.(x, n)
        end
    end

    @testset "norm3df/rnorm3df" begin
        x = Float32[1, 2, 0, 3]
        y = Float32[2, 3, 0, 4]
        z = Float32[2, 6, 5, 12]
        @test gpu_map(CUDACore.norm3df, Float32, x, y, z) ≈ hypot.(x, y, z) rtol=4*eps(Float32)
        @test gpu_map(CUDACore.rnorm3df, Float32, x, y, z) ≈ inv.(hypot.(x, y, z)) rtol=4*eps(Float32)
    end
end
