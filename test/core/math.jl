# Verify PTX lowering of math intrinsics across {f32, f64} × {plain,
# `@fastmath`} × {default, job-wide `fastmath=true`}. Most of these used to
# require `@device_override`s pointing at libdevice; now they're handled by
# Julia + the NVPTX backend + GPUCompiler's `apply_fastmath!` and
# `PTXFDivFastPass`. Each testset pins down the actual PTX so the wiring
# stays put.

@testset "math" begin

@testset "abs" begin
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

@testset "floor/ceil/trunc" begin
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

@testset "isnan/isinf/isfinite" begin
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

@testset "signbit" begin
    for T in (Float32, Float64)
        @test @filecheck CUDA.code_ptx(Tuple{T}) do x
            @check_not "__nv_"
            signbit(x)
        end
    end
end

@testset "copysign" begin
    # NVPTX has no single copysign instruction (custom-lowered to bit ops);
    # we just verify libdevice isn't on the path.
    for T in (Float32, Float64)
        @test @filecheck CUDA.code_ptx(Tuple{T, T}) do x, y
            @check_not "__nv_"
            copysign(x, y)
        end
    end
end

@testset "min/max" begin
    # Plain `min`/`max` propagate NaN (Julia semantics). f32 with sm_80+ &
    # LLVM 14+ gets `min.NaN.f32`/`max.NaN.f32` directly; f64 has to emulate
    # since PTX has no `.NaN` variant for f64.
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

@testset "fma/muladd" begin
    # `Base.fma` lowers to `llvm.fma.fXX` (have_fma branch folded for f32/f64
    # by GPUCompiler; for f16 we keep an explicit override). `Base.muladd`
    # lowers to `fmul contract + fadd contract`, which the backend fuses.
    # Either way: a single `fma.rn` per type.
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

@testset "sqrt" begin
    # Inherits from Julia (`llvm.sqrt.fXX`). Plain → `sqrt.rn.fXX`; per-call
    # `@fastmath` → `sqrt.approx.fXX`; job-wide `fastmath=true` → the FTZ
    # variant via `apply_fastmath!`.
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

@testset "div/inv" begin
    # `Base.{/, inv}` and their fast variants are now handled entirely by
    # GPUCompiler's `PTXFDivFastPass`. `inv(x) = 1/x`; NVPTX pattern-matches
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

    # `@fastmath` on f32: pass picks the non-FTZ `div.approx.f32` since the
    # job isn't fast; f64 always uses rcp+Newton (no native fast f64 fdiv).
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

    # Job-wide `fastmath=true` stamps `afn` on every fdiv → same as @fastmath,
    # and f32 additionally picks up FTZ.
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
