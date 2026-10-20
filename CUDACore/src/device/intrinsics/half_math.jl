module HalfMath

using ..CUDACore: @asmcall, compute_capability, ptx_isa_version, BFloat16, @sv_str
using LLVM.Interop
using GPUCompiler
using Base: @assume_effects

@inline approx_exp2_ftz(value::Float32) =
    @asmcall("ex2.approx.ftz.f32 \$0, \$1;", "=f,f", Float32, Tuple{Float32}, value)
@inline approx_log2_ftz(value::Float32) =
    @asmcall("lg2.approx.ftz.f32 \$0, \$1;", "=f,f", Float32, Tuple{Float32}, value)

@inline function hexp(h::Float16)
    # exp(x) = exp2(x * log2(e)); the negative-zero addend keeps the fma from
    # collapsing into a plain multiply (matches cuda_fp16.hpp's hexp).
    f = fma(Float32(h), log2(Float32(ℯ)), -0f0)
    r = Float16(approx_exp2_ftz(f))

    r = fma(Float16(h == reinterpret(Float16, 0x1F79)), reinterpret(Float16, 0x9400), r)
    r = fma(Float16(h == reinterpret(Float16, 0x25CF)), reinterpret(Float16, 0x9400), r)
    r = fma(Float16(h == reinterpret(Float16, 0xC13B)), reinterpret(Float16, 0x0400), r)
    r = fma(Float16(h == reinterpret(Float16, 0xC1EF)), reinterpret(Float16, 0x0200), r)

    return r
end

@inline function hexp2(h::Float16)
    f = approx_exp2_ftz(Float32(h))
    # one-ULP nudge: ex2.approx.ftz.f32 underestimates by ~½ ULP on average, so
    # bump up by 2^-24 of the result before rounding back to Float16
    return Float16(muladd(f, 2f0^-24, f))
end

@inline function hexp10(h::Float16)
    f = fma(Float32(h), log2(10f0), -0f0)
    r = Float16(approx_exp2_ftz(f))

    r = fma(Float16(h == reinterpret(Float16, 0x34DE)), reinterpret(Float16, 0x9800), r)
    r = fma(Float16(h == reinterpret(Float16, 0x9766)), reinterpret(Float16, 0x9000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x9972)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0xA5C4)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0xBF0A)), reinterpret(Float16, 0x8100), r)

    return r
end

@inline function hlog(h::Float16)
    f = approx_log2_ftz(Float32(h)) * reinterpret(Float32, 0x3f317218)
    r = Float16(f)

    r = fma(Float16(h == reinterpret(Float16, 0x160D)), reinterpret(Float16, 0x9C00), r)
    r = fma(Float16(h == reinterpret(Float16, 0x3BFE)), reinterpret(Float16, 0x8010), r)
    r = fma(Float16(h == reinterpret(Float16, 0x3C0B)), reinterpret(Float16, 0x8080), r)
    r = fma(Float16(h == reinterpret(Float16, 0x6051)), reinterpret(Float16, 0x1C00), r)

    return r
end

@inline function hlog2(h::Float16)
    r = Float16(approx_log2_ftz(Float32(h)))

    # NB: log2 checks the *output* against the special-case pattern, not the input
    r = fma(Float16(r == reinterpret(Float16, 0xA2E2)), reinterpret(Float16, 0x8080), r)
    r = fma(Float16(r == reinterpret(Float16, 0xBF46)), reinterpret(Float16, 0x9400), r)

    return r
end

@inline function hlog10(h::Float16)
    f = approx_log2_ftz(Float32(h)) * reinterpret(Float32, 0x3e9a209b)
    r = Float16(f)

    r = fma(Float16(h == reinterpret(Float16, 0x338F)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x33F8)), reinterpret(Float16, 0x9000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x57E1)), reinterpret(Float16, 0x9800), r)
    r = fma(Float16(h == reinterpret(Float16, 0x719D)), reinterpret(Float16, 0x9C00), r)

    return r
end

@inline function trig_reduce(value::Float32)
    rounded = fma(value, 0.636619772f0, 12582912f0)
    quadrant = reinterpret(UInt32, rounded)
    multiple = rounded - 12582912f0
    reduced = fma(multiple, -1.5707962512969971f0, value)
    reduced = fma(multiple, -7.5497894158615964f-8, reduced)
    reduced, quadrant
end

@inline function trig_polynomial(value::Float32, quadrant::UInt32)
    squared = value * value
    cosine = isodd(quadrant)
    coefficient8 = ifelse(cosine, 2.44331571f-5, -1.95152959f-4)
    coefficient6 = ifelse(cosine, -1.38873163f-3, 8.33216087f-3)
    coefficient4 = ifelse(cosine, 4.16666457f-2, -1.66666546f-1)
    coefficient2 = ifelse(cosine, -0.5f0, 0f0)
    result = fma(coefficient8, squared, coefficient6)
    result = fma(result, squared, coefficient4)
    result = fma(result, squared, coefficient2)
    result = fma(result, ifelse(cosine, squared, value), ifelse(cosine, 1f0, value))
    ifelse(quadrant & UInt32(2) != 0, -result, result)
end

@inline function hsin(value::Float16)
    iszero(value) && return value
    reduced, quadrant = trig_reduce(Float32(value))
    result = Float16(trig_polynomial(reduced, quadrant))
    magnitude = abs(result)
    magnitude = fma(Float16(abs(value) == reinterpret(Float16, 0x32b3)),
                    reinterpret(Float16, 0x0800), magnitude)
    magnitude = fma(Float16(abs(value) == reinterpret(Float16, 0x5cb0)),
                    reinterpret(Float16, 0x9000), magnitude)
    copysign(magnitude, result)
end

@inline function hcos(value::Float16)
    reduced, quadrant = trig_reduce(Float32(value))
    result = Float16(trig_polynomial(reduced, (quadrant & UInt32(3)) + UInt32(1)))
    fma(Float16(abs(value) == reinterpret(Float16, 0x2b7c)),
        reinterpret(Float16, 0x1000), result)
end

@inline approx_exp2(value::Float32) =
    @asmcall("ex2.approx.f32 \$0, \$1;", "=f,f", Float32, Tuple{Float32}, value)
@inline approx_log2(value::Float32) =
    @asmcall("lg2.approx.f32 \$0, \$1;", "=f,f", Float32, Tuple{Float32}, value)

@inline hexp(value::BFloat16) =
    BFloat16(approx_exp2(Float32(value) * reinterpret(Float32, 0x3fb8aa3c)))
@inline hexp2(value::BFloat16) = BFloat16(approx_exp2(Float32(value)))
"""
Match NVIDIA's `cuda_bf16.hpp`, including its `0xbc95 → 0x3f75` fixup.
That one input differs from the correctly rounded result, `0x3f76`.
"""
@inline function hexp10(value::BFloat16)
    result = BFloat16(approx_exp2(Float32(value) * reinterpret(Float32, 0x40549a78)))
    ifelse(reinterpret(UInt16, value) == 0xbc95, reinterpret(BFloat16, 0x3f75), result)
end
@inline hlog2(value::BFloat16) = BFloat16(approx_log2(Float32(value)))
@inline hlog(value::BFloat16) =
    BFloat16(approx_log2(Float32(value)) * reinterpret(Float32, 0x3f317218))
@inline hlog10(value::BFloat16) =
    BFloat16(approx_log2(Float32(value)) * reinterpret(Float32, 0x3e9a209b))
@inline function hsin(value::BFloat16)
    bits = reinterpret(UInt16, value) & 0x7fff
    bits < 0x0080 && return value
    BFloat16(sin(Float32(value)))
end
@inline hcos(value::BFloat16) = BFloat16(cos(Float32(value)))
@inline function htanh(value::BFloat16)
    promoted = Float32(value)
    if compute_capability() >= sv"7.5" && ptx_isa_version() >= sv"7.0"
        return BFloat16(@asmcall("tanh.approx.f32 \$0, \$1;", "=f,f",
                                Float32, Tuple{Float32}, promoted))
    end
    bits = reinterpret(UInt16, value) & 0x7fff
    bits < 0x0080 && return value
    BFloat16(tanh(promoted))
end
@inline hsqrt(value::BFloat16) =
    BFloat16(@asmcall("sqrt.approx.f32 \$0, \$1;", "=f,f",
                     Float32, Tuple{Float32}, Float32(value)))
@inline hrsqrt(value::BFloat16) =
    BFloat16(@asmcall("rsqrt.approx.f32 \$0, \$1;", "=f,f",
                     Float32, Tuple{Float32}, Float32(value)))
@inline hrcp(value::BFloat16) =
    BFloat16(@asmcall("rcp.approx.f32 \$0, \$1;", "=f,f",
                     Float32, Tuple{Float32}, Float32(value)))

const METHODS = Tuple(
    GPUCompiler.DeviceLibraryMethod(getfield(@__MODULE__, Symbol(:h, op)), typ, (typ,);
                                    llvm_name="__cuda_$(op)_$(suffix)")
    for (typ, suffix, ops) in
        ((Float16, "f16", (:exp, :exp2, :exp10, :log, :log2, :log10, :sin, :cos)),
         (BFloat16, "bf16", (:exp, :exp2, :exp10, :log, :log2, :log10,
                            :sin, :cos, :tanh)))
    for op in ops)

library() = GPUCompiler.DeviceLibrary(:cuda_half_math; methods=METHODS, optimize=true)

end
