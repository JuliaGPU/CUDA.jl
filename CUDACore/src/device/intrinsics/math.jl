# math functionality

# we only use libdevice where needed. if possible, we go through LLVM instead,
# ideally relying on Julia's existing definitions.

@public fma, rsqrt, saturate, byte_perm, dp4a, assume
@public add_rn, add_rz, add_rm, add_rp
@public sub_rn, sub_rz, sub_rm, sub_rp
@public mul_rn, mul_rz, mul_rm, mul_rp
@public div_rn, div_rz, div_rm, div_rp
@public fma_rn, fma_rz, fma_rm, fma_rp

using Base: FastMath, @assume_effects
import BFloat16s

## helpers

within(lower, upper) = (val) -> lower <= val <= upper

# Most math functions are implemented as LLVM intrinsics, which `lower_intrinsics!` turns
# into libdevice calls after optimization. Some functions call libdevice directly instead
# (marked "COST MODEL" below), because going through the intrinsic regresses them: LLVM
# costs intrinsics it lowers to calls as a single instruction (`isLoweredToCall` in
# TargetTransformInfoImpl.h), so loops calling them get unrolled before the much larger
# implementation is substituted, and that implementation is then not optimized along with
# the kernel (e.g. the slow-path reduction of `sin` keeps its local-memory array). Remove
# these once LLVM accounts for the cost of lowered math intrinsics; the late lowering
# supports all of them (tested in test/core/device/intrinsics/llvm_math.jl).

# call an LLVM math intrinsic, optionally with the `afn` flag (which `ccall` cannot emit).
# the entry function needs to be marked `alwaysinline`, or it remains an out-of-line call.
@inline @generated function llvm_math(::Val{operation}, ::Val{approximate},
                                     args::Vararg{typ,count}) where {operation,approximate,typ,count}
    element = typ === Float16 ? "half" : typ === Float32 ? "float" :
              typ === Float64 ? "double" : "bfloat"
    suffix = typ === Float16 ? "f16" : typ === Float32 ? "f32" :
             typ === Float64 ? "f64" : "bf16"
    name = "llvm.$operation.$suffix"
    parameter_types = join(Base.fill(element, count), ", ")
    parameters = join(("$element %arg$index" for index in 1:count), ", ")
    flags = approximate ? "afn " : ""
    if operation === :sincos
        result_type = NTuple{2,typ}
        ir = """
            declare {$element, $element} @$name($element)
            define [2 x $element] @entry($parameters) #0 {
                %pair = call $flags{$element, $element} @$name($parameters)
                %sine = extractvalue {$element, $element} %pair, 0
                %cosine = extractvalue {$element, $element} %pair, 1
                %first = insertvalue [2 x $element] undef, $element %sine, 0
                %result = insertvalue [2 x $element] %first, $element %cosine, 1
                ret [2 x $element] %result
            }
            attributes #0 = { alwaysinline }
            """
    else
        result_type = typ
        ir = """
            declare $element @$name($parameter_types)
            define $element @entry($parameters) #0 {
                %result = call $flags$element @$name($parameters)
                ret $element %result
            }
            attributes #0 = { alwaysinline }
            """
    end
    argument_types = Tuple{ntuple(_ -> typ, count)...}
    :(Base.llvmcall(($ir, "entry"), $result_type, $argument_types, args...))
end

for (op, minimum_llvm) in ((:exp, 15), (:exp2, 15), (:exp10, 18),
                           (:log, 15), (:log2, 15), (:log10, 15),
                           (:sin, 15), (:cos, 15), (:tan, 19), (:asin, 19),
                           (:acos, 19), (:atan, 19), (:sinh, 19), (:cosh, 19),
                           (:tanh, 19))
    for (typ, suffix) in ((Float16, "f16"), (BFloat16, "bf16"))
        typ === Float16 && op in (:exp, :exp2, :exp10, :log, :log2, :log10) && continue
        # COST MODEL: these are computed in Float32 instead, through the libdevice-calling
        # implementations (or the Payne-Hanek reduction of the slow path spills)
        typ === Float16 && op in (:sin, :tan) && continue
        typ === BFloat16 && op in (:sin, :cos, :tan) && continue
        if Base.libllvm_version >= VersionNumber(minimum_llvm) &&
           (typ !== BFloat16 || BFloat16s.llvm_arithmetic)
            intrinsic = "llvm.$op.$suffix"
            @eval @device_override Base.$op(value::$typ) =
                ccall($intrinsic, llvmcall, $typ, ($typ,), value)
        else
            implementation = Symbol(:h, op)
            if isdefined(HalfMath, implementation) &&
               hasmethod(getfield(HalfMath, implementation), Tuple{typ})
                @eval @device_override Base.$op(value::$typ) =
                    HalfMath.$implementation(value)
            else
                @eval @device_override Base.$op(value::$typ) = $typ(Base.$op(Float32(value)))
            end
        end
    end
end

for (op, minimum_llvm) in ((:exp, 15), (:exp10, 18), (:log, 15), (:log2, 15),
                           (:log10, 15), (:sin, 15), (:cos, 15), (:tan, 19),
                           (:asin, 19), (:acos, 19), (:atan, 19), (:sinh, 19),
                           (:cosh, 19), (:tanh, 19))
    fast_op = Symbol(op, :_fast)
    isdefined(FastMath, fast_op) || continue
    Base.libllvm_version >= VersionNumber(minimum_llvm) || continue
    for typ in (Float16, Float32, Float64, BFloat16)
        typ === Float32 && op ∉ (:asin, :acos, :atan, :sinh, :cosh, :tanh) && continue
        # Base falls back to `sin` and `cos` (see COST MODEL), and to Julia's own `exp`
        # and `exp10`, which are faster than libdevice's (that lacks fast Float64 variants)
        typ === Float64 && op in (:sin, :cos, :exp, :exp10) && continue
        # computed in Float32 instead (see COST MODEL above)
        typ === Float16 && op === :sin && continue
        typ === BFloat16 && op in (:sin, :cos) && continue
        typ === Float16 && op === :tanh && continue
        typ === BFloat16 && !BFloat16s.llvm_arithmetic && continue
        @eval @device_override FastMath.$fast_op(value::$typ) =
            llvm_math(Val($(QuoteNode(op))), Val(true), value)
    end
end

@device_function rsqrt(value::BFloat16) = HalfMath.hrsqrt(value)
@device_override Base.inv(value::BFloat16) = HalfMath.hrcp(value)

for typ in (Float16, BFloat16)
    typ === BFloat16 && !BFloat16s.llvm_arithmetic && continue
    @eval @device_override Base.:(^)(value::$typ, power::$typ) =
        llvm_math(Val(:pow), Val(false), value, power)
    @eval @device_override FastMath.pow_fast(value::$typ, power::$typ) =
        llvm_math(Val(:pow), Val(true), value, power)
    # COST MODEL: Float16 is computed in Float32 instead
    if Base.libllvm_version >= v"20" && typ === BFloat16
        @eval @device_override Base.atan(value::$typ, other::$typ) =
            llvm_math(Val(:atan2), Val(false), value, other)
        @eval @device_override Base.sincos(value::$typ) =
            llvm_math(Val(:sincos), Val(false), value)
    end
end

@device_override FastMath.pow_fast(value::Float64, power::Float64) =
    llvm_math(Val(:pow), Val(true), value, power)


## trigonometric

# COST MODEL
@device_override Base.cos(x::Float64) = ccall("extern __nv_cos", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cos(x::Float32) = ccall("extern __nv_cosf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.cos_fast(x::Float32) = llvm_math(Val(:cos), Val(true), x)

@device_override Base.cospi(x::Float64) = ccall("extern __nv_cospi", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cospi(x::Float32) = ccall("extern __nv_cospif", llvmcall, Cfloat, (Cfloat,), x)

# COST MODEL
@device_override Base.sin(x::Float64) = ccall("extern __nv_sin", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sin(x::Float32) = ccall("extern __nv_sinf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.sin_fast(x::Float32) = llvm_math(Val(:sin), Val(true), x)

@device_override Base.sinpi(x::Float64) = ccall("extern __nv_sinpi", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sinpi(x::Float32) = ccall("extern __nv_sinpif", llvmcall, Cfloat, (Cfloat,), x)

@static if Base.libllvm_version >= v"19"
    @device_override Base.tan(x::Float64) = ccall("llvm.tan.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.tan(x::Float64) = ccall("extern __nv_tan", llvmcall, Cdouble, (Cdouble,), x)
end
# COST MODEL
@device_override Base.tan(x::Float32) = ccall("extern __nv_tanf", llvmcall, Cfloat, (Cfloat,), x)
@static if Base.libllvm_version >= v"19"
    @device_override FastMath.tan_fast(x::Float32) = llvm_math(Val(:tan), Val(true), x)
else
    @device_override FastMath.tan_fast(x::Float32) = ccall("extern __nv_fast_tanf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"20"
    @device_override Base.sincos(x::Float64) =
        llvm_math(Val(:sincos), Val(false), x)
else
    @device_override function Base.sincos(x::Float64)
        s = Ref{Cdouble}()
        c = Ref{Cdouble}()
        ccall("extern __nv_sincos", llvmcall, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), x, s, c)
        return (s[], c[])
    end
end
# COST MODEL
@device_override function Base.sincos(x::Float32)
    s = Ref{Cfloat}()
    c = Ref{Cfloat}()
    ccall("extern __nv_sincosf", llvmcall, Cvoid, (Cfloat, Ptr{Cfloat}, Ptr{Cfloat}), x, s, c)
    return (s[], c[])
end
# Base has sincos_fast fall back to the native implementation which is presumed faster,
# but that is not the case compared to CUDA's intrinsics
@device_override FastMath.sincos_fast(x::Union{Float64,Float32}) = (FastMath.sin_fast(x), FastMath.cos_fast(x))

@device_override function Base.sincospi(x::Float64)
    s = Ref{Cdouble}()
    c = Ref{Cdouble}()
    ccall("extern __nv_sincospi", llvmcall, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), x, s, c)
    return (s[], c[])
end
@device_override function Base.sincospi(x::Float32)
    s = Ref{Cfloat}()
    c = Ref{Cfloat}()
    ccall("extern __nv_sincospif", llvmcall, Cvoid, (Cfloat, Ptr{Cfloat}, Ptr{Cfloat}), x, s, c)
    return (s[], c[])
end


## inverse trigonometric

@static if Base.libllvm_version >= v"19"
    @device_override Base.acos(x::Float64) = ccall("llvm.acos.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.acos(x::Float64) = ccall("extern __nv_acos", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.acos(x::Float32) = ccall("llvm.acos.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.acos(x::Float32) = ccall("extern __nv_acosf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"19"
    @device_override Base.asin(x::Float64) = ccall("llvm.asin.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.asin(x::Float64) = ccall("extern __nv_asin", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.asin(x::Float32) = ccall("llvm.asin.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.asin(x::Float32) = ccall("extern __nv_asinf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"19"
    @device_override Base.atan(x::Float64) = ccall("llvm.atan.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.atan(x::Float64) = ccall("extern __nv_atan", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.atan(x::Float32) = ccall("llvm.atan.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.atan(x::Float32) = ccall("extern __nv_atanf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"20"
    @device_override Base.atan(x::Float64, y::Float64) = ccall("llvm.atan2.f64", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
else
    @device_override Base.atan(x::Float64, y::Float64) = ccall("extern __nv_atan2", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
end
# COST MODEL
@device_override Base.atan(x::Float32, y::Float32) = ccall("extern __nv_atan2f", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

## hyperbolic

@static if Base.libllvm_version >= v"19"
    @device_override Base.cosh(x::Float64) = ccall("llvm.cosh.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.cosh(x::Float64) = ccall("extern __nv_cosh", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.cosh(x::Float32) = ccall("llvm.cosh.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.cosh(x::Float32) = ccall("extern __nv_coshf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"19"
    @device_override Base.sinh(x::Float64) = ccall("llvm.sinh.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.sinh(x::Float64) = ccall("extern __nv_sinh", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.sinh(x::Float32) = ccall("llvm.sinh.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.sinh(x::Float32) = ccall("extern __nv_sinhf", llvmcall, Cfloat, (Cfloat,), x)
end

@static if Base.libllvm_version >= v"19"
    @device_override Base.tanh(x::Float64) = ccall("llvm.tanh.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.tanh(x::Float64) = ccall("extern __nv_tanh", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"19"
    @device_override Base.tanh(x::Float32) = ccall("llvm.tanh.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.tanh(x::Float32) = ccall("extern __nv_tanhf", llvmcall, Cfloat, (Cfloat,), x)
end
@device_override @inline function FastMath.tanh_fast(x::Float16)
    @static if Base.libllvm_version >= v"19"
        return llvm_math(Val(:tanh), Val(true), x)
    end
    if compute_capability() >= sv"7.5" && ptx_isa_version() >= sv"7.0"
        @asmcall("tanh.approx.f16 \$0, \$1;", "=h,h", Float16, Tuple{Float16}, x)
    else
        Float16(FastMath.tanh_fast(Float32(x)))
    end
end

## inverse hyperbolic

@device_override Base.acosh(x::Float64) = ccall("extern __nv_acosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acosh(x::Float32) = ccall("extern __nv_acoshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.asinh(x::Float64) = ccall("extern __nv_asinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asinh(x::Float32) = ccall("extern __nv_asinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atanh(x::Float64) = ccall("extern __nv_atanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.atanh(x::Float32) = ccall("extern __nv_atanhf", llvmcall, Cfloat, (Cfloat,), x)


## logarithmic

# Float16 strategy for log/exp variants: compute in Float32 using the approximate
# hardware path, then patch the handful of inputs (~2–5) where rounding the
# approximate result back to Float16 disagrees with the IEEE-correct value. The
# fixup table is the one from cuda_fp16.hpp (__hlog/__hexp/etc.); each entry pairs
# a Float16 input bit pattern with the ±1 ULP correction needed at that point.
# After the fixups, the Float16 output matches the IEEE-correct value bit-for-bit,
# so Julia semantics are preserved while still using the fast lg2.approx/ex2.approx
# hardware instead of libdevice's full-precision __nv_log*/__nv_exp*.

@device_override Base.log(x::Float64) = ccall("llvm.log.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log(x::Float32) = ccall("llvm.log.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log(h::Float16) = ccall("llvm.log.f16", llvmcall, Float16, (Float16,), h)
@device_override FastMath.log_fast(x::Float32) = llvm_math(Val(:log), Val(true), x)

@device_override Base.log10(x::Float64) = ccall("llvm.log10.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log10(x::Float32) = ccall("llvm.log10.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log10(h::Float16) = ccall("llvm.log10.f16", llvmcall, Float16, (Float16,), h)
@device_override FastMath.log10_fast(x::Float32) = llvm_math(Val(:log10), Val(true), x)

@device_override Base.log1p(x::Float64) = ccall("extern __nv_log1p", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log1p(x::Float32) = ccall("extern __nv_log1pf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log2(x::Float64) = ccall("llvm.log2.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log2(x::Float32) = ccall("llvm.log2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log2(h::Float16) = ccall("llvm.log2.f16", llvmcall, Float16, (Float16,), h)
@device_override FastMath.log2_fast(x::Float32) = llvm_math(Val(:log2), Val(true), x)

@device_function logb(x::Float64) = ccall("extern __nv_logb", llvmcall, Cdouble, (Cdouble,), x)
@device_function logb(x::Float32) = ccall("extern __nv_logbf", llvmcall, Cfloat, (Cfloat,), x)

@device_function ilogb(x::Float64) = ccall("extern __nv_ilogb", llvmcall, Int32, (Cdouble,), x)
@device_function ilogb(x::Float32) = ccall("extern __nv_ilogbf", llvmcall, Int32, (Cfloat,), x)


## exponential

@device_override Base.exp(x::Float64) = ccall("llvm.exp.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp(x::Float32) = ccall("llvm.exp.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp(h::Float16) = ccall("llvm.exp.f16", llvmcall, Float16, (Float16,), h)
@device_override FastMath.exp_fast(x::Float32) = llvm_math(Val(:exp), Val(true), x)

@device_override Base.exp2(x::Float64) = ccall("llvm.exp2.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp2(x::Float32) = ccall("llvm.exp2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp2(h::Float16) = ccall("llvm.exp2.f16", llvmcall, Float16, (Float16,), h)
@device_override FastMath.exp2_fast(x::Float64) = llvm_math(Val(:exp2), Val(true), x)
@device_override function FastMath.exp2_fast(x::Float32)
    llvm_math(Val(:exp2), Val(true), x)
end
@device_override FastMath.exp2_fast(x::Float16) = llvm_math(Val(:exp2), Val(true), x)

@static if Base.libllvm_version >= v"18"
    @device_override Base.exp10(x::Float64) = ccall("llvm.exp10.f64", llvmcall, Cdouble, (Cdouble,), x)
else
    @device_override Base.exp10(x::Float64) = ccall("extern __nv_exp10", llvmcall, Cdouble, (Cdouble,), x)
end
@static if Base.libllvm_version >= v"18"
    @device_override Base.exp10(x::Float32) = ccall("llvm.exp10.f32", llvmcall, Cfloat, (Cfloat,), x)
else
    @device_override Base.exp10(x::Float32) = ccall("extern __nv_exp10f", llvmcall, Cfloat, (Cfloat,), x)
end
@static if Base.libllvm_version >= v"18"
    @device_override Base.exp10(h::Float16) = ccall("llvm.exp10.f16", llvmcall, Float16, (Float16,), h)
else
    @device_override Base.exp10(h::Float16) = HalfMath.hexp10(h)
end
@static if Base.libllvm_version >= v"18"
    @device_override FastMath.exp10_fast(x::Float32) = llvm_math(Val(:exp10), Val(true), x)
else
    @device_override FastMath.exp10_fast(x::Float32) = ccall("extern __nv_fast_exp10f", llvmcall, Cfloat, (Cfloat,), x)
end

@device_override Base.expm1(x::Float64) = ccall("extern __nv_expm1", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.expm1(x::Float32) = ccall("extern __nv_expm1f", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.expm1(x::Float16) = Float16(CUDACore.expm1(Float32(x)))

@device_override Base.ldexp(x::Float64, y::Int32) = ccall("extern __nv_ldexp", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_override Base.ldexp(x::Float32, y::Int32) = ccall("extern __nv_ldexpf", llvmcall, Cfloat, (Cfloat, Int32), x, y)


## integer handling (bit twiddling)

# where plain LLVM IR selects the same instructions, use it instead of libdevice, which
# also accepts both signednesses without reinterpreting the arguments.

@device_function brev(x::Union{Int32, UInt32}) = bitreverse(x % UInt32)
@device_function brev(x::Union{Int64, UInt64}) = bitreverse(x % UInt64)

@device_function clz(x::Union{Int32, UInt32, Int64, UInt64}) = leading_zeros(x) % Int32

# libdevice's 32-bit version (brev + bfind.shiftamt) is shorter than any generic form
@device_function ffs(x::Union{Int32, UInt32}) =
    assume(within(UInt32(0), UInt32(32)),
           ccall("extern __nv_ffs", llvmcall, Int32, (UInt32,), x % UInt32))
# the leading zeros of the lowest set bit give its (1-based) position from the right
@device_function ffs(x::Union{Int64, UInt64}) = (64 - leading_zeros(x & -x)) % Int32

@device_function function fns(mask::Union{Int32,UInt32}, base::Integer, offset::Integer=0)
    # Reinterpret the input mask instead of letting `ccall` convert them with a range check
    mask %= UInt32

    pos = ccall("llvm.nvvm.fns", llvmcall, Int32,
                (UInt32, Int32, Int32), mask, base, offset)
    assume(within(UInt32(0), UInt32(32)), pos)
end

@device_function popc(x::Union{Int32, UInt32, Int64, UInt64}) = count_ones(x) % Int32

@device_function function byte_perm(x::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8},
                                    y::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8},
                                    z::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8})
    # Reinterpret the input values instead of letting `ccall` convert them with a range check
    x %= UInt32
    y %= UInt32
    z %= UInt32
    ccall("extern __nv_byte_perm", llvmcall, Int32, (UInt32, UInt32, UInt32), x, y, z)
end

"""
    dp4a(a, b, c)

Packed 4-element int8 (or uint8) dot product with 32-bit accumulation, mapped to a single
PTX `dp4a` instruction on sm_61+.

The semantics depend on the signedness of `a` and `b`:

- `dp4a(a::Int32,  b::Int32,  c::Int32)  -> Int32`  — signed × signed
- `dp4a(a::Int32,  b::UInt32, c::Int32)  -> Int32`  — signed × unsigned
- `dp4a(a::UInt32, b::Int32,  c::Int32)  -> Int32`  — unsigned × signed
- `dp4a(a::UInt32, b::UInt32, c::UInt32) -> UInt32` — unsigned × unsigned

Each 32-bit argument `a` and `b` is interpreted as four packed 8-bit integers. The result
is `c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` where the individual byte
extractions respect the signed/unsigned interpretation of each operand.

!!! note
    Requires compute capability sm_61 or higher.
"""
function dp4a end

@static if LLVM.version() >= v"21"
    # LLVM 21 added @llvm.nvvm.idp4a.[us].[us]; prefer the intrinsic over inline PTX so
    # the instruction participates in optimization and instruction selection.
    @device_function function dp4a(a::Int32, b::Int32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.s.s", llvmcall, Int32, (Int32, Int32, Int32), a, b, c)
    end

    @device_function function dp4a(a::Int32, b::UInt32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.s.u", llvmcall, Int32, (Int32, UInt32, Int32), a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::Int32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.u.s", llvmcall, Int32, (UInt32, Int32, Int32), a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::UInt32, c::UInt32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.u.u", llvmcall, UInt32, (UInt32, UInt32, UInt32), a, b, c)
    end
else
    @device_function function dp4a(a::Int32, b::Int32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.s32.s32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{Int32, Int32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::Int32, b::UInt32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.s32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{Int32, UInt32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::Int32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.u32.s32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{UInt32, Int32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::UInt32, c::UInt32)
        require_sm_61()
        @asmcall("dp4a.u32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 UInt32, Tuple{UInt32, UInt32, UInt32}, a, b, c)
    end
end


## floating-point handling

# NOTE: not using libdevice's `__nv_nearbyint`, which calls `llvm.nvvm.round.*`: LLVM 15
#       (Julia 1.10) simplifies that to `llvm.round`, which rounds ties away from zero.
@device_function nearbyint(x::Float64) = ccall("llvm.nearbyint.f64", llvmcall, Cdouble, (Cdouble,), x)
@device_function nearbyint(x::Float32) = ccall("llvm.nearbyint.f32", llvmcall, Cfloat, (Cfloat,), x)

@device_function nextafter(x::Float64, y::Float64) = ccall("extern __nv_nextafter", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_function nextafter(x::Float32, y::Float32) = ccall("extern __nv_nextafterf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## roots and powers

# NVPTX has native `rsqrt.approx.{f32,f64}`; call the intrinsic directly. The
# obvious alternative, `@fastmath 1/sqrt(x)`, also lowers to `rsqrt.approx`
# (via `PTXRSqrtFastPass`), but is too aggressive wrt. fast-math behavior.
@device_function rsqrt(x::Float64) = ccall("llvm.nvvm.rsqrt.approx.d", llvmcall, Cdouble, (Cdouble,), x)
@device_function rsqrt(x::Float32) = ccall("llvm.nvvm.rsqrt.approx.f", llvmcall, Cfloat, (Cfloat,), x)
@device_function rsqrt(x::Float16) = Float16(rsqrt(Float32(x)))

@device_override Base.cbrt(x::Float64) = ccall("extern __nv_cbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cbrt(x::Float32) = ccall("extern __nv_cbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_function rcbrt(x::Float64) = ccall("extern __nv_rcbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_function rcbrt(x::Float32) = ccall("extern __nv_rcbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.:(^)(x::Float64, y::Float64) = ccall("llvm.pow.f64", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
# COST MODEL
@device_override Base.:(^)(x::Float32, y::Float32) = ccall("extern __nv_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override FastMath.pow_fast(x::Float32, y::Float32) = llvm_math(Val(:pow), Val(true), x, y)
# pow_fast: Base methods call llvm.powi which NVPTX cannot lower (#3065)
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float64, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ y  # no fast variant for Float64; uses __nv_pow
end
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float32, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    # __nv_fast_powf computes exp2(y * log2(x)), which is NaN for x < 0, so take the power of
    # abs(x) and copy the sign of x for odd y, like LLVM's AMDGPU expansion of fast `pown`.
    # the parity comes from the integer y, since Float32(y) may round to an even number.
    r = FastMath.pow_fast(abs(x), Float32(y))  # uses __nv_fast_powf
    ifelse(isodd(y), copysign(r, x), r)
end
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float16, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    Float16(FastMath.pow_fast(Float32(x), y))
end

# stay with Base's semantics and avoid the drift of libdevice's integer powers,
# which square at the working precision.
@device_function powi(x::Float64, y::Int32) = ccall("extern __nv_powi", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_function powi(x::Float32, y::Int32) = ccall("extern __nv_powif", llvmcall, Cfloat, (Cfloat, Int32), x, y)

# Base's `^(::Float, ::Integer)` calls `power_by_squaring`, whose
# `trailing_zeros` loop emits `cttz_int`, so convert and defer to `__nv_pow`.
# the float exponent may round to an even number, so take the sign from y.
@device_override @assume_effects :foldable @inline function Base.:(^)(x::Float32, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    r = abs(x) ^ Float32(y)
    ifelse(isodd(y), copysign(r, x), r)
end
@device_override @assume_effects :foldable @inline function Base.:(^)(x::Float64, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    r = abs(x) ^ Float64(y)
    ifelse(isodd(y), copysign(r, x), r)
end

## rounding and selection

# TODO: differentiate in return type, map correctly
#@device_override Base.round(x::Float64) = ccall("extern __nv_llround", llvmcall, Int64, (Cdouble,), x)
#@device_override Base.round(x::Float32) = ccall("extern __nv_llroundf", llvmcall, Int64, (Cfloat,), x)
#@device_override Base.round(x::Float64) = ccall("extern __nv_round", llvmcall, Cdouble, (Cdouble,), x)
#@device_override Base.round(x::Float32) = ccall("extern __nv_roundf", llvmcall, Cfloat, (Cfloat,), x)

# TODO: differentiate in return type, map correctly
#@device_override Base.rint(x::Float64) = ccall("extern __nv_llrint", llvmcall, Int64, (Cdouble,), x)
#@device_override Base.rint(x::Float32) = ccall("extern __nv_llrintf", llvmcall, Int64, (Cfloat,), x)
#@device_override Base.rint(x::Float64) = ccall("extern __nv_rint", llvmcall, Cdouble, (Cdouble,), x)
#@device_override Base.rint(x::Float32) = ccall("extern __nv_rintf", llvmcall, Cfloat, (Cfloat,), x)

#@device_override Base.min(x::Int32, y::Int32) = ccall("extern __nv_min", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.min(x::Int64, y::Int64) = ccall("extern __nv_llmin", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.min(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umin", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.min(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmin", llvmcall, Int64, (Int64, Int64), x, y))
# JuliaGPU/CUDA.jl#2111: fmin semantics wrt. NaN and signed zeros don't match Julia's
#@device_override Base.min(x::Float64, y::Float64) = ccall("extern __nv_fmin", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
#@device_override Base.min(x::Float32, y::Float32) = ccall("extern __nv_fminf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
# Julia's floating-point min/max match IEEE 754-2019 minimum/maximum, i.e.,
# `llvm.minimum`/`llvm.maximum`, which the external back-end legalizes for every
# subtarget: native min.NaN/max.NaN instructions on sm_80+, an expansion with
# NaN/signed-zero fix-ups elsewhere. Don't be tempted to use `llvm.minnum`
# (libdevice's `__nv_fmin`) with a NaN fix-up instead: its loose signed-zero
# semantics leak into constant folding, e.g., folding `min(0.0, -0.0)` to
# `0.0` where the host returns `-0.0`.
# Julia 1.12+ lowers `Base.min` to `llvm.minimum` by itself; keep the overrides
# for uniform codegen on older versions.
@device_override Base.min(x::Float32, y::Float32) =
    ccall("llvm.minimum.f32", llvmcall, Float32, (Float32, Float32), x, y)
@device_override Base.min(x::Float64, y::Float64) =
    ccall("llvm.minimum.f64", llvmcall, Float64, (Float64, Float64), x, y)

#@device_override Base.max(x::Int32, y::Int32) = ccall("extern __nv_max", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.max(x::Int64, y::Int64) = ccall("extern __nv_llmax", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.max(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umax", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.max(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmax", llvmcall, Int64, (Int64, Int64), x, y))
# JuliaGPU/CUDA.jl#2111: fmax semantics wrt. NaN and signed zeros don't match Julia's
#@device_override Base.max(x::Float64, y::Float64) = ccall("extern __nv_fmax", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
#@device_override Base.max(x::Float32, y::Float32) = ccall("extern __nv_fmaxf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.max(x::Float32, y::Float32) =
    ccall("llvm.maximum.f32", llvmcall, Float32, (Float32, Float32), x, y)
@device_override Base.max(x::Float64, y::Float64) =
    ccall("llvm.maximum.f64", llvmcall, Float64, (Float64, Float64), x, y)

# Base's AbstractFloat minmax simply calls min/max, but Julia 1.10/1.11 had
# open-coded definitions for Float32/Float64; override for uniform codegen.
@device_override Base.minmax(x::Float32, y::Float32) = min(x, y), max(x, y)
@device_override Base.minmax(x::Float64, y::Float64) = min(x, y), max(x, y)

@device_function saturate(x::Float32) = ccall("extern __nv_saturatef", llvmcall, Cfloat, (Cfloat,), x)


## division and remainder

@device_override Base.rem(x::Float64, y::Float64) = ccall("extern __nv_fmod", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.rem(x::Float32, y::Float32) = ccall("extern __nv_fmodf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float16, y::Float16) = Float16(rem(Float32(x), Float32(y)))

@device_override Base.rem(x::Float64, y::Float64, ::RoundingMode{:Nearest}) = ccall("extern __nv_remainder", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.rem(x::Float32, y::Float32, ::RoundingMode{:Nearest}) = ccall("extern __nv_remainderf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float16, y::Float16, ::RoundingMode{:Nearest}) = Float16(rem(Float32(x), Float32(y), RoundNearest))

# `Base.FastMath.inv_fast(::AbstractFloat)` is unimplemented upstream (only
# `Complex` has a method) and the catch-all fallback drops `afn`
@device_override FastMath.inv_fast(x::Union{Float16, Float32, Float64}) =
    FastMath.div_fast(one(x), x)


## distributions

# TODO: override StatsFun.jl?

@device_function normcdf(x::Float64) = ccall("extern __nv_normcdf", llvmcall, Cdouble, (Cdouble,), x)
@device_function normcdf(x::Float32) = ccall("extern __nv_normcdff", llvmcall, Cfloat, (Cfloat,), x)

@device_function normcdfinv(x::Float64) = ccall("extern __nv_normcdfinv", llvmcall, Cdouble, (Cdouble,), x)
@device_function normcdfinv(x::Float32) = ccall("extern __nv_normcdfinvf", llvmcall, Cfloat, (Cfloat,), x)



#
# Unsorted
#

@device_override Base.hypot(x::Float64, y::Float64) = ccall("extern __nv_hypot", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.hypot(x::Float32, y::Float32) = ccall("extern __nv_hypotf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

# `Base.fma(::Float16,...)` branches on `jl_have_fma`
#
# Float16 uses `Base.llvmcall`: `ccall(..., llvmcall)` roots the memory its arguments were
# loaded from, which keeps e.g. `Complex{Float16}` fields in memory and leads to redundant
# shared memory loads. switch back once JuliaLang/julia#64079 is merged.
#@device_override Base.fma(x::Float16, y::Float16, z::Float16) =
#    ccall("llvm.fma.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)
@device_override Base.fma(x::Float16, y::Float16, z::Float16) = Base.llvmcall(
    ("""
     declare half @llvm.fma.f16(half, half, half)
     define half @entry(half %x, half %y, half %z) #0 {
       %r = call half @llvm.fma.f16(half %x, half %y, half %z)
       ret half %r
     }
     attributes #0 = { alwaysinline }""", "entry"),
    Float16, Tuple{Float16, Float16, Float16}, x, y, z)

# `Base.muladd(x, y, z) = fma(x, y, z)` is the natural choice on GPU: NVPTX
# always lowers `llvm.fmuladd.fXX` to `fma.rn`, and routing through
# `llvm.fmuladd` (rather than Julia's default `fmul contract + fadd contract`)
# keeps the fusion robust under vectorization (per JuliaGPU/CUDA.jl#3149).
@device_override Base.muladd(x::Float64, y::Float64, z::Float64) =
    ccall("llvm.fmuladd.f64", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
@device_override Base.muladd(x::Float32, y::Float32, z::Float32) =
    ccall("llvm.fmuladd.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
# see `fma` above for why this doesn't use `ccall`
#@device_override Base.muladd(x::Float16, y::Float16, z::Float16) =
#    ccall("llvm.fmuladd.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)
@device_override Base.muladd(x::Float16, y::Float16, z::Float16) = Base.llvmcall(
    ("""
     declare half @llvm.fmuladd.f16(half, half, half)
     define half @entry(half %x, half %y, half %z) #0 {
       %r = call half @llvm.fmuladd.f16(half %x, half %y, half %z)
       ret half %r
     }
     attributes #0 = { alwaysinline }""", "entry"),
    Float16, Tuple{Float16, Float16, Float16}, x, y, z)

# Directed rounding for binary arithmetic and fma. NVPTX exposes
# `{add,mul,div,fma}.{rn,rz,rm,rp}.{f32,f64}` directly; there is no `sub`
# intrinsic, so subtraction reuses add(x, -y) (negation is bit-exact for IEEE
# floats). Names match the CUDA C intrinsics (`__fadd_rn`, `__dadd_rn`, ...) so
# users porting from CUDA C can find the corresponding Julia method by dropping
# the `__f`/`__d` prefix. We don't extend `Base.fma` / `Base.:+` etc. with a
# `RoundingMode` argument because Base has no precedent for per-call rounding
# on hardware floats (BigFloat reads a thread-local mode instead).
for rnd in ("rn", "rz", "rm", "rp")
    for op in (:add, :mul, :div)
        fname = Symbol(op, :_, rnd)
        if rnd == "rn" && op !== :div && Base.libllvm_version < v"18"
            # LLVM 17 and older fold these into a plain fadd/fmul, which the back-end then
            # contracts into an fma (llvm/llvm-project#76870)
            asm_f = "$op.rn.f32 \$0, \$1, \$2;"
            asm_d = "$op.rn.f64 \$0, \$1, \$2;"
            @eval @device_function $fname(x::Float32, y::Float32) =
                @asmcall($asm_f, "=f,f,f", Float32, Tuple{Float32, Float32}, x, y)
            @eval @device_function $fname(x::Float64, y::Float64) =
                @asmcall($asm_d, "=d,d,d", Float64, Tuple{Float64, Float64}, x, y)
            continue
        end
        intrinsic_f = "llvm.nvvm.$(op).$(rnd).f"
        intrinsic_d = "llvm.nvvm.$(op).$(rnd).d"
        @eval @device_function $fname(x::Float32, y::Float32) =
            ccall($intrinsic_f, llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
        @eval @device_function $fname(x::Float64, y::Float64) =
            ccall($intrinsic_d, llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
    end

    # NVPTX has no sub.<rnd> intrinsic; reuse add with negated y.
    sub_fname = Symbol(:sub_, rnd)
    add_fname = Symbol(:add_, rnd)
    @eval @device_function $sub_fname(x::T, y::T) where {T<:Union{Float32, Float64}} =
        $add_fname(x, -y)

    fma_fname = Symbol(:fma_, rnd)
    intrinsic_f = "llvm.nvvm.fma.$(rnd).f"
    intrinsic_d = "llvm.nvvm.fma.$(rnd).d"
    @eval @device_function $fma_fname(x::Float32, y::Float32, z::Float32) =
        ccall($intrinsic_f, llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
    @eval @device_function $fma_fname(x::Float64, y::Float64, z::Float64) =
        ccall($intrinsic_d, llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
end

@device_function sad(x::Int32, y::Int32, z::Int32) = ccall("extern __nv_sad", llvmcall, Int32, (Int32, Int32, Int32), x, y, z)
# NOTE: the unsigned variants declare unsigned argument and return types (the LLVM types
#       are identical), as converting to and from `Int32` would throw for large values.
@device_function sad(x::UInt32, y::UInt32, z::UInt32) = ccall("extern __nv_usad", llvmcall, UInt32, (UInt32, UInt32, UInt32), x, y, z)

@device_function dim(x::Float64, y::Float64) = ccall("extern __nv_fdim", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_function dim(x::Float32, y::Float32) = ccall("extern __nv_fdimf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

# multiplies the low 24 bits of the inputs, keeping the low 32 bits of the product
@device_function mul24(x::Int32, y::Int32) = ((x << 8) >> 8) * ((y << 8) >> 8)
@device_function mul24(x::UInt32, y::UInt32) = (x & 0x00ffffff) * (y & 0x00ffffff)

@device_function mul64hi(x::T, y::T) where {T<:Union{Int64, UInt64}} = (widemul(x, y) >> 64) % T
@device_function mulhi(x::T, y::T) where {T<:Union{Int32, UInt32}} = (widemul(x, y) >> 32) % T

@device_function hadd(x::T, y::T) where {T<:Union{Int32, UInt32}} = ((widen(x) + widen(y)) >> 1) % T
@device_function rhadd(x::T, y::T) where {T<:Union{Int32, UInt32}} = ((widen(x) + widen(y) + 1) >> 1) % T

@device_function scalbn(x::Float64, y::Int32) = ccall("extern __nv_scalbn", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_function scalbn(x::Float32, y::Int32) = ccall("extern __nv_scalbnf", llvmcall, Cfloat, (Cfloat, Int32), x, y)

@device_function norm3df(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_norm3df", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_function rnorm3df(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_rnorm3df", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
