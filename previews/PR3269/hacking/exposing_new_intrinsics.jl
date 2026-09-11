# # Exposing new GPU intrinsics

# This tutorial walks through how to expose a new NVPTX intrinsic in CUDA.jl,
# using the directed-rounding floating-point operations as a worked example.

# ## Identifying the intrinsic

# The PTX ISA documents the available instructions at
# [PTX – Floating Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions).
# Each floating-point arithmetic instruction comes in four directed-rounding
# variants:
#
# ```
# {add,sub,mul,div,fma}.rnd.{f32,f64}      rnd ∈ {.rn, .rz, .rm, .rp}
# ```
#
# where `.rn` is round-to-nearest-even, `.rz` rounds toward zero, `.rm` rounds
# toward minus infinity, and `.rp` rounds toward plus infinity.
#
# Most of these PTX instructions are exposed by LLVM as NVVM intrinsics. The
# canonical list lives in
# [`llvm/include/llvm/IR/IntrinsicsNVVM.td`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsNVVM.td).
# The naming convention there maps `.f32` to `.f` and `.f64` to `.d`, so the
# round-toward-plus-infinity FMA for Float64 is `llvm.nvvm.fma.rp.d`.
#
# If LLVM does not expose an intrinsic for the PTX instruction you need, the
# fallback is to drop down to inline PTX assembly via `@asmcall`.

# ## Calling the intrinsic from Julia

# CUDA.jl invokes LLVM intrinsics through `ccall` with the `llvmcall` calling
# convention. For example, the round-toward-plus-infinity FMA on Float64:

fma_rp(x::Float64, y::Float64, z::Float64) =
    ccall("llvm.nvvm.fma.rp.d", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)

# We can verify the generated PTX contains the expected instruction. (The `io`
# keyword disables syntax highlighting so the listing renders cleanly in these
# docs; at the REPL you can drop it to get colorized output.)

using CUDA

function fma_kernel(out, x, y, z)
    out[] = fma_rp(x, y, z)
    return
end

@device_code_ptx io=IOContext(stdout, :color => false) @cuda launch=false fma_kernel(CuArray{Float64}(undef, 1), 1.0, 1.0, 1.0)

# Look for `fma.rp.f64` in the listing above.

# ## Exposing the full family

# Once the intrinsic for one rounding mode is wired up, the remaining three
# follow the same pattern. CUDA.jl exposes the full
# `{add,sub,mul,div,fma}_{rn,rz,rm,rp}` family for both `Float32` and
# `Float64`. The names mirror CUDA C's `__fadd_rn`/`__dadd_rn`/etc.:

function rounding_demo!(out, x, y, z)
    out[1] = CUDA.fma_rn(x, y, z)
    out[2] = CUDA.fma_rz(x, y, z)
    out[3] = CUDA.fma_rp(x, y, z)
    out[4] = CUDA.fma_rm(x, y, z)
    return
end

out_d = CUDA.zeros(Float64, 4)
@cuda threads=1 rounding_demo!(out_d, 1.0, 1.0, 2.0^-53)
Array(out_d)

# The exact value of `1·1 + 2⁻⁵³` lies on the boundary between `1.0` and
# `nextfloat(1.0)`, so the four entries above are `1.0` (RN, ties to even),
# `1.0` (RZ), `nextfloat(1.0)` (RP), and `1.0` (RM).

# We deliberately don't extend `Base.fma` with a `RoundingMode` argument:
# Julia's Base has no precedent for per-call rounding on hardware floats
# (BigFloat reads a thread-local mode set via `setrounding`), so introducing
# such an API in CUDA.jl would not be portable. Code that needs directed
# rounding should call the `*_rn`/`*_rz`/`*_rm`/`*_rp` functions directly.
