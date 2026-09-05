# CUDA.jl release history

This document lists the noteworthy changes in every release of CUDA.jl, newest
first. It is a hand-written summary aimed at users; for the complete list of
merged pull requests, see the [GitHub
releases](https://github.com/JuliaGPU/CUDA.jl/releases).

CUDA.jl follows [semantic versioning](https://semver.org), so intentional
breaking changes are reserved for major releases. Some minor releases
nevertheless contain changes that are technically breaking, such as stricter
validation of previously invalid uses, corrections to unsupported behavior, or
changes to support floors; those are labeled separately below. Patch releases
are listed as subsections of the minor release they belong to.


## v6.4 (unreleased)

- Restore best-effort CUDA 10.2 and 11 support, including artifact-based installations
  on Jetson Nano. The binary stack selects Tegra toolkit builds and uses NVIDIA's L4T
  compatibility drivers where available; Xavier automatically selects CUDA 12.5 to
  retain vendor-library support for its GPU.
- Require CUDA_Driver_jll 13.3.4, CUDA_Runtime_jll 0.24.4, CUDA_Compiler_jll 0.6.2 and
  CUDA_Runtime_Discovery 2.1.1, including the Tegra artifact-selection fixes and discovery
  of local toolkits without cuSOLVERMg. `CUDA.versioninfo()` identifies a bundled
  forward-compatibility driver, including on systems without NVML.
- Check Tegra profiling permissions before entering CUPTI, and skip activity kinds
  the platform does not support. This permits integrated profiling as root with CUDA
  10.2 and 11 on Tegra, where optional NVTX marker-data records are unavailable.
  Gate allocation tracing on CUPTI's own API version, avoiding a native crash with
  JetPack 5's local CUDA 11.4 and the CUDA 12 compatibility driver.
- Distinguish PTX and cubin target compatibility. Executable compilation rejects
  targets whose cubins cannot load on the device, while PTX reflection remains available.
- Diagnose a selected runtime that requires a newer driver before compiling kernels.
  This can happen when disabling the compatibility driver through the environment
  leaves a previous toolkit selection cached; select a compatible toolkit and restart.

## v6.3 (August 2026)

Compiled kernels are now cached by Julia's compiler. Adapting to GPUCompiler 2,
which builds on CompilerCaching.jl, CUDA.jl dropped its own per-context
dictionary of compiled kernels: the compilation artifacts (the cubin image and
the entry-point name) are attached to the `CodeInstance` that Julia already
caches, keyed on the method instance, world age and compiler configuration,
while the context-specific `CuFunction` handles remain session-local and are
never serialized. On Julia 1.11 and later that cache is Julia's integrated code
cache, so a kernel compiled while a package is being precompiled can travel in
that package's image instead of being recompiled in every fresh session; on
Julia 1.10 the cache stays session-local
([#3185](https://github.com/JuliaGPU/CUDA.jl/pull/3185)).

*Technically breaking changes*:

- The CUDA compiler artifacts (`ptxas` and friends) are now selected
  independently from the CUDA runtime. `CUDA.set_runtime_version!` and
  `CUDA.reset_runtime_version!` write the `version` and `local` preferences for
  both `CUDA_Runtime_jll` and `CUDA_Compiler_jll`, and both UUIDs have to be
  listed in the project's `[extras]` section for those preferences to be picked
  up. The preferences can also be set separately, which is useful to test a
  compiler release against a fixed runtime; keep both within the same CUDA major
  version. The `local_compiler` preference is gone: a local compiler is
  requested through `CUDA_Compiler_jll`'s `local` preference, and is discovered
  with `CUDA_Runtime_Discovery`. Requires CUDA_Compiler_jll 0.5 and
  CUDA_Runtime_jll 0.24
  ([#3227](https://github.com/JuliaGPU/CUDA.jl/pull/3227)).
- On systems without a CUDA driver and without a version preference, no CUDA
  compiler artifact is selected anymore, matching what already happened for the
  runtime. Previously `CUDA_Compiler_jll` fell back to the most recent toolkit,
  which downloaded a large artifact at load time and made `using CUDA` fail
  when that artifact was unavailable or incomplete. The precompilation workload
  that compiles a dummy kernel is skipped in that case; to precompile or
  cross-compile GPU code on such a system, set the toolkit explicitly with
  `CUDA.set_runtime_version!`. Requires CUDA_Compiler_jll 0.6
  ([#3242](https://github.com/JuliaGPU/CUDA.jl/issues/3242)).
- Device intrinsics that need a minimum compute capability now fail at compile
  time with an explanatory error, instead of generating PTX that `ptxas`
  rejects. This affects 16-bit `atomic_cas!`, `dp4a`, `nanosleep`,
  `map_shared_rank` and the WMMA operations. The run-time compute-capability
  checks that used to select a compare-and-swap fallback for `Float16`,
  `Float64` and `BFloat16` atomic addition were removed; the back-end expands
  those on devices without native support
  ([#3202](https://github.com/JuliaGPU/CUDA.jl/pull/3202)).
- The `arch` and `ptx` keyword arguments of `@cuda` and `cufunction` now reject
  anything below compute capability 5.0 and PTX ISA 8.0, which is what the
  oldest supported toolkit (CUDA 12.0) and the oldest supported GPU imply
  ([#3202](https://github.com/JuliaGPU/CUDA.jl/pull/3202)).

*New features*:

- Added support for programmatic dependent launch, which overlaps the tail of
  one kernel with the preamble of the next one in the same stream. The consumer
  is launched with `@cuda dependent=true` (or `launch(...; dependent=true)`),
  the producer signals with `trigger_programmatic_launch_completion()`, and the
  consumer calls `grid_dependency_synchronize()` before reading the producer's
  results. Requires compute capability 9.0 or higher
  ([#3225](https://github.com/JuliaGPU/CUDA.jl/pull/3225)).
- Added a low-level API for conversion-free launches: `KernelCall` converts a
  kernel's function and arguments once, `kernel_compile` compiles the call,
  `kernel_launch` launches it without converting again, and `rebind` replaces a
  single argument. CUDA.jl's KernelAbstractions back-end uses it when selecting
  a workgroup size, which removes a second conversion from common operations
  such as broadcast ([#3197](https://github.com/JuliaGPU/CUDA.jl/pull/3197)).
- The cuDNN wrappers were rebuilt on cuDNN 9's backend graph API. `Graph` and
  `Tensor` describe a computation, `build!` lowers it and picks an execution
  plan through cuDNN's heuristics (constrained by `deterministic`, `math_mode`
  and `max_workspace`), `is_supported` reports whether a plan exists, `execute!`
  binds arrays and runs it, and `cached_graph` caches plans per cuDNN handle. On
  top of that sit `attention!`/`attention_backward!`, `convolution!` and its
  gradients, `maxpool!`/`meanpool!` and their gradients, and the `batchnorm_*`
  operations. The fixed-function wrappers (`cudnnConvolutionForward!`,
  `cudnnActivationForward!`, ...) are unchanged and moved to a `legacy`
  subdirectory. Requires CUDNN_jll 9.24
  ([#3191](https://github.com/JuliaGPU/CUDA.jl/pull/3191)).
- cuDNN's `attention_backward!` is implemented as the composite backward graph
  that the fused-attention engines actually pattern-match, and works for the
  first time. It supports grouped-query attention and dense padding masks, in
  Float16 and BFloat16 ([#3238](https://github.com/JuliaGPU/CUDA.jl/pull/3238)).
- cuDNN gained block-scaled (MXFP8/NVFP4) graph operations
  `block_scale_quantize!` and `block_scale_dequantize!`, for narrow-precision
  matmuls with block-scaled operands, on cuDNN 9.7 and later. Tensors gained a
  `reordering` attribute, and the binding checks of `execute!` are exposed as
  `checked_array_pointer` so packages can bind array types whose layout a dense
  stride comparison cannot express
  ([#3230](https://github.com/JuliaGPU/CUDA.jl/pull/3230)).
- cuDNN's normalization operations accept `:layernorm` and `:rmsnorm` in
  addition to batch normalization, in both `norm_fwd!` and `norm_bwd!`, with
  per-mode parameter and statistic shapes and an optional bias
  ([#3232](https://github.com/JuliaGPU/CUDA.jl/pull/3232)).
- Added support for cuTENSOR 2.6 and 2.7 (requires CUTENSOR_jll 2.7).
  Block-sparse `contract!` and `plan_contraction` take a `reproducible` keyword
  argument for bitwise reproducible contractions, and the compute-descriptor
  list gained the 16BF and FP-emulation descriptors (`9X16BF`, `8XINT8`,
  `4X16F`) that Hopper and Blackwell use
  ([#3239](https://github.com/JuliaGPU/CUDA.jl/pull/3239)).
- Added support for CUDA 13.4: the wrappers were regenerated, `CuMemoryPool`
  adapted to the extended `CUmemPoolProps`, and the compatibility databases
  record PTX ISA 9.4 and `sm_107`
  ([#3217](https://github.com/JuliaGPU/CUDA.jl/pull/3217)).

*Minor changes*:

- A warmed-up `@cuda` launch with `CuArray` arguments allocates nothing on Julia
  1.12 and later, by keeping kernel arguments, launch configuration, attributes
  and driver query outputs in stack-promotable `Ref`s
  ([#3196](https://github.com/JuliaGPU/CUDA.jl/pull/3196)).
- Kernel launch and first-use latency were reduced further: launches with a
  runtime-length argument splat go through a function barrier instead of
  dispatching dynamically at every stage, managed-memory tracking no longer
  allocates and queries the capture state once per launch, and context creation
  and reflection are precompiled. Locally this cut type-unstable launch overhead
  from 2625 ns and 21 allocations to 1653 ns and 6, and first- kernel latency by
  about 150 ms ([#3237](https://github.com/JuliaGPU/CUDA.jl/pull/3237)).
- Fixed several thread-safety issues: the compiler configuration cache is
  protected by its own lock (reflection and direct callers bypass the
  compilation lock), and stream ownership of managed allocations is now
  transferred as part of the launch itself, holding the per-allocation locks
  through submission, so a concurrent task can no longer take ownership between
  argument conversion and the actual launch
  ([#3229](https://github.com/JuliaGPU/CUDA.jl/pull/3229)).
- The compatibility databases were verified against every `ptxas` from CUDA 12.0
  to 13.4. This adds `sm_88`, the `sm_110` family and `sm_101`, which LLVM knew
  about but CUDA.jl did not, and corrects the toolkit lower bounds for compute
  capability 10.3 and 11.0
  ([#3217](https://github.com/JuliaGPU/CUDA.jl/pull/3217)).
- CUPTI now decodes the newest kernel, memcpy, memset and memory activity record
  that each CUDA release emits, instead of pinning them to the CUDA 12.0 layout
  and hiding the fields newer releases report
  ([#3217](https://github.com/JuliaGPU/CUDA.jl/pull/3217)).
- The accuracy of `^` no longer depends on the width of the integer exponent.
  Both widths now take the accurate path; libdevice's cheaper integer powers
  remain available as `powi`
  ([#3218](https://github.com/JuliaGPU/CUDA.jl/pull/3218)).
- `CuDeviceArray` no longer defines its own `axes` method, which was
  invalidating `Base`'s on Julia 1.13
  ([#3221](https://github.com/JuliaGPU/CUDA.jl/pull/3221)).
- When the driver refuses to load a compiled kernel image, the cubin is written
  to a temporary file and its path reported, mirroring what already happened for
  PTX when `ptxas` fails
  ([#3216](https://github.com/JuliaGPU/CUDA.jl/pull/3216)).

*Bug fixes*:

- `mapreducedim!` returns the destination array that was passed in, instead of
  the reshaped view it used internally
  ([#3219](https://github.com/JuliaGPU/CUDA.jl/pull/3219)).
- Fixed the three-argument `ldiv!(x, ::QR, b)` for right-hand sides with more
  than one column, which threw a `DimensionMismatch`
  ([#3199](https://github.com/JuliaGPU/CUDA.jl/pull/3199)).
- Fixed world-age errors when library handles are destroyed from CUDA.jl's
  memory reclamation callback, whose compiled world can predate the owning
  library's destructor ([#3192](https://github.com/JuliaGPU/CUDA.jl/pull/3192)).
- Fixed cuDNN's dropout state cache, which was keyed on the per-task handle
  wrapper instead of the raw cuDNN handle
  ([#3192](https://github.com/JuliaGPU/CUDA.jl/pull/3192)).
- Fixed cuSPARSE conversions of empty COO and BSR matrices
  ([#3192](https://github.com/JuliaGPU/CUDA.jl/pull/3192)).
- Fixed the workspace pointer type of cuTENSOR's block-sparse contraction
  ([#3192](https://github.com/JuliaGPU/CUDA.jl/pull/3192)).
- The precompilation workload now selects a PTX ISA supported by both LLVM and
  `ptxas`, so the artifacts it caches match the ones used at run time
  ([#3198](https://github.com/JuliaGPU/CUDA.jl/pull/3198)).
- Fixed the colors of NVTX ranges in `CUDA.@profile` output, which are ARGB
  while Crayons expects RGB, and lifted the temporary restriction to Crayons 4.1
  ([#3212](https://github.com/JuliaGPU/CUDA.jl/pull/3212),
  [#3215](https://github.com/JuliaGPU/CUDA.jl/pull/3215)).
- Fixed the Enzyme rule for `cudaconvert` with `Const` inputs, which constructed
  a shadow from a value that has none
  ([#3233](https://github.com/JuliaGPU/CUDA.jl/pull/3233)).
- Narrowed the data operand of the `trsm` wrappers to CUDA arrays, resolving a
  dispatch ambiguity with GPUArrays' generic triangular solves
  ([#3240](https://github.com/JuliaGPU/CUDA.jl/pull/3240)).
- Materializing the `Q` factor of a QR decomposition now uses `orgqr` directly,
  avoiding an overflow in `ormqr`'s 32-bit workspace size for tall matrices
  ([#3234](https://github.com/JuliaGPU/CUDA.jl/issues/3234)).
- Applying the `Q` factor of a QR decomposition (`Q * B`, `Q' * b`, `F \ b`) and
  `orgqr!` on very tall matrices now apply the reflectors in blocks, keeping
  cuSOLVER's workspace size within 32-bit range instead of failing with
  `CUSOLVER_STATUS_INVALID_VALUE`
  ([#3234](https://github.com/JuliaGPU/CUDA.jl/issues/3234)).


## v6.2 (June 2026)

Machine-code generation no longer uses the LLVM that ships with Julia. CUDA.jl
now invokes an external `llc` from `NVPTX_LLVM_Backend_jll` (LLVM 22), with the
in-process LLVM only driving the middle end. PTX target selection is therefore
no longer limited by the Julia version, and a number of long-standing codegen
workarounds could be dropped.

*Technically breaking changes*:

- The memory reclamation API was reworked. `CUDA.reclaim(sz::Int)`, which
  returned the number of bytes freed, is gone;
  `CUDA.reclaim(level::CUDA.ReclaimLevel = RECLAIM_DROP)` takes an escalation
  level (`RECLAIM_PURGE`, `RECLAIM_SYNC`, `RECLAIM_GC`, `RECLAIM_DROP`) and
  returns `nothing` ([#3118](https://github.com/JuliaGPU/CUDA.jl/pull/3118)).
- Conversions of library enums to Julia types are no longer `convert` methods:
  use `Type(x)` instead of `convert(Type, x)` for `cudaDataType`,
  `cusparseIndexType_t`, `custatevecComputeType_t` and
  `cutensornetComputeType_t`. The old methods were type piracy and caused
  invalidations downstream
  ([#3126](https://github.com/JuliaGPU/CUDA.jl/pull/3126)).
- Kernel arguments whose host-side layout differs from the device-side one are
  now rejected with an error instead of silently miscompiling. This affects
  aggregates containing `Int128`/`UInt128` fields on Julia 1.10 and 1.11, which
  align 128-bit integers to 8 bytes where the device always uses 16
  ([#3169](https://github.com/JuliaGPU/CUDA.jl/pull/3169),
  [#3170](https://github.com/JuliaGPU/CUDA.jl/pull/3170)).

*New features*:

- Added support for family-specific (`sm_100f`) and architecture-specific
  (`sm_90a`) PTX targets, giving access to architecture-accelerated instructions
  such as `wgmma` and `tcgen05`. A new `SMVersion` type and `sm"..."` string
  macro identify these targets, and `@cuda`/`cufunction` take them through a new
  `arch=` keyword (e.g. `arch=sm"103a"`). The old `cap=` keyword is accepted as
  a deprecated alias, and the default remains the forward-compatible baseline
  target ([#3124](https://github.com/JuliaGPU/CUDA.jl/pull/3124)).
- Relying on the external LLVM 22 back-end adds `sm_88` and `sm_110` targets and
  newer PTX ISA versions, and lowers `llvm.`-prefixed intrinsics the in-process
  LLVM does not know without falling back to the device runtime
  ([#3162](https://github.com/JuliaGPU/CUDA.jl/pull/3162),
  [#3169](https://github.com/JuliaGPU/CUDA.jl/pull/3169)).
- Float16 atomic addition now uses `atomicrmw fadd` instead of inline assembly,
  and BFloat16 arrays support atomic add/sub on Julia 1.11 and later (with a
  compare-and-swap fallback below sm_70 resp. sm_90)
  ([#3169](https://github.com/JuliaGPU/CUDA.jl/pull/3169)).
- Added directed-rounding arithmetic intrinsics `add_*`, `sub_*`, `mul_*`,
  `div_*` and `fma_*` with `rn`/`rz`/`rm`/`rp` suffixes, for `Float32` and
  `Float64`, mirroring the CUDA C `__fadd_rn`-style intrinsics
  ([#2576](https://github.com/JuliaGPU/CUDA.jl/pull/2576)).
- Added the `dp4a` intrinsic, the packed 4-element 8-bit dot product with 32-bit
  accumulate, in all four signedness variants
  ([#3163](https://github.com/JuliaGPU/CUDA.jl/pull/3163)).
- `log`, `log2`, `log10`, `exp`, `exp2` and `exp10` on `Float16` now use the
  hardware approximation instructions with a fix-up table, preserving
  IEEE-correct results, and `tanh_fast` gained a `Float16` method using
  `tanh.approx.f16` ([#2644](https://github.com/JuliaGPU/CUDA.jl/pull/2644)).
- Added support for CUDA toolkit 13.3
  ([#3155](https://github.com/JuliaGPU/CUDA.jl/pull/3155)). This includes cuBLAS
  emulated compute types, selectable through `CUDA.math_mode!(...;
  precision=:BFloat16x9)` for FP32 on tensor cores and `precision=:FixedPoint`
  for FP64, and the generic cuSPARSE SpGEAM API for `geam` on CSR matrices.
- cuDNN now supports `BFloat16`
  ([#3136](https://github.com/JuliaGPU/CUDA.jl/pull/3136)).
- cuStateVec gained `measureBatched!` and `expectationBatched`
  ([#2728](https://github.com/JuliaGPU/CUDA.jl/pull/2728)).
- cuSPARSE: added `similar` methods for `CuSparseMatrixCOO` and
  `CuSparseMatrixBSR`, `adapt_storage` methods for the explicit CSC/CSR/COO
  formats, and a generic `mv!` for `CuSparseMatrixBSR`
  ([#3114](https://github.com/JuliaGPU/CUDA.jl/pull/3114),
  [#3129](https://github.com/JuliaGPU/CUDA.jl/pull/3129),
  [#2929](https://github.com/JuliaGPU/CUDA.jl/pull/2929)).
- cuSPARSE: sparse/dense matrix multiplication now accepts mixed element types
  by promoting the operands, and dense × sparse allocates a dense result
  ([#3137](https://github.com/JuliaGPU/CUDA.jl/pull/3137)).
- `norm(view(x, ...), 2)` now dispatches to cuBLAS instead of falling back to
  scalar indexing ([#2302](https://github.com/JuliaGPU/CUDA.jl/pull/2302)).
- Added `compiler_version()`, reporting the version of the `ptxas` in use
  ([#3139](https://github.com/JuliaGPU/CUDA.jl/pull/3139)).
- `CUDA.versioninfo()` now reports the compilation target selected for each
  device, and the version of the external back-end LLVM
  ([#3124](https://github.com/JuliaGPU/CUDA.jl/pull/3124),
  [#3162](https://github.com/JuliaGPU/CUDA.jl/pull/3162)).

*Minor changes*:

- The CUDA runtime version is now selected based on the devices present in the
  system, not just the driver version, so a machine with e.g. a Volta GPU no
  longer ends up on a CUDA 13 runtime that does not support it
  ([#3134](https://github.com/JuliaGPU/CUDA.jl/pull/3134)).
- Toolkit version detection was improved: the runtime version is now determined
  by inspecting the library on Windows rather than calling into it, and the
  compiler version is parsed from `ptxas` output and used to gate functionality
  ([#3139](https://github.com/JuliaGPU/CUDA.jl/pull/3139)).
- Kernel launches now draw their seed from a private task-local RNG, so
  launching a kernel no longer advances the user-visible `rand()` stream
  ([#3161](https://github.com/JuliaGPU/CUDA.jl/pull/3161)).
- Launch failures caused by register pressure are now reported as such, instead
  of as a generic thread-limit message
  ([#3138](https://github.com/JuliaGPU/CUDA.jl/pull/3138)).
- Device code relies less on `libdevice`, emitting plain LLVM IR that is
  post-processed into the intended PTX instead. This improves optimization and
  compatibility with LLVM tools such as Enzyme
  ([#3149](https://github.com/JuliaGPU/CUDA.jl/pull/3149)).
- `AbstractKernel`, `HostKernel`, `DeviceKernel`, `Const`, `exit` and
  `shfl_recurse` are now marked public
  ([#3156](https://github.com/JuliaGPU/CUDA.jl/pull/3156)).
- `cluster_arrive`, `cluster_arrive_relaxed` and `cluster_wait` are now marked
  as device functions, so calling them outside a kernel raises a proper error
  ([#3153](https://github.com/JuliaGPU/CUDA.jl/pull/3153)).

*Bug fixes*:

- Fixed `cublasHgemm` (`gemm!` on `Float16` arrays), which errored on the
  pointer types of its scalar arguments
  ([#3157](https://github.com/JuliaGPU/CUDA.jl/pull/3157)).
- Dense × COO matrix multiplication no longer requires the COO matrix to be
  sorted beforehand, which previously produced wrong results without warning
  ([#3005](https://github.com/JuliaGPU/CUDA.jl/pull/3005)).
- cuSPARSE: fixed conversion between 0×0 `CuSparseMatrixCSC` and
  `CuSparseMatrixCSR`, and fixed `densetosparse` leaving the pointer array
  uninitialized for empty matrices
  ([#2806](https://github.com/JuliaGPU/CUDA.jl/pull/2806),
  [#2575](https://github.com/JuliaGPU/CUDA.jl/pull/2575)).
- cuSPARSE: `similar` with three or more dimensions now returns a `CuArray`
  rather than a host array, matching what `SparseArrays` does
  ([#3091](https://github.com/JuliaGPU/CUDA.jl/pull/3091)).
- `sortperm` on an empty array no longer fails with a kernel launch error
  ([#3133](https://github.com/JuliaGPU/CUDA.jl/pull/3133)).
- Launch failures unrelated to shared memory are no longer misreported as
  shared-memory overflows for kernels that opted into more than 48 KiB of
  dynamic shared memory
  ([#3135](https://github.com/JuliaGPU/CUDA.jl/pull/3135)).
- cuBLASXt now checks peer capability before granting memory pool access,
  instead of deferring the failure to a later allocation
  ([#3166](https://github.com/JuliaGPU/CUDA.jl/pull/3166)).
- `CUDA.@profile` no longer trips over infinite throughput values for events
  that CUPTI timed at zero nanoseconds
  ([#3151](https://github.com/JuliaGPU/CUDA.jl/pull/3151)).
- Added a weak dependency from `CUDACore` on `CUDA`, preventing the package
  manager from installing an old CUDA.jl alongside the new split packages, which
  resolved but broke at runtime
  ([#3141](https://github.com/JuliaGPU/CUDA.jl/pull/3141)).

### v6.2.1 (July 2026)

- `device_reset!` is now a deprecated no-op: a device's primary context is kept
  alive for the lifetime of the process, so there is no state left to reset.
  Resetting devices never worked reliably
  ([#3178](https://github.com/JuliaGPU/CUDA.jl/pull/3178),
  [#3183](https://github.com/JuliaGPU/CUDA.jl/pull/3183)).
- Fixed cluster barriers silently trapping at launch on Julia 1.10 and 1.11,
  where the `llvm.nvvm.barrier.cluster.*` intrinsics are not known to the
  bundled LLVM ([#3188](https://github.com/JuliaGPU/CUDA.jl/pull/3188)).
- Fixed a stack overflow when calling `findnz` on a COO-formatted sparse matrix
  ([#3190](https://github.com/JuliaGPU/CUDA.jl/pull/3190)).
- Fixed `deepcopy` of a `CuArray` with a zero-size element type by skipping
  device-to-device copies of zero-byte arrays
  ([#3184](https://github.com/JuliaGPU/CUDA.jl/pull/3184)).
- Compute-capability and PTX ISA support queries now ignore patch versions, so a
  `ptxas` or back-end LLVM with a nonzero patch level no longer produces a
  spurious "not fully supported" warning
  ([#3176](https://github.com/JuliaGPU/CUDA.jl/pull/3176)).
- Added the narrow-precision floating-point formats (FP8, FP6, FP4) to the
  `cudaDataType` enumeration
  ([#3180](https://github.com/JuliaGPU/CUDA.jl/pull/3180)).
- `unsafe_cached_load` no longer promotes its index to 64 bits unnecessarily
  ([#3186](https://github.com/JuliaGPU/CUDA.jl/pull/3186)).


## v6.1 (April 2026)

*Technically breaking changes*:

- Random number generation for element types that cuRAND does not support now
  uses GPUArrays' Philox4x32 generator instead of CUDA.jl's own kernel-based
  one, which is considerably faster. `CUDA.RNG` is now an alias for
  `GPUArrays.RNG{CuArray}` and `CUDA.default_rng()` returns that generator; both
  were deprecated bindings to the kernel-based RNG before. The old generator
  remains available as `cuRAND.NativeRNG`, but is only meant for testing. In
  cuRAND, `default_rng` was renamed to `library_rng` and the `RNG` type to
  `LibraryRNG` ([#3102](https://github.com/JuliaGPU/CUDA.jl/pull/3102),
  [#3056](https://github.com/JuliaGPU/CUDA.jl/pull/3056)).
- `plan_fft` and friends now reject duplicate transform dimensions instead of
  silently deduplicating them, and `plan_rfft`/`plan_brfft` raise an
  `ArgumentError` up front when the transform region is not strictly increasing
  ([#3052](https://github.com/JuliaGPU/CUDA.jl/pull/3052)).

*New features*:

- cuFFT plan construction was reworked to pick batching dimensions based on
  performance rather than statically splitting them. This removes a pathological
  slowdown when transforming along a non-innermost dimension of a large array:
  `fft(x, 2)` on a `7×7×1_000_000` array went from about 15 s to 0.1 s. Complex
  transforms now also accept the transform dimensions in any order
  ([#3052](https://github.com/JuliaGPU/CUDA.jl/pull/3052),
  [#119](https://github.com/JuliaGPU/CUDA.jl/pull/119)).
- `@cuda` gained a `backend` keyword and a small dispatch protocol
  (`AbstractBackend`, `kernel_convert`, `kernel_compile`) so alternative
  compilers can plug into the launch syntax. The default remains `LLVMBackend`,
  i.e. `cudaconvert` plus `cufunction`
  ([#3121](https://github.com/JuliaGPU/CUDA.jl/pull/3121)).
- cuTENSOR now wraps the block-sparse API, with a `CuTensorBS` tensor type,
  block-sparse contractions, and a `LinearAlgebra.mul!` method
  ([#3057](https://github.com/JuliaGPU/CUDA.jl/pull/3057)).
- The compiler binaries (`ptxas`, `nvlink`, libdevice) can be taken from a local
  CUDA installation instead of the artifacts by setting the `local_compiler`
  preference. `CUDA.versioninfo()` reports which of the two is in use
  ([#3080](https://github.com/JuliaGPU/CUDA.jl/pull/3080)).
- cuSPARSE `CuSparseMatrixCSR` matrices can be sliced with boolean masks,
  `A[rowmask, colmask]`
  ([#3032](https://github.com/JuliaGPU/CUDA.jl/pull/3032)).
- Wrapped the new cuTensorNet gradient and MPS-projection entry points from the
  cuQuantum 26.3 headers
  ([#3107](https://github.com/JuliaGPU/CUDA.jl/pull/3107)).

*Minor changes*:

- When using a local CUDA installation, CUDA.jl no longer requires the `version`
  preference to match the installed toolkit; the version check now only applies
  to artifact-based installations.
- The `.target` directive in generated PTX now matches the architecture that
  `ptxas` is asked to compile for, instead of whatever LLVM was able to emit
  ([#3120](https://github.com/JuliaGPU/CUDA.jl/pull/3120)).
- `CUDA.fast_div` is now implemented in terms of the fast reciprocal intrinsic,
  and `muladd`/`fma` map onto the corresponding LLVM intrinsics on the device
  ([#3077](https://github.com/JuliaGPU/CUDA.jl/pull/3077),
  [#3078](https://github.com/JuliaGPU/CUDA.jl/pull/3078)).
- The cuSOLVER-backed eigenvalue solvers use the 64-bit API
  ([#3084](https://github.com/JuliaGPU/CUDA.jl/pull/3084)).
- `CuArray` stores its offset in bytes rather than elements, which makes
  reinterpreting a view to a larger element type work when the byte offset is
  not a multiple of the new element size
  ([#3088](https://github.com/JuliaGPU/CUDA.jl/pull/3088)).
- Accessing another device's memory from a kernel is temporarily disabled,
  working around a driver bug that could corrupt memory pools
  ([#3112](https://github.com/JuliaGPU/CUDA.jl/pull/3112)).

*Bug fixes*:

- Unified and host memory is now accounted against system RAM by the eager GC
  heuristic, so allocating unified memory in a loop no longer looks like a leak.
  Early GC is also disabled, with a warning, when querying free memory fails
  because of an earlier CUDA error
  ([#3092](https://github.com/JuliaGPU/CUDA.jl/pull/3092),
  [#3013](https://github.com/JuliaGPU/CUDA.jl/pull/3013)).
- Fixed a GC race in `getindex` on `CuRefValue`/`CuRefArray` where the reference
  could be freed before the copy completed, causing intermittent `invalid
  argument` errors or segfaults under memory pressure
  ([#3087](https://github.com/JuliaGPU/CUDA.jl/pull/3087)).
- `randn` and `randexp` no longer exhaust the per-thread stack in larger
  kernels; the Ziggurat rejection loop no longer recurses
  ([#3086](https://github.com/JuliaGPU/CUDA.jl/pull/3086)).
- `unsafe_cached_load` (`ldg`) works again on LLVM 20 and later, which removed
  the `llvm.nvvm.ldg.global.*` intrinsics in favor of invariant loads
  ([#3094](https://github.com/JuliaGPU/CUDA.jl/pull/3094),
  [#2531](https://github.com/JuliaGPU/CUDA.jl/pull/2531)).
- Added device overrides so that comparing a float with a `Rational`, calling
  `Base.FastMath.pow_fast` with an integer exponent, and hitting a dimension
  mismatch in `reshape` all compile on the GPU
  ([#3093](https://github.com/JuliaGPU/CUDA.jl/pull/3093),
  [#3098](https://github.com/JuliaGPU/CUDA.jl/pull/3098),
  [#3095](https://github.com/JuliaGPU/CUDA.jl/pull/3095),
  [#2681](https://github.com/JuliaGPU/CUDA.jl/pull/2681),
  [#3065](https://github.com/JuliaGPU/CUDA.jl/pull/3065)).
- `cu(x; unified=true)` and friends now propagate the requested memory type to
  sparse arrays instead of always producing device memory
  ([#3090](https://github.com/JuliaGPU/CUDA.jl/pull/3090),
  [#2974](https://github.com/JuliaGPU/CUDA.jl/pull/2974)).
- Scalar `getindex` on sparse arrays no longer assumes sorted, unique indices:
  entries are scanned linearly and duplicates are summed, matching `sparse()`
  ([#3101](https://github.com/JuliaGPU/CUDA.jl/pull/3101),
  [#3100](https://github.com/JuliaGPU/CUDA.jl/pull/3100)).
- `sparse(::Symmetric)` and `sparse(::Hermitian)` on a sparse GPU parent no
  longer overflow the stack, and addition or subtraction of
  `Symmetric`/`Hermitian` sparse matrices no longer falls back to scalar
  indexing ([#3108](https://github.com/JuliaGPU/CUDA.jl/pull/3108),
  [#3042](https://github.com/JuliaGPU/CUDA.jl/pull/3042),
  [#3043](https://github.com/JuliaGPU/CUDA.jl/pull/3043)).
- `LinearAlgebra.normalize` on a `CuArray` no longer triggers scalar indexing
  ([#3097](https://github.com/JuliaGPU/CUDA.jl/pull/3097)).
- Fixed dynamic dispatch in cuBLAS' `unsafe_strided_batch` on Julia 1.12
  ([#3083](https://github.com/JuliaGPU/CUDA.jl/pull/3083)).


## v6.0 (April 2026)

CUDA.jl has been split into several registered packages. `CUDACore` provides the
array type, the kernel compiler and the driver API; `CUDATools` provides the
profiler, code reflection, NVML and CUPTI; and the math libraries now live in
`cuBLAS`, `cuFFT`, `cuRAND`, `cuSOLVER` and `cuSPARSE`. The `CUDA` package
remains, and now is a meta-package that imports and re-exports all of the above,
so `using CUDA` continues to work as before. Code that only needs arrays and
kernels can instead depend on `CUDACore` alone, which avoids loading
`SparseArrays`, `AbstractFFTs` and the library wrappers, and cuts CUDA's own
import time from roughly 700ms to 200ms. All subpackages are versioned in
lockstep with CUDA.jl and released together.

*Breaking changes*:

- The `CUBLAS`, `CUFFT`, `CURAND`, `CUSOLVER` and `CUSPARSE` submodules are now
  separate packages, named `cuBLAS`, `cuFFT`, `cuRAND`, `cuSOLVER` and
  `cuSPARSE`. The old submodule names remain available as deprecated bindings
  ([#3058](https://github.com/JuliaGPU/CUDA.jl/pull/3058)).
- Profiling (`@profile`), code reflection (`@device_code_*`), `versioninfo`,
  NVML and CUPTI moved to the new CUDATools package. `using CUDA` re-exports it,
  so this is only visible when depending on CUDACore directly
  ([#3063](https://github.com/JuliaGPU/CUDA.jl/pull/3063)).
- `CUDA.RNG`, `CUDA.default_rng()` and `CUDA.has_cusolvermg` are deprecated in
  favour of `cuRAND.NativeRNG`, `cuRAND.native_rng()` and
  `cuSOLVER.has_cusolvermg`
  ([#3058](https://github.com/JuliaGPU/CUDA.jl/pull/3058)).
- Removed the long-deprecated `CUDA.Mem` submodule, along with
  `CUDA.memory_status()` and `CUDA.available_memory()`
  ([#3058](https://github.com/JuliaGPU/CUDA.jl/pull/3058)).
- cuDNN, cuTENSOR, cuStateVec and cuTensorNet moved to version 6.0 as part of
  the lockstep versioning, and now depend on CUDACore instead of CUDA
  ([#3058](https://github.com/JuliaGPU/CUDA.jl/pull/3058),
  [#3063](https://github.com/JuliaGPU/CUDA.jl/pull/3063)).

*New features*:

- Added support for thread block clusters on compute capability 9.0 and higher.
  Kernels can be launched with `@cuda clustersize=...` (also available as a
  keyword to the low-level `launch`), and query their position with
  `clusterIdx()`, `clusterDim()`, `blockIdxInCluster()`, `gridClusterDim()`,
  `linearBlockIdxInCluster()` and `linearClusterSize()`. Blocks in a cluster can
  synchronize using `cluster_arrive()`, `cluster_arrive_relaxed()` and
  `cluster_wait()` ([#3017](https://github.com/JuliaGPU/CUDA.jl/pull/3017)).
- Added `CuDistributedSharedArray`, which maps another block's shared memory
  array into the current block for use within a thread block cluster
  ([#3017](https://github.com/JuliaGPU/CUDA.jl/pull/3017)).
- `@device_override`, `@device_function` and `@device_functions` are now public
  API, and can be used from packages that do not have CUDACore in scope
  ([#3066](https://github.com/JuliaGPU/CUDA.jl/pull/3066)).

*Minor changes*:

- Time to first kernel is down from around 7.5s to 1.8s, and the first
  `CUDA.@profile` from around 17s to 3s, by precompiling a dummy kernel through
  the full PTX pipeline and by running the compiler in the world age captured at
  initialization time ([#3064](https://github.com/JuliaGPU/CUDA.jl/pull/3064)).
- All GPU-only functions are now registered in the GPU method table, so they no
  longer leak into host-side or ahead-of-time compilation
  ([#2998](https://github.com/JuliaGPU/CUDA.jl/pull/2998)).
- CUDA.jl and its subpackages now load without warnings on systems where LLVM
  has no NVPTX backend, such as macOS
  ([#3067](https://github.com/JuliaGPU/CUDA.jl/pull/3067)).
- Library workspace buffers are now freed before a larger one is allocated,
  rather than holding both at once, which lowers peak memory use of cuSOLVER and
  cuSPARSE operations ([#3062](https://github.com/JuliaGPU/CUDA.jl/pull/3062)).

*Bug fixes*:

- Fixed the range metadata attached to the index intrinsics, which excluded the
  largest legal value (e.g. a `threadIdx().x` of 1024) and could lead to
  miscompilation ([#3017](https://github.com/JuliaGPU/CUDA.jl/pull/3017)).
- The profiler no longer errors on CUPTI marker data that does not indicate a
  color ([#3075](https://github.com/JuliaGPU/CUDA.jl/pull/3075)).


## v5.11 (March 2026)

*New features*:

- Support for CUDA 13.2, regenerating the driver, CUPTI, cuSOLVER and NVML
  wrappers, and registering PTX ISA 9.0 through 9.2 in the compatibility
  database ([#3053](https://github.com/JuliaGPU/CUDA.jl/pull/3053)).
- Bumped the bundled libraries: cuDNN 9.20, cuTENSOR 2.5 and cuQuantum 26.1 for
  cuStateVec and cuTensorNet
  ([#3054](https://github.com/JuliaGPU/CUDA.jl/pull/3054)).
- cuTensorNet now accepts `BFloat16` as a compute type, and its optimizer
  configuration gained a `gpu_arch` option
  ([#3054](https://github.com/JuliaGPU/CUDA.jl/pull/3054)).

*Minor changes*:

- cuTENSOR supports more mixed-precision combinations for element-wise ternary
  and permutation operations, such as `Float64`/`Float64`/`Float32` and
  `Float32`/`Float16` ([#3054](https://github.com/JuliaGPU/CUDA.jl/pull/3054)).
- Added descriptions for the cuStateVec and cuTensorNet status codes that were
  missing one, so those errors no longer report "no description for this error"
  ([#3054](https://github.com/JuliaGPU/CUDA.jl/pull/3054)).
- Adapted memory pool creation and `access!` to the `CUmemLocation` layout
  change in CUDA 13.2 ([#3054](https://github.com/JuliaGPU/CUDA.jl/pull/3054)).

### v5.11.1 (April 2026)

- `CuArray` now stores its offset in bytes instead of elements, fixing wrong
  results when materializing a `reinterpret` of a view whose byte offset is not
  a multiple of the new element size, e.g. reinterpreting a `CuVector` view as
  `SVector` ([#3088](https://github.com/JuliaGPU/CUDA.jl/pull/3088),
  [#2980](https://github.com/JuliaGPU/CUDA.jl/pull/2980)).
- Device intrinsics are now defined through `@device_function`, keeping NVVM
  intrinsics out of the host method table so that packages using CUDA.jl can be
  compiled ahead of time
  ([#3072](https://github.com/JuliaGPU/CUDA.jl/pull/3072),
  [#2998](https://github.com/JuliaGPU/CUDA.jl/pull/2998)).
- `with_workspaces` frees the GPU workspace before growing it, avoiding an
  out-of-memory error from holding both the old and the new buffer on
  memory-constrained GPUs
  ([#3113](https://github.com/JuliaGPU/CUDA.jl/pull/3113)).

### v5.11.2 (April 2026)

- Added a `local_compiler` preference to take `ptxas`, `nvlink` and libdevice
  from a local CUDA installation, discovered through CUDA_Runtime_Discovery,
  instead of from the `CUDA_Compiler_jll` artifact. `CUDA.versioninfo()` reports
  which of the two is in use
  ([#3081](https://github.com/JuliaGPU/CUDA.jl/pull/3081),
  [#3080](https://github.com/JuliaGPU/CUDA.jl/pull/3080)).

### v5.11.3 (June 2026)

- Compatible with GPUCompiler.jl up to 1.17 and LLVM.jl 9.6, linking libdevice
  lazily so that only the referenced symbols are materialized
  ([#3131](https://github.com/JuliaGPU/CUDA.jl/pull/3131)).
- `sqrt` no longer goes through libdevice, which with recent GPUCompiler.jl
  versions lowered to the approximate `sqrt.approx` instead of `sqrt.rn`.
  `rsqrt` now calls the NVPTX `rsqrt.approx` intrinsics directly
  ([#3149](https://github.com/JuliaGPU/CUDA.jl/pull/3149)).
- The compiler parameters are extensible through a new
  `AbstractCUDACompilerParams` type, for packages like Enzyme.jl that reuse the
  CUDA.jl compilation pipeline
  ([#3168](https://github.com/JuliaGPU/CUDA.jl/pull/3168)).


## v5.10 (March 2026)

*New features*:

- Support for CUDA 13.1
  ([#3040](https://github.com/JuliaGPU/CUDA.jl/pull/3040)).
- Support for Julia 1.13
([#3020](https://github.com/JuliaGPU/CUDA.jl/pull/3020)). *Minor changes*:

- Texture support is broken on Julia 1.13 due to a crash in the LLVM 20 NVPTX
  back-end ([#3037](https://github.com/JuliaGPU/CUDA.jl/pull/3037)). *Bug
fixes*:

- `min` on `Float16` no longer emits a `min.NaN.f16` instruction, which requires
  sm_80 and broke Turing GPUs when building with LLVM 20
  ([#3038](https://github.com/JuliaGPU/CUDA.jl/pull/3038)).
- Non-pivoting `sytrf!` of complex matrices now throws instead of silently
  returning wrong results on cuSOLVER 12.0.9 (CUDA 13.1), where it performs a
  Hermitian instead of a symmetric factorization
  ([#3040](https://github.com/JuliaGPU/CUDA.jl/pull/3040)).

### v5.10.1 (March 2026)

- NVTX injection is now set up on the first use of `CUDA.@profile` instead of at
  package initialization, so that other CUPTI users are not interfered with
  ([#3050](https://github.com/JuliaGPU/CUDA.jl/pull/3050)).


## v5.9 (September 2025)

This release adds support for the CUDA 13 toolkit, and drops support for CUDA
11.

*Technically breaking changes*:

- Dropped support for CUDA 11. CUDA.jl now requires a CUDA 12 or 13 toolkit and
  an NVIDIA driver for CUDA 12 or later, and no longer supports Kepler GPUs: the
  minimum is now compute capability 5.0 (Maxwell). Stay on CUDA.jl v5.8 if you
  need CUDA 11 ([#2870](https://github.com/JuliaGPU/CUDA.jl/pull/2870),
  [#2897](https://github.com/JuliaGPU/CUDA.jl/pull/2897)).
- Raised the lower bounds of several dependencies, among others to GPUCompiler
  1.1, GPUArrays 11.2.4, BFloat16s 0.5, StaticArrays 1.6 and PrettyTables 3
([#2841](https://github.com/JuliaGPU/CUDA.jl/pull/2841)). *New features*:

- Added support for CUDA 13: the wrappers were regenerated against the CUDA 13
  headers, and the device compatibility checks were updated. CUDA 13 itself
  requires a GPU with compute capability 7.5 (Turing) or higher
  ([#2842](https://github.com/JuliaGPU/CUDA.jl/pull/2842),
  [#2867](https://github.com/JuliaGPU/CUDA.jl/pull/2867),
  [#2897](https://github.com/JuliaGPU/CUDA.jl/pull/2897)).
- Added compatibility entries for datacenter Blackwell GPUs, i.e. compute
  capabilities 10.0, 10.3 and 11.0 (GB200 and friends)
  ([#2882](https://github.com/JuliaGPU/CUDA.jl/pull/2882)).
- The CUBLAS wrappers now call the 64-bit (ILP64) entry points, so dimensions
  and strides are no longer limited to what fits in a 32-bit integer
  ([#2845](https://github.com/JuliaGPU/CUDA.jl/pull/2845)).
- Extended the LinearAlgebra support: `norm` of a `Diagonal`,
  `LinearAlgebra.norm2`, in-place `rmul!`/`lmul!` with `Diagonal`s,
  `rmul!`/`lmul!` with mixed element types, and additional `generic_matmatmul!`
methods ([#2854](https://github.com/JuliaGPU/CUDA.jl/pull/2854),
[#2856](https://github.com/JuliaGPU/CUDA.jl/pull/2856),
[#2858](https://github.com/JuliaGPU/CUDA.jl/pull/2858),
[#2860](https://github.com/JuliaGPU/CUDA.jl/pull/2860),
[#2862](https://github.com/JuliaGPU/CUDA.jl/pull/2862)). *Minor changes*:

- Reductions now spread the work over multiple thread blocks when there are too
  few independent slices to saturate the GPU
  ([#2869](https://github.com/JuliaGPU/CUDA.jl/pull/2869),
  [#2880](https://github.com/JuliaGPU/CUDA.jl/pull/2880)).
- Updated the bundled libraries: cuDNN now uses CUDNN_jll 9.12, cuTENSOR uses
  CUTENSOR_jll 2.3, and cuStateVec and cuTensorNet use cuQuantum 25.06
  ([#2874](https://github.com/JuliaGPU/CUDA.jl/pull/2874),
  [#2876](https://github.com/JuliaGPU/CUDA.jl/pull/2876)).
- `CUDA.@profile` works with PrettyTables 3
  ([#2853](https://github.com/JuliaGPU/CUDA.jl/pull/2853),
  [#2891](https://github.com/JuliaGPU/CUDA.jl/pull/2891)).
- Various fixes for Julia 1.12
  ([#2883](https://github.com/JuliaGPU/CUDA.jl/pull/2883),
  [#2888](https://github.com/JuliaGPU/CUDA.jl/pull/2888)).
- On Windows, libraries from the driver store no longer trigger a warning about
  CUDA runtime libraries being loaded from a system path
  ([#2847](https://github.com/JuliaGPU/CUDA.jl/pull/2847)).
- NVTX events are forwarded to CUPTI again on CUDA 13.0 update 1 and later,
  where the NVTX injection bug that disabled this has been fixed
([#2843](https://github.com/JuliaGPU/CUDA.jl/pull/2843),
[#2884](https://github.com/JuliaGPU/CUDA.jl/pull/2884)). *Bug fixes*:

- Fixed `rmul!` on `Transpose` and `Adjoint` wrappers
  ([#2871](https://github.com/JuliaGPU/CUDA.jl/pull/2871)).

### v5.9.1 (October 2025)

- Fixed memory corruption in `mapreduce` that made `sum(x; dims)` and friends
  return wrong results, a regression introduced in v5.9.0
  ([#2907](https://github.com/JuliaGPU/CUDA.jl/pull/2907)).
- Fixed the device compatibility check wrongly rejecting Turing GPUs (compute
  capability 7.5) on CUDA 13
  ([#2939](https://github.com/JuliaGPU/CUDA.jl/pull/2939)).
- Worked around LLVM 18 and earlier emitting nonexistent
  `min.NaN.f64`/`max.NaN.f64` PTX instructions for fast-math `min`/`max`
  ([#2937](https://github.com/JuliaGPU/CUDA.jl/pull/2937)).
- `resize!` on a `CuVector` now over-allocates, doubling small arrays and
  growing arrays over 100 MiB by 32 MiB at a time, so repeatedly growing an
  array no longer reallocates every time
  ([#2828](https://github.com/JuliaGPU/CUDA.jl/pull/2828)).
- Added the 3-argument `dot(x, A, y)`
  ([#2914](https://github.com/JuliaGPU/CUDA.jl/pull/2914)).
- Added `nonzeros`, `nonzeroinds`, `rowvals` and `nnz` for column views of a
  `CuSparseDeviceMatrixCSC`, for use in kernels
  ([#2904](https://github.com/JuliaGPU/CUDA.jl/pull/2904)).
- Fixed `copyto!` between `Symmetric` and `Hermitian` CuMatrices, which had
  started falling back to a dense copy
  ([#2913](https://github.com/JuliaGPU/CUDA.jl/pull/2913)).
- Removed the `sv2!` and `sm2!` wrappers for CSR and CSC matrices, which used
  CUSPARSE functions deprecated since CUDA 11; use `sv!` and `sm!` instead
  ([#2919](https://github.com/JuliaGPU/CUDA.jl/pull/2919),
  [#2926](https://github.com/JuliaGPU/CUDA.jl/pull/2926)).
- Fixed `CUDA.return_type`, CUSOLVER, and `Symmetric`/`Hermitian` multiplication
  on Julia 1.12 ([#2917](https://github.com/JuliaGPU/CUDA.jl/pull/2917),
  [#2923](https://github.com/JuliaGPU/CUDA.jl/pull/2923),
  [#2928](https://github.com/JuliaGPU/CUDA.jl/pull/2928),
  [#2932](https://github.com/JuliaGPU/CUDA.jl/pull/2932)).
- `KernelAbstractions.allocate` and friends are type stable again, and `device!`
  now accepts an `Int` ([#2906](https://github.com/JuliaGPU/CUDA.jl/pull/2906),
  [#2920](https://github.com/JuliaGPU/CUDA.jl/pull/2920)).
- cuTensorNet now passes `Int64` strides, as expected by the library
  ([#2915](https://github.com/JuliaGPU/CUDA.jl/pull/2915)).

### v5.9.2 (October 2025)

- Relaxed the driver and device compatibility checks: the supported devices and
  PTX versions are now derived from the CUDA toolkit version alone, relying on
  driver backwards compatibility. This also removes the warning about a toolkit
  and driver with different major versions
  ([#2941](https://github.com/JuliaGPU/CUDA.jl/pull/2941)).
- `gemmStridedBatchedEx!` now accepts a batch size of 1 for `A` or `B`, matching
  `gemm_strided_batched!`
  ([#2935](https://github.com/JuliaGPU/CUDA.jl/pull/2935)).

### v5.9.3 (November 2025)

- Raised the minimum StaticArrays version to 1.9.8, avoiding breakage with older
  releases ([#2961](https://github.com/JuliaGPU/CUDA.jl/pull/2961)).
- The cuDNN subpackage now requires CUDA.jl v5.9.

### v5.9.4 (November 2025)

- Fixed sparse matrix-vector multiplication with adjoints of complex
  `CuSparseMatrixCSC` matrices, which returned wrong results since v5.9.0
  ([#2957](https://github.com/JuliaGPU/CUDA.jl/pull/2957)).
- `mul!` with empty matrices or vectors now honors `beta` instead of zeroing the
  output, and `rmul!(x, false)` follows Julia's strong-zero semantics, so `NaN *
  false` is `0` ([#2958](https://github.com/JuliaGPU/CUDA.jl/pull/2958)).
- `CUSOLVER.XsyevBatched!` now accepts a 3-dimensional `StridedCuArray` of
  matrices ([#2951](https://github.com/JuliaGPU/CUDA.jl/pull/2951)).
- `SparseArrays.nzrange` on a `CuSparseDeviceMatrixCSC` no longer bounds checks
  ([#2970](https://github.com/JuliaGPU/CUDA.jl/pull/2970)).

### v5.9.5 (November 2025)

- Added the matrix functions `exp`, `cos`, `sin`, `tan`, `cosh`, `sinh`, `tanh`,
  `atan`, `asinh`, `atanh` and `cbrt` for `Symmetric` and `Hermitian` CuMatrices
  ([#2962](https://github.com/JuliaGPU/CUDA.jl/pull/2962)).
- Added `mul!` with a `Diagonal` destination
  ([#2977](https://github.com/JuliaGPU/CUDA.jl/pull/2977)).
- `diagm` on a `CuVector` no longer converts the element type to `Float32`, and
  now uses the GPUArrays implementation
  ([#2975](https://github.com/JuliaGPU/CUDA.jl/pull/2975),
  [#2979](https://github.com/JuliaGPU/CUDA.jl/pull/2979)).

### v5.9.6 (January 2026)

- `eigen` now also handles non-symmetric and non-Hermitian matrices, using
  CUSOLVER's `Xgeev`, and `eigvals` and `eigvecs` were added
  ([#2787](https://github.com/JuliaGPU/CUDA.jl/pull/2787)).
- Added BFloat16 support to the `WMMA` intrinsics, requiring an Ampere or newer
  GPU and a Float32 accumulator
  ([#3009](https://github.com/JuliaGPU/CUDA.jl/pull/3009)).
- Sparse arrays now build on the GPUArrays sparse array hierarchy, sharing
  broadcasting, reductions and several linear algebra operations with other
  back-ends. The `AbstractCuSparseArray` type was removed
  ([#2942](https://github.com/JuliaGPU/CUDA.jl/pull/2942)).
- Added `log` for `Hermitian` CuMatrices
  ([#2993](https://github.com/JuliaGPU/CUDA.jl/pull/2993)).
- cuStateVec and cuTensorNet now use cuQuantum 25.11
  ([#2887](https://github.com/JuliaGPU/CUDA.jl/pull/2887)).
- Removed the dependency on Requires.jl, which package extensions made
  unnecessary ([#2999](https://github.com/JuliaGPU/CUDA.jl/pull/2999)).
- `CUDA.versioninfo()` also reports the GPUArrays, GPUCompiler and
  KernelAbstractions versions
  ([#2983](https://github.com/JuliaGPU/CUDA.jl/pull/2983)).
- Fixed level-1 CUBLAS operations on zero-length arrays
  ([#2994](https://github.com/JuliaGPU/CUDA.jl/pull/2994)).
- Fixed the error message for unsupported `CuTexture` element types
  ([#2990](https://github.com/JuliaGPU/CUDA.jl/pull/2990)).

### v5.9.7 (February 2026)

- The profiler no longer depends on DataFrames.jl, which reduces CUDA.jl's load
  time. `CUDA.@profile` results are now NamedTuples of vectors instead of
  DataFrames ([#3029](https://github.com/JuliaGPU/CUDA.jl/pull/3029)).
- Extended the workaround for LLVM 18 emitting `.NaN` PTX modifiers, which
  require sm_80 or higher, to fast-math `min`/`max` on `Float16` and `Float32`,
  and to `max` on `Float16`
  ([#3016](https://github.com/JuliaGPU/CUDA.jl/pull/3016),
  [#3025](https://github.com/JuliaGPU/CUDA.jl/pull/3025)).
- `transpose!` on a `CuMatrix` of BLAS element types now uses CUBLAS `geam!`
  instead of the generic GPUArrays implementation
  ([#3015](https://github.com/JuliaGPU/CUDA.jl/pull/3015)).
- `accumulate!` now forwards keyword arguments such as `neutral` to `scan!`
  ([#3011](https://github.com/JuliaGPU/CUDA.jl/pull/3011)).
- The documentation now includes an API reference for every wrapped library
  ([#2972](https://github.com/JuliaGPU/CUDA.jl/pull/2972)).


## v5.8 (May 2025)

*New features*:

- `CuSparseVector` now supports broadcasting, matching what was already possible
  with sparse matrices. Zero-preserving operations return a `CuSparseVector`,
  other operations return a dense `CuArray`
  ([#2733](https://github.com/JuliaGPU/CUDA.jl/pull/2733)).
- Added support for CUDA 12.9
  ([#2772](https://github.com/JuliaGPU/CUDA.jl/pull/2772)).
- `CUSPARSE.gemm!` now supports the `CUSPARSE_SPGEMM_ALG2` and
  `CUSPARSE_SPGEMM_ALG3` algorithms, which bound the memory used by sparse
  matrix-matrix multiplication
  ([#2769](https://github.com/JuliaGPU/CUDA.jl/pull/2769)).
- Added a SparseMatricesCSR.jl extension, converting between `SparseMatrixCSR`
  and the `CuSparseMatrix` types, and hooking into `adapt`
  ([#2720](https://github.com/JuliaGPU/CUDA.jl/pull/2720)).
- `CuArray`s can now hold `Symbol`s and other mutable singleton types, and
  `unsafe_wrap` accepts those element types as well
  ([#2753](https://github.com/JuliaGPU/CUDA.jl/pull/2753),
  [#2756](https://github.com/JuliaGPU/CUDA.jl/pull/2756)).
- Implemented the KernelAbstractions 0.9.32 interfaces: `pagelock!`, `ndevices`,
  `device` and `device!`
  ([#2774](https://github.com/JuliaGPU/CUDA.jl/pull/2774)).

*Minor changes*:

- Subpackages were updated to cuDNN 9.10, cuTENSOR 2.2 and cuQuantum 25.03
  ([#2776](https://github.com/JuliaGPU/CUDA.jl/pull/2776)).
- `cuTENSOR` multiplication now preserves the memory type of its inputs instead
  of always returning device memory
  ([#2775](https://github.com/JuliaGPU/CUDA.jl/pull/2775)).

*Bug fixes*:

- Fixed `CUSOLVER.gesvdp!` when only the singular values are requested
  ([#2763](https://github.com/JuliaGPU/CUDA.jl/pull/2763)).
- Fixed a memory leak caused by CUDA log messages piling up, most visible when
  using cuDNN from Pluto.jl
  ([#2750](https://github.com/JuliaGPU/CUDA.jl/pull/2750),
  [#2754](https://github.com/JuliaGPU/CUDA.jl/pull/2754)).
- Fixed a dispatch error in `sum!` and other reductions past a threshold number
  of rows ([#2778](https://github.com/JuliaGPU/CUDA.jl/pull/2778)).
- Fixed matrix-matrix multiplication with `CuSparseMatrixBSR`
  ([#2747](https://github.com/JuliaGPU/CUDA.jl/pull/2747)).

### v5.8.1 (May 2025)

- Fixed several bugs in the new `CuSparseVector` broadcasting
  ([#2780](https://github.com/JuliaGPU/CUDA.jl/pull/2780)).

### v5.8.2 (May 2025)

- `spdiagm` now accepts `Pair`s specifying which diagonals to fill
  ([#2784](https://github.com/JuliaGPU/CUDA.jl/pull/2784)).
- Added `diagm` for `CuVector` inputs
  ([#2786](https://github.com/JuliaGPU/CUDA.jl/pull/2786)).

### v5.8.3 (August 2025)

- `ptxas` and the other compiler tools are now provided by a separately
  versioned `CUDA_Compiler_jll`, picking the newest version compatible with the
  driver instead of the one matching the selected runtime. This avoids code
  generation bugs in older toolkits and makes it possible to target recent
  devices and PTX ISAs regardless of the runtime in use. The version is
  available as `CUDA.compiler_version()`
  ([#2801](https://github.com/JuliaGPU/CUDA.jl/pull/2801)).
- Initial compatibility with CUDA 13
  ([#2834](https://github.com/JuliaGPU/CUDA.jl/pull/2834)).
- `KernelAbstractions.allocate`, `zeros` and `ones` now take a `unified` keyword
  argument to allocate unified memory
  ([#2819](https://github.com/JuliaGPU/CUDA.jl/pull/2819)).
- `CuArray(::Diagonal)` and similar conversion constructors no longer preserve
  the wrapper; they now always collect the diagonal into a dense matrix. Use
  `adapt` for a shape-preserving conversion
  ([#2805](https://github.com/JuliaGPU/CUDA.jl/pull/2805)).
- Fixed a host memory leak in cuTENSOR plan creation
  ([#2794](https://github.com/JuliaGPU/CUDA.jl/pull/2794)).
- `CUSPARSE.sm2` now reports which dimensions mismatched instead of throwing an
  empty `DimensionMismatch`
  ([#2797](https://github.com/JuliaGPU/CUDA.jl/pull/2797)).

### v5.8.4 (September 2025)

Backports for users still on the 5.8 series, from the `release-5.8` branch.
- Fixed illegal memory accesses caused by the `aligned_sizeof` change in v5.8.0,
  e.g. when running `sortperm!` with `--check-bounds=no`
  ([#2838](https://github.com/JuliaGPU/CUDA.jl/pull/2838)).
- Fixed invalid kernel configurations generated by `mapreducedim!` when both
  input and output are `SubArray`s
  ([#2869](https://github.com/JuliaGPU/CUDA.jl/pull/2869),
  [#2880](https://github.com/JuliaGPU/CUDA.jl/pull/2880)).
- `CUPTI.version()` now returns the actual CUPTI version instead of an API
  revision index ([#2843](https://github.com/JuliaGPU/CUDA.jl/pull/2843)).

### v5.8.5 (October 2025)

Further backports for the 5.8 series, released after v5.9.0.
- Fixed memory corruption in multi-block reductions such as `sum(x; dims=1)`
  ([#2907](https://github.com/JuliaGPU/CUDA.jl/pull/2907)).
- Fixed `copyto!` between `Symmetric` or `Hermitian` matrices, which incorrectly
  dispatched to a dense copy
  ([#2913](https://github.com/JuliaGPU/CUDA.jl/pull/2913)).
- cuTensorNet: use `Int64` strides when creating a network descriptor, as the
  library expects ([#2915](https://github.com/JuliaGPU/CUDA.jl/pull/2915)).


## v5.7 (March 2025)

The `CuRef` type was reworked, and CUBLAS now passes scalar arguments through
device memory instead of host memory. Together with the removal of eager
synchronization on small host-to-device copies, this makes CUBLAS calls that
take scalar inputs fully asynchronous, where before every such call had to wait
for the GPU to catch up.

*Technically breaking changes*:

- The minimum supported Julia version is now 1.10.
- `CuRef(x)` now returns a `CuRefValue`, backed by a single device allocation,
  instead of a `CuRefArray` wrapping a one-element `CuArray`
  ([#2645](https://github.com/JuliaGPU/CUDA.jl/pull/2645)).

*New features*:

- Added support for CUDA toolkit 12.8
  ([#2634](https://github.com/JuliaGPU/CUDA.jl/pull/2634)).
- `CuRef` objects now support `getindex` and `setindex!`, like `Base.Ref`
  ([#2645](https://github.com/JuliaGPU/CUDA.jl/pull/2645)).
- CUBLAS now uses device-side pointer mode for scalar arguments, so those calls
  no longer synchronize
  ([#2616](https://github.com/JuliaGPU/CUDA.jl/pull/2616)).
- Kernels can now take `Symbol` arguments
  ([#2624](https://github.com/JuliaGPU/CUDA.jl/pull/2624)).
- CUBLAS: wrapped the Givens rotation methods `rotg!`, `rotm!` and `rotmg!`
  ([#2642](https://github.com/JuliaGPU/CUDA.jl/pull/2642)).
- CUBLAS: integer `gemmEx!` (`Int8` inputs, `Int32` output) is available again;
  it was disabled on all of CUDA 11.3.1 and later because of an NVIDIA bug
  ([#2659](https://github.com/JuliaGPU/CUDA.jl/pull/2659)).
- CUSOLVER: `sytrf!` can perform a symmetric factorization without pivoting, and
  `sytrs!` was added to solve with the resulting factors
  ([#2640](https://github.com/JuliaGPU/CUDA.jl/pull/2640)).
- CUSPARSE: `CuSparseMatrixBSR` is supported by the generic `mm!`
  ([#2639](https://github.com/JuliaGPU/CUDA.jl/pull/2639)).
- CUSPARSE: `Adjoint` and `Transpose` of a `SparseMatrixCSC` can be converted to
  `CuSparseMatrixCOO` ([#2649](https://github.com/JuliaGPU/CUDA.jl/pull/2649)),
  and `CuSparseMatrixCOO` supports row and column indexing
  ([#2668](https://github.com/JuliaGPU/CUDA.jl/pull/2668)).
- cuTENSOR: contractions now accept one-dimensional views of `CuArray`s
  ([#2650](https://github.com/JuliaGPU/CUDA.jl/pull/2650)).

*Minor changes*:

- Device-to-host copies now perform a nonblocking synchronization before calling
  into the driver, so Julia code can keep running while waiting for the GPU
  ([#2648](https://github.com/JuliaGPU/CUDA.jl/pull/2648)). Host-to-device
  copies no longer synchronize eagerly
  ([#2625](https://github.com/JuliaGPU/CUDA.jl/pull/2625)).
- Unified memory is no longer prefetched automatically on multi-GPU systems,
  making it possible to process a single array on several devices
  ([#2626](https://github.com/JuliaGPU/CUDA.jl/pull/2626)).
- `CuDeviceArray` now stores its length, which lets LLVM eliminate bounds checks
  in kernels that already perform their own, such as KernelAbstractions.jl
  kernels ([#2621](https://github.com/JuliaGPU/CUDA.jl/pull/2621)).
- `kron` on dense CUDA arrays now uses the GPUArrays.jl implementation
  ([#2643](https://github.com/JuliaGPU/CUDA.jl/pull/2643)).
- NVTX support on Windows was re-enabled, and NVTX.jl 1.0 is now required
  ([#2665](https://github.com/JuliaGPU/CUDA.jl/pull/2665)).
- `CUDA.@profile` detects an active Nsight Systems session by inspecting the
  session list, fixing profiling under `nsys profile`
  ([#2638](https://github.com/JuliaGPU/CUDA.jl/pull/2638)).
- CUDA.jl now depends on GPUToolbox.jl for functionality shared with other GPU
  back-ends ([#2646](https://github.com/JuliaGPU/CUDA.jl/pull/2646)).

*Bug fixes*:

- Fixed batched `gemv!` with transposed matrices
  ([#2481](https://github.com/JuliaGPU/CUDA.jl/pull/2481)).
- Mixed-precision sparse matrix-vector products, where the matrix and vector
  element types differ, dispatch to CUSPARSE again
  ([#2651](https://github.com/JuliaGPU/CUDA.jl/pull/2651)).
- `similar` on a sparse GPU matrix with new dimensions returns a GPU array
  again, and now accepts the dimensions as a tuple
  ([#2652](https://github.com/JuliaGPU/CUDA.jl/pull/2652)).
- `launch_configuration` no longer errors when `max_threads` exceeds
  `typemax(Cint)` ([#2666](https://github.com/JuliaGPU/CUDA.jl/pull/2666)).
- Cooperative groups now bounds-check their arguments instead of throwing a
  confusing `InexactError`
  ([#2631](https://github.com/JuliaGPU/CUDA.jl/pull/2631)).
- Creating an FFT plan is inferable again
  ([#2683](https://github.com/JuliaGPU/CUDA.jl/pull/2683)).
- cuStateVec: `batchMeasure` with a nonzero offset called the wrong library
  function ([#2671](https://github.com/JuliaGPU/CUDA.jl/pull/2671)).
- A test-coverage campaign over the library wrappers fixed a number of smaller
  bugs and removed invalid conversions and constructors in CUBLAS, CUSPARSE and
  cuStateVec ([#2663](https://github.com/JuliaGPU/CUDA.jl/pull/2663),
  [#2664](https://github.com/JuliaGPU/CUDA.jl/pull/2664),
  [#2668](https://github.com/JuliaGPU/CUDA.jl/pull/2668),
  [#2670](https://github.com/JuliaGPU/CUDA.jl/pull/2670),
  [#2673](https://github.com/JuliaGPU/CUDA.jl/pull/2673),
  [#2677](https://github.com/JuliaGPU/CUDA.jl/pull/2677),
  [#2682](https://github.com/JuliaGPU/CUDA.jl/pull/2682)).

### v5.7.1 (March 2025)

- Fixed a precompilation failure on Julia 1.11.2, caused by duplicate constant
  definitions in the CUSPARSE wrappers
  ([#2703](https://github.com/JuliaGPU/CUDA.jl/pull/2703)).
- Fixed `mul!` between two `CuSparseMatrixCOO`s, which computed the result but
  did not write it to the destination
  ([#2697](https://github.com/JuliaGPU/CUDA.jl/pull/2697)).
- Removed some invalid CUSPARSE descriptor constructors
  ([#2700](https://github.com/JuliaGPU/CUDA.jl/pull/2700)).

### v5.7.2 (April 2025)

- Added `CUDA.enable_synchronization!(::CuArray, ::Bool)` to opt out of the
  implicit synchronization that happens when an array is used from a different
  task or stream. This is an escape hatch for cases such as disjoint slices
  being processed concurrently, and unsafe use will corrupt data
  ([#2662](https://github.com/JuliaGPU/CUDA.jl/pull/2662)).
- Added compatibility entries for compute capabilities 10.0, 10.1 and 12.0, and
  for PTX ISA 8.6 and 8.7
  ([#2717](https://github.com/JuliaGPU/CUDA.jl/pull/2717)).
- Fixed `mapreduce` over sparse matrices when `f(0) != 0`, which made e.g.
  `maximum(abs, ::CuSparseMatrixCSR)` return `Inf`
  ([#2710](https://github.com/JuliaGPU/CUDA.jl/pull/2710)).
- CUSOLVER: `inv` of a triangular `CuMatrix` now returns a triangular matrix,
  `\` with a `Symmetric` matrix no longer errors, and `sytrs!` accepts a vector
  right-hand side ([#2707](https://github.com/JuliaGPU/CUDA.jl/pull/2707),
  [#2712](https://github.com/JuliaGPU/CUDA.jl/pull/2712)).
- CUSPARSE: fixed conversions between COO, CSR and CSC for matrices without
  stored values, `kron` for matrices with non-`Cint` index types, and
  `istriu`/`istril` of triangular wrappers around adjoints and transposes
  ([#2725](https://github.com/JuliaGPU/CUDA.jl/pull/2725),
  [#2726](https://github.com/JuliaGPU/CUDA.jl/pull/2726)).
- cuTensorNet: fixed the error handling and several constructors
  ([#2713](https://github.com/JuliaGPU/CUDA.jl/pull/2713),
  [#2715](https://github.com/JuliaGPU/CUDA.jl/pull/2715)).

### v5.7.3 (April 2025)

- Restored the CUSPARSE descriptor constructors removed in v5.7.1, which had
  broken packages like KrylovPreconditioners.jl and CUSOLVERRF.jl
  ([#2746](https://github.com/JuliaGPU/CUDA.jl/pull/2746)).
- Device-side sparse matrices now implement the SparseArrays interface
  (`rowvals`, `getcolptr`, `getnzval` and `nzrange`), so kernels can use it
  ([#2738](https://github.com/JuliaGPU/CUDA.jl/pull/2738)).
- Implemented `KernelAbstractions.functional` for `CUDABackend`, and taught its
  `Adapt` rules about sparse arrays
  ([#2740](https://github.com/JuliaGPU/CUDA.jl/pull/2740)).
- CUSPARSE: requesting an unknown sparse format now throws an `ArgumentError`
  ([#2744](https://github.com/JuliaGPU/CUDA.jl/pull/2744)).


## v5.6 (January 2025)

The main change in v5.6 is behind the scenes: the release requires GPUArrays.jl
v11, which reimplements all vendor-neutral array kernels on top of
KernelAbstractions.jl. Most code keeps working unchanged, but packages that
hooked into the old `gpu_call` DSL need updating, and simple operations such as
plain broadcasts can be slower than before because KernelAbstractions.jl
indexing is more expensive to amortize.

*Technically breaking changes*:

- CUDA.jl now requires GPUArrays.jl v11 and its KernelAbstractions.jl-based
  kernels. The `gpu_call` DSL is gone, along with CUDA.jl's implementation of
  it: `CuKernelContext`, `CuArrayBackend` and
  `GPUArrays.backend(::Type{<:CuArray})` no longer exist. Use
  KernelAbstractions.jl and `KernelAbstractions.get_backend(::CuArray)` instead
  ([#2524](https://github.com/JuliaGPU/CUDA.jl/pull/2524)).

*New features*:

- CUSOLVER: added `gesv!` and `gels!`
  ([#2406](https://github.com/JuliaGPU/CUDA.jl/pull/2406)), `Xgeev!` for
  nonsymmetric eigenvalue problems
  ([#2513](https://github.com/JuliaGPU/CUDA.jl/pull/2513)), and `XsyevBatched`
  for batched symmetric eigenvalue problems
  ([#2577](https://github.com/JuliaGPU/CUDA.jl/pull/2577)).
- `mul!` now handles products of `Transpose`/`Adjoint` matrices with `Diagonal`
  matrices ([#2518](https://github.com/JuliaGPU/CUDA.jl/pull/2518)), without
  allocating a temporary copy
  ([#2538](https://github.com/JuliaGPU/CUDA.jl/pull/2538)).
- Regenerated the library wrappers against CUDA 12.6.2, exposing the newly added
  cuBLASLt, cuFFT, cuSOLVER and CUPTI APIs
  ([#2512](https://github.com/JuliaGPU/CUDA.jl/pull/2512)).

*Minor changes*:

- CUFFT: `plan_rfft` and `plan_brfft` allocate the scratch buffer needed for
  complex-to-real transforms once, at plan creation, instead of on every
  execution ([#2578](https://github.com/JuliaGPU/CUDA.jl/pull/2578)).
- CUBLAS: the pointer array needed to call batched routines with strided-batched
  inputs is now built on the GPU, which is much faster for large batch counts
  ([#2608](https://github.com/JuliaGPU/CUDA.jl/pull/2608)).
- The library handle cache only triggers the GC when many handles are already in
  use, avoiding a GC run on every failed lookup with large numbers of
  short-lived tasks ([#2583](https://github.com/JuliaGPU/CUDA.jl/pull/2583)).

*Bug fixes*:

- `dot` of strided `CuArray`s that are not vectors no longer goes through the
  CUBLAS level-1 wrappers, which returned wrong results for views into a matrix
  ([#2528](https://github.com/JuliaGPU/CUDA.jl/pull/2528),
  [#2569](https://github.com/JuliaGPU/CUDA.jl/pull/2569)).
- `findall` on an empty `CuArray{Bool}` no longer errors
  ([#2554](https://github.com/JuliaGPU/CUDA.jl/pull/2554)).
- CUSOLVER: fixed `Xgesvdr!`
  ([#2556](https://github.com/JuliaGPU/CUDA.jl/pull/2556)), and several dense
  routines now preserve the memory type of their inputs instead of returning
  device-memory arrays for unified-memory inputs
  ([#2534](https://github.com/JuliaGPU/CUDA.jl/pull/2534)).
- Native RNG: fixed counter overflow when generating very large arrays, and
  fixed the `randn` window calculation
  ([#2561](https://github.com/JuliaGPU/CUDA.jl/pull/2561)).
- `mapreduce` no longer deadlocks when the accumulator type is narrowed, e.g.
  when reducing into an array with a smaller floating-point element type
  ([#2596](https://github.com/JuliaGPU/CUDA.jl/pull/2596)).
- Host memory that is pinned, resized and pinned again is now re-pinned with its
  new size, instead of silently keeping the old registration
  ([#2599](https://github.com/JuliaGPU/CUDA.jl/pull/2599)).
- Fixed an `ArgumentError: array must be non-empty` when looking up a library
  handle ([#2604](https://github.com/JuliaGPU/CUDA.jl/pull/2604)).
- Enzyme: added `make_zero` for `CuArray`s
  ([#2600](https://github.com/JuliaGPU/CUDA.jl/pull/2600)), and marked
  `launch_configuration` and `device_synchronize` as non-differentiable
  ([#2563](https://github.com/JuliaGPU/CUDA.jl/pull/2563),
  [#2605](https://github.com/JuliaGPU/CUDA.jl/pull/2605)).
- Worked around a LinearAlgebra.jl regression in Julia 1.11.2 that broke
  triangular multiplication and division with dense and sparse GPU arrays
  ([#2585](https://github.com/JuliaGPU/CUDA.jl/pull/2585)).

### v5.6.1 (January 2025)

- Added support for the GPUArrays.jl caching allocator. Allocations made inside
  a `GPUArrays.@cached cache begin ... end` block are recorded and reused on the
  next iteration, which helps with repetitive workloads such as training loops
  that would otherwise allocate and free the same buffers over and over. The
  memory is released when the cache is collected, or eagerly with
  `GPUArrays.unsafe_free!(cache)`
  ([#2593](https://github.com/JuliaGPU/CUDA.jl/pull/2593),
  [#2614](https://github.com/JuliaGPU/CUDA.jl/pull/2614)).
- Fixed `resize!` of a `CuArray` when the memory pool is disabled with
  `pool=none` ([#2613](https://github.com/JuliaGPU/CUDA.jl/pull/2613)).
- NVML is no longer used on Tegra devices, working around device lookup failures
  on Jetson Orin ([#2620](https://github.com/JuliaGPU/CUDA.jl/pull/2620)).


## v5.5 (September 2024)

*Technically breaking changes*:

- CUDA.jl's implementation and CI now target Julia 1.10 or later, in
  anticipation of it becoming the next LTS. The package compatibility bound
  remained at Julia 1.8 until v5.7
  ([#2447](https://github.com/JuliaGPU/CUDA.jl/pull/2447)).

*New features*:

- Added support for Julia 1.11
  ([#2492](https://github.com/JuliaGPU/CUDA.jl/pull/2492)).
- Added support for CUDA 12.6
  ([#2461](https://github.com/JuliaGPU/CUDA.jl/pull/2461)).
- CUFFT now supports `Float16` transforms, by switching to the Xt APIs
  ([#2430](https://github.com/JuliaGPU/CUDA.jl/pull/2430)).
- CUSPARSE: added conversions between `CuSparseVector` and the sparse matrix
  types, along with a `gemv` for `CuSparseMatrixCSC * CuSparseVector` that
  preserves sparsity of the result
  ([#2488](https://github.com/JuliaGPU/CUDA.jl/pull/2488),
  [#2489](https://github.com/JuliaGPU/CUDA.jl/pull/2489)).
- CUSPARSE: added `spdiagm`
  ([#2458](https://github.com/JuliaGPU/CUDA.jl/pull/2458)).
- CUBLAS: `gemm_grouped_batched!` and `gemm_grouped_batched` accept arbitrary
  group sizes, by passing vectors of vectors of matrices. The existing methods,
  with a group size of one, still work
  ([#2334](https://github.com/JuliaGPU/CUDA.jl/pull/2334)).
- Enzyme: added reverse-mode support for kernels launched with `@cuda`,
  `cudaconvert` on reverse-mode arguments, and derivatives for `sum`
([#2422](https://github.com/JuliaGPU/CUDA.jl/pull/2422),
[#2471](https://github.com/JuliaGPU/CUDA.jl/pull/2471),
[#2476](https://github.com/JuliaGPU/CUDA.jl/pull/2476)). *Minor changes*:

- Kernel launch overhead was reduced by reusing already-converted arguments.
  This does not apply to kernels obtained through `@cuda launch=false`
  ([#2472](https://github.com/JuliaGPU/CUDA.jl/pull/2472)).
- CUSOLVER's dense wrappers cache their workspace buffers in the handle, cutting
  the number of allocations for repeated calls
  ([#2465](https://github.com/JuliaGPU/CUDA.jl/pull/2465)).
- Reverted the full GC run under very high memory pressure that was introduced
  in v5.4, as it caused unwanted pauses
([#2469](https://github.com/JuliaGPU/CUDA.jl/pull/2469),
[#2467](https://github.com/JuliaGPU/CUDA.jl/pull/2467)). *Bug fixes*:

- The forwards-compatible driver is now probed in a separate process, instead of
  `dlclose`ing the system driver. This fixes segfaults and load failures on some
  systems, including inside PackageCompiler sysimages
  ([#2463](https://github.com/JuliaGPU/CUDA.jl/pull/2463)).
- CUSPARSE: fixed an illegal memory access in the sparse array constructors when
  the input contained duplicate elements
  ([#2495](https://github.com/JuliaGPU/CUDA.jl/pull/2495)).
- Fixed the deprecation of `CUDA.Mem.unregister`, which called `register`
  instead ([#2470](https://github.com/JuliaGPU/CUDA.jl/pull/2470)).
- Fixed a corner case when establishing peer-to-peer access between devices
  ([#2457](https://github.com/JuliaGPU/CUDA.jl/pull/2457)).
- Added missing `GC.@preserve` calls around buffers passed to the driver and
  NVML, e.g. when querying a device's name, UUID or serial, or when reporting a
  PTX JIT error ([#2487](https://github.com/JuliaGPU/CUDA.jl/pull/2487)).

### v5.5.1 (September 2024)

- Updated the cuStateVec wrappers for CUDA 12.6 Update 1
  ([#2499](https://github.com/JuliaGPU/CUDA.jl/pull/2499)).
- Adapted the Enzyme extension to EnzymeCore 0.8
  ([#2490](https://github.com/JuliaGPU/CUDA.jl/pull/2490)).
- The cuDNN, cuTENSOR, cuStateVec and cuTensorNet subpackages now require Julia
  1.10, matching CUDA.jl's supported versions.

### v5.5.2 (September 2024)

- CUFFT: fixed the element type of real-to-complex plans. `plan_rfft` now
  returns an `AbstractFFTs.Plan` parameterized on the input element type, and
  its inverse on the complex one
  ([#2504](https://github.com/JuliaGPU/CUDA.jl/pull/2504)).
- The profiler now demangles kernel names
  ([#2505](https://github.com/JuliaGPU/CUDA.jl/pull/2505)).
- cuDNN.jl now uses CUDNN 9.4
  ([#2507](https://github.com/JuliaGPU/CUDA.jl/pull/2507)).


## v5.4 (May 2024)

This release reworked how CUDA.jl manages memory. Allocations now carry the
device that owns them and the stream that last touched them, which lets the
package enable peer-to-peer access, synchronize streams, and raise a proper
error when memory is inaccessible. The garbage collector is also invoked eagerly
instead of only after an allocation failure.

*Technically breaking changes*:

- The `CUDA.Mem` submodule has been removed, and its contents moved to `CUDA`.
  `Mem.Device` and `Mem.DeviceBuffer` are now `CUDA.DeviceMemory` (similarly for
  host and unified memory), `Mem.set!` is `CUDA.memset`, and `Mem.info` is
  `CUDA.memory_info`. Enum values gained a `MEM_` prefix, e.g.,
  `Mem.ATTACH_GLOBAL` is now `CUDA.MEM_ATTACH_GLOBAL`. Deprecations are in place
  for all of these ([#2335](https://github.com/JuliaGPU/CUDA.jl/pull/2335)).
- `CUDA.memory_status()` has been renamed to `CUDA.pool_status()`, and
  `CUDA.available_memory()` to `CUDA.free_memory()`
  ([#2335](https://github.com/JuliaGPU/CUDA.jl/pull/2335)).
- `unsafe_wrap(CuArray, ptr, dims)` now returns a `HostMemory`-backed array on
  systems without HMM, instead of incorrectly claiming the memory was unified.
  Passing an explicit memory type is now always honored
  ([#2363](https://github.com/JuliaGPU/CUDA.jl/pull/2363),
  [#2372](https://github.com/JuliaGPU/CUDA.jl/pull/2372)).
- `CUDA.run_compute_sanitizer` has been removed; compute-sanitizer is no longer
  part of the CUDA.jl artifacts, and should be obtained from `CUDA_SDK_jll` or a
  local toolkit ([#2374](https://github.com/JuliaGPU/CUDA.jl/pull/2374)).
- `CUDA.device_reset!` now errors on drivers older than CUDA 12, where it is not
  reliable and may crash
  ([#2346](https://github.com/JuliaGPU/CUDA.jl/pull/2346)).

*New features*:

- CUDA.jl now tracks which device owns a piece of memory and which stream last
  accessed it. Peer-to-peer access between devices is configured automatically
  when possible, using memory from an inaccessible device raises a descriptive
  error instead of an illegal memory access, and streams are synchronized
  automatically when data moves between tasks
  ([#2335](https://github.com/JuliaGPU/CUDA.jl/pull/2335),
  [#2348](https://github.com/JuliaGPU/CUDA.jl/pull/2348)).
- The garbage collector is now triggered eagerly, based on tracked memory usage,
  at points such as kernel synchronization. This distributes collection cost
  over time instead of stalling on an out-of-memory error. Set
  `JULIA_CUDA_GC_EARLY=false` to fall back to the old behavior
  ([#2304](https://github.com/JuliaGPU/CUDA.jl/pull/2304)).
- Reading and writing single elements of a unified `CuArray` from the CPU is now
  hundreds of times faster, making it practical to incrementally port code to
  the GPU ([#2340](https://github.com/JuliaGPU/CUDA.jl/pull/2340)).
- Initial support for differentiating mixed host/device code with Enzyme.jl,
  through an `EnzymeCore` package extension
  ([#1869](https://github.com/JuliaGPU/CUDA.jl/pull/1869),
  [#2281](https://github.com/JuliaGPU/CUDA.jl/pull/2281),
  [#2368](https://github.com/JuliaGPU/CUDA.jl/pull/2368),
  [#2369](https://github.com/JuliaGPU/CUDA.jl/pull/2369),
  [#2371](https://github.com/JuliaGPU/CUDA.jl/pull/2371),
  [#2386](https://github.com/JuliaGPU/CUDA.jl/pull/2386)).
- Added support for CUDA 12.5, including PTX ISA 8.3 and 8.4
  ([#2392](https://github.com/JuliaGPU/CUDA.jl/pull/2392),
  [#2396](https://github.com/JuliaGPU/CUDA.jl/pull/2396)).
- Initial support for Julia 1.12
  ([#2277](https://github.com/JuliaGPU/CUDA.jl/pull/2277),
  [#2390](https://github.com/JuliaGPU/CUDA.jl/pull/2390)).
- Tegra devices are now supported by the CUDA.jl artifacts, so a local toolkit
  is no longer required there
  ([#2374](https://github.com/JuliaGPU/CUDA.jl/pull/2374)).
- `CUDA.@profile` now detects when it is running under an external profiler such
  as Nsight, so `external=true` no longer needs to be passed
  ([#2339](https://github.com/JuliaGPU/CUDA.jl/pull/2339)).
- Bumped cuQuantum to 24.03, and made cuTensorNet work with cuTENSOR 2
  ([#2350](https://github.com/JuliaGPU/CUDA.jl/pull/2350),
  [#2351](https://github.com/JuliaGPU/CUDA.jl/pull/2351),
  [#2354](https://github.com/JuliaGPU/CUDA.jl/pull/2354)).
- cuTENSOR operations now accept `AbstractArray` inputs, making it easier for
  other packages to pass their own array types
  ([#2356](https://github.com/JuliaGPU/CUDA.jl/pull/2356)).

*Minor changes*:

- Exceptions thrown from kernels now report a single message instead of one per
  thread, and the exception type is forwarded more accurately
  ([#2342](https://github.com/JuliaGPU/CUDA.jl/pull/2342)).
- Cached library handles are now freed when under memory pressure
  ([#2352](https://github.com/JuliaGPU/CUDA.jl/pull/2352)).
- Reduced the overhead of `CuArray` allocation
  ([#2355](https://github.com/JuliaGPU/CUDA.jl/pull/2355)).

*Bug fixes*:

- cuBLASXt now works with stream-ordered memory, and its contexts are destroyed
  correctly ([#2394](https://github.com/JuliaGPU/CUDA.jl/pull/2394),
  [#2398](https://github.com/JuliaGPU/CUDA.jl/pull/2398)).
- CUBLASLt wrappers now call into the correct library
  ([#2391](https://github.com/JuliaGPU/CUDA.jl/pull/2391)).
- Fixed the launch configuration of the bitonic sorting kernel
  ([#2353](https://github.com/JuliaGPU/CUDA.jl/pull/2353)).
- `code_sass` now works on CUDA 12.4 Update 1, which needs the CUPTI activity
  API to be activated first
  ([#2399](https://github.com/JuliaGPU/CUDA.jl/pull/2399)).
- The garbage collector is no longer run during stream capture, where it cannot
  handle asynchronous frees.

### v5.4.1 (May 2024)

- Fixed the Enzyme rule marking `CuArray` as `noalias`, which had been reverted
  before v5.4.0 was tagged
  ([#2401](https://github.com/JuliaGPU/CUDA.jl/pull/2401)).
- `NVML.compute_processes` now returns an empty dictionary instead of `nothing`
  when no compute processes are running.
- Fixed detection of the CUPTI library version on Windows.

### v5.4.2 (May 2024)

- Fixed allocation when the stream-ordered memory pool is disabled, e.g., on
  systems where it is unavailable or when set through
  `JULIA_CUDA_MEMORY_POOL=none`
  ([#2402](https://github.com/JuliaGPU/CUDA.jl/pull/2402)).

### v5.4.3 (July 2024)

- Bumped cuDNN to 9.1 ([#2404](https://github.com/JuliaGPU/CUDA.jl/pull/2404)).
- CUBLAS: added `getrsBatched`, stopped `getrf_batched` from overwriting its
  inputs, and added support for pre-allocated pivot and info buffers
  ([#2385](https://github.com/JuliaGPU/CUDA.jl/pull/2385),
  [#2431](https://github.com/JuliaGPU/CUDA.jl/pull/2431)).
- cuSOLVER: corrected workspace handling in the dense factorizations, fixing
  failures of eigenvalue decompositions on large matrices
  ([#2437](https://github.com/JuliaGPU/CUDA.jl/pull/2437)).
- CUFFT: `ScaledPlan`-wrapped plans can now be passed to functions taking a plan
  ([#2409](https://github.com/JuliaGPU/CUDA.jl/pull/2409)).
- Fixed the `kron!` launch configuration for matrices with very different
  dimensions ([#2418](https://github.com/JuliaGPU/CUDA.jl/pull/2418)).
- A full garbage collection is now run when under very high memory pressure
  ([#2421](https://github.com/JuliaGPU/CUDA.jl/pull/2421)).
- `mul!` with a strided output vector now dispatches to CUBLAS
  ([#2414](https://github.com/JuliaGPU/CUDA.jl/pull/2414)).
- Added device overrides for two rational-number errors, and conditionalized an
  existing one for Julia 1.11
  ([#2403](https://github.com/JuliaGPU/CUDA.jl/pull/2403),
  [#2411](https://github.com/JuliaGPU/CUDA.jl/pull/2411)).
- `CUDA.@profile` no longer forces a garbage collection before profiling, and
  warms up CUPTI to avoid attributing its start-up cost to the first operation
  ([#2432](https://github.com/JuliaGPU/CUDA.jl/pull/2432)).
- Updated the wrappers for CUDA 12.5.1
  ([#2436](https://github.com/JuliaGPU/CUDA.jl/pull/2436)).


## v5.3 (April 2024)

*New features*:

- Added support for CUDA 12.4
  ([#2282](https://github.com/JuliaGPU/CUDA.jl/pull/2282),
  [#2286](https://github.com/JuliaGPU/CUDA.jl/pull/2286)).
- cuDNN.jl now uses cuDNN 9
  ([#2267](https://github.com/JuliaGPU/CUDA.jl/pull/2267)).
- Multi-dimensional `sort!` uses the bitonic sorting kernel instead of the much
  slower quicksort implementation, and `sortperm`, `sortperm!` and
  `partialsort!` gained support for the `dims` keyword argument
  ([#2308](https://github.com/JuliaGPU/CUDA.jl/pull/2308)).
- CUBLAS now uses the ILP64 API when running on CUDA 12, lifting the 32-bit
  length limit on operations such as `norm`
  ([#2270](https://github.com/JuliaGPU/CUDA.jl/pull/2270)).
- Wrapped the grouped batched GEMM API as `CUBLAS.gemm_grouped_batched`
  ([#2310](https://github.com/JuliaGPU/CUDA.jl/pull/2310)).
- Wrapped `CUSOLVER.larft!`
  ([#2301](https://github.com/JuliaGPU/CUDA.jl/pull/2301)).
- Added `SparseArrays.findnz`, `sparsevec` and `getcolptr` methods for CUSPARSE
  arrays ([#2254](https://github.com/JuliaGPU/CUDA.jl/pull/2254)).
- `CuIterator` now takes an optional adaptor as first argument, so that batches
  can be converted to a different element type, e.g.
  `CuIterator(CuArray{Float32}, batches)`
  ([#2297](https://github.com/JuliaGPU/CUDA.jl/pull/2297)).
- Failing kernel launches are now diagnosed against the device limits, reporting
  which block dimension, grid dimension or shared memory limit was exceeded
  instead of only `ERROR_INVALID_VALUE`
  ([#2245](https://github.com/JuliaGPU/CUDA.jl/pull/2245)).
- `@device_code_sass` now works with kernels that were not compiled by Julia
  ([#2247](https://github.com/JuliaGPU/CUDA.jl/pull/2247)).
- Added a StaticArrays extension, so that conversion errors in device code
  report a proper message instead of failing to compile
  ([#2273](https://github.com/JuliaGPU/CUDA.jl/pull/2273)).
- Workspace buffers used by CUSOLVER, CUSPARSE and cuStateVec are now cached
  instead of being reallocated on every call
([#2279](https://github.com/JuliaGPU/CUDA.jl/pull/2279)). *Minor changes*:

- All calls into the CUDA libraries are now marked GC-safe, fixing hangs of
  multi-threaded applications
  ([#2262](https://github.com/JuliaGPU/CUDA.jl/pull/2262)).
- Reduced locking in the library handle cache, and improved nonblocking
  synchronization ([#2256](https://github.com/JuliaGPU/CUDA.jl/pull/2256),
  [#2272](https://github.com/JuliaGPU/CUDA.jl/pull/2272)).
- `CUDA.@profile` no longer triggers the garbage collector before profiling,
  which could make profiling runs very expensive.
- Tegra systems are now also detected through the device tree
  ([#2251](https://github.com/JuliaGPU/CUDA.jl/pull/2251)).
- Initial support for Julia 1.11
  ([#2291](https://github.com/JuliaGPU/CUDA.jl/pull/2291)).
- cuTENSOR: `CuTensor` stores its modes as integers rather than `Char`s, `mul!`
  accepts scaling factors, and the compute type is now determined as part of the
  plan ([#2246](https://github.com/JuliaGPU/CUDA.jl/pull/2246),
  [#2264](https://github.com/JuliaGPU/CUDA.jl/pull/2264)).
- Device-side exceptions now spell out how to enable stack traces
  ([#2316](https://github.com/JuliaGPU/CUDA.jl/pull/2316)).
- Device-side `BoundsError`s no longer capture the offending array, avoiding a
  GPU allocation when bounds-checking `MArray`s
([#2314](https://github.com/JuliaGPU/CUDA.jl/pull/2314)). *Bug fixes*:

- Fixed broadcasting between arrays of different dimensionality that are backed
  by different memory types
  ([#2290](https://github.com/JuliaGPU/CUDA.jl/pull/2290)).
- `CUDA.rand!` now generates the same numbers for a `CuArray` and for a wrapped
  array of the same size; the launch configuration no longer affects the result
  ([#2307](https://github.com/JuliaGPU/CUDA.jl/pull/2307)).
- Errors raised while handling an out-of-memory condition no longer result in a
  `StackOverflowError` ([#2299](https://github.com/JuliaGPU/CUDA.jl/pull/2299)).
- Fixed method ambiguities when concatenating `CuSparseArrayCSR`s
  ([#2244](https://github.com/JuliaGPU/CUDA.jl/pull/2244)).
- `CUSPARSE.mm!` now sets a default buffer size, avoiding spurious out-of-memory
  errors when multiplying a `CuSparseMatrixCSC` with a `CuMatrix`
  ([#2298](https://github.com/JuliaGPU/CUDA.jl/pull/2298)).
- cuTENSOR, cuTensorNet and cuStateVec now work when using a local CUDA toolkit
  ([#2274](https://github.com/JuliaGPU/CUDA.jl/pull/2274)).

### v5.3.1 (April 2024)

- Kernel launch failures are now also diagnosed against kernel-specific limits,
  such as the maximum number of threads per block imposed by register use
  ([#2329](https://github.com/JuliaGPU/CUDA.jl/pull/2329)).
- Fixed broadcasting between arrays of different dimensionality but identical
  memory type falling back to unified memory
  ([#2327](https://github.com/JuliaGPU/CUDA.jl/pull/2327)).
- Fixed the dispatch of `CUSOLVER.syevd!` and `heevd!`
  ([#2309](https://github.com/JuliaGPU/CUDA.jl/pull/2309)).
- Regenerated the library wrappers, adding cuBLASLt
  ([#2324](https://github.com/JuliaGPU/CUDA.jl/pull/2324)).
- Worked around a CUPTI bug in CUDA 12.4 Update 1
  ([#2330](https://github.com/JuliaGPU/CUDA.jl/pull/2330)).

### v5.3.2 (April 2024)

- `CUDA.@profile` now automatically detects an external profiler such as Nsight
  Systems, making `external=true` unnecessary
  ([#2339](https://github.com/JuliaGPU/CUDA.jl/pull/2339)).
- Fixed `@device_code_sass` on CUDA 12.4 Update 1
  ([#2343](https://github.com/JuliaGPU/CUDA.jl/pull/2343)).
- `cublasLtMatmulDescSetAttribute` now accepts device pointers
  ([#2337](https://github.com/JuliaGPU/CUDA.jl/pull/2337)).
- `CUDA.set_runtime_version!` now stores its preferences as strings.

### v5.3.3 (April 2024)

- Fixed the kernel launch configuration of the bitonic sorting kernel, which
  could exceed the available launch resources
  ([#2353](https://github.com/JuliaGPU/CUDA.jl/pull/2353)).

### v5.3.4 (May 2024)

- Backported the Enzyme.jl extension, providing forward-mode rules for CUDA
  operations ([#2375](https://github.com/JuliaGPU/CUDA.jl/pull/2375)).
- `EnzymeCore` is a weak dependency again, and is no longer loaded
  unconditionally ([#2382](https://github.com/JuliaGPU/CUDA.jl/pull/2382)).

### v5.3.5 (May 2024)

- Extended the Enzyme.jl extension with rules for array allocation functions
  ([#2393](https://github.com/JuliaGPU/CUDA.jl/pull/2393)).


## v5.2 (January 2024)

*Technically breaking changes*:

- cuTENSOR.jl was updated to cuTENSOR 2.0, which revamps the library API in a
  backwards-incompatible way. `contraction!`, `permutation!` and `reduction!`
  are now `contract!`, `permute!` and `reduce!`, elementwise operations are
  performed using `elementwise_binary_execute!` and
  `elementwise_trinary_execute!`, and every operation can be planned ahead of
  time with `plan_contraction` and friends so that the plan can be reused. The
  high-level `CuTensor` interface is mostly unaffected
  ([#2178](https://github.com/JuliaGPU/CUDA.jl/pull/2178),
  [#2234](https://github.com/JuliaGPU/CUDA.jl/pull/2234)).
- Kernels can no longer mutate `Ref` arguments. The support for this that was
  added in v5.1 has been reverted, because broadcast passes ephemeral `Ref`
  boxes for scalar arguments, which could result in illegal memory accesses
  ([#2206](https://github.com/JuliaGPU/CUDA.jl/pull/2206)).
- `CUDA.unsafe_release!` now takes the `CuPrimaryContext` to release, instead of
  a context derived from it
([#2200](https://github.com/JuliaGPU/CUDA.jl/pull/2200)). *New features*:

- `CuSparseArrayCSR`, an N-dimensional batched sparse array type, along with
  `bmm!` for batched sparse-dense matrix multiplication
  ([#1944](https://github.com/JuliaGPU/CUDA.jl/pull/1944)).
- `kron` and `kron!` for dense `CuMatrix` inputs
  ([#2177](https://github.com/JuliaGPU/CUDA.jl/pull/2177)).
- NVML clock queries: `NVML.clock_info`, `NVML.max_clock_info`,
  `NVML.applications_clock`, `NVML.default_applications_clock`,
  `NVML.supported_memory_clocks`, `NVML.supported_graphics_clocks` and
  `NVML.clock_event_reasons`, as well as `NVML.temperature`
  ([#2194](https://github.com/JuliaGPU/CUDA.jl/pull/2194)).
- `CUDA.@profile` and `CUDA.@bprofile` take a `concurrent` keyword argument to
  select between concurrent profiling, which instruments kernels but perturbs
  the application's performance characteristics less, and serial profiling,
  which has lower overhead
  ([#2201](https://github.com/JuliaGPU/CUDA.jl/pull/2201)).
- High-level wrappers for the CUPTI callback API, `CUPTI.CallbackConfig` and
  `CUPTI.enable!`, now used by `code_sass` and the integrated profiler
  ([#2239](https://github.com/JuliaGPU/CUDA.jl/pull/2239)).
- `@cuda` supports shorthand keyword syntax, e.g. `@cuda threads kernel(...)`
  instead of `@cuda threads=threads kernel(...)`
  ([#2189](https://github.com/JuliaGPU/CUDA.jl/pull/2189)).
- Kernel launches that use more parameter memory than the target architecture
  supports now report the limit that was exceeded, instead of failing during PTX
  compilation ([#2180](https://github.com/JuliaGPU/CUDA.jl/pull/2180)).
- CUSOLVER: a `CuSolverParameters` structure for the generic API
  ([#2188](https://github.com/JuliaGPU/CUDA.jl/pull/2188)).
- cuStateVec: batched operations from cuQuantum 23.10, such as
  `applyMatrixBatched!`, `collapseByBitStringBatched!` and `abs2SumArrayBatched`
([#2210](https://github.com/JuliaGPU/CUDA.jl/pull/2210)). *Minor changes*:

- The bundled CUDA toolkit was updated to 12.3.2
  ([#2217](https://github.com/JuliaGPU/CUDA.jl/pull/2217)).
- cuQuantum was updated to 23.10
  ([#2210](https://github.com/JuliaGPU/CUDA.jl/pull/2210)).
- Broadcasting preserves the memory type of its inputs: broadcasting unified
  arrays yields a unified array, and mixing device and unified arrays prefers
  unified memory ([#2203](https://github.com/JuliaGPU/CUDA.jl/pull/2203)).
- CUSPARSE calls `cusparseSpMM_preprocess` and `cusparseSDDMM_preprocess` before
  the actual operation, which can speed up sparse-dense multiplication
  ([#2183](https://github.com/JuliaGPU/CUDA.jl/pull/2183),
  [#2184](https://github.com/JuliaGPU/CUDA.jl/pull/2184)).
- CUDA.jl now requires Adapt 4 and GPUArrays 10
([#2203](https://github.com/JuliaGPU/CUDA.jl/pull/2203)). *Bug fixes*:

- Fixed a segfault during non-blocking synchronization, by reworking how unique
  `CuContext` objects are tracked
  ([#2202](https://github.com/JuliaGPU/CUDA.jl/pull/2202)).
- Constructing a sparse matrix from `I`/`J`/`V` vectors now eagerly combines
  duplicate entries, so that the result displays and behaves correctly
  ([#2213](https://github.com/JuliaGPU/CUDA.jl/pull/2213)).
- `mul!` and the triangular multiplication and division routines accept strided
  outputs, such as a `view` of a `CuMatrix`, instead of falling back to scalar
  indexing ([#2242](https://github.com/JuliaGPU/CUDA.jl/pull/2242)).
- CUDA.jl no longer spawns tasks or registers library log callbacks during
  precompilation, which caused tools like Nsight Compute to attach to the
  precompilation process
  ([#2226](https://github.com/JuliaGPU/CUDA.jl/pull/2226)).
- Removed dynamic dispatch from `cudacall`, `context!` and kernel launches
  ([#2235](https://github.com/JuliaGPU/CUDA.jl/pull/2235)).
- CUSOLVER sparse factorizations no longer cache the library handle, which is
  task- and context-local
  ([#2173](https://github.com/JuliaGPU/CUDA.jl/pull/2173)).
- The `info` structure used by the IC0 and ILU0 preconditioners is now freed
  when a zero pivot is encountered
  ([#2187](https://github.com/JuliaGPU/CUDA.jl/pull/2187)).
- Fixes for Julia 1.11: device-side `Random.seed!`, logical indexing, copying
  arrays with isbits-union element types, and PTX linking
  ([#2199](https://github.com/JuliaGPU/CUDA.jl/pull/2199),
  [#2240](https://github.com/JuliaGPU/CUDA.jl/pull/2240)).
- `NVML.clock_event_reasons` falls back to the throttle-reason API on drivers
  older than 12.2 ([#2206](https://github.com/JuliaGPU/CUDA.jl/pull/2206)).


## v5.1 (November 2023)

This release focused on two parts of the CUDA toolkit: unified memory, which
makes it possible to access GPU memory from the CPU and vice versa, and
cooperative groups, which allow kernels to be written in terms of objects
representing groups of threads.

*New features*:

- Improved support for unified and host memory: `cu` now takes `device`, `host`
  and `unified` keyword arguments, scalar indexing is allowed and efficient on
  unified and host arrays, and the default memory kind can be selected with the
  `default_memory` preference (which is also reported by `CUDA.versioninfo()`)
  ([#2138](https://github.com/JuliaGPU/CUDA.jl/pull/2138)).
- `unsafe_wrap(Array, ::CuArray)` wraps a CPU array around unified GPU memory,
  and `unsafe_wrap(CuArray, ::Array)` does the reverse, using HMM where
  available and page-locking the memory otherwise
  ([#2138](https://github.com/JuliaGPU/CUDA.jl/pull/2138),
  [#2156](https://github.com/JuliaGPU/CUDA.jl/pull/2156)).
- The cooperative groups API has been reworked and extended, covering the
  implicit groups (`this_thread_block`, `this_grid`, `coalesced_threads`) with
  their queries, and collectives such as `shuffle`, `vote` and `memcpy_async`
  ([#2081](https://github.com/JuliaGPU/CUDA.jl/pull/2081)). Explicit groups,
  cluster groups and multi-grid groups are not supported.
- Support for CUDA 12.3 ([#2125](https://github.com/JuliaGPU/CUDA.jl/pull/2125),
  [#2132](https://github.com/JuliaGPU/CUDA.jl/pull/2132)).
- The PTX ISA version is now selected based on what the CUDA toolkit supports,
  instead of being fixed at 6.3
  ([#2088](https://github.com/JuliaGPU/CUDA.jl/pull/2088)).
- Added a `CUDA.exit()` device function to terminate a thread
  ([#2103](https://github.com/JuliaGPU/CUDA.jl/pull/2103)).
- `Ref` objects passed to kernels are now mutable, so kernels can write to them
  ([#2109](https://github.com/JuliaGPU/CUDA.jl/pull/2109)).
- The integrated profiler now reports local memory usage
  ([#2124](https://github.com/JuliaGPU/CUDA.jl/pull/2124)), parses and
  visualizes NVTX marker data
  ([#2137](https://github.com/JuliaGPU/CUDA.jl/pull/2137)), and returns a
  results object instead of writing to `stdout`, which makes it work in Pluto.jl
  and Jupyter ([#2139](https://github.com/JuliaGPU/CUDA.jl/pull/2139)).
  `CUDA.@elapsed` gained a `blocking` keyword argument
  ([#2113](https://github.com/JuliaGPU/CUDA.jl/pull/2113)).
- CUSOLVER: interfaced the generic routines, including `systrs` and `trtri`
  ([#2074](https://github.com/JuliaGPU/CUDA.jl/pull/2074)), added `geqrf!`
  ([#2085](https://github.com/JuliaGPU/CUDA.jl/pull/2085)), `getrf!`
  ([#2100](https://github.com/JuliaGPU/CUDA.jl/pull/2100)), `Xsyevdx!`,
  `Xgesvdr!` ([#2127](https://github.com/JuliaGPU/CUDA.jl/pull/2127)) and
  `Xgesvdp` ([#2128](https://github.com/JuliaGPU/CUDA.jl/pull/2128)), and
  exposed the sparse Cholesky and QR factorizations as `Factorization` objects,
  also supporting `CuSparseMatrixCSC` inputs and multiple right-hand sides
  ([#2121](https://github.com/JuliaGPU/CUDA.jl/pull/2121),
  [#2152](https://github.com/JuliaGPU/CUDA.jl/pull/2152)).
- Added `inv(::CuMatrix)` and additional `\` and `inv` methods backed by
  CUSOLVER ([#2095](https://github.com/JuliaGPU/CUDA.jl/pull/2095),
  [#2117](https://github.com/JuliaGPU/CUDA.jl/pull/2117)).
- Added `mul!` methods where one of the operands is a `Diagonal` matrix
  ([#2096](https://github.com/JuliaGPU/CUDA.jl/pull/2096)).
- CUSPARSE: wrapped the functions that were added in CUDA 12.2
  ([#2116](https://github.com/JuliaGPU/CUDA.jl/pull/2116)).
- Added a ChainRulesCore extension implementing in-place accumulation of
  gradients into `CuArray`s
  ([#2091](https://github.com/JuliaGPU/CUDA.jl/pull/2091)).

*Minor changes*:

- Memory copies are fast again after a regression in 5.0, and constructing
  derived arrays (e.g. `view`s) no longer allocates
  ([#2142](https://github.com/JuliaGPU/CUDA.jl/pull/2142),
  [#2143](https://github.com/JuliaGPU/CUDA.jl/pull/2143)).
- Kernel exceptions now exit the thread, so that a subsequent trap cannot leave
  the process in an unrecoverable state
  ([#2103](https://github.com/JuliaGPU/CUDA.jl/pull/2103)).
- Large documentation update
  ([#2146](https://github.com/JuliaGPU/CUDA.jl/pull/2146)).

*Bug fixes*:

- `minimum` and `maximum` no longer return infinity for arrays containing NaNs:
  libdevice's `fmin`/`fmax` are not used anymore because their NaN behavior
  differs from Julia's ([#2144](https://github.com/JuliaGPU/CUDA.jl/pull/2144)),
  and LLVM's equivalents are avoided on devices older than sm_80, where they
  cannot be lowered ([#2154](https://github.com/JuliaGPU/CUDA.jl/pull/2154)).
- `CUDA.@elapsed` measured the wrong expression when keyword arguments were
  passed ([#2118](https://github.com/JuliaGPU/CUDA.jl/pull/2118)).
- Fixed cooperative groups grid synchronization on sm_61
  ([#2151](https://github.com/JuliaGPU/CUDA.jl/pull/2151)).
- Unified memory is only prefetched when the device supports concurrent access,
  fixing errors on Tegra
  ([#2155](https://github.com/JuliaGPU/CUDA.jl/pull/2155)).
- CUSPARSE: fixed the number of right-hand sides passed to `sm2`-style
  triangular solves when the right-hand side is transposed
  ([#2134](https://github.com/JuliaGPU/CUDA.jl/pull/2134)).
- Random number generation now handles zero-sized inputs
  ([#2098](https://github.com/JuliaGPU/CUDA.jl/pull/2098)).
- CUSOLVER: fixed swapped `jobu`/`jobvt` arguments in `gesvd`
  ([#2101](https://github.com/JuliaGPU/CUDA.jl/pull/2101)).
- The sublibraries (cuDNN, cuTENSOR, cuStateVec, cuTensorNet) were missing a
  dependency on `CUDA_Runtime_Discovery`
  ([#2097](https://github.com/JuliaGPU/CUDA.jl/pull/2097)).
- The profiler no longer crops its output when rendering to a file
  ([#2131](https://github.com/JuliaGPU/CUDA.jl/pull/2131)).

### v5.1.1 (November 2023)

- `CUDA.set_runtime_version!` now forces its setting, so that it also takes
  effect when a preference is set higher up in the environment stack
  ([#2169](https://github.com/JuliaGPU/CUDA.jl/pull/2169)).
- The profiler now shows a time distribution for each entry, and a new
  `CUDA.@bprofile` macro repeatedly executes code before reporting
  ([#2162](https://github.com/JuliaGPU/CUDA.jl/pull/2162)).
- CUSPARSE: in-place triangular solves are supported again with CUDA 12.x
  ([#2164](https://github.com/JuliaGPU/CUDA.jl/pull/2164)), and
  `cusparseSpSV_updateMatrix` is wrapped, along with constructors that allow
  performing the analysis phase without providing a right-hand side
  ([#2159](https://github.com/JuliaGPU/CUDA.jl/pull/2159)).
- The compute-sanitizer helper now passes `--hmm-support` on devices that
  support pageable memory access
  ([#2157](https://github.com/JuliaGPU/CUDA.jl/pull/2157)).

### v5.1.2 (January 2024)

- Reverted mutable `Ref` kernel arguments
  ([#2109](https://github.com/JuliaGPU/CUDA.jl/pull/2109)): broadcast passes
  scalars in ephemeral `Ref` boxes that can be freed while the kernel is still
  running, causing illegal memory accesses.
- Fixed `Random` seeding on Julia 1.11
  ([#2199](https://github.com/JuliaGPU/CUDA.jl/pull/2199)).
- CUSOLVER: don't reuse the sparse handles
  ([#2173](https://github.com/JuliaGPU/CUDA.jl/pull/2173)).


## v5.0 (September 2023)

Two changes dominate this release: `CUDA.@profile` now runs a profiler built
into CUDA.jl instead of merely activating an external one, and task
synchronization no longer relies on stream callbacks, cutting the latency of
waiting for the GPU from at least 25µs to around 5µs. The release also raises
the minimum Julia and CUDA versions.

*Breaking changes*:

- `CUDA.@profile` now runs an integrated profiler, built on CUPTI, that reports
  host-side API calls and the resulting device-side activity. Pass `trace=true`
  for a chronological event list instead of a summary. The previous behaviour,
  activating an external profiler such as Nsight, is now `CUDA.@profile
  external=true` ([#2024](https://github.com/JuliaGPU/CUDA.jl/pull/2024)).
- Local CUDA toolkits are now selected with
  `CUDA.set_runtime_version!(local_toolkit=true)` instead of
  `CUDA.set_runtime_version!("local")`. The `version` and `local` preferences
  are now independent, so the toolkit version is known at precompilation time
  and can be overridden with `CUDA.set_runtime_version!(version;
  local_toolkit=true)` when CUDA is not available during precompilation
  ([#2058](https://github.com/JuliaGPU/CUDA.jl/pull/2058)).
- Julia 1.8 is now required
  ([#2042](https://github.com/JuliaGPU/CUDA.jl/pull/2042)).
- CUDA 11.4 is now the oldest supported toolkit. This cannot be enforced by the
  package manager; pin CUDA.jl to v4.4 or older if you need an older toolkit
  ([#2042](https://github.com/JuliaGPU/CUDA.jl/pull/2042)).
- `code_sass` lost its `verbose` keyword argument, and gained `raw`, which dumps
  `nvdisasm` output without any post-processing
  ([#2019](https://github.com/JuliaGPU/CUDA.jl/pull/2019)).
- `CUDA.unsafe_free!` no longer takes a stream argument
([#2068](https://github.com/JuliaGPU/CUDA.jl/pull/2068)). *New features*:

- Added support for CUDA 12.2, with the artifacts defaulting to 12.2 Update 2
  ([#2034](https://github.com/JuliaGPU/CUDA.jl/pull/2034),
  [#2039](https://github.com/JuliaGPU/CUDA.jl/pull/2039),
  [#2071](https://github.com/JuliaGPU/CUDA.jl/pull/2071)).
- Added support for Julia 1.10
  ([#1946](https://github.com/JuliaGPU/CUDA.jl/pull/1946)).
- The integrated profiler also reports NVTX ranges and markers
  ([#2043](https://github.com/JuliaGPU/CUDA.jl/pull/2043)).
- `@cuda fastmath=true` compiles a kernel with less precise square roots and
  flushing of denormals, and more fast-math functions now have a device
  implementation ([#2030](https://github.com/JuliaGPU/CUDA.jl/pull/2030),
  [#2047](https://github.com/JuliaGPU/CUDA.jl/pull/2047)).
- Added batched CUBLAS `gemm` and `gemv` wrappers, and batched CUSOLVER `svd`
  (`gesvdjBatched` and `gesvdaStridedBatched`, also hooked up to
  `LinearAlgebra.svd` for 3-dimensional inputs)
  ([#1975](https://github.com/JuliaGPU/CUDA.jl/pull/1975),
  [#1981](https://github.com/JuliaGPU/CUDA.jl/pull/1981),
  [#2063](https://github.com/JuliaGPU/CUDA.jl/pull/2063)).
- CUFFT plans may now cover fewer dimensions than the array they are applied to;
  the trailing dimensions are transformed sequentially
  ([#1903](https://github.com/JuliaGPU/CUDA.jl/pull/1903)).
- CUSPARSE: `dot(x, A, y)` no longer materializes `A * y`
  ([#2001](https://github.com/JuliaGPU/CUDA.jl/pull/2001)).
- CUSPARSE: reductions now take an arbitrary function to apply to the elements,
  and no longer sum absolute values
  ([#1987](https://github.com/JuliaGPU/CUDA.jl/pull/1987)).
- Loads through `Base.Experimental.Const` now support vector types
  ([#1993](https://github.com/JuliaGPU/CUDA.jl/pull/1993)).
- `@cuprint` can print types
([#2003](https://github.com/JuliaGPU/CUDA.jl/pull/2003)). *Minor changes*:

- On Julia 1.9 and later, synchronization is performed on a worker thread rather
  than with stream callbacks, which are slow and deprecated by NVIDIA. This is
  faster and has much more predictable latency
  ([#2025](https://github.com/JuliaGPU/CUDA.jl/pull/2025)).
- Synchronization briefly busy-waits before falling back to nonblocking
  synchronization, improving the latency of short operations
  ([#2059](https://github.com/JuliaGPU/CUDA.jl/pull/2059)).
- Nonblocking synchronization can be disabled globally with the
  `nonblocking_synchronization` preference, or per call with `synchronize(x;
  blocking=true)` and `CUDA.@sync blocking=true`. This is only intended for
  latency-critical code such as benchmarks
  ([#2060](https://github.com/JuliaGPU/CUDA.jl/pull/2060)).
- On CUDA 12.2, `JULIA_CUDA_HARD_MEMORY_LIMIT` is enforced by the memory pool
  itself instead of by a check before every allocation, which is much cheaper
  ([#2040](https://github.com/JuliaGPU/CUDA.jl/pull/2040)).
- Array data management moved to GPUArrays, which also makes it possible to
  `resize!` an `unsafe_wrap`ped array
  ([#2068](https://github.com/JuliaGPU/CUDA.jl/pull/2068)).
- The device implementations of SpecialFunctions.jl methods moved to a package
  extension, making SpecialFunctions.jl a weak dependency
  ([#2062](https://github.com/JuliaGPU/CUDA.jl/pull/2062)).
- All initialization errors are now deferred to run time, so importing CUDA.jl
  on a system without a GPU or driver no longer logs an error
  ([#2041](https://github.com/JuliaGPU/CUDA.jl/pull/2041)).
- cuStateVec and cuTensorNet now use cuQuantum 23.6
([#2044](https://github.com/JuliaGPU/CUDA.jl/pull/2044)). *Bug fixes*:

- `rand` in a kernel could return identical numbers across launches, because
  updates to the shared-memory state are not guaranteed to be visible to the
  next kernel. The seed is now passed from the host
  ([#2035](https://github.com/JuliaGPU/CUDA.jl/pull/2035)).
- Fixed `accumulate` returning wrong results
  ([#2005](https://github.com/JuliaGPU/CUDA.jl/pull/2005)).
- Broadcasting a type constructor now compiles
  ([#2000](https://github.com/JuliaGPU/CUDA.jl/pull/2000)).
- `sortperm!` is now resilient to type mismatches between the input and the
  output ([#2051](https://github.com/JuliaGPU/CUDA.jl/pull/2051)).
- cuDNN: the convolution algorithm selection rejected valid algorithms,
  resulting in "No valid algorithm found" warnings and slow convolutions
  ([#1943](https://github.com/JuliaGPU/CUDA.jl/pull/1943)).
- cuDNN: the convolution algorithm cache is now keyed on the contents of the
  descriptors instead of their pointers, so cached choices are actually reused
  ([#1948](https://github.com/JuliaGPU/CUDA.jl/pull/1948),
  [#2048](https://github.com/JuliaGPU/CUDA.jl/pull/2048)).
- Freeing memory no longer errors when the calling thread has no context bound
  ([#2029](https://github.com/JuliaGPU/CUDA.jl/pull/2029)).


## v4.4 (June 2023)

The last feature release of the 4.x series, before development moved to CUDA.jl
5.0.

*New features*:

- CUSPARSE: `UniformScaling` and `Diagonal` operands are now also supported for
  COO matrices and for adjoint/transpose-wrapped sparse matrices, and `*` is
  supported next to `+` and `-`
([#1941](https://github.com/JuliaGPU/CUDA.jl/pull/1941)). *Minor changes*:

- The bundled cuDNN was updated to 8.9, and cuTENSOR to 1.7
  ([#1959](https://github.com/JuliaGPU/CUDA.jl/pull/1959),
  [#1960](https://github.com/JuliaGPU/CUDA.jl/pull/1960)).
- Jetson AGX Orin (compute capability 8.7) can now be used with CUDA 11.4; the
  compatibility database incorrectly required 11.5
  ([#1967](https://github.com/JuliaGPU/CUDA.jl/pull/1967)).
- CUDA.jl now warns when a CUDA runtime library was loaded from a system path,
  e.g. through `LD_LIBRARY_PATH`, while artifact-provided libraries are in use
  ([#1935](https://github.com/JuliaGPU/CUDA.jl/pull/1935)).
- Updated to LLVM.jl 6 and GPUCompiler 0.21
([#1951](https://github.com/JuliaGPU/CUDA.jl/pull/1951),
[#1976](https://github.com/JuliaGPU/CUDA.jl/pull/1976)). *Bug fixes*:

- Kernels containing unreachable control flow no longer result in illegal
  divergent barriers ([#1951](https://github.com/JuliaGPU/CUDA.jl/pull/1951)).
- `sort` no longer fails on Ada (sm_89) GPUs, due to an incorrect launch
  configuration in the bitonic sort implementation
  ([#1979](https://github.com/JuliaGPU/CUDA.jl/pull/1979)).
- `rand` and `randn` no longer trigger illegal memory accesses on large arrays,
  where the `Int32` indices used by the RNG kernels overflowed
  ([#1969](https://github.com/JuliaGPU/CUDA.jl/pull/1969)).
- Matrix multiplication with `StaticArray` element types works again, fixing a
  regression from an earlier `mul!` rework
  ([#1954](https://github.com/JuliaGPU/CUDA.jl/pull/1954)).
- `svd!` no longer crashes when not all outputs are requested, as observed on
  Pascal GPUs with CUDA 12.0
  ([#1934](https://github.com/JuliaGPU/CUDA.jl/pull/1934)).
- Reporting an out-of-memory error before the task-local state has been
  initialized no longer causes a stack overflow
  ([#1937](https://github.com/JuliaGPU/CUDA.jl/pull/1937)).

### v4.4.1 (August 2023)

- Added support for Julia 1.10, reworking the CUBLAS and CUSPARSE `mul!` methods
  to use the new unwrapping mechanism for triangular matrices
  ([#1946](https://github.com/JuliaGPU/CUDA.jl/pull/1946)).
- Fixed `accumulate` and `cumsum` along `dims` returning wrong results for
  arrays with more than 1024 elements in the scanned dimension
  ([#2005](https://github.com/JuliaGPU/CUDA.jl/pull/2005)).
- cuDNN: fixed convolution algorithm selection for `Float32` inputs, where no
  suitable algorithm could be found because of a spurious math-type check
  ([#1943](https://github.com/JuliaGPU/CUDA.jl/pull/1943)).
- Initialization failures, such as not finding a CUDA-capable device, are no
  longer logged at import time but reported when CUDA.jl is actually used
  ([#2041](https://github.com/JuliaGPU/CUDA.jl/pull/2041)).
- `sortperm!` now accepts index arrays whose element type differs from the
  default, and validates axes instead of lengths
  ([#2051](https://github.com/JuliaGPU/CUDA.jl/pull/2051)).
- Fixed an error when freeing memory from a thread without a bound context
  ([#2029](https://github.com/JuliaGPU/CUDA.jl/pull/2029)).

### v4.4.2 (April 2024)

- A late backport on the 4.4 maintenance branch, tagged long after the 5.x
  series had shipped.
- Fixed device-side seeding of the random number generator on Julia 1.11, which
  changed how `Random.seed!` passes seeds
  ([#2199](https://github.com/JuliaGPU/CUDA.jl/pull/2199)).


## v4.3 (May 2023)

*New features*:

- `reverse` and `reverse!` now accept multiple dimensions, e.g. `dims=(1,3)` or
  `dims=:` ([#1899](https://github.com/JuliaGPU/CUDA.jl/pull/1899)).
- `versioninfo()` now also reports the versions of CUDA.jl and its
  `CUDA_Driver_jll`, `CUDA_Runtime_jll` and `CUDA_Runtime_Discovery`
  dependencies ([#1913](https://github.com/JuliaGPU/CUDA.jl/pull/1913)).
- Added the `CUDA.norm3df` and `CUDA.rnorm3df` device functions
([#1916](https://github.com/JuliaGPU/CUDA.jl/pull/1916)). *Minor changes*:

- Precompiling CUDA.jl is around 20% faster, by replacing the `@retry_reclaim`
  and `@check` macros with regular functions
  ([#1906](https://github.com/JuliaGPU/CUDA.jl/pull/1906)).
- Arrays emitted into constant memory, as used by the device-side RNG, are now
  16-byte aligned so that they can be loaded with wide loads
([#1915](https://github.com/JuliaGPU/CUDA.jl/pull/1915)). *Bug fixes*:

- Fixed non-deterministic `sort!` failures on Ada Lovelace GPUs, caused by a bad
  occupancy query, by hard-coding the number of blocks per multiprocessor
  ([#1894](https://github.com/JuliaGPU/CUDA.jl/pull/1894)).
- CUSOLVER: workspace sizes are now passed as a number of elements rather than
  bytes, fixing integer overflows in `svd` of large matrices
  ([#1890](https://github.com/JuliaGPU/CUDA.jl/pull/1890)).
- CUDA.jl now loads on platforms without hardware `Float64` atomics, such as
  PowerPC ([#1912](https://github.com/JuliaGPU/CUDA.jl/pull/1912)).
- cuDNN: convolution algorithms are now only selected when their math type
  matches the descriptor's, fixing spurious failures
  ([#1917](https://github.com/JuliaGPU/CUDA.jl/pull/1917)).

### v4.3.1 (May 2023)

- Hopper (`sm_90`) GPUs can now be used with Julia versions whose LLVM does not
  support `sm_90` yet, by invoking `ptxas` with a higher compute capability than
  the one used for code generation
  ([#1931](https://github.com/JuliaGPU/CUDA.jl/pull/1931)).
- Kernel objects, as returned by `@cuda launch=false`, now have a `show` method
  and no longer print their full type
  ([#1928](https://github.com/JuliaGPU/CUDA.jl/pull/1928)).
- The `cuDNN`, `cuTENSOR`, `cuStateVec` and `cuTensorNet` subpackages now
  declare compatibility with a single CUDA.jl minor version, so that
  incompatible combinations are no longer resolvable
  ([#1929](https://github.com/JuliaGPU/CUDA.jl/pull/1929)).

### v4.3.2 (June 2023)

- Reduced load time by hooking into `LinearAlgebra.generic_matmatmul!` instead
  of defining `mul!` methods for `CuArray` and CUSPARSE matrices
  ([#1904](https://github.com/JuliaGPU/CUDA.jl/pull/1904)).


## v4.2 (May 2023)

*Technically breaking changes*:

- `has_cudnn`, `has_cutensor`, `has_custatevec` and `has_cutensornet` no longer
  take a `show_reason` argument, and now report whether the library was actually
  initialized instead of only whether its JLL is available
  ([#1858](https://github.com/JuliaGPU/CUDA.jl/pull/1858)).
- Allocation statistics are now tracked atomically, so they are accurate when
  allocating from multiple threads. The fields of `CUDA.alloc_stats` are
  `Threads.Atomic`s and need to be dereferenced, e.g.,
`CUDA.alloc_stats.alloc_bytes[]`
([#1884](https://github.com/JuliaGPU/CUDA.jl/pull/1884),
[#1885](https://github.com/JuliaGPU/CUDA.jl/pull/1885)). *New features*:

- cuDNN, cuTENSOR and cuQuantum can now be discovered from a local CUDA
  installation, instead of requiring the corresponding JLLs. This makes those
  libraries usable when CUDA.jl is configured to use a local toolkit
  ([#1858](https://github.com/JuliaGPU/CUDA.jl/pull/1858)).
- CUDA.jl now initializes during precompilation, making it possible for
  downstream packages to precompile GPU code (e.g., with `SnoopPrecompile`)
  ([#1865](https://github.com/JuliaGPU/CUDA.jl/pull/1865)).
- Added compute capability 8.9 (Ada Lovelace) to the compatibility database, so
  CUDA.jl generates `sm_89` code for those GPUs when using CUDA 11.8 or later
  ([#1873](https://github.com/JuliaGPU/CUDA.jl/pull/1873)).
- `CuIterator` now uses `adapt` for both the upload and the free step, so
  batches can be nested objects such as arrays of arrays or named tuples, and
  not just tuples of arrays
  ([#1769](https://github.com/JuliaGPU/CUDA.jl/pull/1769)).
- Implemented `KernelAbstractions.unsafe_free!` for `CuArray`
([#1863](https://github.com/JuliaGPU/CUDA.jl/pull/1863)). *Minor changes*:

- Updated to CUDA 12.1.1
  ([#1883](https://github.com/JuliaGPU/CUDA.jl/pull/1883)).
- Using a device that is newer than what your CUDA toolkit fully supports now
  warns instead of being refused; the test suite no longer skips such devices
  ([#1818](https://github.com/JuliaGPU/CUDA.jl/pull/1818)).
- Removed type piracy of `ndims` and `eltype` on CUSPARSE arrays, reducing
  invalidations ([#1876](https://github.com/JuliaGPU/CUDA.jl/pull/1876),
[#1878](https://github.com/JuliaGPU/CUDA.jl/pull/1878)). *Bug fixes*:

- The driver version check at initialization is now guarded, so a broken driver
  installation leaves CUDA.jl non-functional instead of crashing the session
  ([#1881](https://github.com/JuliaGPU/CUDA.jl/pull/1881)).
- Using CUDA.jl on a system where it is not functional now throws an assertion
  error on first use, instead of an unrelated error about `libcuda` not being
  defined ([#1868](https://github.com/JuliaGPU/CUDA.jl/pull/1868)).


## v4.1 (March 2023)

*Technically breaking changes*:

- CUDA.jl now requires an NVIDIA driver with support for CUDA 11.0 or newer;
  CUDA 10.2 drivers are no longer accepted
  ([#1742](https://github.com/JuliaGPU/CUDA.jl/pull/1742)).
- Constructing a `CuArray` from another `CuArray`, e.g. `CuArray{Int,2}(xs)`,
  now returns a copy instead of the original array, matching `Array` behavior
  ([#1800](https://github.com/JuliaGPU/CUDA.jl/pull/1800)).
- The CUSPARSE level-1 routines `axpyi`, `sctr`, `gthr`, `gthrz` and `roti` have
  been removed, following their removal from CUDA 12
  ([#1742](https://github.com/JuliaGPU/CUDA.jl/pull/1742)).
- `potrf!`, `potrs!` and `potri!` are no longer defined as methods of
  `LinearAlgebra.LAPACK`; call them as `CUSOLVER.potrf!` and friends instead
([#1806](https://github.com/JuliaGPU/CUDA.jl/pull/1806)). *New features*:

- Added support for CUDA 12.0 and 12.1, including compute capability 9.0
  (Hopper) and PTX ISA 8.0. Compute capability 3.5 devices are only supported up
  to CUDA 11.8 ([#1742](https://github.com/JuliaGPU/CUDA.jl/pull/1742),
  [#1793](https://github.com/JuliaGPU/CUDA.jl/pull/1793)).
- The KernelAbstractions back-end is now part of CUDA.jl: `CUDABackend` is
  exported by CUDA.jl and the separate CUDAKernels.jl package is no longer
  needed. This requires KernelAbstractions 0.9
  ([#1772](https://github.com/JuliaGPU/CUDA.jl/pull/1772)).
- `A \ b` now supports rectangular `A`, using a QR factorization for
  overdetermined and an LQ factorization for underdetermined systems
  ([#1802](https://github.com/JuliaGPU/CUDA.jl/pull/1802)).
- CUSPARSE: added `gtsv2!`/`gtsv2` for solving tridiagonal systems
  ([#1795](https://github.com/JuliaGPU/CUDA.jl/pull/1795)).
- CUSPARSE: added `color` for computing a coloring-based reordering, useful
  before an IC(0) or ILU(0) factorization
([#1794](https://github.com/JuliaGPU/CUDA.jl/pull/1794)). *Minor changes*:

- cuDNN was updated to 8.8.1, which is required for CUDA 12
  ([#1792](https://github.com/JuliaGPU/CUDA.jl/pull/1792)).
- `+` and `-` on dense `CuMatrix` of CUBLAS element types now use `geam`
  ([#1775](https://github.com/JuliaGPU/CUDA.jl/pull/1775)).
- Sparse triangular solve now uses the generic CUSPARSE API for CSR and CSC
  matrices on CUDA 12, while COO matrices require CUDA 12 and otherwise throw a
  descriptive error ([#1742](https://github.com/JuliaGPU/CUDA.jl/pull/1742)).
- The `index` argument of the CUSPARSE preconditioner and reordering routines is
  now optional, and the reordering routines no longer convert CSC matrices to
  CSR. `zfd` now returns a column permutation for CSC inputs
  ([#1785](https://github.com/JuliaGPU/CUDA.jl/pull/1785),
  [#1786](https://github.com/JuliaGPU/CUDA.jl/pull/1786)).
- Conversion between sparse formats was reworked: `CuSparseMatrixCOO` can be
  constructed from `Transpose` and `Adjoint` wrappers, and converting COO or BSR
  matrices to `SparseMatrixCSC` takes a shorter path
  ([#1742](https://github.com/JuliaGPU/CUDA.jl/pull/1742),
  [#1774](https://github.com/JuliaGPU/CUDA.jl/pull/1774),
  [#1791](https://github.com/JuliaGPU/CUDA.jl/pull/1791)).
- The CUPTI wrappers were regenerated with support for the versioned-struct
  (`STRUCT_SIZE`) scheme used by recent CUDA toolkits
  ([#1827](https://github.com/JuliaGPU/CUDA.jl/pull/1827)).
- Improved the error message shown when no CUDA runtime could be found.

*Bug fixes*:

- `versioninfo()` no longer fails when NVML denies permission to query device
  information ([#1771](https://github.com/JuliaGPU/CUDA.jl/pull/1771)).
- Fixed the unit diagonal of `lu(A).L` for non-square `A`
  ([#1813](https://github.com/JuliaGPU/CUDA.jl/pull/1813)).

### v4.1.1 (March 2023)

- Fixed `CUDABackend` not actually being available after `using CUDA`, as the
  name was exported but never imported into the `CUDA` module
  ([#1834](https://github.com/JuliaGPU/CUDA.jl/pull/1834)).

### v4.1.2 (March 2023)

- Stopped using `isdefined` checks on `CUDA_Driver_jll` globals, whose result
  could be baked in during precompilation and break initialization. This
  requires CUDA_Driver_jll 0.5
  ([#1832](https://github.com/JuliaGPU/CUDA.jl/pull/1832)).

### v4.1.3 (March 2023)

- Updated to LLVM.jl 5 and GPUCompiler.jl 0.19
  ([#1847](https://github.com/JuliaGPU/CUDA.jl/pull/1847)).

### v4.1.4 (April 2023)

- Fixed `CUDA.system_driver_version()` on platforms not supported by
  CUDA_Driver_jll ([#1854](https://github.com/JuliaGPU/CUDA.jl/pull/1854)).
- `CUDA.set_runtime_version!` now rejects arguments that are not a version
  number or string, instead of writing an unusable preference
  ([#1831](https://github.com/JuliaGPU/CUDA.jl/pull/1831)).


## v4.0 (February 2023)

CUDA.jl 4.0 is a breaking release. The CUDA toolkit is now provided by regular
JLL packages, which makes it possible to build other binary libraries against
the CUDA runtime and use them together with CUDA.jl. At the same time, the cuDNN
and cuTENSOR wrappers were moved out of CUDA.jl into separate packages.

*Breaking changes*:

- CUDNN and CUTENSOR have been split off into separate packages
  ([#1624](https://github.com/JuliaGPU/CUDA.jl/pull/1624)). This should improve
  load times, since most users do not rely on the functionality from these
  modules. The CUTENSORNET and CUSTATEVEC submodules were split off as well,
  into cuTensorNet.jl and cuStateVec.jl.
- Binaries (like the CUDA runtime, CUDNN, etc) are now provided through
  first-class JLL packages
  ([#1629](https://github.com/JuliaGPU/CUDA.jl/pull/1629)). This makes it
  possible to build JLLs for applications and libraries that rely on the CUDA
  runtime. As a result, the `JULIA_CUDA_USE_BINARYBUILDER` and
  `JULIA_CUDA_VERSION` environment variables have been replaced with
  Preferences.jl-based settings; refer to the documentation for more
  information. To select a toolkit, call `CUDA.set_runtime_version!(v"11.7")`,
  or `CUDA.set_runtime_version!("local")` to use a locally-installed toolkit.
- The `CUDA.NVTX` submodule has been removed; use NVTX.jl instead, which is a
  more complete implementation of the NVTX API
  ([#1733](https://github.com/JuliaGPU/CUDA.jl/pull/1733)).
- Deprecated methods have been removed, including `CuCurrentDevice`,
  `CuCurrentContext`, `CuDefaultStream`, `CuStreamLegacy`, `CuStreamPerThread`,
  and `query` ([#1651](https://github.com/JuliaGPU/CUDA.jl/pull/1651)).
- `CuDeviceArray` now only has a single, fully-parameterized constructor,
  `CuDeviceArray{T,N,A}(ptr, dims)`, taking the pointer before the dimensions
  ([#1308](https://github.com/JuliaGPU/CUDA.jl/pull/1308)).

*New features*:

- Support for CUDA 11.8 ([#1620](https://github.com/JuliaGPU/CUDA.jl/pull/1620))
  and Julia 1.9 ([#1731](https://github.com/JuliaGPU/CUDA.jl/pull/1731)).
- Memory limits have been reinstated, in the form of the
  `JULIA_CUDA_SOFT_MEMORY_LIMIT` and `JULIA_CUDA_HARD_MEMORY_LIMIT` environment
  variables ([#1698](https://github.com/JuliaGPU/CUDA.jl/pull/1698)). The former
  is advisory and used to configure the memory pool, the latter is checked
  before every allocation. Values can be a number of bytes, optionally with a
  unit, or a percentage: `100M`, `50%`, `1.5GiB`.
- `@cuda` and `cufunction` accept an `always_inline` keyword argument to force
  inlining of all function calls in the kernel
  ([#1554](https://github.com/JuliaGPU/CUDA.jl/pull/1554)).
- cuFFT plans are now cached, and their work area is managed by CUDA.jl's memory
  pool instead of being allocated by cuFFT
  ([#1734](https://github.com/JuliaGPU/CUDA.jl/pull/1734)).
- Many CUSPARSE improvements: products of `CuSparseMatrixCOO` with `CuVector`
  and `CuMatrix` ([#1592](https://github.com/JuliaGPU/CUDA.jl/pull/1592)),
  `CuMatrix` with `CuSparseVector` and `CuSparseMatrix`
  ([#1632](https://github.com/JuliaGPU/CUDA.jl/pull/1632),
  [#1637](https://github.com/JuliaGPU/CUDA.jl/pull/1637)), and
  `CuSparseMatrixCSC` with `CuSparseMatrixCSC`
  ([#1663](https://github.com/JuliaGPU/CUDA.jl/pull/1663)).
- CUSPARSE `-`, `+` and `*` between sparse matrices and vectors
  ([#1613](https://github.com/JuliaGPU/CUDA.jl/pull/1613),
  [#1640](https://github.com/JuliaGPU/CUDA.jl/pull/1640),
  [#1648](https://github.com/JuliaGPU/CUDA.jl/pull/1648)), sparse BLAS 1
  routines like `dot` on `CuSparseVector`
  ([#1647](https://github.com/JuliaGPU/CUDA.jl/pull/1647)), and the `sv!`,
  `sm!`, `sddmm!` and `gemvi!` routines
  ([#1593](https://github.com/JuliaGPU/CUDA.jl/pull/1593),
  [#1615](https://github.com/JuliaGPU/CUDA.jl/pull/1615)).
- `\` on sparse triangular matrices now uses the new generic forward and
  backward sweep routines, and conversions between sparse and dense formats use
  more recent CUSPARSE routines
  ([#1593](https://github.com/JuliaGPU/CUDA.jl/pull/1593),
  [#1611](https://github.com/JuliaGPU/CUDA.jl/pull/1611)).
- More sparse array functionality: `kron`, `tril`, `triu`, `reshape`, `adjoint`
  and `transpose` ([#1603](https://github.com/JuliaGPU/CUDA.jl/pull/1603)), and
  `kron` with `Diagonal` arguments
  ([#1695](https://github.com/JuliaGPU/CUDA.jl/pull/1695)).
- Right division `/` for `Diagonal` matrices
  ([#1683](https://github.com/JuliaGPU/CUDA.jl/pull/1683)), `rmul!` for QR
  factorizations ([#1739](https://github.com/JuliaGPU/CUDA.jl/pull/1739)), and
  conversion of more `QRPackedQ` objects to `CuArray`
  ([#1662](https://github.com/JuliaGPU/CUDA.jl/pull/1662)).
- `Adapt.adapt` to a partially-parameterized `CuArray` now preserves the
  requested dimensionality and buffer type, e.g.
  `adapt(CuArray{Float64,2,Mem.UnifiedBuffer}, A)`
  ([#1659](https://github.com/JuliaGPU/CUDA.jl/pull/1659)).
- `CuIterator` forwards `length`, `axes` and `eltype` of the underlying iterator
  ([#1602](https://github.com/JuliaGPU/CUDA.jl/pull/1602)), and `copy` of the
  native RNG is now supported
  ([#1719](https://github.com/JuliaGPU/CUDA.jl/pull/1719)).
- Updated wrapped libraries: cuDNN 8.6
  ([#1622](https://github.com/JuliaGPU/CUDA.jl/pull/1622)), cuTENSOR 1.6
  ([#1636](https://github.com/JuliaGPU/CUDA.jl/pull/1636)), cuStateVec 1.1
  ([#1638](https://github.com/JuliaGPU/CUDA.jl/pull/1638)), cuTensorNet 1.1
  ([#1639](https://github.com/JuliaGPU/CUDA.jl/pull/1639),
  [#1654](https://github.com/JuliaGPU/CUDA.jl/pull/1654)) and a newer cuQuantum
  ([#1688](https://github.com/JuliaGPU/CUDA.jl/pull/1688)).

*Minor changes*:

- The CUDA driver is now discovered on the system when `CUDA_Driver_jll` does
  not provide an artifact for the current platform
  ([#1658](https://github.com/JuliaGPU/CUDA.jl/pull/1658)).
- Unsupported element types are reported with an explanation of why they are
  unsupported ([#1596](https://github.com/JuliaGPU/CUDA.jl/pull/1596),
  [#1598](https://github.com/JuliaGPU/CUDA.jl/pull/1598)), and compilation
  errors now include the compiler options that were used
  ([#1657](https://github.com/JuliaGPU/CUDA.jl/pull/1657)).
- The CUBLAS and CUSPARSE wrappers generate less code, reducing compilation time
  ([#1730](https://github.com/JuliaGPU/CUDA.jl/pull/1730)).

*Bug fixes*:

- Fixed the window calculation in the native RNG, which caused elements of large
  arrays to be skipped by `rand!`
  ([#1575](https://github.com/JuliaGPU/CUDA.jl/pull/1575)).
- Arrays with an element type containing inline-allocated union fields are now
  handled correctly ([#1617](https://github.com/JuliaGPU/CUDA.jl/pull/1617),
  [#1717](https://github.com/JuliaGPU/CUDA.jl/pull/1717)).
- Fixed `FastMath.sincos`
  ([#1627](https://github.com/JuliaGPU/CUDA.jl/pull/1627)) and the signatures of
  `rotate!` and `reflect!`
  ([#1604](https://github.com/JuliaGPU/CUDA.jl/pull/1604)).
- Fixed `eigen` for `Hermitian` and `Symmetric` matrices
  ([#1677](https://github.com/JuliaGPU/CUDA.jl/pull/1677)), `\` with non-square
  matrices ([#1584](https://github.com/JuliaGPU/CUDA.jl/pull/1584)), the `LU`
  `getproperty` invoke ([#1714](https://github.com/JuliaGPU/CUDA.jl/pull/1714)),
  and the workspace argument type of the `cusolverDnXgesvdr_bufferSize` wrapper
  ([#1626](https://github.com/JuliaGPU/CUDA.jl/pull/1626)). CUSOLVER handle
  creation is now retried when it fails with an internal error
  ([#1691](https://github.com/JuliaGPU/CUDA.jl/pull/1691)).
- Avoid scalar indexing in `accumulate` on N-dimensional inputs without `dims`
  ([#1681](https://github.com/JuliaGPU/CUDA.jl/pull/1681)) and in `cholcopy`
  ([#1716](https://github.com/JuliaGPU/CUDA.jl/pull/1716)).
- Fixed conversions between `CuSparseMatrixCOO` and `CuSparseMatrixCSC`
  ([#1655](https://github.com/JuliaGPU/CUDA.jl/pull/1655)), and relaxed the
  dispatch of sparse array constructors
  ([#1643](https://github.com/JuliaGPU/CUDA.jl/pull/1643)).

### v4.0.1 (February 2023)

- Renamed the subpackages to follow NVIDIA's capitalization: cuDNN.jl,
  cuTENSOR.jl, cuTensorNet.jl and cuStateVec.jl.
- The subpackages can now be loaded even when the JLLs they need are unavailable
  for the current platform, so they can be used conditionally
  ([#1754](https://github.com/JuliaGPU/CUDA.jl/pull/1754)).
- Warn about GPUs that are deprecated or unsupported by the selected CUDA
  toolkit, and refuse to generate SASS code on sm_35 devices with CUDA 11.7 or
  later, which trips an NVIDIA bug in CUPTI
  ([#1752](https://github.com/JuliaGPU/CUDA.jl/pull/1752)).


## v3.13 (January 2023)

A small release cut from the 3.x branch, days before 4.0 shipped.

*New features*:

- Memory use can be limited through the `JULIA_CUDA_SOFT_MEMORY_LIMIT` and
  `JULIA_CUDA_HARD_MEMORY_LIMIT` environment variables, which is useful when
  sharing a GPU with other users or applications. The soft limit configures the
  memory pool's release threshold, while the hard limit is checked before every
  allocation. Both accept a byte count with an optional unit, or a percentage of
  total device memory (e.g. `1.5GiB`, `50%`)
  ([#1698](https://github.com/JuliaGPU/CUDA.jl/pull/1698)).
- Added an `always_inline` keyword argument to `@cuda`, `cufunction` and the
  code reflection functions, forcing all calls in a kernel to be inlined
  ([#1554](https://github.com/JuliaGPU/CUDA.jl/pull/1554)).

### v3.13.1 (January 2023)

- Removed the `Random.rand`/`randn` methods that CUDA.jl defined for
  `GPUArrays.RNG`, which pirated a type from GPUArrays; those methods are now
  provided by GPUArrays itself
  ([#1735](https://github.com/JuliaGPU/CUDA.jl/pull/1735)).


## v3.12 (July 2022)

The 3.12 line turned into a maintenance series: development moved on to 4.0, and
the two patch releases only appeared half a year later, in January 2023,
backporting bug fixes for users who could not upgrade yet.

*New features*:

- `reshape` is now supported on `CuDeviceArray`, i.e., from within kernels
  ([#1561](https://github.com/JuliaGPU/CUDA.jl/pull/1561)).
- Added low-level wrappers for the cusolverRf and cusolverSp low-level preview
APIs ([#1547](https://github.com/JuliaGPU/CUDA.jl/pull/1547)). *Minor changes*:

- `unsafe_wrap` now takes an `Int` length instead of any `Integer`, so that
  passing another integer type reports a `MethodError` on `unsafe_wrap` itself
  instead of deeper down
([#1552](https://github.com/JuliaGPU/CUDA.jl/pull/1552)). *Bug fixes*:

- Updated the device-side override of `Base.Math.throw_exp_domainerror` to the
  signature used by recent Julia versions
  ([#1546](https://github.com/JuliaGPU/CUDA.jl/pull/1546)).
- Fixed a precompilation failure on Julia 1.9 by explicitly importing
  `QRCompactWY`, `QRCompactWYQ` and `QRPackedQ` from `LinearAlgebra`
  ([#1558](https://github.com/JuliaGPU/CUDA.jl/pull/1558)).

### v3.12.1 (January 2023)

- `ldiv!` and `\` now handle non-square QR factorizations correctly, and gained
  a method for matrix right-hand sides
  ([#1584](https://github.com/JuliaGPU/CUDA.jl/pull/1584)).
- `CuIterator` now implements `length`, `axes`, `eltype`, `IteratorSize` and
  `IteratorEltype`, forwarded from the wrapped iterable
  ([#1602](https://github.com/JuliaGPU/CUDA.jl/pull/1602)).
- Fixed the grid-stride window calculation in the native RNG's `rand!` kernel
  ([#1575](https://github.com/JuliaGPU/CUDA.jl/pull/1575)).
- `FastMath.sincos` again calls the CUDA fast-math intrinsics; Base had switched
  it to the generic implementation, which is slower on the GPU
  ([#1627](https://github.com/JuliaGPU/CUDA.jl/pull/1627)).
- Adapting an array to `CuArray{T,N,B}` now preserves the requested buffer type
  `B`, and `CuArray{T,N}` preserves the dimensionality
  ([#1659](https://github.com/JuliaGPU/CUDA.jl/pull/1659)).
- `getproperty` on an `LU` factorization of a `CuMatrix` works again on Julia
  1.9 ([#1714](https://github.com/JuliaGPU/CUDA.jl/pull/1714)).
- Added a `cholcopy` specialization for Hermitian and symmetric `CuArray`s,
  avoiding scalar indexing in `cholesky`
  ([#1716](https://github.com/JuliaGPU/CUDA.jl/pull/1716)).
- cuDNN convolution algorithm search now calls `CUDA.reclaim()` first, which
  reduces pool fragmentation and allows larger batch sizes before running out of
  memory ([#1711](https://github.com/JuliaGPU/CUDA.jl/pull/1711)).
- CUSOLVER handle creation is now retried after reclaiming memory when it fails
  with `CUSOLVER_STATUS_INTERNAL_ERROR`
  ([#1691](https://github.com/JuliaGPU/CUDA.jl/pull/1691)).
- WMMA fragments holding a single element are no longer wrapped in a struct,
  which generated invalid LLVM IR for e.g. `u8` fragment loads
  ([#1704](https://github.com/JuliaGPU/CUDA.jl/pull/1704)).
- WMMA intrinsics now use the plain `llvmcall` calling convention, as required
  by Julia 1.9 ([#1709](https://github.com/JuliaGPU/CUDA.jl/pull/1709)).
- The CUBLAS `rotate!` and `reflect!` methods are now restricted to `Number`
  coefficients, so that other argument types fall back to the generic GPUArrays
  implementation ([#1604](https://github.com/JuliaGPU/CUDA.jl/pull/1604)).
- `ptxas` and `nvlink` failures now report the invocation arguments alongside
  the log ([#1657](https://github.com/JuliaGPU/CUDA.jl/pull/1657)).

### v3.12.2 (January 2023)

- Rendering a `CuError` before the CUDA driver has been initialized no longer
  recurses, which caused a stack overflow with `JULIA_DEBUG=CUDA` set
  ([#1723](https://github.com/JuliaGPU/CUDA.jl/pull/1723)).
- Stopped defining `rand`/`randn` methods for `GPUArrays.RNG`, which were type
  piracy and broke other GPU back-ends
  ([#1735](https://github.com/JuliaGPU/CUDA.jl/pull/1735)).


## v3.11 (June 2022)

*New features*:

- Added `download_artifacts()`, which pre-populates the artifact cache without
  requiring a working driver, for use in container build scripts. Set
  `JULIA_CUDA_VERSION` to pick the toolkit release to download
  ([#1539](https://github.com/JuliaGPU/CUDA.jl/pull/1539)).
- Added `ldiv!` for LU factorizations of `CuMatrix`
  ([#1532](https://github.com/JuliaGPU/CUDA.jl/pull/1532)).
- Added `mul!` specializations for products of two triangular `CuMatrix`es,
  including transposed and adjoint operands, using CUBLAS `trmm!`
  ([#1538](https://github.com/JuliaGPU/CUDA.jl/pull/1538)).
- Added `+` and `-` methods between `Diagonal` and CUSPARSE CSC/CSR matrices,
  which previously fell back to a dense result
  ([#1514](https://github.com/JuliaGPU/CUDA.jl/pull/1514)).

*Minor changes*:

- Removed the unused `get`, `set` and `CUA_NULL` exports; `CUA_NULL` was
  exported without ever being defined
  ([#1527](https://github.com/JuliaGPU/CUDA.jl/pull/1527),
  [#1545](https://github.com/JuliaGPU/CUDA.jl/pull/1545)).
- Updated to GPUCompiler 0.16, and fixed compatibility with Julia 1.9.

*Bug fixes*:

- Seeding a GPU RNG without an explicit seed now draws from `RandomDevice()`
  instead of the global CPU RNG, so it no longer advances Julia's default RNG
  state ([#1526](https://github.com/JuliaGPU/CUDA.jl/pull/1526),
  [#1530](https://github.com/JuliaGPU/CUDA.jl/pull/1530)).
- The cuDNN convolution algorithm search no longer runs on the destination
  array, which it overwrites with arbitrary values, but on a temporary. This
  fixes wrong results when `beta` is nonzero
  ([#1536](https://github.com/JuliaGPU/CUDA.jl/pull/1536)).


## v3.10 (May 2022)

*New features*:

- Added CUDA 11.7 to the bundled artifacts, together with the matching
  forward-compatible driver, cuDNN and cuTENSOR builds, and support for PTX ISA
  7.7. The CUDA 11.6, 11.5 and 11.4 artifacts were refreshed to 11.6.2, 11.5.2
  and 11.4.4 ([#1507](https://github.com/JuliaGPU/CUDA.jl/pull/1507)).
- `opnorm(A, Inf)` now works on `CuSparseMatrixCSR` and `CuSparseMatrixCSC`
  ([#1466](https://github.com/JuliaGPU/CUDA.jl/pull/1466)).
- The CUSPARSE `mv!` and `mm!` wrappers take an optional trailing argument to
  select the SpMV/SpMM algorithm
([#1201](https://github.com/JuliaGPU/CUDA.jl/pull/1201)). *Minor changes*:

- Artifacts are no longer shipped for CUDA 9.0, 9.2, 10.0 and 10.1; CUDA 10.2 is
  now the oldest toolkit that can be downloaded automatically
  ([#1507](https://github.com/JuliaGPU/CUDA.jl/pull/1507)).
- The cuStateVec and cuTensorNet wrappers moved into `lib/`, and cuTensorNet now
  requires cuTENSOR to be available
  ([#1478](https://github.com/JuliaGPU/CUDA.jl/pull/1478),
  [#1507](https://github.com/JuliaGPU/CUDA.jl/pull/1507)).
- Adapted to GPUCompiler 0.15, which caches compiled code by function type.
  CUDA.jl keeps its own cache of `HostKernel` objects so that kernel launch
  overhead is unaffected
([#1488](https://github.com/JuliaGPU/CUDA.jl/pull/1488),
[#1504](https://github.com/JuliaGPU/CUDA.jl/pull/1504),
[#1510](https://github.com/JuliaGPU/CUDA.jl/pull/1510)). *Bug fixes*:

- CUSPARSE `mv!` now selects a `ComplexF32` compute type for `ComplexF16` inputs
  on CUDA 11.7.2 and later, as required by the library
  ([#1505](https://github.com/JuliaGPU/CUDA.jl/pull/1505)).
- Freeing memory from a finalizer now consults the task-local context, fixing
  spurious `Error while freeing DeviceBuffer` warnings on systems with multiple
  GPUs ([#1454](https://github.com/JuliaGPU/CUDA.jl/pull/1454),
  [#1462](https://github.com/JuliaGPU/CUDA.jl/pull/1462)).
- cuDNN convolution no longer holds its cache lock while allocating memory,
  which blocked finalizers and could result in out-of-memory errors
  ([#1461](https://github.com/JuliaGPU/CUDA.jl/pull/1461),
  [#1491](https://github.com/JuliaGPU/CUDA.jl/pull/1491)).
- The memory pool cleanup task no longer reports an `EOFError` when the REPL
  exits ([#1495](https://github.com/JuliaGPU/CUDA.jl/pull/1495),
  [#1502](https://github.com/JuliaGPU/CUDA.jl/pull/1502)).

### v3.10.1 (May 2022)

- Fixed the Box-Muller transformation used by `randn` with the native RNG:
  complex element types were drawn from the wrong distribution, and radial zeros
  could produce infinities
  ([#1464](https://github.com/JuliaGPU/CUDA.jl/pull/1464),
  [#1515](https://github.com/JuliaGPU/CUDA.jl/pull/1515),
  [#1518](https://github.com/JuliaGPU/CUDA.jl/pull/1518)).
- `opnorm(A, 2)` now works on `CuMatrix`, using `svdvals`
  ([#1516](https://github.com/JuliaGPU/CUDA.jl/pull/1516)).
- Matrix division now promotes its arguments to a common element type instead of
  requiring identical types
  ([#1512](https://github.com/JuliaGPU/CUDA.jl/pull/1512),
  [#1517](https://github.com/JuliaGPU/CUDA.jl/pull/1517)).
- CUSPARSE `mv!` supports mixed real/complex argument types, as offered by the
  library since CUDA 11.0
  ([#1475](https://github.com/JuliaGPU/CUDA.jl/pull/1475)).
- The device override for `SpecialFunctions.lgamma` was moved to
  `SpecialFunctions.loggamma`, which is the name the function has been given
  upstream ([#1528](https://github.com/JuliaGPU/CUDA.jl/pull/1528),
  [#1529](https://github.com/JuliaGPU/CUDA.jl/pull/1529)).


## v3.9 (April 2022)

*Technically breaking changes*:

- Dropped support for CUDA 10.1 and below; a driver for CUDA 10.2 or newer is
  now required ([#1414](https://github.com/JuliaGPU/CUDA.jl/pull/1414)).
- On Julia 1.8 and later, `qr` and `svd` of a `CuMatrix` return the Base `QR`
  and `SVD` factorization objects instead of the CUDA-specific `CuQR` and
  `CuSVD` ([#1449](https://github.com/JuliaGPU/CUDA.jl/pull/1449)).

*New features*:

- `lu` and `lu!` are now implemented for `CuMatrix`, returning a
  `LinearAlgebra.LU` factorization
  ([#1449](https://github.com/JuliaGPU/CUDA.jl/pull/1449),
  [#1193](https://github.com/JuliaGPU/CUDA.jl/pull/1193)).
- Sparse matrices now support general broadcasting: `CuSparseMatrixCSR` and
  `CuSparseMatrixCSC` can be combined with each other, with dense arrays, and
  with scalars. Zero-preserving operations keep the result sparse, others
  produce a dense array ([#1367](https://github.com/JuliaGPU/CUDA.jl/pull/1367),
  [#1380](https://github.com/JuliaGPU/CUDA.jl/pull/1380),
  [#1401](https://github.com/JuliaGPU/CUDA.jl/pull/1401)).
- Sparse operations involving `UniformScaling`, such as `I - 3*S`, are
  implemented on top of that broadcast machinery
  ([#1390](https://github.com/JuliaGPU/CUDA.jl/pull/1390)).
- Sparse-sparse matrix multiplication is now available for `CuSparseMatrixCSR`,
  using the cuSPARSE generic SpGEMM API
  ([#1285](https://github.com/JuliaGPU/CUDA.jl/pull/1285)).
- WMMA gained support for 8-bit integer inputs (`Int8` and `UInt8`) with `Int32`
  accumulators, and for the `m32n8k16` and `m8n32k16` shapes in addition to
  `m16n16k16` ([#1119](https://github.com/JuliaGPU/CUDA.jl/pull/1119),
  [#1442](https://github.com/JuliaGPU/CUDA.jl/pull/1442)).
- Initial wrappers for the cuQuantum SDK, in the form of the `CUSTATEVEC` and
  `CUTENSORNET` subpackages. The binaries are downloaded on demand
  ([#1437](https://github.com/JuliaGPU/CUDA.jl/pull/1437)).
- NCCL binaries are now available as an artifact, exposed through `libnccl` and
  `has_nccl` ([#1450](https://github.com/JuliaGPU/CUDA.jl/pull/1450),
  [#1446](https://github.com/JuliaGPU/CUDA.jl/pull/1446)).
- `CUSPARSE.sparse` accepts unsorted COO inputs, sorting them unless
  `sorted=true` is passed
  ([#1411](https://github.com/JuliaGPU/CUDA.jl/pull/1411)).
- Conversion between `CuSparseMatrixCSR` and `CuSparseMatrixCSC` now works for
  integer and `Float16` element types
  ([#1410](https://github.com/JuliaGPU/CUDA.jl/pull/1410)).
- Added `CuSparseMatrixCOO` constructors, and `similar` with a different element
  type ([#1360](https://github.com/JuliaGPU/CUDA.jl/pull/1360)).
- `atomic_cas!` supports `BFloat16`
  ([#1400](https://github.com/JuliaGPU/CUDA.jl/pull/1400)).

*Minor changes*:

- The compiler now targets compute capability 8.6 and PTX ISA up to 7.5 when the
  LLVM version supports it (LLVM 13 and 14, i.e. Julia 1.8 and later), instead
  of falling back to 8.0 and 7.0. Compute capability 8.7 is recognized as well
  ([#1414](https://github.com/JuliaGPU/CUDA.jl/pull/1414)).
- Matrix division, `svd` and `svdvals` promote element types that CUBLAS does
  not support, such as `Float16` and integers, to a supported floating-point
  type instead of failing
  ([#1453](https://github.com/JuliaGPU/CUDA.jl/pull/1453)).
- Added CUDA 10.2 artifacts for `aarch64`
  ([#1397](https://github.com/JuliaGPU/CUDA.jl/pull/1397)).
- Nsight Systems detection looks at more environment variables
  (`CUDA_INJECTION64_PATH`, `NVTX_INJECTION64_PATH`) when locating the `nsys`
  binary ([#1459](https://github.com/JuliaGPU/CUDA.jl/pull/1459)).
- Compatibility with Julia 1.8 and 1.9
  ([#1432](https://github.com/JuliaGPU/CUDA.jl/pull/1432),
  [#1463](https://github.com/JuliaGPU/CUDA.jl/pull/1463)).
- Requires GPUCompiler 0.14 and GPUArrays 8.3.2
  ([#1441](https://github.com/JuliaGPU/CUDA.jl/pull/1441)).

*Bug fixes*:

- CUTENSOR's `axpy!` and `axpby!` write into their destination tensor instead of
  allocating a new one ([#1416](https://github.com/JuliaGPU/CUDA.jl/pull/1416)).
- `nrm2` of a `ComplexF16` array returns a `Float16` instead of a `ComplexF16`
  ([#1444](https://github.com/JuliaGPU/CUDA.jl/pull/1444)).
- Fixed COO to CSR conversion
  ([#1412](https://github.com/JuliaGPU/CUDA.jl/pull/1412)).
- Fixed the BSR `nnz` count and BSR to CSR conversion, and added indexing of BSR
  matrices ([#1409](https://github.com/JuliaGPU/CUDA.jl/pull/1409)).
- Fixed the output element type of single-argument sparse broadcast
  ([#1405](https://github.com/JuliaGPU/CUDA.jl/pull/1405)).

### v3.9.1 (May 2022)

- Debug info is no longer emitted with CUDA 11.5 and 11.6, working around a
  `ptxas` segfault; it is now only enabled from CUDA 11.7 on
  ([#1473](https://github.com/JuliaGPU/CUDA.jl/pull/1473)).
- `unsafe_wrap` accepts a fully-specified `CuArray{T,N,B}` type, and errors when
  the requested buffer type does not match the pointer
  ([#1483](https://github.com/JuliaGPU/CUDA.jl/pull/1483)).
- `byte_perm` accepts 8- and 16-bit integer arguments
  ([#1420](https://github.com/JuliaGPU/CUDA.jl/pull/1420)).
- Added `CuSparseMatrixCSR` and `CuSparseMatrixCSC` constructors taking a
  `Diagonal` ([#1470](https://github.com/JuliaGPU/CUDA.jl/pull/1470)).
- Fixed a regression that made `A \ b` fail with a `MethodError` when the
  right-hand side was a vector
  ([#1498](https://github.com/JuliaGPU/CUDA.jl/pull/1498),
  [#1476](https://github.com/JuliaGPU/CUDA.jl/pull/1476)).
- Check the result of `cudaRuntimeGetVersion`, avoiding an `InexactError` when
  querying the CUDA version fails
  ([#1490](https://github.com/JuliaGPU/CUDA.jl/pull/1490),
  [#1489](https://github.com/JuliaGPU/CUDA.jl/pull/1489)).


## v3.8 (January 2022)

*Technically breaking changes*:

- The memory pool is now configured to hold on to all memory it has allocated,
  instead of releasing it at every synchronization point. This makes
  synchronization much cheaper, but reported GPU memory usage will be higher;
  call `CUDA.reclaim()` to hand memory back to the driver. In interactive
sessions this is done periodically
([#1344](https://github.com/JuliaGPU/CUDA.jl/pull/1344)). *New features*:

- `copyto!` between `CuArray`s on different devices now works, transparently
  going through the host when needed. Peer-to-peer access is enabled
  automatically when the devices support it, avoiding the host round-trip;
  `can_access_peer` reports whether that is possible
  ([#1284](https://github.com/JuliaGPU/CUDA.jl/pull/1284)).
- `CUDA.run_compute_sanitizer()` restarts the active Julia session under
  NVIDIA's compute sanitizer, which detects memory errors, races and
  synchronization issues. Other tools are selected with the `tool` keyword
  argument. The debugging documentation has been expanded accordingly
  ([#1340](https://github.com/JuliaGPU/CUDA.jl/pull/1340)).
- `CUDA.return_type(f, tt)` returns the return type of `f` as inferred for the
  GPU, i.e. taking device overrides into account
([#1339](https://github.com/JuliaGPU/CUDA.jl/pull/1339)). *Minor changes*:

- `memory_pools_supported(dev)` queries whether a device supports stream-ordered
  allocation, deprecating the internal `has_stream_ordered`
  ([#1344](https://github.com/JuliaGPU/CUDA.jl/pull/1344)).
- The `JULIA_CUDA_MEMORY_POOL` environment variable is now documented.

*Bug fixes*:

- Fixed a regression that broke `mul!` with CUFFT plans whose output element
  type differs from the input, such as real-to-complex transforms
  ([#1341](https://github.com/JuliaGPU/CUDA.jl/pull/1341)).
- `unsafe_wrap` now recovers the host pointer of pinned memory, fixing wrapping
  on devices where the host and device pointer of registered memory differ
  ([#1342](https://github.com/JuliaGPU/CUDA.jl/pull/1342)).
- A missing CUDA driver library is now detected and reported properly, and once
  initialization has failed, `libcuda()` keeps reporting that error instead of
  failing in unrelated places
  ([#1333](https://github.com/JuliaGPU/CUDA.jl/pull/1333),
  [#1335](https://github.com/JuliaGPU/CUDA.jl/pull/1335)).
- Fixed detection of `libcublaslt` with local toolkits that do not provide it,
  and stopped eagerly opening cuDNN sublibraries, which broke older local cuDNN
  installations.
- `dot` no longer uses `Float16` and `Int16` atomics on devices that do not
  support them, and bitonic sort and CUDA graphs no longer use APIs that are
  unavailable on CUDA 10
  ([#1337](https://github.com/JuliaGPU/CUDA.jl/pull/1337)).
- Fixed a `getindex` ambiguity with `CuQRPackedQ`, and avoided an allocating
  error path in `setindex!` on a `Diagonal` from kernels on Julia 1.8
  ([#1337](https://github.com/JuliaGPU/CUDA.jl/pull/1337)).

### v3.8.1 (February 2022)

- The default memory pool is now made accessible to the peer device when
  enabling peer-to-peer access, fixing copies between devices of pool-allocated
  memory. This adds `access!` to control the visibility of a `CuMemoryPool`
  ([#1357](https://github.com/JuliaGPU/CUDA.jl/pull/1357)).
- Non-blocking synchronization no longer hangs when the operation it waits on
  fails: a timer now detects that the completion callback will never fire
  ([#1350](https://github.com/JuliaGPU/CUDA.jl/pull/1350),
  [#1369](https://github.com/JuliaGPU/CUDA.jl/pull/1369)).
- `@fastmath` versions of `log1p`, `tanh` and similar functions no longer fall
  back to libm on Julia 1.6, which failed to compile
  ([#1352](https://github.com/JuliaGPU/CUDA.jl/pull/1352),
  [#1356](https://github.com/JuliaGPU/CUDA.jl/pull/1356)).
- Fixed `resize!` of empty arrays
  ([#1359](https://github.com/JuliaGPU/CUDA.jl/pull/1359)).
- The `setindex!(::Diagonal, ...)` definition added in 3.8.0 was missing its
  `@device_override`, overwriting the LinearAlgebra method on the host
  ([#1364](https://github.com/JuliaGPU/CUDA.jl/pull/1364)).

### v3.8.2 (February 2022)

- `SpecialFunctions.gamma` is now supported in kernels, replacing the
  `CUDA.tgamma` device function
  ([#1361](https://github.com/JuliaGPU/CUDA.jl/pull/1361)).
- Memory buffers now record the context they were allocated from, so that
  freeing memory from a finalizer no longer has to query the active context,
  which could switch tasks. Frees without an explicit stream are now ordered
  against the default stream
  ([#1383](https://github.com/JuliaGPU/CUDA.jl/pull/1383),
  [#1384](https://github.com/JuliaGPU/CUDA.jl/pull/1384)).
- CUFFT plan finalizers now switch to the plan's context before freeing it.
- `errormonitor` is no longer used on Julia 1.6, where it does not exist
  ([#1375](https://github.com/JuliaGPU/CUDA.jl/pull/1375),
  [#1378](https://github.com/JuliaGPU/CUDA.jl/pull/1378),
  [#1382](https://github.com/JuliaGPU/CUDA.jl/pull/1382)).

### v3.8.3 (February 2022)

- Fixed a regression in 3.8.2 that made allocations use the asynchronous API
  even when no memory pool was in use.
- CUDA.jl now always reports why it could not find a usable CUDA installation,
  instead of only saying that none was found. The driver-library error messages
  for Windows and Linux were swapped
  ([#1404](https://github.com/JuliaGPU/CUDA.jl/pull/1404)).
- `CUDA.math_mode!(...; precision)` now actually applies the requested precision
  ([#1392](https://github.com/JuliaGPU/CUDA.jl/pull/1392),
  [#1394](https://github.com/JuliaGPU/CUDA.jl/pull/1394)).
- Precompiling CUDA.jl no longer triggers compilation of device code: the random
  number generator's constant tables are now emitted lazily
  ([#1391](https://github.com/JuliaGPU/CUDA.jl/pull/1391)).

### v3.8.4 (March 2022)

- Out-of-memory errors now report the memory status as it was at the time of the
  failure, instead of when the error is displayed
  ([#1427](https://github.com/JuliaGPU/CUDA.jl/pull/1427),
  [#1428](https://github.com/JuliaGPU/CUDA.jl/pull/1428)).
- Lookups in the cuDNN descriptor and convolution algorithm caches are now
  locked, making them safe to use from multiple threads
  ([#1421](https://github.com/JuliaGPU/CUDA.jl/pull/1421),
  [#1430](https://github.com/JuliaGPU/CUDA.jl/pull/1430)).
- `CUDA.run_compute_sanitizer` now correctly propagates `--project` to the new
  session.

### v3.8.5 (March 2022)

- Fixed an error while reporting the memory status on out-of-memory, as
  introduced in 3.8.4, when no memory pool is in use or when using CUDA before
  11.3.


## v3.7 (January 2022)

*New features*:

- Added support for CUDA 11.6, and made it the preferred toolkit version
  ([#1326](https://github.com/JuliaGPU/CUDA.jl/pull/1326),
  [#1329](https://github.com/JuliaGPU/CUDA.jl/pull/1329)).
- Exposed the multi-GPU cuTENSOR library as `libcutensormg`, available with
  cuTENSOR 1.4 and later
([#1327](https://github.com/JuliaGPU/CUDA.jl/pull/1327)). *Minor changes*:

- Updated the bundled cuDNN to 8.3.2 and cuTENSOR to 1.4.0
  ([#1327](https://github.com/JuliaGPU/CUDA.jl/pull/1327),
  [#1328](https://github.com/JuliaGPU/CUDA.jl/pull/1328)).
- Debug info is now emitted automatically when using CUDA 11.5 or newer, which
  is required to debug kernels with `compute-sanitizer` or `cuda-gdb`. Older
  toolkits keep it disabled to avoid a `ptxas` crash, and the
  `JULIA_CUDA_DEBUG_INFO` environment variable has been removed
  ([#1259](https://github.com/JuliaGPU/CUDA.jl/pull/1259)).
- The forward-compatible driver package is now considered with system drivers
  older than CUDA 11.6, instead of 11.5
  ([#1326](https://github.com/JuliaGPU/CUDA.jl/pull/1326)).
- Kernels can now call `gcd`, `Base.Checked.checked_abs`, and `StepRangeLen`
  constructors, whose error paths previously prevented compilation
([#1315](https://github.com/JuliaGPU/CUDA.jl/pull/1315)). *Bug fixes*:

- `mul!` now works with in-place CUFFT plans, and applying a plan to data of the
  wrong in-placeness throws an error instead of misbehaving. Redundant copies of
  the input are no longer made for in-place and real-to-complex transforms
  ([#1311](https://github.com/JuliaGPU/CUDA.jl/pull/1311),
  [#1313](https://github.com/JuliaGPU/CUDA.jl/pull/1313)).
- Fixed a segfault when using CUDA.jl from a custom system image, caused by the
  `@cfunction` used for stream callbacks
  ([#1314](https://github.com/JuliaGPU/CUDA.jl/pull/1314),
  [#1319](https://github.com/JuliaGPU/CUDA.jl/pull/1319)).

### v3.7.1 (January 2022)

- `CUDA.functional()` and `has_cuda_gpu()` now return `false` consistently when
  the driver is missing or its initialization failed, instead of throwing on
  subsequent calls. This notably affected sessions with an empty
  `CUDA_VISIBLE_DEVICES`
  ([#1331](https://github.com/JuliaGPU/CUDA.jl/pull/1331),
  [#1333](https://github.com/JuliaGPU/CUDA.jl/pull/1333),
  [#1335](https://github.com/JuliaGPU/CUDA.jl/pull/1335),
  [#1336](https://github.com/JuliaGPU/CUDA.jl/pull/1336)).
- Fixed a regression in backwards CUFFT plans, where `mul!` no longer accepted
  an output array of a different element type than the input
  ([#1341](https://github.com/JuliaGPU/CUDA.jl/pull/1341)).
- Fixed `unsafe_wrap` and host-to-device copies involving mapped host memory on
  GPUs where host and device pointers differ
  ([#1342](https://github.com/JuliaGPU/CUDA.jl/pull/1342)).
- Restored compatibility with CUDA 10, where sorting and graph instantiation
  used functionality only available in CUDA 11
  ([#1337](https://github.com/JuliaGPU/CUDA.jl/pull/1337)).
- `dot` no longer uses 16-bit atomics on devices older than Volta, where they
  are unsupported ([#1337](https://github.com/JuliaGPU/CUDA.jl/pull/1337)).


## v3.6 (December 2021)

*Technically breaking changes*:

- The special stream constructors `CuDefaultStream()`, `CuStreamLegacy()` and
  `CuStreamPerThread()` have been renamed to `default_stream()`,
  `legacy_stream()` and `per_thread_stream()`, and `CUDA.query` has been renamed
  to `CUDA.isdone`. The old names are deprecated
  ([#1207](https://github.com/JuliaGPU/CUDA.jl/pull/1207)).
- The `blocking` keyword to `synchronize` has been deprecated. Synchronization
  now always spins for a bounded time before calling into the CUDA API, as CUDA
  relies on those calls to perform deferred work such as releasing memory
  ([#1213](https://github.com/JuliaGPU/CUDA.jl/pull/1213)).
- `sort!` of a `CuVector` now uses a bitonic sort by default. The previous
  implementation is still available as `sort!(x; alg=CUDA.QuickSort)`
  ([#1217](https://github.com/JuliaGPU/CUDA.jl/pull/1217)).
- `CUSPARSE.geam` has been reimplemented on top of `csrgeam2`, and now takes
  both scalars explicitly: `geam(alpha, A, beta, B, index)`
  ([#1195](https://github.com/JuliaGPU/CUDA.jl/pull/1195)).
- cuTENSOR contractions now use a `CuTensorContractionPlan` object that owns its
  workspace. `plan_contraction` returns this type, and it is what the `plan`
  keyword of `contraction!` expects
([#1243](https://github.com/JuliaGPU/CUDA.jl/pull/1243)). *New features*:

- Support for CUDA 11.5, which is now the default toolkit version
  ([#1228](https://github.com/JuliaGPU/CUDA.jl/pull/1228),
  [#1256](https://github.com/JuliaGPU/CUDA.jl/pull/1256),
  [#1267](https://github.com/JuliaGPU/CUDA.jl/pull/1267)).
- `sortperm` and `sortperm!` are now supported for `CuVector`s
  ([#1217](https://github.com/JuliaGPU/CUDA.jl/pull/1217)).
- `randn` and `randexp` can now be called from kernels
  ([#1236](https://github.com/JuliaGPU/CUDA.jl/pull/1236)).
- Shared memory arrays created with `CuStaticSharedArray` and
  `CuDynamicSharedArray` now support isbits-union element types such as
  `Union{Missing,Int}` ([#1288](https://github.com/JuliaGPU/CUDA.jl/pull/1288)).
- Multi-instance GPUs are handled correctly: on CUDA 11.4 and later `uuid`
  returns the UUID of the compute instance, and the new `parent_uuid` returns
  that of the physical device
  ([#1199](https://github.com/JuliaGPU/CUDA.jl/pull/1199)).
- Addition and subtraction of `CuSparseMatrix` objects now handles `Transpose`
  and `Adjoint` operands, as well as mixed CSR/CSC inputs
  ([#1195](https://github.com/JuliaGPU/CUDA.jl/pull/1195)).
- Wrapped the CUSOLVER sparse reordering functions `symrcm`, `symmdq`, `symamd`,
  `metisnd` and `zfd` ([#1198](https://github.com/JuliaGPU/CUDA.jl/pull/1198)).
- Wrapped `CUBLAS.spmv!` and `CUBLAS.spr!`
  ([#1248](https://github.com/JuliaGPU/CUDA.jl/pull/1248)).
- `dot` now works on complex arrays, and between `CuArray`s with different
  element types ([#1240](https://github.com/JuliaGPU/CUDA.jl/pull/1240),
  [#1245](https://github.com/JuliaGPU/CUDA.jl/pull/1245)).
- The `JULIA_CUDA_USE_COMPAT` environment variable controls whether the
  forward-compatible driver package is used. It is disabled automatically when
  Julia runs under a tool that hooks CUDA API calls, which would otherwise
prevent unloading the system driver
([#1228](https://github.com/JuliaGPU/CUDA.jl/pull/1228)). *Minor changes*:

- cuDNN has been updated to 8.3.1, and cuTENSOR to 1.3.3
  ([#1239](https://github.com/JuliaGPU/CUDA.jl/pull/1239),
  [#1267](https://github.com/JuliaGPU/CUDA.jl/pull/1267)).
- `dot` falls back to a deterministic `mapreduce` under `CUDA.PEDANTIC_MATH`
  instead of using atomics
  ([#1245](https://github.com/JuliaGPU/CUDA.jl/pull/1245)).
- Copies between `Array`s and `CuArray`s backed by unified or host memory work
  again, and copies between unified arrays in different contexts are now allowed
  ([#1210](https://github.com/JuliaGPU/CUDA.jl/pull/1210),
  [#1265](https://github.com/JuliaGPU/CUDA.jl/pull/1265),
  [#1277](https://github.com/JuliaGPU/CUDA.jl/pull/1277)).
- Compatible with SpecialFunctions 2
([#1249](https://github.com/JuliaGPU/CUDA.jl/pull/1249)). *Bug fixes*:

- `deepcopy` of an object containing a `CuArray` works again, by defining
  `deepcopy_internal` instead of `deepcopy`
  ([#1221](https://github.com/JuliaGPU/CUDA.jl/pull/1221)).
- Fixed a typo in the CUFFT plan work-area handling that could corrupt memory
  ([#1204](https://github.com/JuliaGPU/CUDA.jl/pull/1204)).
- In-place complex FFTs now update the plan's stream, fixing their use from
  multiple tasks ([#1269](https://github.com/JuliaGPU/CUDA.jl/pull/1269)).
- `CUDA.zeros` and `CUDA.ones` now use `zero(T)` and `one(T)`, so they work with
  user-defined element types
  ([#1278](https://github.com/JuliaGPU/CUDA.jl/pull/1278)).
- `sort!` returns the sorted array again instead of `nothing`
  ([#1272](https://github.com/JuliaGPU/CUDA.jl/pull/1272)).
- `accumulate` with an explicit `init` no longer applies the initializer twice
  ([#1237](https://github.com/JuliaGPU/CUDA.jl/pull/1237)).
- Array offsets are now stored in elements instead of bytes, fixing arrays whose
  dimensions are all singletons
  ([#1255](https://github.com/JuliaGPU/CUDA.jl/pull/1255)).
- Kernels that iterate over the components of a union of more than two types
  compile again ([#1257](https://github.com/JuliaGPU/CUDA.jl/pull/1257)).
- `CuStaticSharedArray` and `CuDynamicSharedArray` accept dimension tuples that
  mix integer types, e.g. a block dimension and a kernel argument
  ([#1211](https://github.com/JuliaGPU/CUDA.jl/pull/1211)).
- Fixed cuTENSOR contractions with `Float16` inputs
  ([#1238](https://github.com/JuliaGPU/CUDA.jl/pull/1238)).
- Fixed sparse `mul!` with an adjoint real matrix on CUDA 11.5
  ([#1234](https://github.com/JuliaGPU/CUDA.jl/pull/1234)).
- CUSPARSE conversions and generic operations now honor the index-base argument
  ([#1214](https://github.com/JuliaGPU/CUDA.jl/pull/1214)).
- The CUDA driver is initialized again when the package is loaded
  ([#1287](https://github.com/JuliaGPU/CUDA.jl/pull/1287)).
- The forward-compatible driver artifact is only resolved when it is actually
  going to be used, avoiding errors about missing or empty artifacts
  ([#1275](https://github.com/JuliaGPU/CUDA.jl/pull/1275)).
- cuDNN now loads CUBLAS first, fixing failures to load `libcudnn`
  ([#1279](https://github.com/JuliaGPU/CUDA.jl/pull/1279)).
- `CUDA.@profile` no longer invokes `nsys stop`, which broke profiling under
  Nsight Systems ([#1282](https://github.com/JuliaGPU/CUDA.jl/pull/1282)).

### v3.6.1 (December 2021)

- Rebuilt the cuDNN artifacts to include zlib on Windows, fixing an error
  loading `cudnn_cnn_infer64_8.dll`
  ([#1293](https://github.com/JuliaGPU/CUDA.jl/pull/1293)).

### v3.6.2 (December 2021)

- Fixed cuDNN convolution algorithm discovery when `CUDA.cached_memory()` is
  unavailable, e.g. when not using the CUDA memory pool
  ([#1295](https://github.com/JuliaGPU/CUDA.jl/pull/1295)).
- `norm` of a complex `CuArray` returns a real number again
  ([#1290](https://github.com/JuliaGPU/CUDA.jl/pull/1290)).

### v3.6.3 (January 2022)

- `CuDeviceArray` now stores the array length next to its dimensions, speeding
  up indexing of high-dimensional arrays
  ([#1303](https://github.com/JuliaGPU/CUDA.jl/pull/1303)).
- Device-side array intrinsics can no longer be called from the host, which
  previously resulted in segfaults when code failed to dispatch correctly
  ([#1305](https://github.com/JuliaGPU/CUDA.jl/pull/1305)).
- Logical indexing of a `CuArray` with a CPU `Array{Bool}` or `BitArray` is
  supported ([#1306](https://github.com/JuliaGPU/CUDA.jl/pull/1306)).
- `CUDA.@atomic` uses `isequal` in its compare-and-swap loop, avoiding a
  deadlock when overwriting `NaN`
  ([#1300](https://github.com/JuliaGPU/CUDA.jl/pull/1300)).
- `device!` now activates the corresponding context, so calling it at the start
  of a session initializes CUDA
  ([#1307](https://github.com/JuliaGPU/CUDA.jl/pull/1307)).
- `sort!` works on `CuVector`s of tuples
  ([#1196](https://github.com/JuliaGPU/CUDA.jl/pull/1196)).

### v3.6.4 (January 2022)

- Fixed the `git-tree-sha1` values of the forward-compatibility artifacts in
  `Artifacts.toml`, which broke installation when using a package server
  ([#1310](https://github.com/JuliaGPU/CUDA.jl/pull/1310)).


## v3.5 (October 2021)

*Technically breaking changes*:

- The low-level device and context getters have been renamed:
  `CuCurrentContext()` and `CuCurrentDevice()` are now `current_context()` and
  `current_device()`, which throw an `UndefRefError` instead of returning
  `nothing` (use `has_context()` and `has_device()` to check first), while
  `CuDevice(::CuContext)` and `CuContext(::Ptr)` are now `device(ctx)` and
  `context(ptr)`. The old names are deprecated
  ([#1135](https://github.com/JuliaGPU/CUDA.jl/pull/1135)).
- `@cuStaticSharedMem` and `@cuDynamicSharedMem` are deprecated in favor of the
  `CuStaticSharedArray` and `CuDynamicSharedArray` functions. Dynamic shared
  memory is now bounds-checked against the amount of shared memory that was
  requested ([#1114](https://github.com/JuliaGPU/CUDA.jl/pull/1114)).
- The CUSPARSE array types carry an additional index type parameter, e.g.
  `CuSparseMatrixCSC{Tv,Ti}`, and are no longer hard-coded to `Cint` indices
  ([#1163](https://github.com/JuliaGPU/CUDA.jl/pull/1163)).
- The low-level CUSPARSE and CUSOLVER wrappers consistently take `Char`s instead
  of raw enum values ([#1181](https://github.com/JuliaGPU/CUDA.jl/pull/1181)).
- Indexing intrinsics like `threadIdx` and `blockIdx` now throw an error when
  called on the CPU, instead of returning a meaningless value
  ([#1117](https://github.com/JuliaGPU/CUDA.jl/pull/1117)).
- `device_synchronize()` is now implemented in Julia, by synchronizing the
  legacy stream, so that it does not block the Julia scheduler. The blocking
  driver call is available as `synchronize(::CuContext)`
([#1147](https://github.com/JuliaGPU/CUDA.jl/pull/1147)). *New features*:

- CUDA.jl implements CUDA's forward compatibility mode: when the system driver
  is too old, a newer driver library is loaded from an artifact, making it
  possible to use a newer toolkit (on x86_64, powerpc64le and aarch64 Linux).
  `CUDA.system_version()` reports the version of the system driver, and
  `versioninfo()` now lists both
  ([#1182](https://github.com/JuliaGPU/CUDA.jl/pull/1182)).
- `unsafe_wrap(CuArray, ptr, dims)` supports pointers to host memory
  ([#1131](https://github.com/JuliaGPU/CUDA.jl/pull/1131)).
- `sparse` is implemented for `CuArray`s, with a `fmt` keyword argument to
  select between the `:csc`, `:csr`, `:bsr` and `:coo` formats, as well as a
  method taking `I`, `J` and `V` vectors
  ([#1093](https://github.com/JuliaGPU/CUDA.jl/pull/1093)).
- Sparse arrays have device-side counterparts, so they can be passed to kernels
  ([#1106](https://github.com/JuliaGPU/CUDA.jl/pull/1106),
  [#1154](https://github.com/JuliaGPU/CUDA.jl/pull/1154)).
- Sparse matrices can be constructed from a `Transpose` or `Adjoint` of a sparse
  matrix, turning a CSC matrix into a CSR one (and vice versa) without permuting
  the data ([#1132](https://github.com/JuliaGPU/CUDA.jl/pull/1132)).
- `reinterpret(reshape, T, ::CuArray)` is supported
  ([#1149](https://github.com/JuliaGPU/CUDA.jl/pull/1149)).
- `mul!` supports `Hermitian` and `Symmetric` matrices, using the CUBLAS `symv`,
  `hemv`, `symm` and `hemm` routines
  ([#217](https://github.com/JuliaGPU/CUDA.jl/pull/217)).
- The random number interface is more complete: `rand`, `randn`, `rand_logn` and
  `rand_poisson` now accept dimensions as varargs, can generate scalars, and can
  fill CPU arrays ([#1146](https://github.com/JuliaGPU/CUDA.jl/pull/1146)).
*Minor changes*:

- Device arrays preserve 32-bit hardware indices instead of promoting them to
  64-bit integers, reducing register pressure in kernels that index using
  `Int32` values ([#1153](https://github.com/JuliaGPU/CUDA.jl/pull/1153),
  [#1167](https://github.com/JuliaGPU/CUDA.jl/pull/1167)).
- Reductions use warp shuffle instructions for all integer types, as well as for
  `Float16` and `ComplexF16`
  ([#1130](https://github.com/JuliaGPU/CUDA.jl/pull/1130)).
- `@captured` disables the garbage collector during capture, so that memory
  operations cannot break the capture
  ([#1137](https://github.com/JuliaGPU/CUDA.jl/pull/1137)).
- Discovery of a local CUDA toolkit no longer relies on a database of library
  versions: unversioned libraries are now used when available, and the toolkit
  version is determined using the CUDA runtime library instead of `ptxas`. That
  library is also exposed as `libcudart` for use by other packages.
  `JULIA_CUDA_VERSION` now selects a CUDA release, e.g. `11.4`, instead of an
  exact toolkit version ([#1121](https://github.com/JuliaGPU/CUDA.jl/pull/1121),
  [#1134](https://github.com/JuliaGPU/CUDA.jl/pull/1134),
  [#1159](https://github.com/JuliaGPU/CUDA.jl/pull/1159)).
- Compatible with BFloat16s.jl 0.2
  ([#1156](https://github.com/JuliaGPU/CUDA.jl/pull/1156)) and with Julia 1.8
  ([#1183](https://github.com/JuliaGPU/CUDA.jl/pull/1183)).

*Bug fixes*:

- Memory operations on a `CuArray` switch to the array's context first, fixing
  illegal memory accesses when working with multiple devices, e.g. after calling
  `device!` in the REPL
  ([#1176](https://github.com/JuliaGPU/CUDA.jl/pull/1176)).
- Reclaiming memory now performs actual synchronization API calls, which the
  driver requires before it releases memory held by the stream-ordered
  allocator. This fixes spurious out-of-memory errors
  ([#1157](https://github.com/JuliaGPU/CUDA.jl/pull/1157),
  [#1177](https://github.com/JuliaGPU/CUDA.jl/pull/1177)).
- Reductions on large arrays work with operators that have no neutral element,
  e.g. `sum!` into a preallocated destination
  ([#1174](https://github.com/JuliaGPU/CUDA.jl/pull/1174)).
- Broadcasting a type, as in `convert.(ComplexF32, x)`, works on the GPU
  ([#1109](https://github.com/JuliaGPU/CUDA.jl/pull/1109)).
- `normalize` works on complex arrays
  ([#1151](https://github.com/JuliaGPU/CUDA.jl/pull/1151)).
- Sparse GPU arrays are displayed like their CPU counterparts, and printing a
  custom sparse array type no longer overflows the stack
  ([#1129](https://github.com/JuliaGPU/CUDA.jl/pull/1129)).
- `byte_perm` no longer inserts sign checks when called with `Int32` values
  ([#1166](https://github.com/JuliaGPU/CUDA.jl/pull/1166)).
- `CUDA.cached_memory()` no longer fails on CUDA 11.2, where the amount of
  memory reserved by the pool cannot be queried.


## v3.4 (August 2021)

This release drops the Julia-side memory pools in favor of CUDA's stream-ordered
allocator, which requires a driver supporting CUDA 11.2 or later and is
incompatible with the legacy `cuIpc` APIs used by OpenMPI. The 3.3 series was
maintained alongside 3.4 for users affected by either limitation.

*Technically breaking changes*:

- Removed the Julia-side `binned`, `split` and `simple` memory pools.
  `JULIA_CUDA_MEMORY_POOL` now only accepts `cuda` (CUDA's stream-ordered pool)
  or `none`; running without a pool degrades allocation performance
  ([#1015](https://github.com/JuliaGPU/CUDA.jl/pull/1015)).
- Removed support for GPU memory limits through `JULIA_CUDA_MEMORY_LIMIT`
  ([#1015](https://github.com/JuliaGPU/CUDA.jl/pull/1015)).
- `CuArray` gained a third type parameter identifying the buffer that backs it,
  e.g. `CuArray{Int,1,CUDA.Mem.DeviceBuffer}`
  ([#1016](https://github.com/JuliaGPU/CUDA.jl/pull/1016),
  [#1023](https://github.com/JuliaGPU/CUDA.jl/pull/1023)).
- Streamlined atomics: the low-level `atomic_` functions now only cover
  operations the hardware supports natively, so e.g. `atomic_mul!` no longer
  exists and `atomic_sub!` no longer works on `Float64`. Use `CUDA.@atomic` for
  those instead ([#1059](https://github.com/JuliaGPU/CUDA.jl/pull/1059)).
- `@atomic` is no longer exported on Julia 1.7 and later, to avoid conflicting
  with the macro in Base. Always spell it `CUDA.@atomic`
([#1097](https://github.com/JuliaGPU/CUDA.jl/pull/1097)). *New features*:

- Arrays can be backed by unified memory, allocated with `cu(x; unified=true)`
  or by constructing a `CuArray{T,N,CUDA.Mem.UnifiedBuffer}`, and used from
  multiple devices. The `is_unified` predicate tests for this
  ([#1023](https://github.com/JuliaGPU/CUDA.jl/pull/1023)).
- `CUDA.@atomic` now falls back to a compare-and-swap loop for operations that
  do not map onto a single atomic instruction, so any operation works as long as
  `CUDA.atomic_cas!` supports the element type. More in-place operations are
  recognized ([#1059](https://github.com/JuliaGPU/CUDA.jl/pull/1059),
  [#1098](https://github.com/JuliaGPU/CUDA.jl/pull/1098)).
- The native random number generator introduced in CUDA.jl 3.0 now supports
  normally distributed and complex numbers, and is used as the default fallback
  when CURAND does not support the requested element type. It is both faster and
  of better statistical quality than the previous GPUArrays.jl-based fallback
  ([#1082](https://github.com/JuliaGPU/CUDA.jl/pull/1082)).
- Kernels can branch on hardware properties using `compute_capability()` and
  `ptx_isa_version()`, compared against version literals from the `sv"..."`
  string macro. The branches are folded away during optimization, without
  requiring re-inference
  ([#1060](https://github.com/JuliaGPU/CUDA.jl/pull/1060)).
- Added support for CUDA 11.4 and CUDA 11.4 Update 1, and bumped the bundled
  cuDNN to 8.2.2 ([#1024](https://github.com/JuliaGPU/CUDA.jl/pull/1024),
  [#1084](https://github.com/JuliaGPU/CUDA.jl/pull/1084)).
- CUBLAS, CUSOLVER and CUSPARSE wrappers now accept strided inputs instead of
  requiring contiguous arrays
  ([#1038](https://github.com/JuliaGPU/CUDA.jl/pull/1038)).
- Added half-precision batched `gemm`
([#1080](https://github.com/JuliaGPU/CUDA.jl/pull/1080)). *Minor changes*:

- Memory pools can now be created with a specific handle type, e.g.
  `CuMemoryPool(device(); handle_type=CUDA.HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)`
  ([#1036](https://github.com/JuliaGPU/CUDA.jl/pull/1036)).
- `libcuda` is now located with `Libdl.find_library`, covering more installation
  layouts, and reports an actionable error when the driver library cannot be
  found ([#1030](https://github.com/JuliaGPU/CUDA.jl/pull/1030)).
- `device!` returns the device that was activated.
- Array reference counting moved to an atomic field on the array's storage
  object, removing a lock from the allocation path
  ([#1016](https://github.com/JuliaGPU/CUDA.jl/pull/1016)).
- Reduced package load time and time-to-first-kernel
  ([#1069](https://github.com/JuliaGPU/CUDA.jl/pull/1069)).
- The library handle cache is now thread-safe
([#1074](https://github.com/JuliaGPU/CUDA.jl/pull/1074)). *Bug fixes*:

- Integer powers with a constant exponent, such as `x^2`, are now lowered
  directly instead of going through a generic `pow`
  ([#1033](https://github.com/JuliaGPU/CUDA.jl/pull/1033)).
- `rem` now uses the correct device intrinsics
  ([#1041](https://github.com/JuliaGPU/CUDA.jl/pull/1041)).
- `sincos` now comes from libdevice, avoiding illegal global loads
  ([#1086](https://github.com/JuliaGPU/CUDA.jl/pull/1086)).
- CUSPARSE: fixed matrix-matrix multiplication, conversion from sparse to dense
  matrices, and multiplication involving empty matrices
  ([#1073](https://github.com/JuliaGPU/CUDA.jl/pull/1073),
  [#1083](https://github.com/JuliaGPU/CUDA.jl/pull/1083),
  [#1096](https://github.com/JuliaGPU/CUDA.jl/pull/1096)).
- `CuArray(Q)` and `CuMatrix(Q)` on the `Q` factor of a `qr` factorization now
  yield the compact `Q`, matching `Matrix(Q)`
  ([#1063](https://github.com/JuliaGPU/CUDA.jl/pull/1063)).
- Fixed `cholesky` and a `Base.unsafe_length` deprecation warning on Julia 1.8
  ([#1045](https://github.com/JuliaGPU/CUDA.jl/pull/1045),
  [#1049](https://github.com/JuliaGPU/CUDA.jl/pull/1049)).
- Faster `mapreduce` when reducing many independent slices at once, e.g.
  `sum(CUDA.zeros(Int, 40000, 100); dims=1)`
  ([#1012](https://github.com/JuliaGPU/CUDA.jl/pull/1012)).

### v3.4.1 (August 2021)

- Restored `CUDA.cached_memory()`, which was removed along with the old memory
  pools and is used by cuDNN's convolution algorithm search
  ([#1101](https://github.com/JuliaGPU/CUDA.jl/pull/1101),
  [#1103](https://github.com/JuliaGPU/CUDA.jl/pull/1103)).
- Restored the `JULIA_CUDA_MEMORY_POOL` environment variable: setting it to
  `none` disables the pool again instead of warning and being ignored
  ([#1103](https://github.com/JuliaGPU/CUDA.jl/pull/1103)).
- Removed a use of `Base.unsafe_length` that triggered a deprecation warning on
  Julia 1.8 ([#1103](https://github.com/JuliaGPU/CUDA.jl/pull/1103)).

### v3.4.2 (August 2021)

- Fixed `CUDA.cached_memory()` on CUDA 11.2, where the memory pool's reserved
  memory cannot be queried.
- The library handle cache now uses the documented `GC.enable_finalizers` API
  instead of the undocumented `GC.disable_finalizers`
  ([#1111](https://github.com/JuliaGPU/CUDA.jl/pull/1111)).
- The method table is no longer embedded in the generated AST, which makes it
  possible to use Revise.jl on CUDA.jl again
  ([#1112](https://github.com/JuliaGPU/CUDA.jl/pull/1112)).


## v3.3 (June 2021)

*Technically breaking changes*:

- Scalar indexing of `CuArray`s is now only allowed in interactive sessions,
  such as the REPL. In scripts and applications it is disallowed by default, and
  there is no global switch to re-enable it: use `CUDA.@allowscalar` or
  `CUDA.allowscalar() do ... end` to mark the expressions that need it
  ([#964](https://github.com/JuliaGPU/CUDA.jl/pull/964)).
- The device-side random number generator is now seeded at run time from the GPU
  clock instead of by the host at compile time, so repeatedly launching the same
  kernel no longer yields the same sequence of numbers
([#932](https://github.com/JuliaGPU/CUDA.jl/pull/932)). *New features*:

- `CuArray` supports isbits union element types, e.g. `CuArray([1, nothing,
  3])`. Such arrays can be passed to kernels and used with the usual array
  operations, which makes it possible to represent missing values on the GPU
  ([#941](https://github.com/JuliaGPU/CUDA.jl/pull/941)).
- Kernels are now compiled with location information, and optionally debug
  information. The level is taken from Julia's `-g` flag: the default `-g1` only
  emits line-number information, which profilers use to correlate instructions
  to source code; `-g2` additionally emits DWARF debug information and compiles
  in debug mode, for use with `cuda-gdb`; `-g0` disables both. Emission can also
  be turned off with the `JULIA_CUDA_DEBUG_INFO` environment variable
  ([#891](https://github.com/JuliaGPU/CUDA.jl/pull/891)).
- Toolkit selection now follows CUDA's Enhanced Compatibility rules, so a CUDA
  11.x toolkit can be used with any driver that supports CUDA 11.0 or later,
  instead of requiring a driver for that exact minor release
  ([#936](https://github.com/JuliaGPU/CUDA.jl/pull/936)).
- Support for CUDA 11.3 Update 1
  ([#945](https://github.com/JuliaGPU/CUDA.jl/pull/945)).
- Compatibility with Kepler GPUs (compute capability 3.5) has been reinstated. A
  warning is now shown when the driver only supports CUDA below 11.2
  ([#923](https://github.com/JuliaGPU/CUDA.jl/pull/923)).
- CUBLAS and CUSPARSE now support `Float16` and `ComplexF16`: `dot`, `norm`,
  `axpy!`, `axpby!` and `scal!` on the dense side, and sparse matrix-vector and
  matrix-matrix products and conversions on the sparse side
  ([#904](https://github.com/JuliaGPU/CUDA.jl/pull/904)).
- `CuDevice(ptr)` and `CuContext(ptr)` identify the device and context a pointer
  was allocated in ([#935](https://github.com/JuliaGPU/CUDA.jl/pull/935)).
- `devices()` now has a `show` method that lists the available GPUs
  ([#915](https://github.com/JuliaGPU/CUDA.jl/pull/915)).
- Exceptions thrown from outlined Base functions (bounds errors, inexact
  conversions, domain errors, integer overflow) now print a message from the
  kernel instead of failing to compile
  ([#874](https://github.com/JuliaGPU/CUDA.jl/pull/874)).
- CUBLAS and cuDNN log messages are forwarded to Julia's logging system, at a
  severity matching the message
([#953](https://github.com/JuliaGPU/CUDA.jl/pull/953),
[#966](https://github.com/JuliaGPU/CUDA.jl/pull/966)). *Minor changes*:

- Kernels are now compiled with the CUDA toolkit's `ptxas` instead of the
  driver's embedded JIT compiler. This is what makes Enhanced Compatibility
  possible; it does mean a local toolkit installation needs to provide `ptxas`
  ([#892](https://github.com/JuliaGPU/CUDA.jl/pull/892)).
- `using CUDA` is faster: library discovery and device context creation are now
  lazy, and less work is done during module initialization
  ([#910](https://github.com/JuliaGPU/CUDA.jl/pull/910)).
- Fixes for Julia 1.7 ([#949](https://github.com/JuliaGPU/CUDA.jl/pull/949)).
*Bug fixes*:

- `CUBLAS.iamax` and `CUBLAS.iamin` no longer throw a `MethodError` when called
  on an array ([#913](https://github.com/JuliaGPU/CUDA.jl/pull/913)).
- Fixed two-step `mapreduce` when the output is a wrapped array such as a view
  ([#925](https://github.com/JuliaGPU/CUDA.jl/pull/925)).
- Fixed sorting with fewer threads than the block size
  ([#959](https://github.com/JuliaGPU/CUDA.jl/pull/959)).
- `CUDA.@profile` now stops the profiler even when the profiled expression
  throws ([#914](https://github.com/JuliaGPU/CUDA.jl/pull/914)).
- CUFFT plans now update their work area correctly, and eagerly free the old
  one, avoiding out-of-memory errors when repeatedly creating plans
  ([#921](https://github.com/JuliaGPU/CUDA.jl/pull/921),
  [#927](https://github.com/JuliaGPU/CUDA.jl/pull/927)).
- `NVML.compute_processes` no longer races with processes that start while the
  list is being queried ([#933](https://github.com/JuliaGPU/CUDA.jl/pull/933)).

### v3.3.1 (June 2021)

- `CUDA.@atomic` now converts the value to the element type of the array
  ([#990](https://github.com/JuliaGPU/CUDA.jl/pull/990)).
- `CUDA.reclaim()` is now implemented for the stream-ordered memory allocator
  used with CUDA 11.2 and later
  ([#983](https://github.com/JuliaGPU/CUDA.jl/pull/983)).
- Reduced kernel launch overhead and the cost of retrieving library handles
  ([#986](https://github.com/JuliaGPU/CUDA.jl/pull/986),
  [#993](https://github.com/JuliaGPU/CUDA.jl/pull/993),
  [#996](https://github.com/JuliaGPU/CUDA.jl/pull/996),
  [#997](https://github.com/JuliaGPU/CUDA.jl/pull/997),
  [#1000](https://github.com/JuliaGPU/CUDA.jl/pull/1000),
  [#1002](https://github.com/JuliaGPU/CUDA.jl/pull/1002)).
- Library log messages are only emitted when Julia is started with `-g2` or when
  `JULIA_DEBUG` is set, and installing the log callbacks no longer initializes
  the libraries ([#987](https://github.com/JuliaGPU/CUDA.jl/pull/987),
  [#992](https://github.com/JuliaGPU/CUDA.jl/pull/992)).
- Fixed disambiguation of a local CUDA 11.1 installation using CUSOLVER
  ([#972](https://github.com/JuliaGPU/CUDA.jl/pull/972)).

### v3.3.2 (July 2021)

- Added support for CUDA 11.4, though it is not selected by default yet
  ([#1024](https://github.com/JuliaGPU/CUDA.jl/pull/1024)).
- Fixed discovery and loading of toolkit artifacts, including empty artifact
  directories and the load order of CUBLAS and CUTENSOR
  ([#1006](https://github.com/JuliaGPU/CUDA.jl/pull/1006),
  [#1007](https://github.com/JuliaGPU/CUDA.jl/pull/1007),
  [#1010](https://github.com/JuliaGPU/CUDA.jl/pull/1010)).
- Fixed the memory pool on platforms without atomic `Float64` support, such as
  PowerPC ([#1009](https://github.com/JuliaGPU/CUDA.jl/pull/1009)).
- Fixed a name clash between `ExprTools` and `LLVM` that warned on import
  ([#1026](https://github.com/JuliaGPU/CUDA.jl/pull/1026)).
- Fixes for Julia 1.7 ([#1013](https://github.com/JuliaGPU/CUDA.jl/pull/1013)).

### v3.3.3 (July 2021)

- Compatibility with LLVM.jl 4.0
  ([#1022](https://github.com/JuliaGPU/CUDA.jl/pull/1022)).
- `CuMemoryPool` now takes `alloc_type` and `handle_type` keyword arguments,
  e.g. to create a pool that exports POSIX file descriptor handles
  ([#1036](https://github.com/JuliaGPU/CUDA.jl/pull/1036)).

### v3.3.4 (July 2021)

- Fixed `cholesky` dispatch on Julia 1.8
  ([#1049](https://github.com/JuliaGPU/CUDA.jl/pull/1049)).
- Adapted to the deprecation of `Base.unsafe_length` on Julia 1.8
  ([#1045](https://github.com/JuliaGPU/CUDA.jl/pull/1045)).

### v3.3.5 (August 2021)

- Support for CUDA 11.4.1 and cuDNN 8.2.2
  ([#1084](https://github.com/JuliaGPU/CUDA.jl/pull/1084)).
- `rem` now uses the correct device intrinsic
  ([#1041](https://github.com/JuliaGPU/CUDA.jl/pull/1041)).
- Fixed illegal memory accesses when evaluating complex exponentials with a
  large imaginary part, by using libdevice's `sincos`
  ([#1086](https://github.com/JuliaGPU/CUDA.jl/pull/1086)).
- The library handle cache is now thread-safe, fixing errors when calling into
  CUBLAS from multiple threads
  ([#1074](https://github.com/JuliaGPU/CUDA.jl/pull/1074)).
- CUSPARSE: fixed a `DivideError` when multiplying a sparse matrix with an empty
  dense matrix, and fixed conversion of a sparse matrix to a dense one
  ([#1073](https://github.com/JuliaGPU/CUDA.jl/pull/1073),
  [#1083](https://github.com/JuliaGPU/CUDA.jl/pull/1083)).
- Added `CuArray` and `CuMatrix` constructors for the `Q` factor of a QR
  factorization ([#1063](https://github.com/JuliaGPU/CUDA.jl/pull/1063)).
- Fixed detection of a local CUDA 11.3 Update 1 installation
  ([#1089](https://github.com/JuliaGPU/CUDA.jl/pull/1089)).

### v3.3.6 (August 2021)

- CUSPARSE: fixed sparse matrix-matrix multiplication with a transposed operand
  ([#1096](https://github.com/JuliaGPU/CUDA.jl/pull/1096)).


## v3.2 (May 2021)

*New features*:

- Added wrappers for the CUDA graph API. `capture`, `instantiate`, `launch` and
  `update` give full control over recording and replaying a graph of GPU
  operations, while the `@captured` macro records a block of code and caches the
  resulting executable graph, updating it when the operations change
  ([#877](https://github.com/JuliaGPU/CUDA.jl/pull/877),
  [#65](https://github.com/JuliaGPU/CUDA.jl/pull/65)).
- `@cuprintln` and `@cushow` can print tuples
  ([#880](https://github.com/JuliaGPU/CUDA.jl/pull/880)).
- Integer intrinsics (`brev`, `clz`, `ffs`, `popc`, `byte_perm`) accept unsigned
  inputs, and their results now carry range information so that conversions to
  `Int` do not need a check. This also fixes an `InexactError` when using the
  result of `activemask()`
  ([#881](https://github.com/JuliaGPU/CUDA.jl/pull/881)).

*Minor changes*:

- The device-side RNG now uses Philox2x32, from Random123.jl. It passes
  SmallCrush, uses much less shared memory, and no longer requires `rand()` to
  be called uniformly by all threads in a block. As a result, `CUDA.RNG` is no
  longer experimental ([#879](https://github.com/JuliaGPU/CUDA.jl/pull/879),
  [#882](https://github.com/JuliaGPU/CUDA.jl/pull/882),
  [#803](https://github.com/JuliaGPU/CUDA.jl/pull/803)).
- Artifacts now come from unified JLLs, CUDA 11.3 is used by default, and cuDNN
  was updated to 8.2 and cuTENSOR to 1.3. cuDNN older than 8.0, or cuTENSOR
  other than 1.3, now warns
  ([#889](https://github.com/JuliaGPU/CUDA.jl/pull/889)).
- The cuDNN and cuTENSOR artifacts are only downloaded when those libraries are
  first used, instead of on installation
  ([#890](https://github.com/JuliaGPU/CUDA.jl/pull/890)).
- `mean`, `var` and `std` moved to GPUArrays.jl
  ([#888](https://github.com/JuliaGPU/CUDA.jl/pull/888)).
- `CUDA.@profile` uses the profiler API to start capture, which results in
  sharper traces ([#878](https://github.com/JuliaGPU/CUDA.jl/pull/878)).

*Bug fixes*:

- `view` treats `CartesianIndex` arguments as scalar indices, so that more views
  return a contiguous `CuArray` instead of a `SubArray`
  ([#886](https://github.com/JuliaGPU/CUDA.jl/pull/886)).
- Unparseable values of the `CI` and `JULIA_CUDA_USE_BINARYBUILDER` environment
  variables no longer make initialization fail
  ([#887](https://github.com/JuliaGPU/CUDA.jl/pull/887)).

### v3.2.1 (May 2021)

- Reworked synchronization: `synchronize` and `CUDA.@sync` now first spin, then
  yield, and finally block on an event. This keeps latency low for short
  operations while no longer burning CPU time waiting for long-running kernels.
  Note that `blocking=false` now means "spin", where it previously meant "yield"
  ([#896](https://github.com/JuliaGPU/CUDA.jl/pull/896),
  [#838](https://github.com/JuliaGPU/CUDA.jl/pull/838),
  [#839](https://github.com/JuliaGPU/CUDA.jl/pull/839),
  [#893](https://github.com/JuliaGPU/CUDA.jl/pull/893)).
- CUFFT plans manage their own work area, fixing memory leaks with repeated use
  of `fft` ([#902](https://github.com/JuliaGPU/CUDA.jl/pull/902),
  [#894](https://github.com/JuliaGPU/CUDA.jl/pull/894)).
- Local CUDA installations without cuSOLVERMg are usable again, instead of being
  rejected ([#900](https://github.com/JuliaGPU/CUDA.jl/pull/900)).
- Rebuilt the artifacts so that the correct cuDNN and cuTENSOR builds are
  selected for the CUDA toolkit in use, and improved the error message when an
  artifact is broken ([#901](https://github.com/JuliaGPU/CUDA.jl/pull/901),
  [#899](https://github.com/JuliaGPU/CUDA.jl/pull/899)).


## v3.1 (April 2021)

*New features*:

- Added support for CUDA 11.3
  ([#858](https://github.com/JuliaGPU/CUDA.jl/pull/858)).
- `cov` and `cor` from `Statistics` now work on `CuMatrix`
  ([#509](https://github.com/JuliaGPU/CUDA.jl/pull/509)).
- Implemented `partialsort` and `partialsort!` for `CuArray`, covering the same
  methods as Base ([#93](https://github.com/JuliaGPU/CUDA.jl/pull/93),
  [#864](https://github.com/JuliaGPU/CUDA.jl/pull/864)).
- `@atomic` now supports `*=` and `/=`
  ([#842](https://github.com/JuliaGPU/CUDA.jl/pull/842)).
- Added half-precision device intrinsics: `fma`, `rem`, `rsqrt` and `expm1` for
  `Float16`, atomic addition of `Float16` values, and warp shuffles of `Float16`
values ([#871](https://github.com/JuliaGPU/CUDA.jl/pull/871)). *Minor changes*:

- Reductions now use half as much shared memory, and no longer round the number
  of threads down to a power of two
  ([#843](https://github.com/JuliaGPU/CUDA.jl/pull/843),
  [#853](https://github.com/JuliaGPU/CUDA.jl/pull/853)).
- Library handles are kept in a common cache that bounds how many are retained,
  lowering memory usage ([#868](https://github.com/JuliaGPU/CUDA.jl/pull/868)).
- `device_reset!` can be used again together with the stream-ordered allocator,
  except on CUDA 11.2 where the underlying driver bug lives
([#858](https://github.com/JuliaGPU/CUDA.jl/pull/858)). *Bug fixes*:

- Fixed sorting of arrays that need more blocks than fit in a single grid
  dimension, which failed with an invalid launch configuration
  ([#852](https://github.com/JuliaGPU/CUDA.jl/pull/852),
  [#854](https://github.com/JuliaGPU/CUDA.jl/pull/854)).
- Fixed nondeterministic incorrect results when sorting multidimensional arrays;
  median selection now uses a bitonic sort
  ([#845](https://github.com/JuliaGPU/CUDA.jl/pull/845)).
- cuFFT plans and cuRAND generators are no longer tied to the stream they were
  created on, fixing memory corruption and segfaults when using them from
  multiple tasks ([#859](https://github.com/JuliaGPU/CUDA.jl/pull/859),
  [#867](https://github.com/JuliaGPU/CUDA.jl/pull/867),
  [#869](https://github.com/JuliaGPU/CUDA.jl/pull/869)).
- Fixed cuDNN convolutions of small images, where the requested padding, stride
  and dilation were silently clamped to the input size
  ([#848](https://github.com/JuliaGPU/CUDA.jl/pull/848),
  [#873](https://github.com/JuliaGPU/CUDA.jl/pull/873)).
- `unsafe_wrap(CuArray, ptr, dims; own=true)` now identifies the kind of memory
  it wraps, so that unified memory is freed correctly
  ([#737](https://github.com/JuliaGPU/CUDA.jl/pull/737),
  [#857](https://github.com/JuliaGPU/CUDA.jl/pull/857)).
- Worked around an offset calculation bug in `cuMemcpy3DAsync` that broke
  `unsafe_copy3d!` when using the stream-ordered allocator
  ([#863](https://github.com/JuliaGPU/CUDA.jl/pull/863),
  [#872](https://github.com/JuliaGPU/CUDA.jl/pull/872)).
- Worked around a buggy NVML initialization that crashed CUDA.jl on WSL
  ([#860](https://github.com/JuliaGPU/CUDA.jl/pull/860),
  [#861](https://github.com/JuliaGPU/CUDA.jl/pull/861)).


## v3.0 (April 2021)

Execution is now task-local: every Julia task gets its own stream, library
handles and device selection, so independent operations can overlap on the GPU.
Because operations from different tasks are no longer implicitly ordered,
sharing data between tasks requires explicit synchronization, which is what
makes this release breaking.

*Breaking changes*:

- Each Julia task now enqueues work on its own private stream, and blocking
  operations have been replaced by ones that yield to the Julia scheduler. Data
  created in one task and used in another needs an explicit `synchronize()` in
  between ([#662](https://github.com/JuliaGPU/CUDA.jl/pull/662)).
- `synchronize()` now only waits for the task-local stream. Use the new
  `device_synchronize()` to wait for all work on the device
  ([#662](https://github.com/JuliaGPU/CUDA.jl/pull/662)).
- Device-specific math functions such as `CUDA.sin` no longer exist. Kernels
  call `Base.sin` and friends directly, dispatching to GPU-specific method
  overrides that CUDA.jl installs through Julia 1.6's `AbstractInterpreter`.
  This also fixes many cases where the CUDA version lacked methods the Base one
  had, such as `sincos` on complex numbers
  ([#750](https://github.com/JuliaGPU/CUDA.jl/pull/750),
  [#776](https://github.com/JuliaGPU/CUDA.jl/pull/776)).
- The `@cufunc` macro has been removed, as have the ForwardDiff-specific rules
  that were built on it ([#750](https://github.com/JuliaGPU/CUDA.jl/pull/750)).
- Calling a GPU-only intrinsic such as `CUDA.saturate` on the CPU now throws an
  error instead of crashing the process
  ([#750](https://github.com/JuliaGPU/CUDA.jl/pull/750)).
- The cuDNN wrappers have been rewritten. The new interface maps the cuDNN API
  onto Julia functions and descriptor objects (`cudnnPoolingForward`,
  `cudnnPoolingForward!`, `cudnnPoolingDescriptor`, ...) so that downstream
  packages can reach advanced cuDNN features without dropping to C calls
  ([#523](https://github.com/JuliaGPU/CUDA.jl/pull/523)).
- The NNlib layer on top of cuDNN (convolutions, batch normalization,
  activations, pooling) has been removed from CUDA.jl and now lives in
  NNlibCUDA.jl, a subpackage of NNlib.jl
  ([#753](https://github.com/JuliaGPU/CUDA.jl/pull/753)).

*New features*:

- CUDA 11.2 is now the preferred toolkit, and is used by default if the driver
  supports it ([#719](https://github.com/JuliaGPU/CUDA.jl/pull/719),
  [#768](https://github.com/JuliaGPU/CUDA.jl/pull/768)).
- On CUDA 11.2, memory is allocated with the stream-ordered allocator and
  CUDA.jl's own caching layer is disabled. The pool is now chosen at run time
  instead of at precompilation time, so `JULIA_CUDA_MEMORY_POOL=binned` selects
  the old allocator without a rebuild
  ([#679](https://github.com/JuliaGPU/CUDA.jl/pull/679),
  [#745](https://github.com/JuliaGPU/CUDA.jl/pull/745)).
- `rand()` can now be called from inside kernels, backed by a combined
  Tausworthe generator that shares 32 bytes of state across a warp. The same
  generator is available on the host as `CUDA.RNG`, though `rand!(::CuArray)`
  still uses CURAND by default
  ([#772](https://github.com/JuliaGPU/CUDA.jl/pull/772),
  [#788](https://github.com/JuliaGPU/CUDA.jl/pull/788)).
- Host memory can be page-locked with `Mem.pin`, which makes copies to and from
  the GPU asynchronous instead of blocking the whole process
  ([#760](https://github.com/JuliaGPU/CUDA.jl/pull/760)).
- New `stream()` and `stream!` functions to query and set the task-local stream
  ([#662](https://github.com/JuliaGPU/CUDA.jl/pull/662)).
- Wrappers for cuSolverMg, providing multi-GPU dense linear algebra:
  `CUSOLVER.mg_potrf!`, `mg_getrf!`, `mg_syevd!` and related routines
  ([#308](https://github.com/JuliaGPU/CUDA.jl/pull/308)).
- Warp-synchronous vote intrinsics `vote_all_sync`, `vote_any_sync`,
  `vote_uni_sync` and `vote_ballot_sync`, plus the non-synchronizing `vote_uni`
  ([#723](https://github.com/JuliaGPU/CUDA.jl/pull/723)).
- New device intrinsics `laneid()` and `active_mask()`
  ([#801](https://github.com/JuliaGPU/CUDA.jl/pull/801)).
- `reinterpret` is now implemented for `CuDeviceArray`, so it can be used on
  shared memory and other device-side arrays
  ([#755](https://github.com/JuliaGPU/CUDA.jl/pull/755)).
- `@cushow` accepts multiple values and can print `LLVMPtr`s
  ([#709](https://github.com/JuliaGPU/CUDA.jl/pull/709)).
- `CUBLAS.getri_strided_batched!`, a batched matrix inverse operating on a
  3-dimensional array instead of a vector of pointers
  ([#682](https://github.com/JuliaGPU/CUDA.jl/pull/682)).

*Minor changes*:

- cuDNN was upgraded to 8.1, and cuTENSOR artifacts are available for CUDA 11.2
  ([#701](https://github.com/JuliaGPU/CUDA.jl/pull/701)).
- A portion of GPU memory is now kept in reserve for allocations made by CUBLAS,
  cuDNN and other libraries, by splitting the memory limit into a soft and a
  hard limit. This avoids confusing errors that were really out-of-memory
  conditions ([#718](https://github.com/JuliaGPU/CUDA.jl/pull/718)).
- Each device gets its own memory pool
  ([#746](https://github.com/JuliaGPU/CUDA.jl/pull/746)).
- Quicksort is faster: batch partitioning now uses a cumulative sum instead of a
  merge sort, and small arrays are sorted with a bitonic sort instead of a
  bubble sort ([#762](https://github.com/JuliaGPU/CUDA.jl/pull/762)).
- Array construction and allocation are faster
  ([#792](https://github.com/JuliaGPU/CUDA.jl/pull/792)), as is bounds checking
  for `view` ([#804](https://github.com/JuliaGPU/CUDA.jl/pull/804)).
- Logical indexing with a `CuArray{Bool}` mask no longer falls back to scalar
  iteration ([#724](https://github.com/JuliaGPU/CUDA.jl/pull/724)).
- Freeing memory from a finalizer now uses the legacy stream so that it orders
  against work on other streams. Call `unsafe_free!` explicitly to release
  memory on the task-local stream without that synchronization
  ([#778](https://github.com/JuliaGPU/CUDA.jl/pull/778),
  [#782](https://github.com/JuliaGPU/CUDA.jl/pull/782),
  [#783](https://github.com/JuliaGPU/CUDA.jl/pull/783)).
- Detection of when a kernel needs the CUDA device runtime linked in has been
  fixed, which speeds up compilation of kernels that do not use dynamic
  parallelism ([#811](https://github.com/JuliaGPU/CUDA.jl/pull/811)).
- Linker errors now include the JIT error log
  ([#712](https://github.com/JuliaGPU/CUDA.jl/pull/712)).

*Bug fixes*:

- `mod` on the device no longer uses the libdevice implementation, which
  disagreed with `Base.mod` for negative numbers
  ([#805](https://github.com/JuliaGPU/CUDA.jl/pull/805)).
- Fixed the bounds check in `reverse` when passing a `dims` keyword argument
  ([#806](https://github.com/JuliaGPU/CUDA.jl/pull/806)).
- Memory pool operations are now performed in the right context, fixing frees in
  multi-GPU programs ([#732](https://github.com/JuliaGPU/CUDA.jl/pull/732)), as
  are various operations performed from finalizers
  ([#761](https://github.com/JuliaGPU/CUDA.jl/pull/761)).
- Fixed a race during multi-threaded initialization
  ([#687](https://github.com/JuliaGPU/CUDA.jl/pull/687)), and several CUPTI
  issues that surfaced when profiling multi-threaded code
  ([#693](https://github.com/JuliaGPU/CUDA.jl/pull/693)).
- Library handles are now tracked so that they cannot be garbage-collected while
  still in use ([#704](https://github.com/JuliaGPU/CUDA.jl/pull/704)).
- Fixed `CURAND.set_stream`
  ([#698](https://github.com/JuliaGPU/CUDA.jl/pull/698)).
- Fixed memory pinning ([#781](https://github.com/JuliaGPU/CUDA.jl/pull/781))
  and out-of-memory handling with the stream-ordered allocator
  ([#809](https://github.com/JuliaGPU/CUDA.jl/pull/809),
  [#818](https://github.com/JuliaGPU/CUDA.jl/pull/818)).
- CUDA.jl no longer errors out when it cannot parse the version of a local CUDA
  installation ([#740](https://github.com/JuliaGPU/CUDA.jl/pull/740)).
- `cublasLt` is now loaded eagerly, so a system copy cannot get picked up
  instead ([#729](https://github.com/JuliaGPU/CUDA.jl/pull/729)).

### v3.0.1 (April 2021)

- Fixed `sort!` overwriting values in the target array
  ([#823](https://github.com/JuliaGPU/CUDA.jl/pull/823)).

### v3.0.2 (April 2021)

- Expressions entered at the REPL are now synchronized before returning, because
  displaying the value happens on a different task and would otherwise observe
  unsynchronized data ([#837](https://github.com/JuliaGPU/CUDA.jl/pull/837)).

### v3.0.3 (April 2021)

- Only synchronize REPL expressions when CUDA is configured, fixing an
  initialization error when CUDA.jl is loaded at the REPL on a system without a
  working CUDA setup ([#840](https://github.com/JuliaGPU/CUDA.jl/pull/840)).


## v2.6 (January 2021)

*New features*:

- Added GPU sorting: `sort` and `sort!` on `CuArray`s, implemented as a
  quicksort that uses dynamic parallelism, supporting the `lt`, `by`, `rev` and
  `dims` keyword arguments
  ([#431](https://github.com/JuliaGPU/CUDA.jl/pull/431)).
- `@cuda` now converts values captured by a closure, so a kernel can be a
  closure over `CuArray`s and other objects that need conversion
  ([#625](https://github.com/JuliaGPU/CUDA.jl/pull/625)).
- `view`, `reshape` and `reinterpret` now work on unmanaged arrays, i.e. arrays
  created with `unsafe_wrap`
  ([#663](https://github.com/JuliaGPU/CUDA.jl/pull/663)).
- Regenerated the library wrappers against CUDA 11.1 and 11.2, exposing
  newly-added APIs of the driver, cuSOLVER, cuSPARSE, CUPTI and NVML
([#638](https://github.com/JuliaGPU/CUDA.jl/pull/638)). *Minor changes*:

- `fill!` now uses an asynchronous memset instead of blocking
  ([#669](https://github.com/JuliaGPU/CUDA.jl/pull/669)).
- Loading a kernel module now reclaims memory and retries when the GPU is out of
  memory, instead of failing outright
  ([#665](https://github.com/JuliaGPU/CUDA.jl/pull/665)).
- `code_sass` reuses the regular compilation pipeline, so it now also works for
  kernels that need the CUDA device runtime
  ([#654](https://github.com/JuliaGPU/CUDA.jl/pull/654)).
- Relaxed compatibility bounds to allow Adapt 3.1, AbstractFFTs 1.0 and Reexport
  1.0 ([#640](https://github.com/JuliaGPU/CUDA.jl/pull/640),
[#646](https://github.com/JuliaGPU/CUDA.jl/pull/646)). *Bug fixes*:

- Worked around a `ptxas` bug that produced wrong results for kernels combining
  shared memory with divergent early exits, by disabling early kernel exits on
  pre-Volta devices and with drivers older than 460
  ([#656](https://github.com/JuliaGPU/CUDA.jl/pull/656)).
- NVML is now initialized using `nvmlInitWithFlags`, fixing initialization
  failures on Windows with driver 460 and later
  ([#641](https://github.com/JuliaGPU/CUDA.jl/pull/641)).
- Fixed version checks that compared against Julia's `VERSION` instead of the
  CUDA or cuBLAS version, which selected the wrong cuBLAS math mode and the
  wrong primary-context APIs
  ([#671](https://github.com/JuliaGPU/CUDA.jl/pull/671)).
- cuBLAS log messages are now written out directly, avoiding a crash when
  cuBLASXt invokes the logging callback from a foreign thread
  ([#649](https://github.com/JuliaGPU/CUDA.jl/pull/649)).
- `CUDA.launch(::Function)` no longer leaks an async condition on every call
  ([#650](https://github.com/JuliaGPU/CUDA.jl/pull/650)).
- Updated the Windows artifacts for compatibility with Julia 1.6
  ([#639](https://github.com/JuliaGPU/CUDA.jl/pull/639)).

### v2.6.1 (January 2021)

- `@cuda` now keeps the kernel function alive for the duration of the launch, so
  a closure cannot be collected while its kernel is still running
  ([#674](https://github.com/JuliaGPU/CUDA.jl/pull/674)).
- Kernels that cannot throw an exception no longer register an exception flag
  when loaded ([#675](https://github.com/JuliaGPU/CUDA.jl/pull/675)).
- Version and device queries (`CUDA.version()`, `CUDA.release()`, `ndevices()`,
  `CUDNN.version()`) are memoized instead of calling into the driver every time
  ([#676](https://github.com/JuliaGPU/CUDA.jl/pull/676),
  [#677](https://github.com/JuliaGPU/CUDA.jl/pull/677)).
- `JIT_OPTIMIZATION_LEVEL` can now be passed as a JIT option when loading or
  linking a module.
- Added `show` methods for `CuStream` and `CuLink`
  ([#677](https://github.com/JuliaGPU/CUDA.jl/pull/677)).

### v2.6.2 (March 2021)

- Maintenance release from the `release-2.6` branch. Added support for CUDA 11.2
  Update 1 and Update 2, and refreshed the bundled artifacts to CUDA 11.2.2,
  cuDNN 8.1 and cuTENSOR 1.2.2
  ([#771](https://github.com/JuliaGPU/CUDA.jl/pull/771)).
- Fixed the linkage of the exception-flag global, which broke kernels that
  require device-code linking on Julia 1.6
  ([#694](https://github.com/JuliaGPU/CUDA.jl/pull/694)).
- Failing to parse the version of a local CUDA installation now reports an error
  and falls back to the artifacts, instead of aborting initialization
  ([#740](https://github.com/JuliaGPU/CUDA.jl/pull/740)).
- Recognize the `nvdisasm` of an additional CUDA 11.0 build found in some Docker
  containers ([#680](https://github.com/JuliaGPU/CUDA.jl/pull/680)).

### v2.6.3 (April 2021)

- Fixed detection of whether a kernel needs the CUDA device runtime: the check
  compared LLVM function objects against names and never matched, so kernels
  using `@cuprintf`, dynamic allocation or assertions were needlessly linked
  with `cudadevrt` and compiled as relocatable device code
  ([#811](https://github.com/JuliaGPU/CUDA.jl/pull/811),
  [#813](https://github.com/JuliaGPU/CUDA.jl/pull/813)).


## v2.5 (January 2021)

v2.5 is the Julia 1.6 continuation of the v2.4 series: the two releases carry
the same features, but v2.5 requires Julia 1.6 while v2.4 remains the release
for Julia 1.5.

*Technically breaking changes*:

- CUDA.jl now requires Julia 1.6. Users on Julia 1.5 stay on the v2.4 series,
  which the package manager selects automatically
  ([#566](https://github.com/JuliaGPU/CUDA.jl/pull/566)).
- The `config` callback keyword argument to `@cuda` has been removed, and the
  equivalent argument to `cudacall` is deprecated. Use `@cuda launch=false`
  instead ([#569](https://github.com/JuliaGPU/CUDA.jl/pull/569)).

*New features*:

- `@cuda launch=false` returns a compiled but unlaunched kernel object, which
  can be introspected (e.g. with `CUDA.registers`) before launching it by
  calling it with the kernel arguments. This replaces the manual
  `cudaconvert`/`cufunction` dance and the `config` callback
  ([#569](https://github.com/JuliaGPU/CUDA.jl/pull/569)).
- Support for CUDA 11.2, both for local toolkits and as artifacts. Because cuDNN
  and cuTENSOR are not available for CUDA 11.2 yet, it is not selected
  automatically; set `JULIA_CUDA_VERSION=11.2` to opt in
  ([#607](https://github.com/JuliaGPU/CUDA.jl/pull/607),
  [#622](https://github.com/JuliaGPU/CUDA.jl/pull/622),
  [#624](https://github.com/JuliaGPU/CUDA.jl/pull/624)).
- Wrapped `cudaGL.h`, adding the OpenGL interop APIs such as
  `cuGraphicsGLRegisterBuffer`
  ([#612](https://github.com/JuliaGPU/CUDA.jl/pull/612),
  [#621](https://github.com/JuliaGPU/CUDA.jl/pull/621)).
- `LinearAlgebra.lmul!`, `rmul!`, `mul!`, `ldiv!` and `rdiv!` are now defined
  for all triangular wrapper types over dense and sparse CUDA matrices,
  including their transposed and adjoint forms, instead of only a handpicked set
  of `\` methods ([#575](https://github.com/JuliaGPU/CUDA.jl/pull/575)).
- CUSPARSE now covers all combinations of element types and transpose/adjoint
  wrappers in its `LinearAlgebra` integration, fixing missing methods like `A' *
  B` ([#535](https://github.com/JuliaGPU/CUDA.jl/pull/535)).
- `Statistics.varm` and `stdm` are implemented for `CuArray`, which makes
  `StatsBase.mean_and_std` and `ZScoreTransform` work on the GPU
  ([#583](https://github.com/JuliaGPU/CUDA.jl/pull/583)).
- Compatibility with Adapt 3.0.

*Minor changes*:

- `findmin` and `findmax` use a single-pass reduction kernel, which handles
  `NaN` elements correctly and is faster for most array shapes and sizes
  ([#320](https://github.com/JuliaGPU/CUDA.jl/pull/320),
  [#484](https://github.com/JuliaGPU/CUDA.jl/pull/484),
  [#576](https://github.com/JuliaGPU/CUDA.jl/pull/576)).
- The CUDA toolkit artifacts now ship `cuda-memcheck`, and the test runner
  gained a `--memcheck` option to run tests under it
  ([#571](https://github.com/JuliaGPU/CUDA.jl/pull/571),
  [#573](https://github.com/JuliaGPU/CUDA.jl/pull/573)).
- Artifacts are loaded through `LazyArtifacts` instead of `Pkg`, avoiding a
  deprecation warning on Julia 1.6
  ([#570](https://github.com/JuliaGPU/CUDA.jl/pull/570),
  [#574](https://github.com/JuliaGPU/CUDA.jl/pull/574)).
- The warning about a GPU with an unsupported compute capability is now only
  shown once per device.

*Bug fixes*:

- Broadcasting `angle` over a `CuArray` no longer falls back to a slow and
  GPU-incompatible `Base` implementation
  ([#618](https://github.com/JuliaGPU/CUDA.jl/pull/618)).
- `CUDA.used_memory`, and with it `CUDA.memory_status()`, reports actual usage
  again ([#579](https://github.com/JuliaGPU/CUDA.jl/pull/579),
  [#582](https://github.com/JuliaGPU/CUDA.jl/pull/582)).
- cuBLASXt calls now synchronize their input data first, since the API does not
  respect stream semantics. This fixes sporadic `xt_trsm` failures
  ([#124](https://github.com/JuliaGPU/CUDA.jl/pull/124),
  [#536](https://github.com/JuliaGPU/CUDA.jl/pull/536),
  [#577](https://github.com/JuliaGPU/CUDA.jl/pull/577)).
- Fixed data races in the shared-memory callback and in the `cfunction` lookup
  used by kernel launches, which could crash reductions called from multiple
  host threads ([#564](https://github.com/JuliaGPU/CUDA.jl/pull/564),
  [#588](https://github.com/JuliaGPU/CUDA.jl/pull/588),
  [#589](https://github.com/JuliaGPU/CUDA.jl/pull/589)).
- All errors raised during initialization are now caught and reported, instead
  of only those from `cuInit`
  ([#593](https://github.com/JuliaGPU/CUDA.jl/pull/593)).
- Fixed the error message shown when initialization fails
  ([#603](https://github.com/JuliaGPU/CUDA.jl/pull/603),
  [#604](https://github.com/JuliaGPU/CUDA.jl/pull/604)).


## v2.4 (January 2021)

CUDA.jl 2.4 and 2.5 are the same release built for two different Julia versions:
2.4 is restricted to Julia 1.5, while 2.5 requires Julia 1.6. This is the last
feature release for Julia 1.5; the later patch releases in this series are
backports from the release branch.

*Technically breaking changes*:

- The `config` keyword argument of `@cuda` and `cudacall`, used to compute a
  launch configuration from the compiled kernel, is deprecated in favour of
  `@cuda launch=false` ([#569](https://github.com/JuliaGPU/CUDA.jl/pull/569)).
- `CUDA.jl` now requires Julia 1.5 exactly (`julia = "~1.5"`). Julia 1.6 users
  need CUDA.jl 2.5 ([#623](https://github.com/JuliaGPU/CUDA.jl/pull/623)).

*New features*:

- `@cuda launch=false` compiles a kernel without launching it and returns the
  kernel object, which can be introspected (e.g. with `CUDA.registers`) and
  called later. Kernel objects now convert their arguments with `cudaconvert`
  when called, so no manual conversion is needed
  ([#569](https://github.com/JuliaGPU/CUDA.jl/pull/569)).
- Added support for CUDA 11.2. The 11.2 artifact is not selected automatically
  because cuDNN and cuTENSOR were not compatible with it yet; set
  `JULIA_CUDA_VERSION=11.2` to opt in
  ([#622](https://github.com/JuliaGPU/CUDA.jl/pull/622),
  [#624](https://github.com/JuliaGPU/CUDA.jl/pull/624)).
- `findmin` and `findmax` use a new single-pass reduction kernel. This fixes
  results for arrays containing `NaN`, and is generally faster
  ([#576](https://github.com/JuliaGPU/CUDA.jl/pull/576)).

*Minor changes*:

- `Adapt` 3.0 is now supported
  ([#623](https://github.com/JuliaGPU/CUDA.jl/pull/623)).
- `libgomp` is now provided by `CompilerSupportLibraries_jll` instead of being
  looked up on the system.
- The warning about a GPU with an unsupported compute capability is now only
  shown once per device.
- Removed the warning about using a CUDA toolkit newer than the driver, which is
  valid with CUDA 11's minor version compatibility
  ([#622](https://github.com/JuliaGPU/CUDA.jl/pull/622)).

*Bug fixes*:

- Fixed texture fetches, which failed to compile because the texture intrinsics
  were declared with the wrong return type
  ([#554](https://github.com/JuliaGPU/CUDA.jl/pull/554)).
- libdevice is no longer cached across compilations, fixing errors about missing
  `__nv_*` symbols such as `__nv_sqrt`.
- Fixed `CUDA.used_memory()`, which reported wrong values
  ([#582](https://github.com/JuliaGPU/CUDA.jl/pull/582)).
- Initialization failures from any stage are now caught and reported, instead of
  only those from driver initialization
  ([#593](https://github.com/JuliaGPU/CUDA.jl/pull/593),
  [#604](https://github.com/JuliaGPU/CUDA.jl/pull/604)).
- `launch_configuration` is now thread-safe when passing a `shmem` callback
  ([#564](https://github.com/JuliaGPU/CUDA.jl/pull/564),
  [#589](https://github.com/JuliaGPU/CUDA.jl/pull/589)).
- Fixed a method ambiguity between `reverse`/`reverse!` on a `CuVector` and
  `Base`.
- Fixed the `@cuassert` macro, which failed to compile.
- Broadcasting `angle` over a `CuArray` now uses the device intrinsic
  ([#618](https://github.com/JuliaGPU/CUDA.jl/pull/618)).
- Fixed the `DimensionMismatch` message thrown by `reshape`
  ([#562](https://github.com/JuliaGPU/CUDA.jl/pull/562)).

### v2.4.1 (January 2021)

- Allow `Reexport` 1.0 ([#640](https://github.com/JuliaGPU/CUDA.jl/pull/640)).
- Updated the bundled CUDA artifacts to newer builds of `CUDA_jll`, fixing use
  on Windows ([#639](https://github.com/JuliaGPU/CUDA.jl/pull/639)).
- NVML is now initialized with `nvmlInitWithFlags`, fixing a segfault on Windows
  ([#641](https://github.com/JuliaGPU/CUDA.jl/pull/641)).
- `cublasXt` log messages are written out directly instead of through the Julia
  runtime, because the logging callback can be invoked from a foreign thread
  ([#649](https://github.com/JuliaGPU/CUDA.jl/pull/649)).
- The async condition used to schedule host functions is now closed after use,
  fixing a resource leak ([#650](https://github.com/JuliaGPU/CUDA.jl/pull/650)).

### v2.4.2 (March 2021)

- Maintenance release from the 2.4 branch, tagged after CUDA.jl 2.5 and 2.6 had
  shipped. Backported support for local CUDA 11.2 Update 1 and Update 2
  installations, as well as an additional CUDA 11.0 build
  ([#680](https://github.com/JuliaGPU/CUDA.jl/pull/680),
  [#770](https://github.com/JuliaGPU/CUDA.jl/pull/770)).
- Toolkit discovery no longer errors out when the version of a local CUDA
  installation cannot be parsed or is not recognized
  ([#770](https://github.com/JuliaGPU/CUDA.jl/pull/770)).

### v2.4.3 (April 2021)

- Relaxed the `AbstractFFTs` compat bound to allow 1.0.


## v2.3 (November 2020)

*New features*:

- `CUBLAS.gemm_strided_batched` and `gemm_strided_batched!` now accept any
  strided array, including `PermutedDimsArray` and NNlib's batched adjoint and
  transpose wrappers, and allow one operand to have a batch size of 1 so it is
  reused across the batch
  ([#539](https://github.com/JuliaGPU/CUDA.jl/pull/539)).

*Minor changes*:

- The `NNlib` requirement was raised to 0.7.7, and batched matrix multiplication
  now hooks into NNlib's `_batched_gemm!` interface instead of defining
  `batched_mul!` methods directly
  ([#539](https://github.com/JuliaGPU/CUDA.jl/pull/539)).
- Device-to-device copies now use `CuDefaultStream()` instead of explicitly
  requesting `CuStreamPerThread()`; since CUDA.jl calls the per-thread API entry
  points, the two refer to the same stream. The documentation for
  `CuStreamLegacy` and `CuStreamPerThread` explains when you still need them
  ([#551](https://github.com/JuliaGPU/CUDA.jl/pull/551)).

*Bug fixes*:

- Fix the address calculation for `__ldg` loads, which offset the pointer by
  elements instead of bytes and caused misaligned-address errors when reading
  from a `Base.Experimental.Const` array of multi-byte elements
  ([#548](https://github.com/JuliaGPU/CUDA.jl/pull/548),
  [#549](https://github.com/JuliaGPU/CUDA.jl/pull/549)).
- Fix the bounds check in `getindex(::CuSparseMatrixCSR, ::Integer, ::Colon)`,
  which passed its indices in the wrong order
  ([#545](https://github.com/JuliaGPU/CUDA.jl/pull/545)).


## v2.2 (November 2020)

*New features*:

- The bundled CUDA artifacts were upgraded to CUDA 11.1 Update 1, and a
  locally-installed CUDA 11.1 toolkit is now recognized too
  ([#530](https://github.com/JuliaGPU/CUDA.jl/pull/530)).
- Added a `powerpc64le` artifact for CUDA 11.1
  ([#530](https://github.com/JuliaGPU/CUDA.jl/pull/530)).
- `sv2` and `sv2!` take a `unit_diag` keyword argument, and `\` on a
  `UnitUpperTriangular` or `UnitLowerTriangular` sparse matrix now uses it
([#540](https://github.com/JuliaGPU/CUDA.jl/pull/540)). *Bug fixes*:

- `svd!` on a `CuMatrix` no longer throws a `MethodError`
  ([#531](https://github.com/JuliaGPU/CUDA.jl/pull/531)).
- Sparse triangular solves no longer call `istril` to determine the fill mode,
  which caused scalar indexing
  ([#540](https://github.com/JuliaGPU/CUDA.jl/pull/540)).

### v2.2.1 (November 2020)

- Re-tagged v2.2.0 so that documentation would be built; no functional changes.


## v2.1 (October 2020)

The array wrapper changes from v2.0 have been partly reverted: using Base's
`ReshapedArray`, `ReinterpretArray` and `SubArray` regressed precompilation and
load times, so these wrappers are implemented as part of the `CuArray` type
again.

*Technically breaking changes*:

- `reshape`, `reinterpret` and contiguous `view`s of a `CuArray` return a
  `CuArray` again instead of the corresponding wrapper type from Base.
  `DenseCuArray` is consequently an alias for `CuArray`; use it for methods that
  need contiguous GPU memory, and `StridedCuArray` for methods that also accept
  non-contiguous views ([#498](https://github.com/JuliaGPU/CUDA.jl/pull/498)).
- The `filter_mode` keyword argument of `CuTexture` has been replaced by
  `interpolation`, which takes `NearestNeighbour()`, `LinearInterpolation()` or
  `CubicInterpolation()` ([#460](https://github.com/JuliaGPU/CUDA.jl/pull/460)).

*New features*:

- Textures support cubic interpolation, implemented in software on top of the
  hardware's linear interpolation. It is available for 1D and 2D textures with
  non-normalized coordinates
  ([#460](https://github.com/JuliaGPU/CUDA.jl/pull/460)).
- CUDA 11.1 is now selected automatically when the driver supports it, and cuDNN
  and cuTENSOR artifacts for CUDA 11.1 have been added
  ([#506](https://github.com/JuliaGPU/CUDA.jl/pull/506)).
- `inv` of a triangular `CuMatrix` is now implemented using CUBLAS
  ([#487](https://github.com/JuliaGPU/CUDA.jl/pull/487)).
- `cu` converts complex arrays to `ComplexF32`, matching how real arrays are
  converted to `Float32` ([#489](https://github.com/JuliaGPU/CUDA.jl/pull/489)).
- `CUDA.precompile_runtime()` builds the GPU runtime library ahead of time, for
  every compute capability supported by the available LLVM
  ([#465](https://github.com/JuliaGPU/CUDA.jl/pull/465)).

*Minor changes*:

- Bundled cuDNN 8.0.4 and cuTENSOR 1.2.1, and adapted to cuTENSOR's new
  compute-type enumeration
  ([#506](https://github.com/JuliaGPU/CUDA.jl/pull/506)).
- Removed the dependency on BinaryProvider
  ([#475](https://github.com/JuliaGPU/CUDA.jl/pull/475)).

*Bug fixes*:

- cuDNN operations now pass scaling parameters using the type cuDNN expects
  (`Float32` for `Float16` tensors), fixing `Float16` convolutions that returned
  only zeros ([#92](https://github.com/JuliaGPU/CUDA.jl/pull/92),
  [#454](https://github.com/JuliaGPU/CUDA.jl/pull/454)).
- Out-of-place broadcasts of the cuDNN-accelerated activation functions (`σ`,
  `relu`, `relu6`, `elu`, `tanh`, `leakyrelu`) no longer overwrite their input
  ([#515](https://github.com/JuliaGPU/CUDA.jl/pull/515)).
- `mapreduce` no longer assumes the reduction operator is commutative
  ([#484](https://github.com/JuliaGPU/CUDA.jl/pull/484),
  [#500](https://github.com/JuliaGPU/CUDA.jl/pull/500)).
- The CUSPARSE `mul!` methods now demand dense operands, so that non-contiguous
  views fall back to a generic implementation instead of producing wrong results
  ([#493](https://github.com/JuliaGPU/CUDA.jl/pull/493),
  [#495](https://github.com/JuliaGPU/CUDA.jl/pull/495)).
- On Linux, the driver library is loaded as `libcuda.so.1` when the unversioned
  `libcuda.so` symlink is absent, as is the case with driver-only installations
  ([#502](https://github.com/JuliaGPU/CUDA.jl/pull/502),
  [#503](https://github.com/JuliaGPU/CUDA.jl/pull/503)).
- Fixed discovery of versioned dynamic libraries on macOS, e.g.
  `libnvToolsExt.dylib.1` ([#482](https://github.com/JuliaGPU/CUDA.jl/pull/482),
  [#486](https://github.com/JuliaGPU/CUDA.jl/pull/486)).
- Fixed loading CUDA.jl on platforms for which no artifacts are available
  ([#473](https://github.com/JuliaGPU/CUDA.jl/pull/473)).
- `reshape` with `Colon()` dimensions works again on Julia versions before 1.6
  ([#511](https://github.com/JuliaGPU/CUDA.jl/pull/511),
  [#512](https://github.com/JuliaGPU/CUDA.jl/pull/512)).
- NVML is now found on Windows, where the installer does not add its directory
  to `PATH` ([#518](https://github.com/JuliaGPU/CUDA.jl/pull/518)).
- cuTENSOR works on Windows: `CuTensor` stores indices as `Char` instead of
  `Cwchar_t`, and modes are converted to `Cint` when passed to the library
  ([#422](https://github.com/JuliaGPU/CUDA.jl/pull/422),
  [#519](https://github.com/JuliaGPU/CUDA.jl/pull/519)).
- `@cuprintf` promotes unsigned `char` and `short` arguments to `Cint`, as C's
  default argument promotions require
  ([#508](https://github.com/JuliaGPU/CUDA.jl/pull/508)).


## v2.0 (October 2020)

A breaking release, centered on CUDA's simplified stream model, low- and
mixed-precision operations, and a rework of the array wrappers and the sparse
array support.

*Breaking changes*:

- CUDA.jl now requires Julia 1.5, a GPU with compute capability 5.0 (Maxwell) or
  higher, and a driver supporting CUDA 10.1 or newer.
- Switched to per-thread default stream semantics
  ([#395](https://github.com/JuliaGPU/CUDA.jl/pull/395)). Operations on the
  default stream no longer serialize with operations on explicitly-created
  streams, and every thread gets its own default stream. The legacy and
  per-thread streams are available as `CuStreamLegacy()` and
  `CuStreamPerThread()`.
- Tensor cores are now used by default. Use `CUDA.math_mode!(CUDA.PEDANTIC)` to
  get the old behavior ([#424](https://github.com/JuliaGPU/CUDA.jl/pull/424)).
- Views, reshapes and reinterpretations of a `CuArray` are now represented by
  the Base array wrappers instead of by a derived `CuArray`
  ([#437](https://github.com/JuliaGPU/CUDA.jl/pull/437),
  [#438](https://github.com/JuliaGPU/CUDA.jl/pull/438)). New type unions
  `DenseCuArray`, `StridedCuArray` and `AnyCuArray`, along with their `Vector`,
  `Matrix` and `VecOrMat` variants, are exported for use in method signatures.
- `DevicePtr` has been replaced by `Core.LLVMPtr`, affecting kernel code that
  works with raw pointers
  ([#199](https://github.com/JuliaGPU/CUDA.jl/pull/199)).
- The CUSPARSE `switch2csr`, `switch2csc` and `switch2bsr` functions have been
  removed in favor of `convert` and the sparse array constructors
  ([#409](https://github.com/JuliaGPU/CUDA.jl/pull/409)).

*New features*:

- Added support for CUDA 11.1, including compute capability 8.6 and PTX ISA 7.1
  ([#445](https://github.com/JuliaGPU/CUDA.jl/pull/445)). Since no compatible
  CUDNN or CUTENSOR is available yet, this version is not selected
  automatically; set `JULIA_CUDA_VERSION=11.1` to force it.
- CUBLAS `mul!` now supports `Float16` and `BFloat16` inputs, as well as
  mixed-precision multiplications, by using `cublasGemmEx`
  ([#417](https://github.com/JuliaGPU/CUDA.jl/pull/417)).
- Added a task-local math mode,
  `CUDA.math_mode!(CUDA.PEDANTIC|CUDA.DEFAULT_MATH|CUDA.FAST_MATH)`, which
  configures the CUDA libraries to trade accuracy for speed
  ([#424](https://github.com/JuliaGPU/CUDA.jl/pull/424)). The `FAST_MATH` mode
  takes a `precision` keyword to have CUBLAS down-cast 32-bit inputs.
- Added the `CuSparseMatrixCOO` sparse matrix type
  ([#421](https://github.com/JuliaGPU/CUDA.jl/pull/421)), and made conversions
  between all sparse formats available through `convert` and the constructors
  ([#451](https://github.com/JuliaGPU/CUDA.jl/pull/451)).
- Sparse arrays now implement the 5-argument `mul!`, and `cu` converts
  `SparseVector` and `SparseMatrixCSC` to their GPU counterparts
  ([#409](https://github.com/JuliaGPU/CUDA.jl/pull/409)).
- Added `LinearAlgebra.reflect!` and `rotate!`, wrapping the CUBLAS `rot!`
  family ([#427](https://github.com/JuliaGPU/CUDA.jl/pull/427)).
- CUDA libraries can now be called with strided inputs
  ([#435](https://github.com/JuliaGPU/CUDA.jl/pull/435)), and `mul!` accepts
  strided vectors and matrix views
  ([#450](https://github.com/JuliaGPU/CUDA.jl/pull/450)).
- Dynamic parallelism now supports kernels with more than five arguments
  ([#407](https://github.com/JuliaGPU/CUDA.jl/pull/407)).

*Minor changes*:

- CUDNN: the fastest convolution algorithm is now selected automatically,
  activations and softmax are faster, and convolution, bias addition and
  activation are fused where possible
  ([#321](https://github.com/JuliaGPU/CUDA.jl/pull/321)).
- Device-to-device `copyto!` is now asynchronous
  ([#428](https://github.com/JuliaGPU/CUDA.jl/pull/428)).
- CUBLAS is now given a workspace backed by the memory pool, fixing use with
  CUDA 11.0 Update 1 ([#443](https://github.com/JuliaGPU/CUDA.jl/pull/443)).
- CUFFT no longer leaks the temporary copies it makes for in-place transforms
  ([#428](https://github.com/JuliaGPU/CUDA.jl/pull/428)).
- Reduced package load and first-use latency
  ([#403](https://github.com/JuliaGPU/CUDA.jl/pull/403)).
- Initialization failures now print a more helpful message, pointing at
  `JULIA_DEBUG=CUDA` for details
  ([#425](https://github.com/JuliaGPU/CUDA.jl/pull/425)).

*Bug fixes*:

- Bounds checking a view with GPU index arrays no longer errors with a scalar
  indexing error ([#404](https://github.com/JuliaGPU/CUDA.jl/pull/404)).
- Fixed a performance regression in `reverse`, and fixed `reverse` on wrapped
  arrays ([#429](https://github.com/JuliaGPU/CUDA.jl/pull/429),
  [#439](https://github.com/JuliaGPU/CUDA.jl/pull/439)).
- Fixed finalization of copied arrays
  ([#444](https://github.com/JuliaGPU/CUDA.jl/pull/444)).
- Fixed several wrong pointer types in the CUDNN wrappers
  ([#408](https://github.com/JuliaGPU/CUDA.jl/pull/408)).
- CURAND now retries seed generation when it runs out of memory
  ([#436](https://github.com/JuliaGPU/CUDA.jl/pull/436)).

### v2.0.1 (October 2020)

- Added `CUDA.precompile_runtime()` to compile the GPU runtime library ahead of
  time, for use with read-only depots and system images
  ([#465](https://github.com/JuliaGPU/CUDA.jl/pull/465)).

### v2.0.2 (October 2020)

- Fixed initialization of global state when a `CuDevice` is passed to another
  process before any API call has been made there
  ([#471](https://github.com/JuliaGPU/CUDA.jl/pull/471)).
- Removed the custom `view` implementation, fixing `view(x, :)`
  ([#472](https://github.com/JuliaGPU/CUDA.jl/pull/472)).
- Fixed loading CUDA.jl on platforms for which no artifacts are available
  ([#473](https://github.com/JuliaGPU/CUDA.jl/pull/473)).
- Dropped the `BinaryProvider` dependency
  ([#475](https://github.com/JuliaGPU/CUDA.jl/pull/475)).


## v1.3 (August 2020)

This release makes it possible to use multiple GPUs from a single Julia process.
The memory pool and the handles of every wrapped library are now tracked per
device, and the active device is part of the task-local state, so tasks and
threads can each work with a different GPU.

*Technically breaking changes*:

- `device(ctx::CuContext)` has been replaced by `CuDevice(ctx)`, and the
  low-level driver query for the device bound to the calling thread is now
  `CuCurrentDevice()`. The exported `device()` returns the device of the current
  task instead ([#356](https://github.com/JuliaGPU/CUDA.jl/pull/356)).
- The `CUDA.atcontextswitch` hook has been replaced by `CUDA.atdeviceswitch` and
  `CUDA.atdevicereset`. Switch callbacks no longer take thread or task arguments
  ([#356](https://github.com/JuliaGPU/CUDA.jl/pull/356),
  [#253](https://github.com/JuliaGPU/CUDA.jl/pull/253)).
- `Mem.Buffer` has been renamed to `Mem.AbstractBuffer`
  ([#364](https://github.com/JuliaGPU/CUDA.jl/pull/364)).
- The memory pool selected with `JULIA_CUDA_MEMORY_POOL` is now baked in at
  precompilation time; changing it requires recompiling CUDA.jl. The `dummy`
  pool has been renamed to `none`
([#253](https://github.com/JuliaGPU/CUDA.jl/pull/253)). *New features*:

- Multiple GPUs can be used within one process. Select a device with `device!`,
  query it with `device()` or `deviceid()`, and release it with `device_reset!`.
  Memory allocations are bound to the device they were allocated on, so only use
  an array while its device is active
  ([#253](https://github.com/JuliaGPU/CUDA.jl/pull/253),
  [#356](https://github.com/JuliaGPU/CUDA.jl/pull/356)).
- GPU state is task-local: each task can bind itself to a different device with
  `device!`, and gets its own library handles, so libraries like CUBLAS and
  CUDNN can be configured independently per task
  ([#356](https://github.com/JuliaGPU/CUDA.jl/pull/356),
  [#253](https://github.com/JuliaGPU/CUDA.jl/pull/253)).
- `ndevices()` returns the number of available devices
  ([#356](https://github.com/JuliaGPU/CUDA.jl/pull/356)).
- Support for CUDA 11.0 Update 1. Toolkit libraries are now selected by full
  toolkit version rather than by release, since CUDA 11 versions its libraries
  independently ([#374](https://github.com/JuliaGPU/CUDA.jl/pull/374)).
- CUDNN has been upgraded to 8.0.2, which is now also used with CUDA 10.1 (it
  previously used CUDNN 7)
  ([#353](https://github.com/JuliaGPU/CUDA.jl/pull/353)).
- Support for Julia 1.3 has been reinstated
  ([#377](https://github.com/JuliaGPU/CUDA.jl/pull/377)).
- Debug logs from CUBLAS and CUDNN can be shown by starting Julia with
  `JULIA_DEBUG=CUBLAS` or `JULIA_DEBUG=CUDNN`
  ([#367](https://github.com/JuliaGPU/CUDA.jl/pull/367)).
- `rand` and `rand!` now work with `Bool` and `Int8` arrays, via GPUArrays 5.1.

*Minor changes*:

- `versioninfo()` numbers and iterates devices the way CUDA does, and queries
  device properties through NVML when available so it no longer needs to create
  a context per device ([#375](https://github.com/JuliaGPU/CUDA.jl/pull/375)).
- In interactive sessions, a background task reclaims unused pool memory after
  the pool has been idle for five minutes
  ([#253](https://github.com/JuliaGPU/CUDA.jl/pull/253)).
- Copies between CPU and GPU arrays with Cartesian indices are performed using
  linear slices, via GPUArrays 5.1.

*Bug fixes*:

- `sum!` and the other in-place reduction shorthands no longer accumulate into a
  nonzero destination, via GPUArrays 5.1
  ([#370](https://github.com/JuliaGPU/CUDA.jl/pull/370)).
- Fixed `show` of `Mem.ArrayBuffer`, which made displaying a `CuTextureArray`
  fail ([#363](https://github.com/JuliaGPU/CUDA.jl/pull/363),
  [#357](https://github.com/JuliaGPU/CUDA.jl/pull/357)).
- Failing to load CUTENSOR now reports an error instead of aborting
  initialization, which on Windows is usually caused by a missing Visual C++
  redistributable ([#355](https://github.com/JuliaGPU/CUDA.jl/pull/355)).

### v1.3.1 (August 2020)

- Wrapped `cusparseSpMV` from the generic cuSPARSE API, restoring sparse
  matrix-vector multiplication on CUDA 11. This also raised the minimum required
  CUDA version to 10.1, and was reverted in v1.3.2
  ([#351](https://github.com/JuliaGPU/CUDA.jl/pull/351)).
- Fixed an error in `versioninfo()` on systems without NVML
  ([#385](https://github.com/JuliaGPU/CUDA.jl/pull/385)).

### v1.3.2 (August 2020)

- Reverted the cuSPARSE generic API work from v1.3.1, restoring support for CUDA
  9.0, 9.2 and 10.0. The new cuSPARSE wrappers returned in CUDA.jl 2.0, which
  drops those toolkits ([#351](https://github.com/JuliaGPU/CUDA.jl/pull/351)).

### v1.3.3 (August 2020)

- Non-contiguous `view`s of a `CuArray` now move their index arrays to the GPU
  when the view is constructed, instead of uploading them again on every kernel
  launch that uses the view
  ([#388](https://github.com/JuliaGPU/CUDA.jl/pull/388),
  [#384](https://github.com/JuliaGPU/CUDA.jl/pull/384)).
- Allow DataStructures 0.18
  ([#389](https://github.com/JuliaGPU/CUDA.jl/pull/389)).


## v1.2 (July 2020)

This release adds support for the CUDA 11.0 toolkit. CUDA 11 removed a large
part of the legacy cuSPARSE API, so several sparse array operations had to be
dropped and stayed unavailable throughout the remainder of the 1.x series.

*Technically breaking changes*:

- Removed `CuSparseMatrixHYB` and the cuSPARSE operations that CUDA 11
  deprecated or removed, among them `mm!`, `sv`, `sm`, `geam`, `gemm`, `ic0!`,
  `ilu0!`, `gtsv!` and `doti!`
  ([#291](https://github.com/JuliaGPU/CUDA.jl/pull/291)).
- `adapt(CuArray, x)` now preserves the element type. Only `cu(x)` still
  converts arrays of floating-point numbers to `Float32`, using a dedicated
  adaptor ([#278](https://github.com/JuliaGPU/CUDA.jl/pull/278),
  [#281](https://github.com/JuliaGPU/CUDA.jl/pull/281)).

*New features*:

- Support for the CUDA 11.0 toolkit, with artifacts for the toolkit itself and
  for cuTENSOR ([#291](https://github.com/JuliaGPU/CUDA.jl/pull/291)).
- `CUDA.@sync` takes a `blocking` keyword argument. `CUDA.@sync blocking=false
  ex` spins instead of blocking, which lowers the synchronization overhead when
  timing very short operations
  ([#279](https://github.com/JuliaGPU/CUDA.jl/pull/279),
  [#280](https://github.com/JuliaGPU/CUDA.jl/pull/280)).
- The cuTENSOR wrappers (`contraction!`, `reduction!`, `permutation!`,
  `plan_contraction`) accept host `Array`s next to `CuArray`s, and `Mem.pin` was
  added to page-lock host memory
  ([#243](https://github.com/JuliaGPU/CUDA.jl/pull/243)).

*Minor changes*:

- CUDA.jl warns during initialization when it loads a cuDNN or cuTENSOR version
  it does not support ([#134](https://github.com/JuliaGPU/CUDA.jl/pull/134),
  [#284](https://github.com/JuliaGPU/CUDA.jl/pull/284)).

*Bug fixes*:

- Fixed an `LLVM error: Cannot cast between two non-generic address spaces`
  failure during kernel compilation
  ([#286](https://github.com/JuliaGPU/CUDA.jl/pull/286)).
- The libdevice bitcode file is no longer opened twice when it is loaded
  ([#294](https://github.com/JuliaGPU/CUDA.jl/pull/294)).

### v1.2.1 (July 2020)

- Fixed discovery of CUDA 11 and cuDNN 8 libraries, in particular on Windows
  ([#300](https://github.com/JuliaGPU/CUDA.jl/pull/300),
  [#301](https://github.com/JuliaGPU/CUDA.jl/pull/301),
  [#310](https://github.com/JuliaGPU/CUDA.jl/pull/310),
  [#324](https://github.com/JuliaGPU/CUDA.jl/pull/324),
  [#326](https://github.com/JuliaGPU/CUDA.jl/pull/326),
  [#328](https://github.com/JuliaGPU/CUDA.jl/pull/328),
  [#335](https://github.com/JuliaGPU/CUDA.jl/pull/335)).
- Added artifacts for cuDNN 8.0.1 and cuTENSOR 1.2, including Windows and
  ppc64le builds of cuTENSOR
  ([#325](https://github.com/JuliaGPU/CUDA.jl/pull/325)).
- Reduced kernel launch overhead, and improved the performance of broadcast and
  other element-wise operations by switching to grid-stride kernels with a block
  size that no longer only maximizes occupancy
  ([#298](https://github.com/JuliaGPU/CUDA.jl/pull/298),
  [#299](https://github.com/JuliaGPU/CUDA.jl/pull/299),
  [#302](https://github.com/JuliaGPU/CUDA.jl/pull/302),
  [#307](https://github.com/JuliaGPU/CUDA.jl/pull/307),
  [#312](https://github.com/JuliaGPU/CUDA.jl/pull/312),
  [#313](https://github.com/JuliaGPU/CUDA.jl/pull/313)).
- Optimized `mapreduce` ([#316](https://github.com/JuliaGPU/CUDA.jl/pull/316))
  and `fill!` ([#339](https://github.com/JuliaGPU/CUDA.jl/pull/339)).
- `CuArray` no longer carries a third type parameter for its parent array, and
  is spelled `CuArray{T,N}` again
  ([#295](https://github.com/JuliaGPU/CUDA.jl/pull/295)).
- `CUDA.rand` and `Random.rand!` fall back to the generic GPUArrays random
  number generator for element types that cuRAND does not support
  ([#327](https://github.com/JuliaGPU/CUDA.jl/pull/327)).
- Added `NVML.utilization_rates` to query compute and memory utilization.
  `NVML.compute_processes` now returns `missing` when per-process memory use is
  unavailable ([#314](https://github.com/JuliaGPU/CUDA.jl/pull/314),
  [#329](https://github.com/JuliaGPU/CUDA.jl/pull/329)).
- Fixed the alignment of kernel parameters passed with dynamic parallelism,
  which could reorder the fields of a struct
  ([#263](https://github.com/JuliaGPU/CUDA.jl/pull/263),
  [#338](https://github.com/JuliaGPU/CUDA.jl/pull/338)).
- Fixed the type instability of `Statistics.var` and `Statistics.std` when
  passing `dims` ([#336](https://github.com/JuliaGPU/CUDA.jl/pull/336),
  [#337](https://github.com/JuliaGPU/CUDA.jl/pull/337)).
- Fixed a possible hang when freeing memory from a finalizer while another
  thread held the memory pool lock
  ([#25](https://github.com/JuliaGPU/CUDA.jl/pull/25)).
- Fixed an invalid pointer-to-signed-integer conversion in the run-time `ccall`
  cache ([#317](https://github.com/JuliaGPU/CUDA.jl/pull/317)).


## v1.1 (July 2020)

*Technically breaking changes*:

- Julia 1.4 is now the minimum supported version; support for Julia 1.3 has been
  dropped ([#275](https://github.com/JuliaGPU/CUDA.jl/pull/275)).
- The WMMA intrinsics now require Julia 1.5, following the rename of
  `Core.AddrSpacePtr` to `Core.LLVMPtr`
  ([#258](https://github.com/JuliaGPU/CUDA.jl/pull/258)).
- The toolkit discovery helpers (`find_toolkit`, `find_cuda_library`,
  `find_cuda_binary`, `find_libdevice`, ...) inherited from CUDAapi.jl are no
  longer exported ([#274](https://github.com/JuliaGPU/CUDA.jl/pull/274)).

*New features*:

- Experimental support for the GPU's texture hardware, using the new `CuTexture`
  and `CuTextureArray` types (ported from CuTextures.jl by @cdsousa). Textures
  can be passed to kernels and indexed there, with hardware filtering,
  configurable address modes and optional normalized coordinates
  ([#209](https://github.com/JuliaGPU/CUDA.jl/pull/209)).
- New `CUDA.NVML` submodule wrapping the NVIDIA Management Library, for querying
  the system and its devices: `NVML.devices`, `NVML.name`, `NVML.brand`,
  `NVML.uuid`, `NVML.serial`, `NVML.power_usage`, `NVML.energy_consumption`,
  `NVML.memory_info`, `NVML.compute_capability`, `NVML.compute_processes`, and
  `NVML.driver_version` ([#248](https://github.com/JuliaGPU/CUDA.jl/pull/248),
  [#251](https://github.com/JuliaGPU/CUDA.jl/pull/251)).
- `CUDA.versioninfo()` reports on the CUDA toolkit and where it came from, the
  CUDA and NVIDIA driver versions, the versions of all wrapped libraries, the
  Julia/LLVM toolchain with its supported PTX ISAs and device capabilities, and
  the available devices with their free and total memory
  ([#245](https://github.com/JuliaGPU/CUDA.jl/pull/245)).
- `uuid(::CuDevice)` returns a device's UUID
  ([#248](https://github.com/JuliaGPU/CUDA.jl/pull/248)).
- Support for Julia 1.5, through compatibility with LLVM.jl 2.0 and
  GPUCompiler.jl 0.5 ([#258](https://github.com/JuliaGPU/CUDA.jl/pull/258),
  [#275](https://github.com/JuliaGPU/CUDA.jl/pull/275)).
- Locally-installed CUDA 11 toolkits are now discovered correctly, including
  their independently-versioned libraries. Set
  `JULIA_CUDA_USE_BINARYBUILDER=false` to use one; the artifacts still top out
  at CUDA 10.2 ([#274](https://github.com/JuliaGPU/CUDA.jl/pull/274)).

*Minor changes*:

- cuTENSOR artifacts have been upgraded to v1.1.0
  ([#269](https://github.com/JuliaGPU/CUDA.jl/pull/269)).
- cuDNN is now provided by Yggdrasil-built artifacts
  ([#272](https://github.com/JuliaGPU/CUDA.jl/pull/272)).
- Nsight Systems is detected more reliably when Julia runs under `nsys`, and its
  location can be set with the `JULIA_CUDA_NSYS` environment variable
  ([#234](https://github.com/JuliaGPU/CUDA.jl/pull/234)).

*Bug fixes*:

- Library handle creation (cuBLAS, cuDNN, ...) close to running out of memory
  failed with a misleading `NOT_INITIALIZED` or `ALLOC_FAILED` error; these
  calls now retry after reclaiming memory
  ([#268](https://github.com/JuliaGPU/CUDA.jl/pull/268)).
- `cudnnCreate` is retried on `CUDNN_STATUS_INTERNAL_ERROR` and
  `CUDNN_STATUS_NOT_INITIALIZED`
  ([#244](https://github.com/JuliaGPU/CUDA.jl/pull/244)).
- Fixed a spurious "Your LLVM does not support the NVPTX back-end" error when
  loading CUDA.jl in a local project environment
  ([#252](https://github.com/JuliaGPU/CUDA.jl/pull/252)).


## v1.0 (June 2020)

CUDA.jl 1.0 is the first stable release of the package that merges CUDAdrv.jl,
CUDAnative.jl, CuArrays.jl and CUDAapi.jl into one. From here on, `using CUDA`
is the single entry point for CUDA programming in Julia: the `CuArray` type, the
`@cuda` kernel compiler, the driver API wrappers and the wrappers for CUBLAS,
CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN and CUTENSOR all live in this package,
and the four predecessor packages are no longer developed. The package requires
Julia 1.3 or later.

*New features*:

- Added support for Julia 1.5
  ([#194](https://github.com/JuliaGPU/CUDA.jl/pull/194)).
- Added compatibility data for CUDA 11: the `sm_80` device capability and PTX
  ISA 7.0, as well as the toolkit versions that dropped support for `sm_30` and
  `sm_32` ([#221](https://github.com/JuliaGPU/CUDA.jl/pull/221)).
- `CUBLAS.gemmEx!` wraps `cublasGemmEx`, which can use tensor cores and
  mixed-precision inputs, and takes the algorithm to use as an `algo` keyword
  argument ([#196](https://github.com/JuliaGPU/CUDA.jl/pull/196)).
- `LinearAlgebra.LAPACK.potri!` now works on a `CuMatrix`, computing the inverse
  from a Cholesky factorization. Requires CUDA 10.1 or later
  ([#179](https://github.com/JuliaGPU/CUDA.jl/pull/179)).
- Added `CUSOLVER.potrfBatched!` and `CUSOLVER.potrsBatched!` for batched
  Cholesky factorizations and solves
  ([#192](https://github.com/JuliaGPU/CUDA.jl/pull/192),
  [#193](https://github.com/JuliaGPU/CUDA.jl/pull/193)).
- cuTENSOR contractions take a compute type that may differ from the element
  type of the output, for mixed-precision contractions
([#200](https://github.com/JuliaGPU/CUDA.jl/pull/200)). *Minor changes*:

- `mul!` dispatches more argument combinations to CUBLAS, including `Transpose`-
  and `Adjoint`-wrapped vectors, and no longer conflicts with the more generic
  methods from Base and GPUArrays.jl
  ([#213](https://github.com/JuliaGPU/CUDA.jl/pull/213),
  [#214](https://github.com/JuliaGPU/CUDA.jl/pull/214)).
- `Base.mightalias` is specialized for `CuArray`s, comparing the actual memory
  ranges of two views into the same parent array. Broadcasting into a view no
  longer copies the array when the operands provably do not overlap
  ([#211](https://github.com/JuliaGPU/CUDA.jl/pull/211)).
- `copy` of a `CuArray` allocates and performs a device-to-device copy instead
  of going through `similar` and `copyto!`.
- `CuArray{T}(xs::AbstractArray)` converts the array to `T` on the CPU before
  uploading it, instead of uploading and broadcasting the conversion on the GPU.
- When the `CI` environment variable is set and `JULIA_CUDA_USE_BINARYBUILDER`
  is not, a locally-installed CUDA toolkit is preferred over the artifacts
  ([#198](https://github.com/JuliaGPU/CUDA.jl/pull/198)).
- The documentation now covers the API of the merged package, including the
  kernel programming, compiler and driver APIs that used to be documented in
  CUDAnative.jl and CUDAdrv.jl
([#23](https://github.com/JuliaGPU/CUDA.jl/pull/23)). *Bug fixes*:

- `mul!` with a sparse matrix and a transposed dense matrix no longer misses the
  CUSPARSE method and falls back to a generic implementation that performs
  scalar indexing ([#77](https://github.com/JuliaGPU/CUDA.jl/pull/77),
  [#180](https://github.com/JuliaGPU/CUDA.jl/pull/180)).
- `CUSPARSE.gemm` on `CuSparseMatrixCSC` operands returned a wrong result, using
  the wrong transpose operations and index base. It now computes the product
  correctly, returning a `CuSparseMatrixCSR`
  ([#181](https://github.com/JuliaGPU/CUDA.jl/pull/181),
  [#185](https://github.com/JuliaGPU/CUDA.jl/pull/185)).
- `unsafe_copy3d!` did not scale the `x` component of `srcPos` and `dstPos` by
  the element size, reading from and writing to the wrong offsets
  ([#27](https://github.com/JuliaGPU/CUDA.jl/pull/27),
  [#197](https://github.com/JuliaGPU/CUDA.jl/pull/197)).
- `CUBLAS.gels_batched!` verified the wrong dimensions of its inputs, and
  `gels_batched` only made a shallow copy of the vector of matrices, so it
  overwrote the caller's arrays
  ([#191](https://github.com/JuliaGPU/CUDA.jl/pull/191)).
- `mean` on a `CuArray` works again on Julia 1.5, whose Statistics stdlib
  changed the internal method that CUDA.jl overrides
  ([#194](https://github.com/JuliaGPU/CUDA.jl/pull/194)).
- Broadcasting into an empty array no longer causes a stack overflow
  ([#82](https://github.com/JuliaGPU/CUDA.jl/pull/82)).
- The exception flag emitted into every kernel is no longer eliminated by the
  optimizer, which caused a stream of API failures when running under
  `cuda-memcheck` ([#18](https://github.com/JuliaGPU/CUDA.jl/pull/18)).
- The memory pool no longer protects its data structures with spin locks, which
  cannot yield to other tasks while contended
  ([#22](https://github.com/JuliaGPU/CUDA.jl/pull/22)).

### v1.0.1 (June 2020)

- Require LLVM.jl 1.5.2, which no longer mistakes unrelated libraries with
  `LLVM` in their name for the LLVM library Julia was built against, avoiding a
  `Multiple LLVM libraries loaded by Julia` error at load time.

### v1.0.2 (June 2020)

- `findfirst(vals::CuArray, xs::CuArray)` no longer extends `Base.findfirst`,
  whose meaning it did not match. It is now available as the internal
  `CUDA.findfirstval` ([#230](https://github.com/JuliaGPU/CUDA.jl/pull/230)).
- Allow NNlib 0.7 ([#232](https://github.com/JuliaGPU/CUDA.jl/pull/232)).
- Added a chapter to the documentation on array programming, and enabled
  doctests ([#227](https://github.com/JuliaGPU/CUDA.jl/pull/227)).


## v0.1 (May 2020)

The initial release, published as a call for testing. The sources of the four
predecessor packages were imported with their history
([#14](https://github.com/JuliaGPU/CUDA.jl/pull/14)), so this is not new
development: it is the point at which the Julia CUDA stack became one package
with one version number, exporting from a single `CUDA` module what used to be
spread over four namespaces. Existing code should keep working after switching
the imports over.

*New features*:

- Three interfaces in one package: the `CuArray` type for array programming,
  `@cuda` for writing kernels in Julia, and wrappers for the CUDA driver API.
- Wrappers for CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN and CUTENSOR,
  available as submodules and hooked into the corresponding Julia interfaces, so
  that e.g. `mul!`, `fft` and `rand!` on a `CuArray` dispatch to the vendor
  libraries. CUPTI and NVTX are wrapped for profiling.
- The CUDA toolkit is downloaded automatically using artifacts, covering CUDA
  9.0 up to 10.2 along with cuDNN 7.6.5 and cuTENSOR 1.0.1, with a fallback to a
  locally-installed toolkit. Set `JULIA_CUDA_USE_BINARYBUILDER=false` to always
  use a local installation.
- The package loads even when no GPU or CUDA installation is available:
  initialization is deferred until first use, and `CUDA.functional()` reports
  whether the GPU stack can be used. `JULIA_CUDA_SILENT` and
  `JULIA_CUDA_VERBOSE` control the initialization message.

*Minor changes*:

- The `CUARRAYS_MEMORY_LIMIT`, `CUARRAYS_MEMORY_POOL` and
  `CUARRAYS_MANAGED_POOL` environment variables have been renamed to
  `JULIA_CUDA_MEMORY_LIMIT`, `JULIA_CUDA_MEMORY_POOL` and
  `JULIA_CUDA_MEMORY_POOL_MANAGED`. The old names still work, but warn.
- Julia 1.3 or higher is required.
