# Native intrinsics

export
    # Indexing and dimensions
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize,

    # Memory management
    sync_threads,
    cuSharedMem



#
# Indexing and dimensions
#

for dim in (:x, :y, :z)
    # Thread index
    fname = Symbol("threadIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.tid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, Tuple{}) + Int32(1)
    end

    # Block size (#threads per block)
    fname = Symbol("blockDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ntid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, Tuple{})
    end

    # Block index
    fname = Symbol("blockIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, Tuple{}) + Int32(1)
    end

    # Grid size (#blocks per grid)
    fname = Symbol("gridDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.nctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, Tuple{})
    end
end

# Tuple accessors
threadIdx() = CuDim3(threadIdx_x(), threadIdx_y(), threadIdx_z())
blockDim() =  CuDim3(blockDim_x(),  blockDim_y(),  blockDim_z())
blockIdx() =  CuDim3(blockIdx_x(),  blockIdx_y(),  blockIdx_z())
gridDim() =   CuDim3(gridDim_x(),   gridDim_y(),   gridDim_z())

# Warpsize
warpsize() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
        ret i32 %1"""),
    Int32, Tuple{})


#
# Thread management
#

# Synchronization
sync_threads() = Base.llvmcall(
    ("""declare void @llvm.nvvm.barrier0() readnone nounwind""",
     """call void @llvm.nvvm.barrier0()
        ret void"""),
    Void, Tuple{})


#
# Shared memory
#

# FIXME: this is broken, cannot declare global variables in `llvmcall`
# TODO: this is nasty
#       - box-like semantics?
#       - Ptr{AS}?
# TODO: this is inefficient, converting to global pointers (does propagation suffice?)
# TODO: static shared memory

cuSharedMem(::Type{Int64}) = Base.llvmcall(
    ("""@shmem_i64 = external addrspace(3) global [0 x i64]""",
     """%1 = getelementptr inbounds [0 x i64], [0 x i64] addrspace(3)* @shmem_i64, i64 0, i64 0
        %2 = addrspacecast i64 addrspace(3)* %1 to i64 addrspace(0)*
        ret i64* %2"""),
    Ptr{Int64}, Tuple{})

cuSharedMem(::Type{Float32}) = Base.llvmcall(
    ("""@shmem_f32 = external addrspace(3) global [0 x float]""",
     """%1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @shmem_f32, i64 0, i64 0
        %2 = addrspacecast float addrspace(3)* %1 to float addrspace(0)*
        ret float* %2"""),
    Ptr{Float32}, Tuple{})

cuSharedMem(::Type{Float64}) = Base.llvmcall(
    ("""@shmem_f64 = external addrspace(3) global [0 x double]""",
     """%1 = getelementptr inbounds [0 x double], [0 x double] addrspace(3)* @shmem_f64, i64 0, i64 0
        %2 = addrspacecast double addrspace(3)* %1 to double addrspace(0)*
        ret double* %2"""),
    Ptr{Float64}, Tuple{})


#
# Math
#

# Trigonometric
sin(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_sinf(float)""",
     """%2 = call float @__nv_sinf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
sin(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_sin(double)""",
     """%2 = call double @__nv_sin(double %0)
        ret double %2"""),
    Float64, Tuple{Float64} , x)
cos(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_cosf(float)""",
     """%2 = call float @__nv_cosf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
cos(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_cos(double)""",
     """%2 = call double @__nv_cos(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)

# Rounding
floor(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_floorf(float)""",
     """%2 = call float @__nv_floorf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
floor(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_floor(double)""",
     """%2 = call double @__nv_floor(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)
ceil(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_ceilf(float)""",
     """%2 = call float @__nv_ceilf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
ceil(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_ceil(double)""",
     """%2 = call double @__nv_ceil(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)
abs(x::Int32) = Base.llvmcall(
    ("""declare i32 @__nv_abs(i32)""",
     """%2 = call i32 @__nv_abs(i32 %0)
        ret i32 %2"""),
    Int32, Tuple{Int32}, x)
abs(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_fabsf(float)""",
     """%2 = call float @__nv_fabsf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
abs(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_fabs(double)""",
     """%2 = call double @__nv_fabs(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)

# Square root
sqrt(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_sqrtf(float)""",
     """%2 = call float @__nv_sqrtf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
sqrt(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_sqrt(double)""",
     """%2 = call double @__nv_sqrt(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)

# Log and exp
exp(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_expf(float)""",
     """%2 = call float @__nv_expf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
exp(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_exp(double)""",
     """%2 = call double @__nv_exp(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)
log(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_logf(float)""",
     """%2 = call float @__nv_logf(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
log(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_log(double)""",
     """%2 = call double @__nv_log(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)
log10(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_log10f(float)""",
     """%2 = call float @__nv_log10f(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
log10(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_log10(double)""",
     """%2 = call double @__nv_log10(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)

erf(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_erff(float)""",
     """%2 = call float @__nv_erff(float %0)
        ret float %2"""),
    Float32, Tuple{Float32}, x)
erf(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_erf(double)""",
     """%2 = call double @__nv_erf(double %0)
        ret double %2"""),
    Float64, Tuple{Float64}, x)
