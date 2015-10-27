# Native intrinsics

export
    # Indexing and dimensions
    # TODO: dynamically export
    threadIdx_x, threadIdx_y, threadIdx_z,
    blockDim_x, blockDim_y, blockDim_z,
    blockIdx_x, blockIdx_y, blockIdx_z,
    gridDim_x, gridDim_y, gridDim_z,
    warpsize,

    # Memory management
    sync_threads,
    cuSharedMem, setCuSharedMem, getCuSharedMem

#
# Indexing and dimensions
#

for dim in (:x, :y, :z)
    # Thread index
    fname = symbol("threadIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.tid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, ()) + 1
    end

    # Block dimension (#threads per block)
    fname = symbol("blockDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ntid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, ())
    end

    # Block index
    fname = symbol("blockIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, ()) + 1
    end

    # Grid dimension (#blocks)
    fname = symbol("gridDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.nctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($("""declare i32 @$intrinsic() readnone nounwind"""),
             $("""%1 = tail call i32 @$intrinsic()
                  ret i32 %1""")),
            Int32, ())
    end
end

# Tuple accessors
# TODO: these get boxed no matter what -- avoid that!
#threadId() = (threadIdx_x(), threadIdx_y(), threadIdx_z())
#blockDim() = (blockDim_x(), blockDim_y(), blockDim_z())
#blockId() = (blockIdx_x(), blockIdx_y(), blockIdx_z())
#gridDim() = (gridDim_x(), gridDim_y(), gridDim_z())

# Warpsize
warpsize() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
        ret i32 %1"""),
    Int32, ())


#
# Memory management
#

# Synchronization
sync_threads() = Base.llvmcall(
    ("""declare void @llvm.nvvm.barrier0() readnone nounwind""",
     """call void @llvm.nvvm.barrier0()
        ret void"""),
    Void, ())

# Shared memory
# TODO: generalize for types
# TODO: static shared memory
# TODO: wrap this in a class, using get and setindex
# FIXME: it is a hack to declare the p0,p3 intrinsic in cuSharedMem,
#        but declaring it in the setters and getters results in two declarations
cuSharedMem() = Base.llvmcall(
    ("""@shmem = external addrspace(3) global [0 x float]
        declare float* @llvm.nvvm.ptr.shared.to.gen.p0f32.p3f32(float addrspace(3)*)
        declare float addrspace(3)* @llvm.nvvm.ptr.gen.to.shared.p3f32.p0f32(float*)""",
     """%1 = getelementptr inbounds [0 x float] addrspace(3)* @shmem, i64 0, i64 0
        %2 = tail call float* @llvm.nvvm.ptr.shared.to.gen.p0f32.p3f32( float addrspace(3)* %1 )
        ret float* %2"""),
    Ptr{Float32}, ())
setCuSharedMem(shmem, index, value) = Base.llvmcall(
     """%4 = tail call float addrspace(3)* @llvm.nvvm.ptr.gen.to.shared.p3f32.p0f32( float* %0 )
        %5 = getelementptr inbounds float addrspace(3)* %4, i64 %1
        store float %2, float addrspace(3)* %5
        ret void""",
    Void, (Ptr{Float32}, Int64, Float32), shmem, index-1, value)
getCuSharedMem(shmem, index) = Base.llvmcall(
     """%3 = tail call float addrspace(3)* @llvm.nvvm.ptr.gen.to.shared.p3f32.p0f32( float* %0 )
        %4 = getelementptr inbounds float addrspace(3)* %3, i64 %1
        %5 = load float addrspace(3)* %4
        ret float %5""",
    Float32, (Ptr{Float32}, Int64), shmem, index-1)

#
# Math
#

# Trigonometric
sin(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_sinf(float)""",
     """%2 = call float @__nv_sinf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
sin(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_sin(double)""",
     """%2 = call double @__nv_sin(double %0)
        ret double %2"""),
    Float64, (Float64,), x)
cos(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_cosf(float)""",
     """%2 = call float @__nv_cosf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
cos(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_cos(double)""",
     """%2 = call double @__nv_cos(double %0)
        ret double %2"""),
    Float64, (Float64,), x)

# Rounding
floor(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_floorf(float)""",
     """%2 = call float @__nv_floorf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
floor(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_floor(double)""",
     """%2 = call double @__nv_floor(double %0)
        ret double %2"""),
    Float64, (Float64,), x)
abs(x::Int32) = Base.llvmcall(
    ("""declare i32 @__nv_abs(i32)""",
     """%2 = call i32 @__nv_abs(i32 %0)
        ret i32 %2"""),
    Int32, (Int32,), x)
abs(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_fabsf(float)""",
     """%2 = call float @__nv_fabsf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
abs(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_fabs(double)""",
     """%2 = call double @__nv_fabs(double %0)
        ret double %2"""),
    Float64, (Float64,), x)


# Square root
sqrt(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_sqrtf(float)""",
     """%2 = call float @__nv_sqrtf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
sqrt(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_sqrt(double)""",
     """%2 = call double @__nv_sqrt(double %0)
        ret double %2"""),
    Float64, (Float64,), x)

# Log and exp
exp(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_expf(float)""",
     """%2 = call float @__nv_expf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
exp(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_exp(double)""",
     """%2 = call double @__nv_exp(double %0)
        ret double %2"""),
    Float64, (Float64,), x)
log(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_logf(float)""",
     """%2 = call float @__nv_logf(float %0)
        ret float %2"""),
    Float32, (Float32,), x)
log(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_log(double)""",
     """%2 = call double @__nv_log(double %0)
        ret double %2"""),
    Float64, (Float64,), x)