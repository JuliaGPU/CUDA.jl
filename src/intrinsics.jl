# Native intrinsics

export
    # Indexing and dimensions
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize,

    # Memory management
    sync_threads,
    @cuSharedMem



#
# Indexing and dimensions
#

for dim in (:x, :y, :z)
    # Thread index
    fname = Symbol("threadIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.tid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($"""declare i32 @$intrinsic() readnone nounwind""",
             $"""%1 = tail call i32 @$intrinsic()
                 ret i32 %1"""),
            Int32, Tuple{}) + Int32(1)
    end

    # Block size (#threads per block)
    fname = Symbol("blockDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ntid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($"""declare i32 @$intrinsic() readnone nounwind""",
             $"""%1 = tail call i32 @$intrinsic()
                 ret i32 %1"""),
            Int32, Tuple{})
    end

    # Block index
    fname = Symbol("blockIdx_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.ctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($"""declare i32 @$intrinsic() readnone nounwind""",
             $"""%1 = tail call i32 @$intrinsic()
                 ret i32 %1"""),
            Int32, Tuple{}) + Int32(1)
    end

    # Grid size (#blocks per grid)
    fname = Symbol("gridDim_$dim")
    intrinsic = "llvm.nvvm.read.ptx.sreg.nctaid.$dim"
    @eval begin
        $fname() = Base.llvmcall(
            ($"""declare i32 @$intrinsic() readnone nounwind""",
             $"""%1 = tail call i32 @$intrinsic()
                 ret i32 %1"""),
            Int32, Tuple{})
    end
end

# Tuple accessors
threadIdx() = CUDAdrv.CuDim3(threadIdx_x(), threadIdx_y(), threadIdx_z())
blockDim() =  CUDAdrv.CuDim3(blockDim_x(),  blockDim_y(),  blockDim_z())
blockIdx() =  CUDAdrv.CuDim3(blockIdx_x(),  blockIdx_y(),  blockIdx_z())
gridDim() =   CUDAdrv.CuDim3(gridDim_x(),   gridDim_y(),   gridDim_z())

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

const typemap = Dict{Type,Symbol}(
    Int32   => :i32,
    Int64   => :i64,
    Float32 => :float,
    Float64 => :double
)

# FIXME: this adds module-scope declarations by means of `llvmcall`, which is unsupported
# TODO: return an Array-like object (containing the number of elements) instead of a raw pointer
# TODO: downcasting pointers to global AS might be inefficient
#       -> check if AS propagation resolves this
#       -> Ptr{AS}, ASPtr{AS}, ...?
# BUG: calling a device function referencing a static-memory @cuSharedMem will reference
#      the same memory -- how does this work in CUDA?

"""
    @cuSharedMem(typ::Type, [nel::Integer=0]) -> Ptr{typ}

Get a reference to a chunk of shared memory. The type `typ` and number of elements `nel`
should either be known at parse time (in which case the macro will directly generate the
necessary code), or after inference by means of a generated function. Note that due to
restrictions of generated functions, any errors will not be printed but result in a dynamic
call to said generated function.

If the number of elements is 0, dynamic shared memory will be referenced (ie. you need to
allocated the shared memory when calling the kernel), otherwise statically-allocated memory
will be used.
"""
macro cuSharedMem(typ, nel=0)
    if haskey(typemap, typ) && isa(nel, Integer)
        # easy case: both typ and N are known
        return generate_shmem_llvmcall(typ, nel)
    else
        # either one is now known (often typ being a typevar), retry after inference
        return esc(:(CUDAnative.generated_shmem($typ, Val{$nel})))
    end
end

shmem_id = 0
function generate_shmem_llvmcall(jltyp::Type, nel::Int)
    if !haskey(typemap, jltyp)
        error("cuSharedMem: unsupported type '$jltyp'")
    end
    llvmtyp = typemap[jltyp]

    if !isa(nel, Integer)
        error("cuSharedMem: second argument should be a positive integer")
    end

    global shmem_id
    var = "shmem$(shmem_id::Int += 1)"

    if nel == 0
        decl = """@$var = external addrspace(3) global [0 x $llvmtyp]"""
    else
        # TODO: alignment, why 4 (clang)? unify with dynamic shmem
        decl = """@$var = internal addrspace(3) global [$nel x $llvmtyp] zeroinitializer, align 4"""
    end

    return quote
        Base.llvmcall(
            ($decl,
             $"""%1 = getelementptr inbounds [$nel x $llvmtyp], [$nel x $llvmtyp] addrspace(3)* @$var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{})
    end
end

@generated function generated_shmem{T,N}(::Type{T}, ::Type{Val{N}})
    # TODO: if this actually errors, the error won't be print but a generic call to
    #       generate_shmem will be emitted instead (obviously incompatible with PTX)
    return generate_shmem_llvmcall(T, N)
end

# NOTE: this might be a neater approach (with a user-end macro for hiding the `Val{N}`):

# for typ in ((Int64,   :i64),
#             (Float32, :float),
#             (Float64, :double))
#     T, U = typ
#     @eval begin
#         cuSharedMem{T}(::Type{$T}) = Base.llvmcall(
#             ($"""@shmem_$U = external addrspace(3) global [0 x $U]""",
#              $"""%1 = getelementptr inbounds [0 x $U], [0 x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                  %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                  ret $U* %2"""),
#             Ptr{$T}, Tuple{})
#         cuSharedMem{T,N}(::Type{$T}, ::Val{N}) = Base.llvmcall(
#             ($"""@shmem_$U = internal addrspace(3) global [$N x $llvmtyp] zeroinitializer, align 4""",
#              $"""%1 = getelementptr inbounds [$N x $U], [$N x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                  %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                  ret $U* %2"""),
#             Ptr{$T}, Tuple{})
#     end
# end

# However, it requires a change to `llvmcall`, as now calling the static case twice results in
#          a reference to the same memory


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
