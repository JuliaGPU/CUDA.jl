#
# Indexing and dimensions
#

# Thread indexes
threadId_x() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
        ret i32 %1"""),
    Int32, ()) + 1
threadId_y() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
        ret i32 %1"""),
    Int32, ()) + 1
threadId_z() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.tid.z() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
        ret i32 %1"""),
    Int32, ()) + 1

# Block dimensions (num of threads per block)
numThreads_x() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
        ret i32 %1"""),
    Int32, ())
numThreads_y() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
        ret i32 %1"""),
    Int32, ())
numThreads_z() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
        ret i32 %1"""),
    Int32, ())

# Block indexes
blockId_x() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() readnone nounwind""", 
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
        ret i32 %1"""),
    Int32, ()) + 1
blockId_y() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
        ret i32 %1"""),
    Int32, ()) + 1
blockId_z() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
        ret i32 %1"""),
    Int32, ()) + 1

# Grid dimensions (num blocks per grid)
numBlocks_x() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
        ret i32 %1"""),
    Int32, ())
numBlocks_y() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
       ret i32 %1"""),
    Int32, ())
numBlocks_z() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
        ret i32 %1"""),
    Int32, ())

# Warpsize
warpsize() = Base.llvmcall(
    ("""declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() readnone nounwind""",
     """%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
        ret i32 %1"""),
    Int32, ())

# Barriers
sync_threads() = Base.llvmcall(
    ("""declare void @llvm.nvvm.barrier0() readnone nounwind""",
     """call void @llvm.nvvm.barrier0()
        ret void"""),
    Void, ())


#
# Math
#

# Trigonometric
sin(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_sinf(float %0)""",
     """%1 = call float @__nv_sinf(float %0)
        ret float %1"""),
    Float32, (Float32,), x)
sin(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_sin(double %0)""",
     """%1 = call double @__nv_sin(double %0)
        ret double %1"""),
    Float64, (Float64,), x)
cos(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_cosf(float %0)""",
     """%1 = call float @__nv_cosf(float %0)
        ret float %1"""),
    Float32, (Float32,), x)
cos(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_cos(double %0)""",
     """%1 = call double @__nv_cos(double %0)
        ret double %1"""),
    Float64, (Float64,), x)

# Rounding
floor(x::Float32) = Base.llvmcall(
    ("""declare float @__nv_floorf(float %0)""",
     """%1 = call float @__nv_floorf(float %0)
        ret float %1"""),
    Float32, (Float32,), x)
floor(x::Float64) = Base.llvmcall(
    ("""declare double @__nv_floor(double %0)""",
     """%1 = call double @__nv_floor(double %0)
        ret double %1"""),
    Float64, (Float64,), x)
