# Thread ID
threadId_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
                                                                           ret i32 %1""", Int32, ()) + 1 # ::Int32 # This gives error
threadId_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() readnone nounwind
                                                                           ret i32 %1""", Int32, ()) + 1 # ::Int32 # This gives error
threadId_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z() readnone nounwind
                                                                           ret i32 %1""", Int32, ()) + 1
# Block Dim (num threads per block)
numThreads_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
numThreads_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
numThreads_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
# Block ID
blockId_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() readnone nounwind
                                                                          ret i32 %1""", Int32, ()) + 1
blockId_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() readnone nounwind
                                                                          ret i32 %1""", Int32, ()) + 1
blockId_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() readnone nounwind
                                                                          ret i32 %1""", Int32, ()) + 1
# Grid Dim (num blocks per grid)
numBlocks_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
numBlocks_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
numBlocks_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() readnone nounwind
                                                                                ret i32 %1""", Int32, ())
# Warpsize
warpsize() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize() readnone nounwind
                                                                         ret i32 %1""", Int32, ())
# Barrier
sync_threads() = Base.llvmcall("""call void @llvm.nvvm.barrier0()
                                                                                 ret void""", Void, ())

#
# Math
#

# Trigonometric
sin(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_sinf( float %0 )
                      ret float %2""", Float32, (Float32,), x)
sin(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_sin( double %0 )
                      ret double %2""", Float64, (Float64,), x)

cos(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_cosf( float %0 )
                      ret float %2""", Float32, (Float32,), x)
cos(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_cos( double %0 )
                      ret double %2""", Float64, (Float64,), x)

# Rounding
floor(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_floorf( float %0 )
                      ret float %2""", Float32, (Float32,), x)
floor(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_floor( double %0 )
                      ret double %2""", Float64, (Float64,), x)
