# Native support for cuda

# intrinsics
threadId_x() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1
threadId_y() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1
threadId_z() = Base.llvmcall("""%1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z() readnone nounwind
                				ret i32 %1""", Int32, ()) + 1


# transfer datatypes
type In{T}
	data::T
end
length(i::In) = length(i.data)
size(i::In) = size(i.data)
eltype{T}(i::In{T}) = T

type Out{T}
	data::T
end
length(o::Out) = length(o.data)
size(o::Out) = size(o.data)
eltype{T}(o::Out{T}) = T

type InOut{T}
	data::T
end
length(io::InOut) = length(io.data)
size(io::InOut) = size(io.data)
eltype{T}(io::InOut{T}) = T
