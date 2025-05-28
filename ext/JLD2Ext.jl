module JLD2Ext

using CUDA: CUDA
using JLD2: JLD2

JLD2.writeas(::Type{CUDA.CuArray{T, N, M}}) where {T, N, M} = Array{T, N}

end
