# Julia values as CCCL types, values and iterators

cccl_type(::Type{Int8}) = CCCL_INT8
cccl_type(::Type{Int16}) = CCCL_INT16
cccl_type(::Type{Int32}) = CCCL_INT32
cccl_type(::Type{Int64}) = CCCL_INT64
cccl_type(::Type{UInt8}) = CCCL_UINT8
cccl_type(::Type{UInt16}) = CCCL_UINT16
cccl_type(::Type{UInt32}) = CCCL_UINT32
cccl_type(::Type{UInt64}) = CCCL_UINT64
cccl_type(::Type{Float16}) = CCCL_FLOAT16
cccl_type(::Type{Float32}) = CCCL_FLOAT32
cccl_type(::Type{Float64}) = CCCL_FLOAT64
cccl_type(::Type{Bool}) = CCCL_BOOLEAN
# other bits types are opaque storage, only usable with Julia operators
cccl_type(::Type) = CCCL_STORAGE

function type_info(T::Type)
    isbitstype(T) || throw(ArgumentError("CCCL only supports bits types, got $T"))
    cccl_type_info(sizeof(T), Base.datatype_alignment(T), cccl_type(T))
end

const null_op = cccl_op_t(CCCL_STATELESS, C_NULL, C_NULL, 0, CCCL_OP_LLVM_IR, 0, 0,
                          C_NULL, C_NULL, C_NULL, 0, C_NULL)

function pointer_iterator(T::Type, ptr)
    cccl_iterator_t(sizeof(Ptr{Cvoid}), Base.datatype_alignment(Ptr{Cvoid}), CCCL_POINTER,
                    null_op, null_op, type_info(T), reinterpret(Ptr{Cvoid}, ptr), C_NULL)
end

iterator(x::DenseCuArray{T}) where {T} = pointer_iterator(T, pointer(x))

# keys-only sorts are requested by passing a null value iterator
null_iterator(T::Type=UInt8) = pointer_iterator(T, C_NULL)

# a host value whose address is passed to the library; keep `ref` alive across the call
cccl_value(ref::Base.RefValue{T}) where {T} =
    cccl_value_t(type_info(T), Base.unsafe_convert(Ptr{Cvoid}, ref))
