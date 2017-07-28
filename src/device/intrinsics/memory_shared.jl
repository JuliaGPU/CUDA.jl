# Shared Memory (part of B.2)

export @cuStaticSharedMem, @cuDynamicSharedMem

# FIXME: `shmem_id` increment in the macro isn't correct, as multiple parametrically typed
#        functions will alias the id (but the size might be a parameter). but incrementing in
#        the @generated function doesn't work, as it is supposed to be pure and identical
#        invocations will erroneously share (and even cause multiple shmem globals).
shmem_id = 0

# add a shared memory definition to the module, and get a pointer to the first item
# FIXME: this adds module-scope declarations by means of `llvmcall`, which is unsupported
function emit_shmem(id, llvmtyp, len, align)
    var = Symbol("@shmem", id)
    jltyp = jltypes[llvmtyp]

    @gensym ptr
    quote
        $ptr = Base.llvmcall(
            ($"""$var = external addrspace(3) global [$len x $llvmtyp], align $align""",
             $"""%1 = getelementptr inbounds [$len x $llvmtyp], [$len x $llvmtyp] addrspace(3)* $var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{})
        DevicePtr{$jltyp,AS.Shared}($ptr)
    end
end

# IDEA: merge static and dynamic shared memory, specializing on whether the shape is known
#       (ie. a number) or an expression/symbol, communicating the necessary dynamic memory
#       to `@cuda`

"""
    @cuStaticSharedMem(typ::Type, dims) -> CuDeviceArray{typ,Shared}

Get an array of type `typ` and dimensions `dims` (either an integer length or tuple shape)
pointing to a statically-allocated piece of shared memory. The type should be statically
inferable and the dimensions should be constant (without requiring constant propagation, see
JuliaLang/julia#5560), or an error will be thrown and the generator function will be called
dynamically.

Multiple statically-allocated shared memory arrays can be requested by calling this macro
multiple times.
"""
macro cuStaticSharedMem(typ, dims)
    global shmem_id
    id = shmem_id::Int += 1

    return :(generate_static_shmem(Val{$id}, $(esc(typ)), Val{$(esc(dims))}))
end

# types with known corresponding LLVM type
function emit_static_shmem{N, T<:LLVMTypes}(id::Integer, jltyp::Type{T}, shape::NTuple{N,Int})
    llvmtyp = llvmtypes[jltyp]

    len = prod(shape)
    align = datatype_align(jltyp)

    @gensym ptr
    return quote
        Base.@_inline_meta
        $ptr = $(emit_shmem(id, llvmtyp, len, align))
        CuDeviceArray($shape, $ptr)
    end
end

# fallback for unknown types
function emit_static_shmem{N}(id::Integer, jltyp::Type, shape::NTuple{N,<:Integer})
    if !isbits(jltyp)
        error("cuStaticSharedMem: non-isbits type '$jltyp' is not supported")
    end

    len = prod(shape) * sizeof(jltyp)
    align = datatype_align(jltyp)

    @gensym ptr
    return quote
        Base.@_inline_meta
        $ptr = $(emit_shmem(id, :i8, len, align))
        CuDeviceArray($shape, Base.convert(DevicePtr{$jltyp}, $ptr))
    end
end

@generated function generate_static_shmem{ID,T,D}(::Type{Val{ID}}, ::Type{T}, ::Type{Val{D}})
    return emit_static_shmem(ID, T, tuple(D...))
end


"""
    @cuDynamicSharedMem(typ::Type, dims, offset::Integer=0) -> CuDeviceArray{typ,Shared}

Get an array of type `typ` and dimensions `dims` (either an integer length or tuple shape)
pointing to a dynamically-allocated piece of shared memory. The type should be statically
inferable and the dimension and offset parameters should be constant (without requiring
constant propagation, see JuliaLang/julia#5560), or an error will be thrown and the
generator function will be called dynamically.

Dynamic shared memory also needs to be allocated beforehand, when calling the kernel.

Optionally, an offset parameter indicating how many bytes to add to the base shared memory
pointer can be specified. This is useful when dealing with a heterogeneous buffer of dynamic
shared memory; in the case of a homogeneous multi-part buffer it is preferred to use `view`.

Note that calling this macro multiple times does not result in different shared arrays; only
a single dynamically-allocated shared memory array exists.
"""
macro cuDynamicSharedMem(typ, dims, offset=0)
    global shmem_id
    id = shmem_id::Int += 1

    return :(generate_dynamic_shmem(Val{$id}, $(esc(typ)), $(esc(dims)), $(esc(offset))))
end

# TODO: boundscheck against %dynamic_smem_size (currently unsupported by LLVM)

# types with known corresponding LLVM type
function emit_dynamic_shmem{T<:LLVMTypes}(id::Integer, jltyp::Type{T}, shape::Union{Expr,Symbol}, offset)
    llvmtyp = llvmtypes[jltyp]

    align = datatype_align(jltyp)

    @gensym ptr
    return quote
        Base.@_inline_meta
        $ptr = $(emit_shmem(id, llvmtyp, 0, align)) + $offset
        CuDeviceArray($shape, $ptr)
    end
end

# fallback for unknown types
function emit_dynamic_shmem(id::Integer, jltyp::Type, shape::Union{Expr,Symbol}, offset)
    if !isbits(jltyp)
        error("cuDynamicSharedMem: non-isbits type '$jltyp' is not supported")
    end

    align = datatype_align(jltyp)

    @gensym ptr
    return quote
        Base.@_inline_meta
        $ptr = $(emit_shmem(id, :i8, 0, align)) + $offset
        CuDeviceArray($shape, Base.convert(DevicePtr{$jltyp}, $ptr))
    end
end

@generated function generate_dynamic_shmem{ID,T}(::Type{Val{ID}}, ::Type{T}, dims, offset)
    return emit_dynamic_shmem(ID, T, :(dims), :(offset))
end

# IDEA: a neater approach (with a user-end macro for hiding the `Val{N}`):
#
#   for typ in ((Int64,   :i64),
#               (Float32, :float),
#               (Float64, :double))
#       T, U = typ
#       @eval begin
#           cuSharedMem{T}(::Type{$T}) = Base.llvmcall(
#               ($"""@shmem_$U = external addrspace(3) global [0 x $U]""",
#                $"""%1 = getelementptr inbounds [0 x $U], [0 x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                    %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                    ret $U* %2"""),
#               Ptr{$T}, Tuple{})
#           cuSharedMem{T,N}(::Type{$T}, ::Val{N}) = Base.llvmcall(
#               ($"""@shmem_$U = internal addrspace(3) global [$N x $llvmtyp] zeroinitializer, align 4""",
#                $"""%1 = getelementptr inbounds [$N x $U], [$N x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                    %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                    ret $U* %2"""),
#               Ptr{$T}, Tuple{})
#       end
#   end
#
# Requires a change to `llvmcall`, as calling the static case twice references the same memory.
