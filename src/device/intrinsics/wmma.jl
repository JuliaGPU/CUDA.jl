export WMMA
module WMMA

import ..LLVM
using ..CUDA: AS
using Core: LLVMPtr

################################################################################
# CONSTANTS
################################################################################

# Maps PTX types to Julia array types
const map_ptx_to_jl_array = Dict(
                                 "u8"  => UInt8,
                                 "s8"  => Int8,
                                 "s32" => Int32,
                                 "f16" => Float16,
                                 "f32" => Float32
                                )

# Maps PTX types to Julia fragment types
const map_ptx_to_jl_frag = Dict(
                                "u8"  => UInt32,
                                "s8"  => UInt32,
                                "s32" => Int32,
                                "f16" => NTuple{2, VecElement{Float16}},
                                "f32" => Float32
                               )

# Maps matrix & PTX types to fragment sizes, information retrieved from 
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-wmma

const map_frag_sizes = Dict(
                            # A
                            "a.u8.m16n16k16"  => 2,
                            "a.u8.m8n32k16"   => 1,
                            "a.u8.m32n8k16"   => 4,

                            "a.s8.m16n16k16"  => 2,
                            "a.s8.m8n32k16"   => 1,
                            "a.s8.m32n8k16"   => 4,

                            "a.f16.m16n16k16" => 8,
                            "a.f16.m8n32k16"  => 8,
                            "a.f16.m32n8k16"  => 8,
                            # B
                            "b.u8.m16n16k16"  => 2,
                            "b.u8.m8n32k16"   => 4,
                            "b.u8.m32n8k16"   => 1,

                            "b.s8.m16n16k16"  => 2,
                            "b.s8.m8n32k16"   => 4,
                            "b.s8.m32n8k16"   => 1,

                            "b.f16.m16n16k16" => 8,
                            "b.f16.m8n32k16"  => 8,
                            "b.f16.m32n8k16"  => 8,
                            # C
                            "c.s32.m16n16k16" => 8,
                            "c.s32.m8n32k16"  => 8,
                            "c.s32.m32n8k16"  => 8,

                            "c.f16.m16n16k16" => 4,
                            "c.f16.m8n32k16"  => 4,
                            "c.f16.m32n8k16"  => 4,

                            "c.f32.m16n16k16" => 8,
                            "c.f32.m8n32k16"  => 8,
                            "c.f32.m32n8k16"  => 8,
                            # D
                            "d.s32.m16n16k16" => 8,
                            "d.s32.m8n32k16"  => 8,
                            "d.s32.m32n8k16"  => 8,

                            "d.f16.m16n16k16" => 4,
                            "d.f16.m8n32k16"  => 4,
                            "d.f16.m32n8k16"  => 4,

                            "d.f32.m16n16k16" => 8,
                            "d.f32.m8n32k16"  => 8,
                            "d.f32.m32n8k16"  => 8,
                           )

# Maps PTX AS to CUDA.AS
const map_ptx_as_to_as_ty = Dict(
                                 ""       => AS.Generic,
                                 "shared" => AS.Shared,
                                 "global" => AS.Global
                                )

# Valid WMMA Operation configurations: Shape (M,N,K), Matrix, Element Type

# Half-Precision Floating Point
const ldst_half_ab_ops = [(16,16,16), (32,8,16), (8,32,16)], ["a", "b"], ["f16"]
const ldst_half_cd_ops = [(16,16,16), (32,8,16), (8,32,16)], ["c", "d"], ["f16", "f32"]
const wmma_half_ops    = [(16,16,16), (32,8,16), (8,32,16)], ["f16"], ["f16", "f32"], ["f16", "f32"]
# Integer
const ldst_int_ab_ops = [(16,16,16), (32,8,16), (8,32,16)], ["a", "b"], ["u8", "s8"]
const ldst_int_cd_ops = [(16,16,16), (32,8,16), (8,32,16)], ["c", "d"], ["s32"]
const wmma_int_ops    = [(16,16,16), (32,8,16), (8,32,16)], ["s8", "u8"], ["s32"], ["s32"]

const all_ldst_ops = vcat(ldst_half_ab_ops, ldst_half_cd_ops,
                          ldst_int_ab_ops,  ldst_int_cd_ops)
const all_wmma_ops = vcat(wmma_half_ops, wmma_int_ops)

# Valid WMMA operation shapes
const valid_shapes = [(16, 16, 16), (32, 8, 16), (8, 32, 16)]

################################################################################
# HELPER FUNCTIONS
################################################################################

# Returns shape information as a string
function get_hl_shape(M, N, K)
    if (M, N, K) in valid_shapes
        return "m$(M)n$(N)k$(K)"
    end
    error("Invalid shape for WMMA: (M, N, K) = ($M, $N, $K)")
end

# Returns (Julia array type, Julia fragment type, fragment size)
get_frag_info(matrix, ptx_el_type, shape) = (
        map_ptx_to_jl_array[ptx_el_type],
        map_ptx_to_jl_frag[ptx_el_type],
        map_frag_sizes["$matrix.$ptx_el_type.$shape"]
        )

get_addrspace_info(addr_space) = convert(Int, map_ptx_as_to_as_ty[addr_space])

# Fix for https://github.com/JuliaGPU/CUDAnative.jl/issues/587.
# Instead of ccall'ing the intrinsics with NTuple{N, T} (which gets lowered to
# [N x llvmT]), we generate custom structs LLVMStructN{T}, containing N fields
# of type T, and use those as return type. After
# https://github.com/JuliaLang/julia/pull/34996, these structs are lowered to
# { llvmT, llvmT, ... }, which is the return type LLVM expects.
for N in unique(values(map_frag_sizes))
    struct_ty = Symbol("LLVMStruct$N")

    @eval struct $struct_ty{T}
        Base.Cartesian.@nexprs $N i -> x_i::T
    end

    @eval Base.convert(::Type{NTuple{$N, T}}, x::$struct_ty{T}) where {T} = ntuple(i -> getfield(x, i), $N)
end

################################################################################
# LOW LEVEL API
################################################################################

# -----------
# Matrix load
# -----------

@doc """
    WMMA.llvm_wmma_load_{matrix}_{layout}_{shape}_{addr_space}_stride_{elem_type}(src_addr, stride)

Wrapper around the LLVM intrinsic `@llvm.nvvm.wmma.load.{matrix}.sync.{layout}.{shape}.{addr_space}.stride.{elem_type}`.

# Arguments
- `src_addr`: The memory address to load from.
- `stride`: The leading dimension of the matrix, in numbers of elements.

# Placeholders
- `{matrix}`: The matrix to load. Can be `a`, `b` or `c`.
- `{layout}`: The storage layout for the matrix. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively.
- `{shape}`: The overall shape of the MAC operation. Valid values are `m16n16k16`, `m32n8k16`, and `m8n32k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. For `a` and `b` matrices, valid values are `u8` (byte unsigned integer),
                `s8` (byte signed integer), and `f16` (half precision floating point). For `c` and `d` matrices, valid values are
                `s32` (32-bit signed integer), `f16` (half precision floating point), and `f32` (full precision floating point).
"""
llvm_wmma_load() = error("Cannot call llvm_wmma_load without values for placeholders!")
export llvm_wmma_load

for ops in all_ldst_ops,
    mnk in ops[1],
    mat in ops[2],
    elem_type in ops[3],
    layout in ["col", "row"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"]

    shape = get_hl_shape(mnk[1], mnk[2], mnk[3])
    # TODO: Non-stride versions?

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type]), "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.load.$mat.$layout.stride.$elem_type.p$(addr_space_int)"
    if LLVM.version() < v"17"
        llvm_intr *= "i8"
    end

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type, shape)

    ccall_name = "$llvm_intr"

    ptr_ty = :(LLVMPtr{$arr_ty, $addr_space_int})

    if sz == 1
        @eval $func_name(src_addr, stride) = tuple(ccall($ccall_name, llvmcall, $frag_ty, ($ptr_ty, Int32), src_addr, stride))
    else
        struct_ty = Symbol("LLVMStruct$sz")
        @eval $func_name(src_addr, stride) = convert(NTuple{$sz, $frag_ty}, ccall($ccall_name, llvmcall, $struct_ty{$frag_ty}, ($ptr_ty, Int32), src_addr, stride))
    end
    @eval export $func_name
    @eval @doc (@doc llvm_wmma_load) $func_name
end

# ------------
# Matrix store
# ------------

@doc """
    WMMA.llvm_wmma_store_d_{layout}_{shape}_{addr_space}_stride_{elem_type}(dst_addr, data, stride)

Wrapper around the LLVM intrinsic `@llvm.nvvm.wmma.store.d.sync.{layout}.{shape}.{addr_space}.stride.{elem_type}`.

# Arguments
- `dst_addr`: The memory address to store to.
- `data`: The ``D`` fragment to store.
- `stride`: The leading dimension of the matrix, in numbers of elements.

# Placeholders
- `{layout}`: The storage layout for the matrix. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively.
- `{shape}`: The overall shape of the MAC operation. Valid values are `m16n16k16`, `m32n8k16`, and `m8n32k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. For `a` and `b` matrices, valid values are `u8` (byte unsigned integer),
                `s8` (byte signed integer), and `f16` (half precision floating point). For `c` and `d` matrices, valid values are
                `s32` (32-bit signed integer), `f16` (half precision floating point), and `f32` (full precision floating point).
"""
llvm_wmma_store() = error("Cannot call llvm_wmma_store without values for placeholders!")
export llvm_wmma_store

    for ops in all_ldst_ops,
        mnk in ops[1],
        mat in ops[2],
        elem_type in ops[3],
        layout in ["col", "row"],
        addr_space in ["", "shared", "global"],
        stride in ["stride"]

    if mat != "d"
        continue
    end

    shape = get_hl_shape(mnk[1], mnk[2], mnk[3])

    # TODO: Non-stride versions?

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type]), "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.store.$mat.$layout.stride.$elem_type.p$(addr_space_int)"
    if LLVM.version() < v"17"
        llvm_intr *= "i8"
    end

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type, shape)

    ccall_name = "$llvm_intr"
    frag_types = ntuple(i -> frag_ty, sz)
    frag_vars = ntuple(i -> :(data[$i]), sz)

    ptr_ty = :(LLVMPtr{$arr_ty, $addr_space_int})

    @eval $func_name(dst_addr, data, stride) = ccall($ccall_name, llvmcall, Nothing, ($ptr_ty, $(frag_types...), Int32), dst_addr, $(frag_vars...), stride)
    @eval export $func_name
    @eval @doc (@doc llvm_wmma_store) $func_name
end

# --------------------------
# Matrix multiply accumulate
# --------------------------

@doc """
    WMMA.llvm_wmma_mma_{a_layout}_{b_layout}_{shape}_{d_elem_type}_{c_elem_type}(a, b, c) or
    WMMA.llvm_wmma_mma_{a_layout}_{b_layout}_{shape}_{a_elem_type}(a, b, c)

For floating point operations: wrapper around the LLVM intrinsic `@llvm.nvvm.wmma.mma.sync.{a_layout}.{b_layout}.{shape}.{d_elem_type}.{c_elem_type}`
For all other operations: wrapper around the LLVM intrinsic `@llvm.nvvm.wmma.mma.sync.{a_layout}.{b_layout}.{shape}.{a_elem_type}`

# Arguments
- `a`: The WMMA fragment corresponding to the matrix ``A``.
- `b`: The WMMA fragment corresponding to the matrix ``B``.
- `c`: The WMMA fragment corresponding to the matrix ``C``.

# Placeholders
- `{a_layout}`: The storage layout for matrix ``A``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{b_layout}`: The storage layout for matrix ``B``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{shape}`: The overall shape of the MAC operation. Valid values are `m16n16k16`, `m32n8k16`, and `m8n32k16`.
- `{a_elem_type}`: The type of each element in the ``A`` matrix. Valid values are `u8` (byte unsigned integer), `s8` (byte signed integer), and `f16` (half precision floating point).
- `{d_elem_type}`: The type of each element in the resultant ``D`` matrix. Valid values are `s32` (32-bit signed integer), `f16` (half precision floating point), and `f32` (full precision floating point).
- `{c_elem_type}`: The type of each element in the ``C`` matrix. Valid values are `s32` (32-bit signed integer), `f16` (half precision floating point), and `f32` (full precision floating point).

!!! warning

    Remember that the shape, type and layout of all operations (be it MMA, load or store) **MUST** match.
    Otherwise, the behaviour is undefined!
"""
llvm_wmma_mma() = error("Cannot call llvm_wmma_mma without values for placeholders!")
export llvm_wmma_mma

for ops in all_wmma_ops,
    a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    mnk in ops[1],
    d_elem_type in ops[4],
    c_elem_type in ops[3],
    b_elem_type in ops[2]

    a_elem_type = b_elem_type
    shape = get_hl_shape(mnk[1], mnk[2], mnk[3])

    # Name of the LLVM intrinsic
    # If integer/sub-byte/bit A/B types, name is determined by A/B types
    if d_elem_type == "s32"
        llvm_intr = "llvm.nvvm.wmma.$shape.mma.$a_layout.$b_layout.$a_elem_type"
        # Name of the Julia wrapper function
        func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "mma", a_layout, b_layout, shape, a_elem_type]), "_"))
    else # Name defined by D/C types
        llvm_intr = "llvm.nvvm.wmma.$shape.mma.$a_layout.$b_layout.$d_elem_type.$c_elem_type"
        # Name of the Julia wrapper function
        func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type]), "_"))
    end

    # Determine types + size for the (matrix, elem_type) combinations for matrix A, B, C and D
    a_arr_ty, a_frag_ty, a_sz = get_frag_info("a", a_elem_type, shape)
    b_arr_ty, b_frag_ty, b_sz = get_frag_info("b", b_elem_type, shape)
    c_arr_ty, c_frag_ty, c_sz = get_frag_info("c", c_elem_type, shape)
    d_arr_ty, d_frag_ty, d_sz = get_frag_info("d", d_elem_type, shape)

    ccall_name = "$llvm_intr"

    a_types = ntuple(i -> a_frag_ty, a_sz)
    b_types = ntuple(i -> b_frag_ty, b_sz)
    c_types = ntuple(i -> c_frag_ty, c_sz)

    a_vars = ntuple(i -> :(a[$i]), a_sz)
    b_vars = ntuple(i -> :(b[$i]), b_sz)
    c_vars = ntuple(i -> :(c[$i]), c_sz)

    if d_sz == 1
        @eval $func_name(a, b, c) = tuple(ccall($ccall_name, llvmcall, $d_frag_ty, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...)))
    else
        struct_ty = Symbol("LLVMStruct$d_sz")
        @eval $func_name(a, b, c) = convert(NTuple{$d_sz, $d_frag_ty}, ccall($ccall_name, llvmcall, $struct_ty{$d_frag_ty}, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...)))
    end
    @eval export $func_name
    @eval @doc (@doc llvm_wmma_mma) $func_name
end

################################################################################
# FLATTENING/UNFLATTENING LOGIC
################################################################################

# Base case (Float16, Float32, ...)
flatten_recurse(typ, e) = [:($e)]
unflatten_recurse(typ, e, idx) = :($e[$idx]), idx + 1

# VecElements
flatten_recurse(typ::Type{VecElement{T}}, e) where T = [:($e.value)]
unflatten_recurse(typ::Type{VecElement{T}}, e, idx) where T = :(VecElement{$T}($e[$idx])), idx + 1

# NTuples
function flatten_recurse(typ::Type{T}, e) where {T <: NTuple}
    ret = Expr[]

    for (i, eltyp) in enumerate(typ.types)
        append!(ret, flatten_recurse(eltyp, :($e[$i])))
    end

    return ret
end

function unflatten_recurse(typ::Type{T}, e, idx) where {T<:NTuple}
    ret = Expr(:tuple)

    for (i, eltyp) in enumerate(typ.types)
        arg, idx = unflatten_recurse(eltyp, e, idx)
        push!(ret.args, arg)
    end

    return ret, idx
end

@generated flatten(x::typ) where typ = Expr(:tuple, flatten_recurse(typ, :x)...)
@generated unflatten(::Type{typ}, x) where typ = unflatten_recurse(typ, :x, 1)[1]

################################################################################
# HIGH LEVEL (CUDA-STYLE API)
################################################################################

# -------------
# WMMA fragment
# -------------

export FragmentLayout, RowMajor, ColMajor, Unspecified

"""
    WMMA.FragmentLayout

Abstract type that specifies the storage layout of a matrix.

Possible values are [`WMMA.RowMajor`](@ref), [`WMMA.ColMajor`](@ref) and [`WMMA.Unspecified`](@ref).
"""
abstract type FragmentLayout end

"""
    WMMA.RowMajor

Type that represents a matrix stored in row major (C style) order.
"""
struct RowMajor <: FragmentLayout end

"""
    WMMA.ColMajor

Type that represents a matrix stored in column major (Julia style) order.
"""
struct ColMajor <: FragmentLayout end

"""
    WMMA.Unspecified

Type that represents a matrix stored in an unspecified order.

!!! warning

    This storage format is not valid for all WMMA operations!
"""
struct Unspecified <: FragmentLayout end


export MatrixA, MatrixB, Accumulator

abstract type FragmentUse end
struct MatrixA <: FragmentUse end
struct MatrixB <: FragmentUse end
struct Accumulator <: FragmentUse end


export Fragment

"""
    WMMA.Fragment

Type that represents per-thread intermediate results of WMMA operations.

You can access individual elements using the `x` member or `[]` operator, but beware that the exact ordering of elements is unspecified.
"""
struct Fragment{M, N, K, FS, T, L <: FragmentLayout, U <: FragmentUse}
    x::NTuple{FS, T}
end

# ----------------------
# WMMA fragment indexing
# ----------------------

for f in (:getindex, :setindex!, :firstindex, :lastindex)
    @eval Base.$f(frag::Fragment, args...) = $f(frag.x, args...)
end

# ------------------
# WMMA configuration
# ------------------

export Config

"""
    WMMA.Config{M, N, K, d_type}

Type that contains all information for WMMA operations that cannot be inferred from the argument's types.

WMMA instructions calculate the matrix multiply-accumulate operation ``D = A \\cdot B + C``, where ``A`` is a ``M \\times K`` matrix,
``B`` a ``K \\times N`` matrix, and ``C`` and ``D`` are ``M \\times N`` matrices.

`d_type` refers to the type of the elements of matrix ``D``, and can be either `Float16` or `Float32`.

All WMMA operations take a `Config` as their final argument.

# Examples
```jldoctest
julia> config = WMMA.Config{16, 16, 16, Float32}
CUDA.WMMA.Config{16, 16, 16, Float32}
```
"""
struct ConfigRounding{M, N, K, d_type, rounding} end

Config{M, N, K, d_type} = ConfigRounding{M, N, K, d_type, RoundNearest}

# ---------
# Constants
# ---------

# Maps Julia array types to string
const map_jl_array_to_str = Dict(val => key for (key, val) in map_ptx_to_jl_array)

# Maps CUDA.AS types to string
const map_as_ty_to_str = Dict(val => key for (key, val) in map_ptx_as_to_as_ty)

# Maps layout types to string
const map_layout_ty_to_str = Dict(
                                  RowMajor => "row",
                                  ColMajor => "col"
                                 )

# Maps matrix & type to number of elements (size after flattening)
const map_num_elems = Dict(
                           ("a", Float16) => 16,
                           ("b", Float16) => 16,
                           ("c", Float16) => 8,
                           ("c", Float32) => 8,
                           ("d", Float16) => 8,
                           ("d", Float32) => 8
                          )

# Maps matrix to its use
const map_matrix_to_use = Dict(
                               "a" => MatrixA,
                               "b" => MatrixB,
                               "c" => Accumulator,
                               "d" => Accumulator
                              )

# ----------------
# Helper functions
# ----------------

function get_hl_as_info(AS)
    try
        return map_as_ty_to_str[AS]
    catch
        error("Invalid address space for WMMA: $AS")
    end
end

function get_hl_layout(L)
    try
        return map_layout_ty_to_str[L]
    catch
        error("Invalid layout for WMMA: $L")
    end
end

get_hl_mat_use(mat) = map_matrix_to_use[mat]

function get_hl_frag_info(matrix, T, shape)
    ptx_ty = nothing

    try
        ptx_ty = map_jl_array_to_str[T]
    catch
        error("Invalid element type for WMMA: $T")
    end

    try
        return (map_num_elems[(matrix, T)],
                map_frag_sizes["$matrix.$ptx_ty.$shape"],
                map_ptx_to_jl_frag[ptx_ty],
                ptx_ty)
    catch
        error("Invalid type $T for matrix $matrix")
    end
end

# ---------
# WMMA load
# ---------

export load_a, load_b, load_c

"""
    WMMA.load_a(addr, stride, layout, config)
    WMMA.load_b(addr, stride, layout, config)
    WMMA.load_c(addr, stride, layout, config)

Load the matrix `a`, `b` or `c` from the memory location indicated by `addr`, and return the resulting [`WMMA.Fragment`](@ref).

# Arguments
- `addr`: The address to load the matrix from.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`WMMA.RowMajor`](@ref) and [`WMMA.ColMajor`](@ref).
- `config`: The WMMA configuration that should be used for loading this matrix. See [`WMMA.Config`](@ref).

See also: [`WMMA.Fragment`](@ref), [`WMMA.FragmentLayout`](@ref), [`WMMA.Config`](@ref)

!!! warning

    All threads in a warp **MUST** execute the load operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
load_a, load_b, load_c

for mat in ["a", "b", "c"]
    func_name = Symbol("load_$mat")

    @eval @generated function $func_name(addr::LLVMPtr{T, AS},
                                         stride::Number,
                                         layout::Type{L},
                                         config::Type{Config{M, N, K, D_TYPE}}) where {T, AS, L, M, N, K, D_TYPE}

        as_str                 = get_hl_as_info(AS)
        layout                 = get_hl_layout(L)
        shape                  = get_hl_shape(M, N, K)
        num_els, _, _, arr_str = get_hl_frag_info($mat, T, shape)
        U                      = get_hl_mat_use($mat)
        L_ret                  = ($mat == "c") ? Unspecified : L

        # Name of the Julia wrapper
        wrapper = Symbol(join(filter(!isempty, ["llvm", "wmma", "load", $mat, layout, shape, as_str, "stride", arr_str]), "_"))

        return quote
            x = flatten($wrapper(addr, stride))
            return Fragment{$M, $N, $K, $num_els, $T, $L_ret, $U}(x)
        end
    end
end


# ------------------------
# WMMA multiply-accumulate
# ------------------------

export mma

"""
    WMMA.mma(a, b, c, conf)

Perform the matrix multiply-accumulate operation ``D = A \\cdot B + C``.

# Arguments

- `a`: The [`WMMA.Fragment`](@ref) corresponding to the matrix ``A``.
- `b`: The [`WMMA.Fragment`](@ref) corresponding to the matrix ``B``.
- `c`: The [`WMMA.Fragment`](@ref) corresponding to the matrix ``C``.
- `conf`: The [`WMMA.Config`](@ref) that should be used in this WMMA operation.

!!! warning

    All threads in a warp **MUST** execute the `mma` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
mma

@generated function mma(a::Fragment{M, N, K, A_SZ, A_T, A_L, MatrixA},
                        b::Fragment{M, N, K, B_SZ, B_T, B_L, MatrixB},
                        c::Fragment{M, N, K, C_SZ, C_T, Unspecified, Accumulator},
                        config::Type{Config{M, N, K, D_T}}) where {M, N, K, A_SZ, A_T, A_L, B_SZ, B_T, B_L, C_SZ, C_T, D_T}

    a_layout = get_hl_layout(A_L)
    b_layout = get_hl_layout(B_L)
    shape = get_hl_shape(M, N, K)

    _, a_frag_sz, a_frag_ty, _         = get_hl_frag_info("a", A_T, shape)
    _, b_frag_sz, b_frag_ty, _         = get_hl_frag_info("b", B_T, shape)
    _, c_frag_sz, c_frag_ty, c_arr_str = get_hl_frag_info("c", C_T, shape)
    d_num_els, _, _, d_arr_str         = get_hl_frag_info("d", D_T, shape)



    # Name of the Julia wrapper
    wrapper = Symbol(join(filter(!isempty, ["llvm", "wmma", "mma", a_layout, b_layout, shape, d_arr_str, c_arr_str]), "_"))

    return quote
        a_unfl = unflatten(NTuple{$a_frag_sz, $a_frag_ty}, a.x)
        b_unfl = unflatten(NTuple{$b_frag_sz, $b_frag_ty}, b.x)
        c_unfl = unflatten(NTuple{$c_frag_sz, $c_frag_ty}, c.x)

        x = flatten($wrapper(a_unfl, b_unfl, c_unfl))
        return Fragment{$M, $N, $K, $d_num_els, $D_T, Unspecified, Accumulator}(x)
    end
end


# ----------
# WMMA store
# ----------

export store_d

"""
    WMMA.store_d(addr, d, stride, layout, config)

Store the result matrix `d` to the memory location indicated by `addr`.

# Arguments
- `addr`: The address to store the matrix to.
- `d`: The [`WMMA.Fragment`](@ref) corresponding to the `d` matrix.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`WMMA.RowMajor`](@ref) and [`WMMA.ColMajor`](@ref).
- `config`: The WMMA configuration that should be used for storing this matrix. See [`WMMA.Config`](@ref).

See also: [`WMMA.Fragment`](@ref), [`WMMA.FragmentLayout`](@ref), [`WMMA.Config`](@ref)

!!! warning

    All threads in a warp **MUST** execute the `store` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
store_d

@generated function store_d(addr::LLVMPtr{T, AS},
                            d::Fragment{M, N, K, D_SZ, T, Unspecified, Accumulator},
                            stride::Number,
                            layout::Type{L},
                            config::Type{Config{M, N, K, T}}) where {T, AS, M, N, K, D_SZ, L}

    as_str                             = get_hl_as_info(AS)
    layout                             = get_hl_layout(L)
    shape                              = get_hl_shape(M, N, K)
    num_els, frag_sz, frag_ty, arr_str = get_hl_frag_info("d", T, shape)

    # Name of the Julia wrapper
    wrapper = Symbol(join(filter(!isempty, ["llvm", "wmma", "store", "d", layout, shape, as_str, "stride", arr_str]), "_"))

    return quote
        d_unfl = unflatten(NTuple{$frag_sz, $frag_ty}, d.x)
        $wrapper(addr, d_unfl, stride)
        return nothing
    end
end


# ------------------
# WMMA fill fragment
# ------------------

export fill_c

"""
    WMMA.fill_c(value, config)

Return a [`WMMA.Fragment`](@ref) filled with the value `value`.

This operation is useful if you want to implement a matrix multiplication (and thus want to set ``C = O``).

# Arguments
- `value`: The value used to fill the fragment. Can be a `Float16` or `Float32`.
- `config`: The WMMA configuration that should be used for this WMMA operation. See [`WMMA.Config`](@ref).
"""
fill_c

@generated function fill_c(value::T,
                           config::Type{Config{M, N, K, D_TYPE}}) where {T, M, N, K, D_TYPE}

    # We can't use closures in @generated functions, so we'll have to do it this way instead of
    # ntuple(i -> val, $num_els)
    shape = get_hl_shape(M, N, K)
    num_els, _, _ = get_hl_frag_info("c", T, shape)

    args = [:value for i=1:num_els]
    expr = :(tuple($(args...)))

    return quote
        return Fragment{$M, $N, $K, $num_els, $T, Unspecified, Accumulator}($expr)
    end
end

################################################################################
# BROADCASTING OVER WMMA FRAGMENTS
################################################################################

# Based on broadcasting implementation of Tuples in
# https://github.com/JuliaLang/julia/blob/master/base/broadcast.jl


# Custom broadcast style for Fragments
struct FragmentBroadcastStyle <: Broadcast.BroadcastStyle end

# Use this broadcasting style for Fragments
Base.BroadcastStyle(::Type{<:Fragment}) = FragmentBroadcastStyle()

# Broadcast style precedence rules
# If we broadcast a fragment with a scalar, we want the Fragment style to take precedence
Base.BroadcastStyle(s::FragmentBroadcastStyle, t::Broadcast.DefaultArrayStyle{0}) = s

# We don't want to convert fragments before broadcasting
Base.broadcastable(frag::Fragment) = frag

# Needed for broadcast machinery
Base.axes(frag::Fragment) = axes(frag.x)

# Helper functions to get element at specified index
@inline get_index(x, i) = x                    # scalar
@inline get_index(frag::Fragment, i) = frag[i] # Fragment

# Helper functions to get first fragment in broadcast call
@inline find_first_fragment(args::Tuple) = find_first_fragment(args[1], Base.tail(args))
@inline find_first_fragment(a::Fragment, tail) = a
@inline find_first_fragment(::Any, tail) = find_first_fragment(tail)

# Custom broadcast implementation that returns a Fragment
@inline function Base.copy(bc::Broadcast.Broadcasted{FragmentBroadcastStyle})
    dim = Broadcast.combine_axes(bc.args...)

    if length(dim) != 1
        throw(DimensionMismatch("WMMA fragment broadcast only supports one dimension!"))
    end

    N = length(dim[1])

    tuple = ntuple(i -> bc.f(map(arg -> get_index(arg, i), bc.args)...), Val(N))

    frag_ty = typeof(find_first_fragment(bc.args))
    return frag_ty(tuple)
end

end
