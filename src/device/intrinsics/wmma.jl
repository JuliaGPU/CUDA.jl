export WMMA
module WMMA

using ..CUDA: AS
using Core: LLVMPtr

################################################################################
# CONSTANTS
################################################################################

# Maps PTX types to Julia array types
const map_ptx_to_jl_array = Dict(
                                 "f16" => Float16,
                                 "f32" => Float32
                                )

# Maps PTX types to Julia fragment types
const map_ptx_to_jl_frag = Dict(
                                "f16" => NTuple{2, VecElement{Float16}},
                                "f32" => Float32
                               )

# Maps matrix & PTX types to fragment sizes
const map_frag_sizes = Dict(
                            "a.f16" => 8,
                            "b.f16" => 8,
                            "c.f16" => 4,
                            "c.f32" => 8,
                            "d.f16" => 4,
                            "d.f32" => 8
                           )

# Maps PTX AS to CUDA.AS
const map_ptx_as_to_as_ty = Dict(
                                 ""       => AS.Generic,
                                 "shared" => AS.Shared,
                                 "global" => AS.Global
                                )

################################################################################
# HELPER FUNCTIONS
################################################################################

# Returns (Julia array type, Julia fragment type, fragment size)
get_frag_info(matrix, ptx_el_type) = (
        map_ptx_to_jl_array[ptx_el_type],
        map_ptx_to_jl_frag[ptx_el_type],
        map_frag_sizes["$matrix.$ptx_el_type"]
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
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point). Note that `f32` is only valid for the matrix ``C``.
"""
llvm_wmma_load() = error("Cannot call llvm_wmma_load without values for placeholders!")
export llvm_wmma_load

for mat in ["a", "b", "c"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    # Float32 is only supported for C
    if (elem_type == "f32") && (mat != "c")
        continue
    end

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type]), "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.load.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"

    ptr_ty = LLVMPtr{arr_ty, addr_space_int}
    struct_ty = Symbol("LLVMStruct$sz")

    @eval $func_name(src_addr, stride) = convert(NTuple{$sz, $frag_ty}, ccall($ccall_name, llvmcall, $struct_ty{$frag_ty}, ($ptr_ty, Int32), src_addr, stride))
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
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).
"""
llvm_wmma_store() = error("Cannot call llvm_wmma_store without values for placeholders!")
export llvm_wmma_store

for mat in ["d"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type]), "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.store.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"
    frag_types = ntuple(i -> frag_ty, sz)
    frag_vars = ntuple(i -> :(data[$i]), sz)

    ptr_ty = LLVMPtr{arr_ty, addr_space_int}

    @eval $func_name(dst_addr, data, stride) = ccall($ccall_name, llvmcall, Nothing, ($ptr_ty, $(frag_types...), Int32), dst_addr, $(frag_vars...), stride)
    @eval export $func_name
    @eval @doc (@doc llvm_wmma_store) $func_name
end

# --------------------------
# Matrix multiply accumulate
# --------------------------

@doc """
    WMMA.llvm_wmma_mma_{a_layout}_{b_layout}_{shape}_{d_elem_type}_{c_elem_type}(a, b, c)

Wrapper around the LLVM intrinsic `@llvm.nvvm.wmma.mma.sync.{a_layout}.{b_layout}.{shape}.{d_elem_type}.{c_elem_type}`.

# Arguments
- `a`: The WMMA fragment corresponding to the matrix ``A``.
- `b`: The WMMA fragment corresponding to the matrix ``B``.
- `c`: The WMMA fragment corresponding to the matrix ``C``.

# Placeholders
- `{a_layout}`: The storage layout for matrix ``A``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{b_layout}`: The storage layout for matrix ``B``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{d_elem_type}`: The type of each element in the resultant ``D`` matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).
- `{c_elem_type}`: The type of each element in the ``C`` matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).

!!! warning

    Remember that the shape, type and layout of all operations (be it MMA, load or store) **MUST** match.
    Otherwise, the behaviour is undefined!
"""
llvm_wmma_mma() = error("Cannot call llvm_wmma_mma without values for placeholders!")
export llvm_wmma_mma

for a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    shape in ["m16n16k16"],
    d_elem_type in ["f16", "f32"],
    c_elem_type in ["f16", "f32"],
    b_elem_type in ["f16"],
    a_elem_type in ["f16"]

    # Name of the Julia wrapper function
    func_name = Symbol(join(filter(!isempty, ["llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type]), "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.mma.$a_layout.$b_layout.$d_elem_type.$c_elem_type"

    # Determine types + size for the (matrix, elem_type) combinations for matrix A, B, C and D
    a_arr_ty, a_frag_ty, a_sz = get_frag_info("a", a_elem_type)
    b_arr_ty, b_frag_ty, b_sz = get_frag_info("b", b_elem_type)
    c_arr_ty, c_frag_ty, c_sz = get_frag_info("c", c_elem_type)
    d_arr_ty, d_frag_ty, d_sz = get_frag_info("d", d_elem_type)

    ccall_name = "extern $llvm_intr"

    a_types = ntuple(i -> a_frag_ty, a_sz)
    b_types = ntuple(i -> b_frag_ty, b_sz)
    c_types = ntuple(i -> c_frag_ty, c_sz)

    a_vars = ntuple(i -> :(a[$i]), a_sz)
    b_vars = ntuple(i -> :(b[$i]), b_sz)
    c_vars = ntuple(i -> :(c[$i]), c_sz)

    struct_ty = Symbol("LLVMStruct$d_sz")

    @eval $func_name(a, b, c) = convert(NTuple{$d_sz, $d_frag_ty}, ccall($ccall_name, llvmcall, $struct_ty{$d_frag_ty}, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...)))
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
function flatten_recurse(typ::Type{NTuple{N, T}}, e) where {N, T}
    ret = Expr[]

    for (i, eltyp) in enumerate(typ.types)
        append!(ret, flatten_recurse(eltyp, :($e[$i])))
    end

    return ret
end

function unflatten_recurse(typ::Type{NTuple{N, T}}, e, idx) where {N, T}
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
julia> config = Config{16, 16, 16, Float32}
Config{16,16,16,Float32}
```
"""
struct Config{M, N, K, d_type} end

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

function get_hl_shape(M, N, K)
    if (M, N, K) != (16, 16, 16)
        error("Invalid shape for WMMA: (M, N, K) = ($M, $N, $K)")
    end

    return "m$(M)n$(N)k$(K)"
end

get_hl_mat_use(mat) = map_matrix_to_use[mat]

function get_hl_frag_info(matrix, T)
    ptx_ty = nothing

    try
        ptx_ty = map_jl_array_to_str[T]
    catch
        error("Invalid element type for WMMA: $T")
    end

    try
        return (map_num_elems[(matrix, T)],
                map_frag_sizes["$matrix.$ptx_ty"],
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
        num_els, _, _, arr_str = get_hl_frag_info($mat, T)
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

    _, a_frag_sz, a_frag_ty, _         = get_hl_frag_info("a", A_T)
    _, b_frag_sz, b_frag_ty, _         = get_hl_frag_info("b", B_T)
    _, c_frag_sz, c_frag_ty, c_arr_str = get_hl_frag_info("c", C_T)
    d_num_els, _, _, d_arr_str         = get_hl_frag_info("d", D_T)

    a_layout = get_hl_layout(A_L)
    b_layout = get_hl_layout(B_L)
    shape = get_hl_shape(M, N, K)

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
    num_els, frag_sz, frag_ty, arr_str = get_hl_frag_info("d", T)

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
    num_els, _, _ = get_hl_frag_info("c", T)

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
