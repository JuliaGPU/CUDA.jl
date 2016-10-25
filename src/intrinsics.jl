# Native intrinsics

export
    # I/O
    @cuprintf,

    # Indexing and dimensions
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize, nearest_warpsize,

    # Memory management
    sync_threads,
    @cuStaticSharedMem, @cuDynamicSharedMem



#
# Support functionality
#

# TODO: compute capability checks

const llvmtypes = Dict{Type,Symbol}(
    Void    => :void,
    Int32   => :i32,
    Int64   => :i64,
    Float32 => :float,
    Float64 => :double
)

const jltypes = Dict{Symbol,Type}(v => k for (k,v) in llvmtypes)

"""
Decode an expression of the form:

   function(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type

Returns a tuple containing the function name, a vector of argument, a vector of argument
types and the return type (all in symbolic form).
"""
function decode_call(e)
    @assert e.head == :(::)
    rettype = e.args[2]::Symbol

    call = e.args[1]
    @assert call.head == :call

    fn = Symbol(call.args[1])
    args = Symbol[arg.args[1] for arg in call.args[2:end]]
    argtypes = Symbol[arg.args[2] for arg in call.args[2:end]]

    return fn, args, argtypes, rettype
end

"""
Generate a `llvmcall` statement calling an intrinsic specified as follows:

    intrinsic(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type [attr]

The argument types should be valid LLVM type identifiers (eg. i32, float, double).
Conversions to the corresponding Julia type are automatically generated; make sure the
actual arguments are of the same type to make these conversions no-ops. The optional
argument `attr` indicates which LLVM function attributes (such as `readnone` or `nounwind`)
to add to the intrinsic declaration.

For example, the following call:
    @wrap __some_intrinsic(x::float, y::double)::float

will yield the following `llvmcall`:

    Base.llvmcall(("declare float @__somme__intr(float, double)",
                   "%3 = call float @__somme__intr(float %0, double %1)
                    ret float %3"),
                  Float32, Tuple{Float32,Float64},
                  convert(Float32,x), convert(Float64,y))
"""
macro wrap(call, attrs="")
    intrinsic, args, argtypes, rettype = decode_call(call)

    llvm_args = String["%$i" for i in 0:length(argtypes)]
    if rettype == :void
        llvm_ret_asgn = ""
        llvm_ret = "void"
    else
        llvm_ret_var = "%$(length(argtypes)+1)"
        llvm_ret_asgn = "$llvm_ret_var = "
        llvm_ret = "$rettype $llvm_ret_var"
    end
    llvm_declargs = join(argtypes, ", ")
    llvm_defargs = join(("$t $arg" for (t,arg) in zip(argtypes, llvm_args)), ", ")

    julia_argtypes = (jltypes[t] for t in argtypes)
    julia_args = (:(convert($argtype, $arg)) for (arg, argtype) in zip(args, julia_argtypes))

    return quote
        Base.llvmcall(
            ($"""declare $rettype @$intrinsic($llvm_declargs)""",
             $"""$llvm_ret_asgn call $rettype @$intrinsic($llvm_defargs)
                 ret $llvm_ret"""),
            $(jltypes[rettype]), Tuple{$(julia_argtypes...)}, $(julia_args...))
    end
end

"""
Make a string literal safe to embed in LLVM IR.

This is a custom, simplified version of Base.escape_string, replacing non-printable
characters with their two-digit hex code.
"""
function escape_llvm_string(io, s::AbstractString, esc::AbstractString)
    i = start(s)
    while !done(s,i)
        c, j = next(s,i)
        c == '\\'       ? print(io, "\\\\") :
        c in esc        ? print(io, '\\', c) :
        isprint(c)      ? print(io, c) :
                          print(io, "\\", hex(c, 2))
        i = j
    end
end
escape_llvm_string(s::AbstractString) = sprint(endof(s), escape_llvm_string, s, "\"")


#
# I/O
#

const cuprintf_fmts = Vector{String}()

"""
Print a formatted string in device context on the host standard output:

    @cuprintf("%Fmt", args...)

Note that this is not a fully C-compliant `printf` implementation; see the CUDA
documentation for supported options and inputs.

Also beware that it is an untyped, and unforgiving `printf` implementation. Type widths need
to match, eg. printing a Julia integer requires the `%ld` formatting string.

"""
macro cuprintf(fmt::String, args...)
    # NOTE: we can't pass fmt by Val{}, so save it in a global buffer
    push!(cuprintf_fmts, "$fmt\0")
    id = length(cuprintf_fmts)

    return esc(:(CUDAnative.generated_cuprintf(Val{$id}, $(args...))))
end

@generated function generated_cuprintf{ID}(::Type{Val{ID}}, argspec...)
    args = [:( argspec[$i] ) for i in 1:length(argspec)]
    return emit_vprintf(ID, argspec, args...)
end

function emit_vprintf(id::Integer, argtypes, args...)
    fmt = cuprintf_fmts[id]
    fmtlen = length(fmt)

    llvm_argtypes = [llvmtypes[jltype] for jltype in argtypes]

    decls = Vector{String}()
    push!(decls, """declare i32 @vprintf(i8*, i8*)""")
    push!(decls, """%print$(id)_argtyp = type { $(join(llvm_argtypes, ", ")) }""")
    push!(decls, """@print$(id)_fmt = private unnamed_addr constant [$fmtlen x i8] c"$(escape_llvm_string(fmt))", align 1""")

    ir = Vector{String}()
    push!(ir, """%args = alloca %print$(id)_argtyp""")
    arg = 0
    tmp = length(args)+1
    for jltype in argtypes
        llvmtype = llvmtypes[jltype]
        push!(ir, """%$tmp = getelementptr inbounds %print$(id)_argtyp, %print$(id)_argtyp* %args, i32 0, i32 $arg""")
        push!(ir, """store $llvmtype %$arg, $llvmtype* %$tmp, align 4""")
        arg+=1
        tmp+=1
    end
    push!(ir, """%argptr = bitcast %print$(id)_argtyp* %args to i8*""")
    push!(ir, """%$tmp = call i32 @vprintf(i8* getelementptr inbounds ([$fmtlen x i8], [$fmtlen x i8]* @print$(id)_fmt, i32 0, i32 0), i8* %argptr)""")
    push!(ir, """ret void""")

    return quote
        Base.llvmcall(($(join(decls, "\n")),
                       $(join(ir,    "\n"))),
                      Void, Tuple{$argtypes...}, $(args...)
                     )
    end
end



#
# Indexing and dimensions
#

for dim in (:x, :y, :z)
    # Thread index
    fn = Symbol("threadIdx_$dim")
    @eval @inline $fn() = (@wrap llvm.nvvm.read.ptx.sreg.tid.$dim()::i32    "readnone nounwind")+Int32(1)

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    @eval @inline $fn() =  @wrap llvm.nvvm.read.ptx.sreg.ntid.$dim()::i32   "readnone nounwind"

    # Block index
    fn = Symbol("blockIdx_$dim")
    @eval @inline $fn() = (@wrap llvm.nvvm.read.ptx.sreg.ctaid.$dim()::i32  "readnone nounwind")+Int32(1)

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    @eval @inline $fn() =  @wrap llvm.nvvm.read.ptx.sreg.nctaid.$dim()::i32 "readnone nounwind"
end

# Tuple accessors
@inline threadIdx() = CUDAdrv.CuDim3(threadIdx_x(), threadIdx_y(), threadIdx_z())
@inline blockDim() =  CUDAdrv.CuDim3(blockDim_x(),  blockDim_y(),  blockDim_z())
@inline blockIdx() =  CUDAdrv.CuDim3(blockIdx_x(),  blockIdx_y(),  blockIdx_z())
@inline gridDim() =   CUDAdrv.CuDim3(gridDim_x(),   gridDim_y(),   gridDim_z())

# NOTE: we often need a const warpsize (eg. for shared memory), sp keep this fixed for now
# @inline warpsize() = @wrap llvm.nvvm.read.ptx.sreg.warpsize()::i32 "readnone nounwind"
const warpsize = Int32(32)

"Return the nearest multiple of a warpsize, a common requirement for the amount of threads."
@inline nearest_warpsize(threads) =  threads + (warpsize - threads % warpsize) % warpsize



#
# Parallel Synchronization and Communication
#

## barriers

# TODO: rename to syncthreads
@inline sync_threads() = @wrap llvm.nvvm.barrier0()::void "readnone nounwind"


## voting

export vote_all, vote_any, vote_ballot

const all_asm = """{
    .reg .pred %p1;
    .reg .pred %p2;
    setp.ne.u32 %p1, \$1, 0;
    vote.all.pred %p2, %p1;
    selp.s32 \$0, 1, 0, %p2;
}"""
function vote_all(pred::Bool)
    return Base.llvmcall(
        """%2 = call i32 asm sideeffect "$all_asm", "=r,r"(i32 %0)
           ret i32 %2""",
        Int32, Tuple{Int32}, convert(Int32, pred)) != Int32(0)
end

const any_asm = """{
    .reg .pred %p1;
    .reg .pred %p2;
    setp.ne.u32 %p1, \$1, 0;
    vote.any.pred %p2, %p1;
    selp.s32 \$0, 1, 0, %p2;
}"""
function vote_any(pred::Bool)
    return Base.llvmcall(
        """%2 = call i32 asm sideeffect "$any_asm", "=r,r"(i32 %0)
           ret i32 %2""",
        Int32, Tuple{Int32}, convert(Int32, pred)) != Int32(0)
end

const ballot_asm = """{
   .reg .pred %p1;
   setp.ne.u32 %p1, \$1, 0;
   vote.ballot.b32 \$0, %p1;
}"""
function vote_ballot(pred::Bool)
    return Base.llvmcall(
        """%2 = call i32 asm sideeffect "$ballot_asm", "=r,r"(i32 %0)
           ret i32 %2""",
        UInt32, Tuple{Int32}, convert(Int32, pred))
end


#
# Shared memory
#

# FIXME: this adds module-scope declarations by means of `llvmcall`, which is unsupported
# TODO: return an Array-like object (containing the number of elements) instead of a raw pointer
# TODO: downcasting pointers to global AS might be inefficient
#       -> check if AS propagation resolves this
#       -> Ptr{AS}, ASPtr{AS}, ...?
# BUG: calling a device function referencing a static-memory @cuSharedMem will reference
#      the same memory -- how does this work in CUDA?

shmem_id = 0

"""
    @cuStaticSharedMem(typ::Type, dims) -> CuDeviceArray{typ}

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

    return esc(:(CUDAnative.generate_static_shmem(Val{$id}, $typ, Val{$dims})))
end

@generated function generate_static_shmem{ID,T,D}(::Type{Val{ID}}, ::Type{T}, ::Type{Val{D}})
    return emit_static_shmem(ID, T, tuple(D...))
end

function emit_static_shmem{N}(id::Integer, jltyp::Type, shape::NTuple{N,Int})
    if !haskey(llvmtypes, jltyp)
        error("cuStaticSharedMem: unsupported type '$jltyp'")
    end
    llvmtyp = llvmtypes[jltyp]

    var = Symbol(:@shmem, id)
    len = prod(shape)

    return quote
        CuDeviceArray{$jltyp}($shape, Base.llvmcall(
            ($"""$var = internal addrspace(3) global [$len x $llvmtyp] zeroinitializer, align 4""",
             $"""%1 = getelementptr inbounds [$len x $llvmtyp], [$len x $llvmtyp] addrspace(3)* $var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{}))
    end
end


"""
    @cuDynamicSharedMem(typ::Type, dims, offset::Integer=0) -> CuDeviceArray{typ}

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

    return esc(:(CUDAnative.generate_dynamic_shmem(Val{$id}, $typ, $dims, $offset)))
end

@generated function generate_dynamic_shmem{ID,T}(::Type{Val{ID}}, ::Type{T}, dims, offset)
    return emit_dynamic_shmem(ID, T, :(dims), :(offset))
end

# TODO: boundscheck against %dynamic_smem_size (currently unsupported by LLVM)
function emit_dynamic_shmem(id::Integer, jltyp::Type, shape::Union{Expr,Symbol}, offset::Symbol)
    if !haskey(llvmtypes, jltyp)
        error("cuDynamicSharedMem: unsupported type '$jltyp'")
    end
    llvmtyp = llvmtypes[jltyp]

    var = Symbol(:@shmem, id)

    return quote
        CuDeviceArray{$jltyp}($shape, Base.llvmcall(
            ($"""$var = external addrspace(3) global [0 x $llvmtyp]""",
             $"""%1 = getelementptr inbounds [0 x $llvmtyp], [0 x $llvmtyp] addrspace(3)* $var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{}) + $offset)
    end
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
# Data movement and conversion
#

## shuffling

# TODO: should shfl_idx conform to 1-based indexing?

# narrow
for typ in ((Int32,   :i32, :i32),
            (UInt32,  :i32, :i32),
            (Float32, :f32, :float))
    jl, intr, llvm = typ

    for op in ((:up,   Int32(0x00)),
               (:down, Int32(0x1f)),
               (:bfly, Int32(0x1f)),
               (:idx,  Int32(0x1f)))
        mode, mask = op
        fname = Symbol("shfl_$mode")
        pack_expr = :(((warpsize - Int32(width)) << 8) | $mask)
        @static if VersionNumber(Base.libllvm_version) >= v"3.9-"
            intrinsic = Symbol("llvm.nvvm.shfl.$mode.$intr")
            @eval begin
                export $fname
                @inline $fname(val::$jl, srclane::Integer, width::Integer=warpsize) = Base.llvmcall(
                        ($"""declare $llvm @$intrinsic($llvm, i32, i32)""",
                         $"""%4 = call $llvm @$intrinsic($llvm %0, i32 %1, i32 %2)
                             ret $llvm %4"""),
                        $jl, Tuple{$jl, Int32, Int32}, val, Int32(srclane),
                        $pack_expr)
            end
        else
            instruction = Symbol("shfl.$mode.b32")  # NOTE: only b32 available, no i32/f32
            @eval begin
                export $fname
                @inline $fname(val::$jl, srclane::Integer, width::Integer=warpsize) = Base.llvmcall(
                        $"""%4 = call $llvm asm sideeffect "$instruction \$0, \$1, \$2, \$3;", "=r,r,r,r"($llvm %0, i32 %1, i32 %2)
                            ret $llvm %4""",    # "
                        $jl, Tuple{$jl, Int32, Int32}, val, Int32(srclane),
                        $pack_expr)
            end
        end
    end
end

@inline decode(val::UInt64) = trunc(UInt32,  val & 0x00000000ffffffff),
                              trunc(UInt32, (val & 0xffffffff00000000)>>32)

@inline encode(x::UInt32, y::UInt32) = UInt64(x) | UInt64(y)<<32

# wide
# NOTE: we only reuse the i32 shuffle, does it make any difference using eg. f32 shuffle for f64 values?
for typ in (Int64, UInt64, Float64)
    for mode in (:up, :down, :bfly, :idx)
        fname = Symbol("shfl_$mode")
        @eval begin
            export $fname
            @inline function $fname(val::$typ, srclane::Integer, width::Integer=warpsize)
                x,y = decode(reinterpret(UInt64, val))
                x = $fname(x, srclane, width)
                y = $fname(y, srclane, width)
                reinterpret($typ, encode(x,y))
            end
        end
    end
end


#
# Math
#

## trigonometric

@inline cos(x::Float64) = @wrap __nv_cos(x::double)::double
@inline cos(x::Float32) = @wrap __nv_cosf(x::float)::float

@inline cospi(x::Float64) = @wrap __nv_cospi(x::double)::double
@inline cospi(x::Float32) = @wrap __nv_cospif(x::float)::float

@inline sin(x::Float64) = @wrap __nv_sin(x::double)::double
@inline sin(x::Float32) = @wrap __nv_sinf(x::float)::float

@inline sinpi(x::Float64) = @wrap __nv_sinpi(x::double)::double
@inline sinpi(x::Float32) = @wrap __nv_sinpif(x::float)::float

@inline tan(x::Float64) = @wrap __nv_tan(x::double)::double
@inline tan(x::Float32) = @wrap __nv_tanf(x::float)::float


## inverse trigonometric

@inline acos(x::Float64) = @wrap __nv_acos(x::double)::double
@inline acos(x::Float32) = @wrap __nv_acosf(x::float)::float

@inline asin(x::Float64) = @wrap __nv_asin(x::double)::double
@inline asin(x::Float32) = @wrap __nv_asinf(x::float)::float

@inline atan(x::Float64) = @wrap __nv_atan(x::double)::double
@inline atan(x::Float32) = @wrap __nv_atanf(x::float)::float

@inline atan2(x::Float64, y::Float64) = @wrap __nv_atan2(x::double, y::double)::double
@inline atan2(x::Float32, y::Float32) = @wrap __nv_atan2f(x::float, y::float)::float


## hyperbolic

@inline cosh(x::Float64) = @wrap __nv_cosh(x::double)::double
@inline cosh(x::Float32) = @wrap __nv_coshf(x::float)::float

@inline sinh(x::Float64) = @wrap __nv_sinh(x::double)::double
@inline sinh(x::Float32) = @wrap __nv_sinhf(x::float)::float

@inline tanh(x::Float64) = @wrap __nv_tanh(x::double)::double
@inline tanh(x::Float32) = @wrap __nv_tanhf(x::float)::float


## inverse hyperbolic

@inline acosh(x::Float64) = @wrap __nv_acosh(x::double)::double
@inline acosh(x::Float32) = @wrap __nv_acoshf(x::float)::float

@inline asinh(x::Float64) = @wrap __nv_asinh(x::double)::double
@inline asinh(x::Float32) = @wrap __nv_asinhf(x::float)::float

@inline atanh(x::Float64) = @wrap __nv_atanh(x::double)::double
@inline atanh(x::Float32) = @wrap __nv_atanhf(x::float)::float


## logarithmic

@inline log(x::Float64) = @wrap __nv_log(x::double)::double
@inline log(x::Float32) = @wrap __nv_logf(x::float)::float

@inline log10(x::Float64) = @wrap __nv_log10(x::double)::double
@inline log10(x::Float32) = @wrap __nv_log10f(x::float)::float

@inline log1p(x::Float64) = @wrap __nv_log1p(x::double)::double
@inline log1p(x::Float32) = @wrap __nv_log1pf(x::float)::float

@inline log2(x::Float64) = @wrap __nv_log2(x::double)::double
@inline log2(x::Float32) = @wrap __nv_log2f(x::float)::float

@inline logb(x::Float64) = @wrap __nv_logb(x::double)::double
@inline logb(x::Float32) = @wrap __nv_logbf(x::float)::float

@inline ilogb(x::Float64) = @wrap __nv_ilogb(x::double)::i32
@inline ilogb(x::Float32) = @wrap __nv_ilogbf(x::float)::i32


## exponential

@inline exp(x::Float64) = @wrap __nv_exp(x::double)::double
@inline exp(x::Float32) = @wrap __nv_expf(x::float)::float

@inline exp2(x::Float64) = @wrap __nv_exp2(x::double)::double
@inline exp2(x::Float32) = @wrap __nv_exp2f(x::float)::float

@inline exp10(x::Float64) = @wrap __nv_exp10(x::double)::double
@inline exp10(x::Float32) = @wrap __nv_exp10f(x::float)::float

@inline expm1(x::Float64) = @wrap __nv_expm1(x::double)::double
@inline expm1(x::Float32) = @wrap __nv_expm1f(x::float)::float

@inline ldexp(x::Float64, y::Int32) = @wrap __nv_ldexp(x::double, y::i32)::double
@inline ldexp(x::Float32, y::Int32) = @wrap __nv_ldexpf(x::float, y::i32)::float


## error

@inline erf(x::Float64) = @wrap __nv_erf(x::double)::double
@inline erf(x::Float32) = @wrap __nv_erff(x::float)::float

@inline erfinv(x::Float64) = @wrap __nv_erfinv(x::double)::double
@inline erfinv(x::Float32) = @wrap __nv_erfinvf(x::float)::float

@inline erfc(x::Float64) = @wrap __nv_erfc(x::double)::double
@inline erfc(x::Float32) = @wrap __nv_erfcf(x::float)::float

@inline erfcinv(x::Float64) = @wrap __nv_erfcinv(x::double)::double
@inline erfcinv(x::Float32) = @wrap __nv_erfcinvf(x::float)::float

@inline erfcx(x::Float64) = @wrap __nv_erfcx(x::double)::double
@inline erfcx(x::Float32) = @wrap __nv_erfcxf(x::float)::float


## integer handling (bit twiddling)

@inline brev(x::Int32) =   @wrap __nv_brev(x::i32)::i32
@inline brev(x::Int64) =   @wrap __nv_brevll(x::i64)::i64

@inline clz(x::Int32) =   @wrap __nv_clz(x::i32)::i32
@inline clz(x::Int64) =   @wrap __nv_clzll(x::i64)::i32

@inline ffs(x::Int32) = @wrap __nv_ffs(x::i32)::i32
@inline ffs(x::Int64) = @wrap __nv_ffsll(x::i64)::i32

@inline byte_perm(x::Int32, y::Int32, z::Int32) = @wrap __nv_byte_perm(x::i32, y::i32, z::i32)::i32

@inline popc(x::Int32) = @wrap __nv_popc(x::i32)::i32
@inline popc(x::Int64) = @wrap __nv_popcll(x::i64)::i32


## floating-point handling

@inline isfinite(x::Float32) = (@wrap __nv_finitef(x::float)::i32) != 0
@inline isfinite(x::Float64) = (@wrap __nv_isfinited(x::double)::i32) != 0

@inline isinf(x::Float32) = (@wrap __nv_isinfd(x::double)::i32) != 0
@inline isinf(x::Float64) = (@wrap __nv_isinff(x::float)::i32) != 0

@inline isnan(x::Float32) = (@wrap __nv_isnand(x::double)::i32) != 0
@inline isnan(x::Float64) = (@wrap __nv_isnanf(x::float)::i32) != 0

@inline nearbyint(x::Float64) = @wrap __nv_nearbyint(x::double)::double
@inline nearbyint(x::Float32) = @wrap __nv_nearbyintf(x::float)::float

@inline nextafter(x::Float64, y::Float64) = @wrap __nv_nextafter(x::double, y::double)::double
@inline nextafter(x::Float32, y::Float32) = @wrap __nv_nextafterf(x::float, y::float)::float


## sign handling

@inline signbit(x::Float64) = (@wrap __nv_signbitd(x::double)::i32) != 0
@inline signbit(x::Float32) = (@wrap __nv_signbitf(x::float)::i32) != 0

@inline copysign(x::Float64, y::Float64) = @wrap __nv_copysign(x::double, y::double)::double
@inline copysign(x::Float32, y::Float32) = @wrap __nv_copysignf(x::float, y::float)::float

@inline abs(x::Int32) =   @wrap __nv_abs(x::i32)::i32
@inline abs(f::Float64) = @wrap __nv_fabs(f::double)::double
@inline abs(f::Float32) = @wrap __nv_fabsf(f::float)::float
@inline abs(x::Int64) =   @wrap __nv_llabs(x::i64)::i64


## roots and powers

@inline sqrt(x::Float64) = @wrap __nv_sqrt(x::double)::double
@inline sqrt(x::Float32) = @wrap __nv_sqrtf(x::float)::float

@inline rsqrt(x::Float64) = @wrap __nv_rsqrt(x::double)::double
@inline rsqrt(x::Float32) = @wrap __nv_rsqrtf(x::float)::float

@inline cbrt(x::Float64) = @wrap __nv_cbrt(x::double)::double
@inline cbrt(x::Float32) = @wrap __nv_cbrtf(x::float)::float

@inline rcbrt(x::Float64) = @wrap __nv_rcbrt(x::double)::double
@inline rcbrt(x::Float32) = @wrap __nv_rcbrtf(x::float)::float

@inline pow(x::Float64, y::Float64) = @wrap __nv_pow(x::double, y::double)::double
@inline pow(x::Float32, y::Float32) = @wrap __nv_powf(x::float, y::float)::float
@inline pow(x::Float64, y::Int32) =   @wrap __nv_powi(x::double, y::i32)::double
@inline pow(x::Float32, y::Int32) =   @wrap __nv_powif(x::float, y::i32)::float


## rounding and selection

# TODO: differentiate in return type, map correctly
# @inline round(x::Float64) = @wrap __nv_llround(x::double)::i64
# @inline round(x::Float32) = @wrap __nv_llroundf(x::float)::i64
# @inline round(x::Float64) = @wrap __nv_round(x::double)::double
# @inline round(x::Float32) = @wrap __nv_roundf(x::float)::float

# TODO: differentiate in return type, map correctly
# @inline rint(x::Float64) = @wrap __nv_llrint(x::double)::i64
# @inline rint(x::Float32) = @wrap __nv_llrintf(x::float)::i64
# @inline rint(x::Float64) = @wrap __nv_rint(x::double)::double
# @inline rint(x::Float32) = @wrap __nv_rintf(x::float)::float

# TODO: would conflict with trunc usage in this module
# @inline trunc(x::Float64) = @wrap __nv_trunc(x::double)::double
# @inline trunc(x::Float32) = @wrap __nv_truncf(x::float)::float

@inline ceil(x::Float64) = @wrap __nv_ceil(x::double)::double
@inline ceil(x::Float32) = @wrap __nv_ceilf(x::float)::float

@inline floor(f::Float64) = @wrap __nv_floor(f::double)::double
@inline floor(f::Float32) = @wrap __nv_floorf(f::float)::float

@inline min(x::Int32, y::Int32) = @wrap __nv_min(x::i32, y::i32)::i32
@inline min(x::Int64, y::Int64) = @wrap __nv_llmin(x::i64, y::i64)::i64
@inline min(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umin(x::i32, y::i32)::i32)
@inline min(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmin(x::i64, y::i64)::i64)
@inline min(x::Float64, y::Float64) = @wrap __nv_fmin(x::double, y::double)::double
@inline min(x::Float32, y::Float32) = @wrap __nv_fminf(x::float, y::float)::float

@inline max(x::Int32, y::Int32) = @wrap __nv_max(x::i32, y::i32)::i32
@inline max(x::Int64, y::Int64) = @wrap __nv_llmax(x::i64, y::i64)::i64
@inline max(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umax(x::i32, y::i32)::i32)
@inline max(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmax(x::i64, y::i64)::i64)
@inline max(x::Float64, y::Float64) = @wrap __nv_fmax(x::double, y::double)::double
@inline max(x::Float32, y::Float32) = @wrap __nv_fmaxf(x::float, y::float)::float

@inline saturate(x::Float32) = @wrap __nv_saturatef(x::float)::float


## division and remainder

@inline mod(x::Float64, y::Float64) = @wrap __nv_fmod(x::double, y::double)::double
@inline mod(x::Float32, y::Float32) = @wrap __nv_fmodf(x::float, y::float)::float

@inline rem(x::Float64, y::Float64) = @wrap __nv_remainder(x::double, y::double)::double
@inline rem(x::Float32, y::Float32) = @wrap __nv_remainderf(x::float, y::float)::float


## gamma function

@inline lgamma(x::Float64) = @wrap __nv_lgamma(x::double)::double
@inline lgamma(x::Float32) = @wrap __nv_lgammaf(x::float)::float

@inline tgamma(x::Float64) = @wrap __nv_tgamma(x::double)::double
@inline tgamma(x::Float32) = @wrap __nv_tgammaf(x::float)::float


## Bessel

@inline j0(x::Float64) = @wrap __nv_j0(x::double)::double
@inline j0(x::Float32) = @wrap __nv_j0f(x::float)::float

@inline j1(x::Float64) = @wrap __nv_j1(x::double)::double
@inline j1(x::Float32) = @wrap __nv_j1f(x::float)::float

@inline jn(n::Int32, x::Float64) = @wrap __nv_jn(n::i32, x::double)::double
@inline jn(n::Int32, x::Float32) = @wrap __nv_jnf(n::i32, x::float)::float

@inline y0(x::Float64) = @wrap __nv_y0(x::double)::double
@inline y0(x::Float32) = @wrap __nv_y0f(x::float)::float

@inline y1(x::Float64) = @wrap __nv_y1(x::double)::double
@inline y1(x::Float32) = @wrap __nv_y1f(x::float)::float

@inline yn(n::Int32, x::Float64) = @wrap __nv_yn(n::i32, x::double)::double
@inline yn(n::Int32, x::Float32) = @wrap __nv_ynf(n::i32, x::float)::float



## distributions

@inline normcdf(x::Float64) = @wrap __nv_normcdf(x::double)::double
@inline normcdf(x::Float32) = @wrap __nv_normcdff(x::float)::float

@inline normcdfinv(x::Float64) = @wrap __nv_normcdfinv(x::double)::double
@inline normcdfinv(x::Float32) = @wrap __nv_normcdfinvf(x::float)::float



#
# Unsorted
#

@inline hypot(x::Float64, y::Float64) = @wrap __nv_hypot(x::double, y::double)::double
@inline hypot(x::Float32, y::Float32) = @wrap __nv_hypotf(x::float, y::float)::float

@inline fma(x::Float64, y::Float64, z::Float64) = @wrap __nv_fma(x::double, y::double, z::double)::double
@inline fma(x::Float32, y::Float32, z::Float32) = @wrap __nv_fmaf(x::float, y::float, z::float)::float

@inline sad(x::Int32, y::Int32, z::Int32) = @wrap __nv_sad(x::i32, y::i32, z::i32)::i32
@inline sad(x::UInt32, y::UInt32, z::UInt32) = convert(UInt32, @wrap __nv_usad(x::i32, y::i32, z::i32)::i32)

@inline dim(x::Float64, y::Float64) = @wrap __nv_fdim(x::double, y::double)::double
@inline dim(x::Float32, y::Float32) = @wrap __nv_fdimf(x::float, y::float)::float

@inline mul24(x::Int32, y::Int32) = @wrap __nv_mul24(x::i32, y::i32)::i32
@inline mul24(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umul24(x::i32, y::i32)::i32)

@inline mul64hi(x::Int64, y::Int64) = @wrap __nv_mul64hi(x::i64, y::i64)::i64
@inline mul64hi(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_umul64hi(x::i64, y::i64)::i64)
@inline mulhi(x::Int32, y::Int32) = @wrap __nv_mulhi(x::i32, y::i32)::i32
@inline mulhi(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umulhi(x::i32, y::i32)::i32)

@inline hadd(x::Int32, y::Int32) = @wrap __nv_hadd(x::i32, y::i32)::i32
@inline hadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_uhadd(x::i32, y::i32)::i32)

@inline rhadd(x::Int32, y::Int32) = @wrap __nv_rhadd(x::i32, y::i32)::i32
@inline rhadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_urhadd(x::i32, y::i32)::i32)

@inline scalbn(x::Float64, y::Int32) = @wrap __nv_scalbn(x::double, y::i32)::double
@inline scalbn(x::Float32, y::Int32) = @wrap __nv_scalbnf(x::float, y::i32)::float
