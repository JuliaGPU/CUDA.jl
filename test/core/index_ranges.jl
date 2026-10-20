using CUDACore, Test

# Defined after CUDACore's generators, as in a downstream package.
@inline test_guard(s, f::F, args::AT) where {F,AT<:Tuple} = CUDACore.check_index_ranges(s, f, AT, args)
@inline test_guard(s, f::F, TT, args) where F = CUDACore.check_index_ranges(s, f, TT, args)

struct FactArray
    ptr::Ptr{Float32}
    dims::Tuple{Int,Int}
    len::Int
    bytes::Int
end
function CUDACore.index_ranges(::Type{FactArray})
    ranges = ()
    for i in 1:2
        ranges = (ranges..., (:dims, i) => 0:typemax(Int32))
    end
    return (ranges..., (:len,) => 0:typemax(Int32))
end

struct BoundedValue{T,L,H}
    value::T
end
CUDACore.index_ranges(::Type{BoundedValue{T,L,H}}) where {T,L,H} =
    ((:value,) => L:H,)
const SignedIndex = BoundedValue{Int,Int(typemin(Int32)),Int(typemax(Int32))}

struct FactWrapper{T}
    value::T
    offset::Int
end

# Exercise range validation independently from the classifier's no-useful-facts result.
fits_ranges(x) = CUDACore.check_argument_ranges(x, CUDACore.index_ranges(typeof(x)))
@noinline classify_allocations(f::F, args::TT) where {F,TT<:Tuple} = @allocated CUDACore.argument_index_ranges(f, args)
@noinline guard_allocations(s, f::F, args::TT) where {F,TT<:Tuple} = @allocated test_guard(s, f, args)

struct SpecializationHolder{F}
    specialization::Tuple
    f::F
end
@inline held_guard(k::SpecializationHolder, args) = test_guard(k.specialization, k.f, args)
@noinline held_allocations(k, args) = @allocated held_guard(k, args)

struct TypedSpecializationHolder{F,TT}
    specialization::Tuple
    f::F
end
@inline typed_guard(k::TypedSpecializationHolder{F,TT}, args) where {F,TT} =
    test_guard(k.specialization, k.f, TT, args[])
@noinline typed_allocations(k, args) = @allocated typed_guard(k, args)

@testset "argument specialization" begin
    classify = CUDACore.argument_index_ranges
    guard = test_guard
    @test CUDACore.index_ranges(Int) == ()
    @test CUDACore.index_ranges(Base.OneTo{Int}) == ((:stop,) => 0:typemax(Int32),)
    @test classify(identity, (typemax(Int),)) === ()
    @test classify(identity, (typemax(UInt),)) === ()
    @test classify(identity, ()) === ()
    @test classify(identity, (Ptr{Int}(typemax(UInt)),)) === ()
    @test classify(identity, (Ref(typemax(Int)),)) === ()
    @test guard((), identity, (typemax(Int),))

    a = FactArray(C_NULL, (10, 20), 200, typemax(Int))
    s = classify(identity, (a,))
    @test s !== ()
    @test s isa Tuple
    @test guard(s, identity, (a,))
    @test !guard(s, identity, ())
    @test !guard(s, identity, (a, a))
    @test guard(s, sin, (a,)) # same asserted fields; function identity belongs to signature cache
    typed = classify(identity, (Float32, a))
    @test typed !== ()
    @test guard(typed, identity, (Float32, a))
    @test guard(typed, identity, (Float64, a)) # same range facts and physical layout
    @test !guard(typed, identity, Tuple{Type{Float32},FactArray}, (Float64, a))
    th = TypedSpecializationHolder{typeof(identity),Tuple{Type{Float32},FactArray}}(typed, identity)
    type_args = Ref{Tuple{DataType,FactArray}}((Float32, a))
    @test typed_guard(th, type_args)
    typed_allocations(th, type_args)
    @test typed_allocations(th, type_args) == 0
    type_args[] = (Float64, a)
    @test !typed_guard(th, type_args)
    @test typed_allocations(th, type_args) == 0
    @test classify(identity, (FactArray(C_NULL, (5, 6), 30, 120),)) === s
    @test classify(identity, (FactArray(C_NULL, (-1, 20), 200, 0),)) === ()
    @test classify(identity, (FactWrapper(a, typemax(Int)),)) !== ()
    @test CUDACore.index_ranges(typeof(FactWrapper(a, -1))) ==
          ((:value, :dims, 1) => 0:typemax(Int32),
           (:value, :dims, 2) => 0:typemax(Int32),
           (:value, :len) => 0:typemax(Int32))

    if sizeof(Int) == 8
        huge = Int(1) << 40
        @test classify(identity, (SignedIndex(huge),)) === ()
        @test classify(identity, (Base.OneTo(huge),)) === ()
        @test classify(identity, (CartesianIndices((10, huge)),)) === ()
        @test classify(identity, (CartesianIndices((10, 20)),)) !== ()
        @test classify(identity, (-huge:10,)) === ()
        @test classify(identity, (-10:10,)) !== ()
        @test classify(identity, (-10:2:10,)) !== ()
        @test classify(identity, (0:huge:huge,)) === ()
        @test classify(identity, (CartesianIndex(-10, 20),)) !== ()
        @test classify(identity, (CartesianIndex(huge, 20),)) === ()
        @test classify(identity, (FactArray(C_NULL, (10, huge), 200, 0),)) === ()
        capture(x) = () -> x
        @test classify(capture(a), ()) !== ()
        @test classify(capture(SignedIndex(huge)), ()) === ()
        captured = classify(capture(SignedIndex(1)), ())
        @test !guard(captured, capture(SignedIndex(huge)), ())
        # All-or-nothing classification: no partial combinations of facts.
        @test classify(identity, (a, Base.OneTo(huge))) === ()
        @test classify(identity, (a, Base.OneTo(1))) !== ()
        # Unrelated scalar payloads do not select or disable a specialization.
        @test classify(identity, (a, huge)) !== ()
    end
    many = ntuple(i -> SignedIndex(i), 100)
    smany = classify(identity, many)
    @test smany !== ()
    @test @inferred(guard(smany, identity, many))
    # The candidate tuple for this many arguments is too large to inline as a constant.
    # It is returned from a constant-folded call, which coverage instrumentation defeats.
    coverage = Base.JLOptions().code_coverage != 0
    classify_allocations(identity, many)
    guard_allocations(smany, identity, many)
    @test classify_allocations(identity, many) == 0 skip=coverage
    @test guard_allocations(smany, identity, many) == 0 skip=coverage
    holder = SpecializationHolder(smany, identity)
    held_allocations(holder, many)
    @test held_allocations(holder, many) == 0 skip=coverage
    classify_allocations(identity, (a,))
    @test classify_allocations(identity, (a,)) == 0
    # A full-domain candidate adds no information and must not create another variant.
    full = BoundedValue{Int,typemin(Int),typemax(Int)}(1)
    @test classify(identity, (full,)) === ()
end

@testset "integer domains" begin
    fits = fits_ranges
    # Exhaustive checks catch modular-subtraction errors on either side of zero.
    for x in typemin(Int8):typemax(Int8)
        @test fits(BoundedValue{Int8,-100,87}(x)) == (-100 <= x <= 87)
    end
    for x in typemin(UInt8):typemax(UInt8)
        @test fits(BoundedValue{UInt8,17,211}(x)) == (17 <= x <= 211)
    end
    for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128)
        lo, hi = typemin(T), typemax(T)
        @test fits(BoundedValue{T,lo,hi}(lo))
        @test fits(BoundedValue{T,lo,hi}(hi))
        @test fits(BoundedValue{T,lo,lo}(lo))
        @test !fits(BoundedValue{T,lo,lo}(hi))
        @test fits(BoundedValue{T,hi,hi}(hi))
        @test !fits(BoundedValue{T,hi,hi}(lo))
        # The empty path describes a scalar itself, without redefining a built-in's policy.
        ranges = (() => lo:hi,)
        @test CUDACore.check_argument_ranges(lo, ranges)
        @test CUDACore.check_argument_ranges(hi, ranges)
    end
    @test fits(BoundedValue{Bool,false,true}(false))
    @test fits(BoundedValue{Bool,false,true}(true))
    @test fits(BoundedValue{Bool,true,true}(true))
    @test !fits(BoundedValue{Bool,true,true}(false))
end

