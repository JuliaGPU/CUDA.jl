using CUDACore: LLVM
using .LLVM: Context, supports_typed_pointers, run!, verify
using KernelAbstractions

# the narrowing pass itself, on hand-written IR
@testset "IR" begin

# run the narrowing pass on a module, and print it for FileCheck. the tests only use
# pointers to floats, which are spelled `ptr`, or `float*` with typed pointers.
function narrow(ir::String; kwargs...)
    Context() do ctx
        if supports_typed_pointers(ctx)
            ir = replace(ir, r"\bptr\b" => "float*")
        end
        mod = parse(LLVM.Module, ir)
        run!(CUDACore.IndexNarrowing.NarrowIndicesPass(; kwargs...), mod)
        verify(mod)
        print(string(mod))
    end
end

@testset "constant bounds" begin
    # indices bounded by assumptions are recomputed in 32 bits
    ir = """
        declare void @llvm.assume(i1)
        define void @f(ptr %p, i64 %i, i64 %j) {
          %ci = icmp ult i64 %i, 1024
          call void @llvm.assume(i1 %ci)
          %cj = icmp ult i64 %j, 1024
          call void @llvm.assume(i1 %cj)
          %m = mul i64 %i, 1024
          %idx = add i64 %m, %j
          %g = getelementptr float, ptr %p, i64 %idx
          store float 0.0, ptr %g
          ret void
        }"""
    @test @filecheck begin
        @check_label "define void @f"
        @check_dag "[[I:%.*]] = trunc i64 %i to i32"
        @check_dag "[[J:%.*]] = trunc i64 %j to i32"
        @check "[[M:%.*]] = mul nuw nsw i32 [[I]], 1024"
        @check "[[IDX:%.*]] = add nuw nsw i32 [[M]], [[J]]"
        @check "[[W:%.*]] = zext {{(nneg )?}}i32 [[IDX]] to i64"
        @check "getelementptr float, {{ptr|float\\*}} %p, i64 [[W]]"
        @check_not "mul i64"
        narrow(ir)
    end
end

@testset "non-constant bounds" begin
    # LLVM's analyses don't use assumptions that bound a value by another value, like the
    # bounds check of an array access (`i < len`), with the length bounded separately
    ir = """
        declare void @llvm.assume(i1)
        define float @f(ptr %p, i64 %len, i64 %i) {
          %clen = icmp ule i64 %len, 2147483647
          call void @llvm.assume(i1 %clen)
          %ci = icmp ult i64 %i, %len
          call void @llvm.assume(i1 %ci)
          %idx = add i64 %i, 1
          %g = getelementptr float, ptr %p, i64 %idx
          %v = load float, ptr %g
          ret float %v
        }"""
    @test @filecheck begin
        @check_label "define float @f"
        @check "[[I:%.*]] = trunc i64 %i to i32"
        @check "[[IDX:%.*]] = add nuw nsw i32 [[I]], 1"
        @check "[[W:%.*]] = zext {{(nneg )?}}i32 [[IDX]] to i64"
        @check "getelementptr float, {{ptr|float\\*}} %p, i64 [[W]]"
        narrow(ir)
    end
end

@testset "no facts" begin
    ir = """
        define float @f(ptr %p, i64 %i, i64 %j) {
          %m = mul i64 %i, 1024
          %idx = add i64 %m, %j
          %g = getelementptr float, ptr %p, i64 %idx
          %v = load float, ptr %g
          ret float %v
        }"""
    @test @filecheck begin
        @check_label "define float @f"
        @check "%idx = add i64 %m, %j"
        @check "getelementptr float, {{ptr|float\\*}} %p, i64 %idx"
        @check_not "i32"
        narrow(ir)
    end
end

@testset "loops" begin
    # induction variables are left wide, so that loop strength reduction can optimize them
    ir = """
        define void @f(ptr %p) {
        entry:
          br label %loop
        loop:
          %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
          %idx = mul i64 %i, 3
          %g = getelementptr float, ptr %p, i64 %idx
          store float 0.0, ptr %g
          %i.next = add nuw nsw i64 %i, 1
          %c = icmp ult i64 %i.next, 100
          br i1 %c, label %loop, label %exit
        exit:
          ret void
        }"""
    @test @filecheck begin
        @check_label "define void @f"
        @check "%idx = mul i64 %i, 3"
        @check "getelementptr float, {{ptr|float\\*}} %p, i64 %idx"
        narrow(ir)
    end
    @test @filecheck begin
        @check_label "define void @f"
        @check "mul nuw nsw i32"
        narrow(ir; skip_addrec=false)
    end
end

@testset "compares" begin
    ir = """
        declare void @llvm.assume(i1)
        define i1 @f(i64 %i, i64 %n) {
          %ci = icmp ult i64 %i, 1024
          call void @llvm.assume(i1 %ci)
          %cn = icmp ult i64 %n, 1024
          call void @llvm.assume(i1 %cn)
          %j = add i64 %i, 1
          %c = icmp ult i64 %j, %n
          ret i1 %c
        }"""
    @test @filecheck begin
        @check_label "define i1 @f"
        @check "icmp ult i32"
        @check_not "icmp ult i64 %j"
        narrow(ir)
    end
    @test @filecheck begin
        @check_label "define i1 @f"
        @check "icmp ult i64 %j, %n"
        narrow(ir; compares=false)
    end
end

@testset "poison" begin
    # `assume(idx >= 0)` would turn a poison index into undefined behavior, so it is only
    # emitted where a poison index is undefined behavior already (here: it is loaded from)
    ir = """
        declare void @llvm.assume(i1)
        define float @f(ptr %p, i64 %i, i1 %c) {
          %ci = icmp ult i64 %i, 1024
          call void @llvm.assume(i1 %ci)
          %idx = add i64 %i, 1
          %g = getelementptr float, ptr %p, i64 %idx
          %v = load float, ptr %g
          %q = select i1 %c, ptr %g, ptr %p
          ret float %v
        }
        define ptr @g(ptr %p, i64 %i) {
          %ci = icmp ult i64 %i, 1024
          call void @llvm.assume(i1 %ci)
          %idx = add i64 %i, 1
          %g = getelementptr float, ptr %p, i64 %idx
          ret ptr %g
        }"""
    @test @filecheck begin
        @check_label "define float @f"
        @check "icmp sge i32"
        @check "call void @llvm.assume"
        @check_label "define {{.*}}@g("
        @check_not "icmp sge i32"
        @check "zext {{(nneg )?}}i32"
        narrow(ir)
    end
end

@testset "boundaries" begin
    # zext(trunc(x)) == x for x in [0, 2^32), with nneg only for [0, 2^31); sext(trunc(x)) == x
    # for x in [-2^31, 2^31). one past each boundary, the index must stay wide.
    gep(bound, offset; pred="ult", lower=nothing) = """
        declare void @llvm.assume(i1)
        define float @f(ptr %p, i64 %x) {
          %c = icmp $pred i64 %x, $bound
          call void @llvm.assume(i1 %c)
          $(lower === nothing ? "" : "%l = icmp sge i64 %x, $lower
          call void @llvm.assume(i1 %l)")
          %a = add i64 %x, $offset
          %g = getelementptr float, ptr %p, i64 %a
          %v = load float, ptr %g
          ret float %v
        }"""

    # idx in [1, 2^31 - 1]
    @test @filecheck begin
        @check "[[A:%.*]] = add nuw nsw i32 {{%.*}}, 1"
        @check "zext {{(nneg )?}}i32 [[A]] to i64"
        narrow(gep(2147483647, 1))
    end
    # idx in [1, 2^31]: no nneg, no nsw, and no nonneg assumption
    @test @filecheck begin
        @check "[[A:%.*]] = add nuw i32 {{%.*}}, 1"
        @check_not "icmp sge i32"
        @check "zext i32 [[A]] to i64"
        narrow(gep(2147483648, 1))
    end
    # idx in [1, 2^32 - 1]
    @test @filecheck begin
        @check "zext i32"
        narrow(gep(4294967295, 1))
    end
    # idx in [1, 2^32]: the truncation of 2^32 is 0
    @test @filecheck begin
        @check_not "trunc"
        @check "getelementptr float, {{ptr|float\\*}} %p, i64 %a"
        narrow(gep(4294967296, 1))
    end
    # idx in [-2^31, -2]: sext, nsw but no nuw
    @test @filecheck begin
        @check "[[A:%.*]] = add nsw i32 {{%.*}}, -1"
        @check "sext i32 [[A]] to i64"
        narrow(gep(0, -1; pred="slt", lower=-2147483647))
    end
    # idx in [-2^31 - 1, -2]
    @test @filecheck begin
        @check_not "trunc"
        narrow(gep(0, -1; pred="slt", lower=-2147483648))
    end
end

@testset "wrapping" begin
    # the narrow expression is exact modulo 2^32, even if intermediate values wrap; flags of
    # the wide operations are never copied, only derived from the ranges
    ir = """
        declare void @llvm.assume(i1)
        define float @f(ptr %p, i64 %i, i64 %j, i64 %d) {
          %ci = icmp ult i64 %i, 65536
          call void @llvm.assume(i1 %ci)
          %cj = icmp ult i64 %j, 65536
          call void @llvm.assume(i1 %cj)
          %cd = icmp ult i64 %d, 65536
          call void @llvm.assume(i1 %cd)
          %a = mul nuw nsw i64 %i, %d
          %b = mul nuw nsw i64 %j, %d
          %idx = sub nuw nsw i64 %a, %b
          %c = icmp ult i64 %idx, 1024
          call void @llvm.assume(i1 %c)
          %g = getelementptr float, ptr %p, i64 %idx
          %v = load float, ptr %g
          ret float %v
        }"""
    @test @filecheck begin
        # the products fit in an unsigned i32 (nuw), but not in a signed one (no nsw), and
        # so does their difference, which is bounded by the assumption
        @check "[[A:%.*]] = mul nuw i32"
        @check "[[B:%.*]] = mul nuw i32"
        @check "sub nuw i32 [[A]], [[B]]"
        narrow(ir)
    end
end

end

function copy3d!(B, A)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if i <= size(A,1) && j <= size(A,2) && k <= size(A,3)
        @inbounds B[i,j,k] = A[i,j,k]
    end
    return
end

@testset "kernels" begin
    tt = Tuple{CuDeviceArray{Float32,3,AS.Global}, CuDeviceArray{Float32,3,AS.Global}}

    # with facts about the extents of the arguments, indices are computed in 32 bits
    @test @filecheck begin
        @check "call void @llvm.assume"
        @check "[[IDX:%nidx[0-9]*]] = zext nneg i32"
        @check "getelementptr float, {{.*}} i64 [[IDX]]"
        CUDA.code_llvm(copy3d!, tt; kernel=true, index_facts=true)
    end
    @test @filecheck begin
        @check_not "%nidx"
        CUDA.code_llvm(copy3d!, tt; kernel=true, index_facts=false)
    end

    # `@cuda` derives the facts from the arguments
    A = CUDA.rand(Float32, 7, 5, 3)
    B = similar(A)
    k = @cuda threads=(7, 5, 3) copy3d!(B, A)
    @test k.index_facts
    @test Array(B) == Array(A)
end

# with facts, `last(r) > typemax(Int32)` is folded to `false`
function exceeds(out, r)
    out[1] = last(r) > typemax(Int32)
    return
end

@kernel function ka_exceeds(out, r)
    out[1] = last(r) > typemax(Int32)
end

world_helper() = 1
function world_kernel(out, r)
    out[1] = world_helper() + (last(r) > typemax(Int32))
    return
end

@testset "launch" begin
    small = Base.OneTo(10)
    huge = Base.OneTo(2^40)
    out = CuArray([false])

    k = @cuda exceeds(out, huge)
    @test !k.index_facts
    @test Array(out) == [true]

    # compiled with facts, and launched with arguments that don't fit them
    k = @cuda launch=false exceeds(out, small)
    @test k.index_facts
    k(out, small)
    @test Array(out) == [false]
    k(out, huge)
    @test Array(out) == [true]

    # the facts belong to the compiled function, also when reconstructing the kernel
    k′ = typeof(k)(k.f, k.fun, k.state)
    @test k′.index_facts
    k′(out, huge)
    @test Array(out) == [true]

    # captured by the kernel function, and replaced after compilation
    captured(r) = out -> exceeds(out, r)
    k = @cuda launch=false captured(small)(out)
    @test k.index_facts
    k′ = typeof(k)(captured(huge), k.fun, k.state)
    k′(out)
    @test Array(out) == [true]

    # captured by the kernel function
    k = @cuda (out -> exceeds(out, huge))(out)
    @test !k.index_facts

    # kernels compiled from types assume that the arguments fit, and check them on launch
    fill!(out, false)
    ka_exceeds(CUDABackend())(out, huge; ndrange=1)
    @test Array(out) == [true]
    ka_exceeds(CUDABackend())(out, small; ndrange=1)
    @test Array(out) == [false]

    # the fallback is compiled in the world of the original kernel
    out = CuArray([0])
    k = @cuda launch=false world_kernel(out, small)
    @eval world_helper() = 2
    Base.invokelatest(k, out, huge)
    @test Array(out) == [2]
end
