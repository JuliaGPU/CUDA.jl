# narrowing of 64-bit index arithmetic to 32 bits
#
# GPUs execute most 64-bit integer operations as pairs of 32-bit instructions, so index
# computations that are known to fit in 32 bits are cheaper to compute in 32 bits. This pass
# recomputes the i64 indices of GEPs (and operands of i64 comparisons) that are proven to fit
# in 32 bits at their use, as an i32 expression, and feeds the use with a single extension of
# the narrow value. As add, sub, mul, shl, and, or and xor commute with truncation (they are
# exact modulo 2^32), the narrow expression computes the same low 32 bits as the wide one
# regardless of intermediate overflow; the range of the root then guarantees that extending
# them gives back the wide value.
#
# The ranges come from LLVM's analyses (lazy value info, value tracking, scalar evolution),
# extended with assumptions that bound a value by another value (like `i <= length(A)`),
# which LLVM's analyses only use with constant bounds. Narrowed expressions are memoized, so
# that values used by multiple roots are narrowed once; wide values that are used elsewhere
# are truncated instead. Loop-variant affine indices are left alone, so that loop strength
# reduction can still optimize them in the wide type.
#
# The pass is target-independent. It lives here until it has proven itself, and should then
# move to GPUCompiler.jl.

module IndexNarrowing

using LLVM, LLVM.IR, LLVM.Build, LLVM.Passes, LLVM.Analysis

export NarrowIndicesPass, narrow_indices!


## options

# The behavior of the pass (with the defaults of the reference implementation):
# - `skip_addrec`: leave loop-variant affine (add-recurrence) indices wide;
# - `compares`: also narrow i64 integer comparisons;
# - `flags`, `range_flags`: set nuw/nsw on narrow operations when ranges prove them (with
#   `range_flags`, using signed and unsigned ranges, otherwise only for [0, 2^31));
# - `chains`: narrow split GEP chains `gep(gep(p, a), b)` whose combined index is bounded
#   by an assumption;
# - `max_depth`: the maximal depth of the narrowed expressions;
# - `assume_dead`: users that only (transitively) feed assumptions don't keep wide values
#   alive;
# - `assume_nonneg`: emit `assume(idx >= 0)` for narrowed nonnegative indices (where a
#   poison index is undefined behavior already);
# - `mulwide`: rewrite operands of 64-bit multiplications outside of loops that fit in 32
#   bits as `zext(trunc(x))`, so that back-ends can select a widening multiplication;
# - `deep_range`: propagate ranges through add/sub/mul/shl.
Base.@kwdef struct NarrowingOptions
    skip_addrec::Bool = true
    compares::Bool = true
    flags::Bool = true
    range_flags::Bool = true
    chains::Bool = true
    max_depth::Int = 12
    assume_dead::Bool = true
    assume_nonneg::Bool = true
    mulwide::Bool = true
    deep_range::Bool = true
end

# parse the options of a pass in a pipeline, like LLVM's passes: `narrow-idx<opts>`, with
# `;`-separated options `name`, `no-name` or `name=N` (e.g. `no-compares;max-depth=8`),
# where the names are those of the fields of `T` with dashes instead of underscores
function parse_options(::Type{T}, str::AbstractString; defaults...) where {T}
    opts = Dict{Symbol,Any}(defaults)
    for opt in split(str, ';'; keepempty=false)
        key, value = '=' in opt ? split(opt, '='; limit=2) :
                     startswith(opt, "no-") ? (opt[4:end], false) : (opt, true)
        field = Symbol(replace(key, '-' => '_'))
        F = field in fieldnames(T) ? fieldtype(T, field) :
            throw(ArgumentError("unknown option '$key'"))
        opts[field] = value isa AbstractString ? parse(F, value) : convert(F, value)
    end
    return T(; opts...)
end


## analysis

mutable struct Narrower
    const fun::LLVM.Function
    const am::FunctionAnalysisManager
    const opts::NarrowingOptions
    const domtree::DomTree
    const assumptions::AssumptionCache
    const lvi::LazyValueInfo
    const scev::ScalarEvolution
    const loops::LoopInfo
    const datalayout::LLVM.DataLayout
    const builder::IRBuilder
    const i32::LLVMType
    const i64::LLVMType
    # wide value -> narrow i32 value
    const narrowed::Dict{Value,Value}
    changed::Bool
end

function Narrower(f::LLVM.Function, am::FunctionAnalysisManager, opts::NarrowingOptions,
                  builder::IRBuilder)
    Narrower(f, am, opts, am[DomTree], am[AssumptionCache], am[LazyValueInfo],
             am[ScalarEvolution], am[LoopInfo], f.parent.datalayout, builder,
             LLVM.Int32Type(), LLVM.Int64Type(), Dict{Value,Value}(), false)
end

is_i64(v::Value) = v.value_type isa LLVM.IntegerType && v.value_type.width == 64

isassume(v::Value) = v isa CallInst && LLVM.isintrinsic(v.called_operand, Intrinsic("llvm.assume"))

# LLVM's analyses only use assumptions `icmp pred v, C` with a constant `C`; also use
# `icmp pred v, w` (e.g. `i <= length(A)`), with the range of `w` at the assumption.
function assume_range(n::Narrower, v::Value, at::Instruction, depth::Int)
    nbits = v.value_type.width
    r = ConstantRange(nbits)
    depth > 3 && return r
    for entry in n.assumptions[v]
        entry.bundle_index === nothing || continue
        assume = entry.assume
        is_valid_assume_for_context(assume, at; domtree=n.domtree) || continue
        cond = first(assume.operands)
        cond isa ICmpInst || continue
        pred = cond.predicate
        lhs, rhs = cond.operands[1], cond.operands[2]
        if lhs == v
            other = rhs
        elseif rhs == v
            other = lhs
            pred = swapped_predicate(pred)
        else
            continue
        end
        signed = is_signed_predicate(pred)
        bound = value_range(n, other, assume, signed, depth + 1)
        isempty(bound) && continue
        r = intersect_with(r, allowed_icmp_region(pred, bound);
                           prefer=signed ? :signed : :unsigned)
    end
    return r
end

# the range of an integer value at an instruction
function value_range(n::Narrower, v::Value, at::Instruction, signed::Bool, depth::Int=0)
    prefer = signed ? :signed : :unsigned
    r = assume_range(n, v, at, depth)
    r = intersect_with(r, ConstantRange(n.lvi, v; at); prefer)
    r = intersect_with(r, ConstantRange(v; signed, at, assumptions=n.assumptions,
                                        domtree=n.domtree); prefer)
    known = KnownBits(v; at, assumptions=n.assumptions, domtree=n.domtree)
    r = intersect_with(r, ConstantRange(known; signed); prefer)
    if n.opts.deep_range && depth < 4 && v isa Instruction &&
       v.opcode in (LLVM.Opcode.Add, LLVM.Opcode.Sub, LLVM.Opcode.Mul, LLVM.Opcode.Shl)
        lhs, rhs = v.operands[1], v.operands[2]
        a = value_range(n, lhs, at, signed, depth + 1)
        b = rhs isa ConstantInt ? ConstantRange(rhs) : value_range(n, rhs, at, signed, depth + 1)
        # multiplications ignore the flags, like the reference implementation
        nowrap = v.opcode == LLVM.Opcode.Mul ? (; nuw=false, nsw=false) :
                 (; nuw=v.nuw, nsw=v.nsw)
        r = intersect_with(r, binary_op(v.opcode, a, b; nowrap...); prefer)
    end
    # every i64 value is SCEVable
    r = intersect_with(r, ConstantRange(n.scev, n.scev[v]; signed); prefer)
    return r
end

# empty ranges (for unreachable code) don't fit anything
fits_u32(n, v, at) = (r = value_range(n, v, at, false); !isempty(r) && r.unsigned_max <= typemax(UInt32))
fits_nonneg31(n, v, at) = (r = value_range(n, v, at, false); !isempty(r) && r.unsigned_max <= typemax(Int32))
function fits_s32(n, v, at)
    r = value_range(n, v, at, true)
    return !isempty(r) && r.signed_min >= typemin(Int32) && r.signed_max <= typemax(Int32)
end

const arith_opcodes = (LLVM.Opcode.Add, LLVM.Opcode.Sub, LLVM.Opcode.Mul, LLVM.Opcode.Shl,
                       LLVM.Opcode.And, LLVM.Opcode.Or, LLVM.Opcode.Xor)

# is the expression rooted at v worth narrowing (does it contain arithmetic)?
function has_arith(n::Narrower, v::Value, depth::Int=0)
    v isa Instruction && depth <= n.opts.max_depth || return false
    v.opcode in arith_opcodes && return true
    if v isa SelectInst
        return has_arith(n, v.operands[2], depth + 1) || has_arith(n, v.operands[3], depth + 1)
    end
    return false
end

# opcodes of instructions that are recomputed in i32 rather than truncated
function is_narrowable(inst::Instruction)
    op = inst.opcode
    op in (LLVM.Opcode.Add, LLVM.Opcode.Sub, LLVM.Opcode.Mul, LLVM.Opcode.And,
           LLVM.Opcode.Or, LLVM.Opcode.Xor, LLVM.Opcode.Select) && return true
    if op == LLVM.Opcode.Shl
        amount = inst.operands[2]
        return amount isa ConstantInt && convert(UInt64, amount) < 32
    end
    return false
end

function collect_nodes!(n::Narrower, v::Value, nodes::Set{Instruction}, depth::Int=0)
    v isa Instruction && is_i64(v) && depth <= n.opts.max_depth && is_narrowable(v) ||
        return
    v in nodes && return
    push!(nodes, v)
    ops = collect(v.operands)
    for op in (v isa SelectInst ? ops[2:end] : ops)
        collect_nodes!(n, op, nodes, depth + 1)
    end
end

# whether an instruction (transitively) only feeds assumptions, so that its value is dead
# once the back end drops the assumptions
function only_feeds_assumes(n::Narrower, inst::Instruction, depth::Int=0)
    if n.opts.assume_dead
        (depth > 8 || may_have_side_effects(inst) || inst isa PHIInst ||
         isterminator(inst)) && return false
        users = collect(inst.users)
        isempty(users) && return false
        for user in users
            isassume(user) && continue
            user isa Instruction && only_feeds_assumes(n, user, depth + 1) || return false
        end
        return true
    else
        inst isa ICmpInst || return false
        return all(isassume, inst.users)
    end
end

function is_loop_affine(n::Narrower, v::Value)
    is_i64(v) || return false
    return contains_scev(n.scev[v], SCEVAddRecExpr)
end


## transformation

# where the narrow copy of a leaf goes: right after its definition (in the entry block for
# arguments). returns `nothing` for values without such a point (e.g. callbr results, or an
# invoke that doesn't dominate its normal destination), whose leaves are truncated right
# before the narrowed user instead.
function insertion_point_after(n::Narrower, v::Value)
    v isa Argument && return LLVM.after_phis(v.parent.entry)
    v isa Instruction || return nothing
    ip = LLVM.after_def(v)
    ip === nothing && return nothing
    # the normal destination of an invoke may have other predecessors
    v isa InvokeInst && !dominates(n.domtree, v, Instruction(ip.anchor)) && return nothing
    return ip
end

# the i32 value equal to trunc(v), available at `use_at` (the insertion point of the user)
function narrow!(n::Narrower, v::Value, nodes::Set{Instruction}, use_at, depth::Int=0)
    b = n.builder
    if v isa Constant
        # poison, undef, or a constant expression (e.g. ptrtoint of a global)
        folded = constant_fold_cast(LLVM.Opcode.Trunc, v, n.i32, n.datalayout)
        folded === nothing || return folded
        position!(b, LLVM.after_phis(n.fun.entry))
        return trunc!(b, v, n.i32, "c.t32")
    end
    haskey(n.narrowed, v) && return n.narrowed[v]

    narrow = nothing
    if v isa Instruction && depth <= n.opts.max_depth &&
       (v in nodes || v isa ZExtInst || v isa SExtInst)
        op = v.opcode
        if op == LLVM.Opcode.ZExt || op == LLVM.Opcode.SExt
            src = v.operands[1]
            srcbits = src.value_type.width
            if srcbits == 32
                narrow = src
            elseif srcbits < 32
                position!(b, LLVM.after(v))
                narrow = op == LLVM.Opcode.ZExt ? zext!(b, src, n.i32, "nzx") :
                                                   sext!(b, src, n.i32, "nsx")
            end
        elseif op in (LLVM.Opcode.Add, LLVM.Opcode.Sub, LLVM.Opcode.Mul, LLVM.Opcode.And,
                      LLVM.Opcode.Or, LLVM.Opcode.Xor)
            # take the insertion point first: a leaf truncated at its use goes right after
            # `v` as well, and the narrow operation must come after it
            ip = LLVM.after(v)
            lhs = narrow!(n, v.operands[1], nodes, ip, depth + 1)
            rhs = narrow!(n, v.operands[2], nodes, ip, depth + 1)
            position!(b, ip)
            narrow = binop!(b, op, lhs, rhs, v.name * ".n32")
            # the narrow operation may have been folded to a constant
            if n.opts.flags && narrow isa Instruction &&
               op in (LLVM.Opcode.Add, LLVM.Opcode.Sub, LLVM.Opcode.Mul)
                set_narrow_flags!(n, narrow, v)
            end
        elseif op == LLVM.Opcode.Shl
            amount = v.operands[2]
            if amount isa ConstantInt && convert(UInt64, amount) < 32
                ip = LLVM.after(v)
                lhs = narrow!(n, v.operands[1], nodes, ip, depth + 1)
                position!(b, ip)
                narrow = shl!(b, lhs, ConstantInt(n.i32, convert(UInt64, amount)),
                              v.name * ".n32")
            end
        elseif op == LLVM.Opcode.Select
            ip = LLVM.after(v)
            lhs = narrow!(n, v.operands[2], nodes, ip, depth + 1)
            rhs = narrow!(n, v.operands[3], nodes, ip, depth + 1)
            position!(b, ip)
            narrow = select!(b, v.operands[1], lhs, rhs, v.name * ".n32")
        end
    end
    if narrow === nothing
        # a leaf: truncate the wide value (free on GPUs: the low half of the register)
        ip = insertion_point_after(n, v)
        if ip === nothing
            # no single point after the definition: truncate at this use only
            position!(b, use_at)
            return trunc!(b, v, n.i32, v.name * ".t32")
        end
        position!(b, ip)
        narrow = trunc!(b, v, n.i32, v.name * ".t32")
    end
    n.narrowed[v] = narrow
    return narrow
end

# the narrow operation doesn't wrap if its operands and result all fit (then it computes
# the same value as the wide one). wide flags are never copied.
function set_narrow_flags!(n::Narrower, narrow::Instruction, wide::Instruction)
    lhs, rhs = wide.operands[1], wide.operands[2]
    if n.opts.range_flags
        if fits_s32(n, wide, wide) && fits_s32(n, lhs, wide) && fits_s32(n, rhs, wide)
            narrow.nsw = true
        end
        if fits_u32(n, wide, wide) && fits_u32(n, lhs, wide) && fits_u32(n, rhs, wide)
            narrow.nuw = true
        end
    elseif fits_nonneg31(n, wide, wide) && fits_nonneg31(n, lhs, wide) &&
           fits_nonneg31(n, rhs, wide)
        narrow.nsw = true
        wide.opcode == LLVM.Opcode.Sub || (narrow.nuw = true)
    end
end

# assume(narrow >= 0) turns a poison value into immediate UB, so only emit it where the
# program is already undefined if the wide value is poison (e.g., because it feeds a load
# or a store). a narrow value is only poison if the wide value it replaces is.
function emit_nonneg!(n::Narrower, narrow::Value, wide::Vector{<:Instruction})
    narrow isa Constant && return
    any(program_undefined_if_poison, wide) || return
    b = n.builder
    cond = icmp!(b, LLVM.IntPredicate.SGE, narrow, ConstantInt(n.i32, 0), "nidx.nn")
    push!(n.assumptions, assume!(b, cond))
end

# record a rewrite, and report it as an optimization remark
function rewritten!(n::Narrower, inst::Instruction, message::String)
    remark!(n.am, :passed, "narrow-idx", "Narrowed", inst, message)
    n.changed = true
end

struct GEPRoot
    gep::GetElementPtrInst
    index::Int          # operand index
    zext::Bool          # extend with zext (otherwise sext)
    nonneg::Bool        # fits in [0, 2^31)
end

struct ChainRoot
    outer::GetElementPtrInst
    inner::GetElementPtrInst
    a::Value
    b::Value
end

function narrow_indices!(n::Narrower)
    opts = n.opts
    f = n.fun

    # find the candidate roots: i64 GEP indices and comparisons
    geps = Tuple{GetElementPtrInst,Int}[]
    cmps = ICmpInst[]
    for bb in f.blocks, inst in bb.instructions
        if inst isa GetElementPtrInst
            for (i, idx) in enumerate(inst.operands)
                i == 1 && continue
                is_i64(idx) && !(idx isa Constant) && push!(geps, (inst, i))
            end
        elseif inst isa ICmpInst && opts.compares
            is_i64(inst.operands[1]) && !only_feeds_assumes(n, inst) && push!(cmps, inst)
        end
    end

    # phase 1: roots whose value provably fits in 32 bits at the use
    groots = GEPRoot[]
    for (gep, i) in geps
        idx = gep.operands[i]
        has_arith(n, idx) || continue
        opts.skip_addrec && is_loop_affine(n, idx) && continue
        z = fits_u32(n, idx, gep)
        s = !z && fits_s32(n, idx, gep)
        z || s || continue
        push!(groots, GEPRoot(gep, i, z, fits_nonneg31(n, idx, gep)))
    end
    croots = ICmpInst[]
    for cmp in cmps
        a, b = cmp.operands[1], cmp.operands[2]
        ok = if is_signed_predicate(cmp.predicate)
            fits_s32(n, a, cmp) && fits_s32(n, b, cmp)
        else
            fits_u32(n, a, cmp) && fits_u32(n, b, cmp)
        end
        ok || continue
        opts.skip_addrec && (is_loop_affine(n, a) || is_loop_affine(n, b)) && continue
        push!(croots, cmp)
    end

    # phase 1b: InstCombine splits `gep p, (a + b)` into `gep (gep p, a), b`, after which
    # neither part is bounded on its own. match the combined index a + b against values
    # that are bounded by assumptions (modulo a constant offset), using scalar evolution.
    kroots = ChainRoot[]
    if opts.chains
        bounded = Tuple{Value,CallInst}[]
        for assume in n.assumptions
            cond = first(assume.operands)
            cond isa ICmpInst && is_i64(cond.operands[1]) || continue
            for op in cond.operands
                op isa Constant || push!(bounded, (op, assume))
            end
        end
        isroot = Set(r.gep for r in groots)
        for bb in f.blocks, outer in bb.instructions
            outer isa GetElementPtrInst && length(outer.operands) == 2 && !(outer in isroot) ||
                continue
            inner = outer.operands[1]
            inner isa GetElementPtrInst && length(inner.operands) == 2 &&
                inner.source_element_type == outer.source_element_type || continue
            a, b = inner.operands[2], outer.operands[2]
            is_i64(a) && is_i64(b) && !(a isa Constant) && !(b isa Constant) || continue
            sum = scev_add(n.scev, n.scev[a], n.scev[b])
            opts.skip_addrec && contains_scev(sum, SCEVAddRecExpr) && continue
            ok = false
            for (v, assume) in bounded
                is_valid_assume_for_context(assume, outer; domtree=n.domtree) || continue
                d = scev_minus(n.scev, sum, n.scev[v])
                d isa SCEVConstant || continue
                rv = value_range(n, v, outer, false)
                rs = rv + ConstantRange(d.value)
                if !isfullset(rv) && !isempty(rv) && !iswrappedset(rv) && !iswrappedset(rs) &&
                   rs.unsigned_max <= typemax(Int32) && rv.unsigned_max <= typemax(Int32)
                    ok = true
                    break
                end
            end
            ok && push!(kroots, ChainRoot(outer, inner, a, b))
        end
    end

    # phase 2: candidate nodes to recompute in i32
    nodes = Set{Instruction}()
    for r in groots
        collect_nodes!(n, r.gep.operands[r.index], nodes)
    end
    for k in kroots
        collect_nodes!(n, k.a, nodes)
        collect_nodes!(n, k.b, nodes)
    end
    for c in croots
        collect_nodes!(n, c.operands[1], nodes)
        collect_nodes!(n, c.operands[2], nodes)
    end

    # phase 3: a node whose wide value stays live anyway is truncated instead (which is
    # free), so only keep nodes all of whose users are narrowed nodes or roots
    root_users = Set{Instruction}(Instruction[(r.gep for r in groots)..., croots...])
    chain_geps = Set{Instruction}(Instruction[(k.outer for k in kroots)...,
                                              (k.inner for k in kroots)...])
    pruned = true
    while pruned
        pruned = false
        for node in collect(nodes)
            for user in node.users
                ok = user isa Instruction &&
                     (user in nodes || only_feeds_assumes(n, user) || user in chain_geps)
                if user isa Instruction && !ok && user in root_users
                    if user isa GetElementPtrInst
                        # only the rewritten index operands count
                        ok = true
                        for (i, op) in enumerate(user.operands)
                            i == 1 && continue
                            if op == node && !any(r -> r.gep == user && r.index == i, groots)
                                ok = false
                            end
                        end
                    else
                        ok = true
                    end
                end
                if !ok
                    delete!(nodes, node)
                    pruned = true
                    break
                end
            end
        end
    end

    # phase 4: rewrite
    b = n.builder
    for r in groots
        idx = r.gep.operands[r.index]
        # an index that isn't recomputed would only become zext(trunc(wide))
        idx isa Instruction && idx in nodes || continue
        narrow = narrow!(n, idx, nodes, LLVM.before(r.gep))
        position!(b, LLVM.before(r.gep))
        opts.assume_nonneg && r.nonneg && emit_nonneg!(n, narrow, Instruction[idx, r.gep])
        wide = if r.zext
            w = zext!(b, narrow, n.i64, "nidx")
            r.nonneg && w isa ZExtInst && LLVM.version() >= v"18" && (w.nneg = true)
            w
        else
            sext!(b, narrow, n.i64, "nidx")
        end
        r.gep.operands[r.index] = wide
        rewritten!(n, r.gep, "computed index $(r.index - 1) in 32 bits")
    end
    for k in kroots
        # p + a + b with a + b in [0, 2^31): recompute the sum in i32
        na = narrow!(n, k.a, nodes, LLVM.before(k.outer))
        nb = narrow!(n, k.b, nodes, LLVM.before(k.outer))
        position!(b, LLVM.before(k.outer))
        sum = add!(b, na, nb, "chain.n32")
        opts.assume_nonneg && emit_nonneg!(n, sum, Instruction[k.outer])
        wide = zext!(b, sum, n.i64, "nidx")
        wide isa ZExtInst && LLVM.version() >= v"18" && (wide.nneg = true)
        # the new GEP computes the same address from a different decomposition, so it
        # cannot be inbounds
        gep = gep!(b, k.outer.source_element_type, k.inner.operands[1], [wide])
        name = k.outer.name
        k.outer.name = ""
        gep.name = name
        replace_uses!(k.outer, gep)
        rewritten!(n, gep, "computed the combined index of a GEP chain in 32 bits")
    end
    for c in croots
        na = narrow!(n, c.operands[1], nodes, LLVM.before(c))
        nb = narrow!(n, c.operands[2], nodes, LLVM.before(c))
        position!(b, LLVM.before(c))
        new = icmp!(b, c.predicate, na, nb, c.name * ".n32")
        replace_uses!(c, new)
        rewritten!(n, c, "compared in 32 bits")
    end

    # delete the wide nodes that are now dead, and the operands that only they used
    if n.changed
        erase_trivially_dead!([inst for bb in f.blocks for inst in bb.instructions
                               if is_trivially_dead(inst)])
    end

    if opts.mulwide
        muls = Instruction[inst for bb in f.blocks for inst in bb.instructions
                           if inst.opcode == LLVM.Opcode.Mul && is_i64(inst)]
        for mul in muls
            # keep loop-variant arithmetic in a form that loop strength reduction understands
            opts.skip_addrec && (is_loop_affine(n, mul) || n.loops[mul.parent] !== nothing) &&
                continue
            for i in 1:2
                v = mul.operands[i]
                (v isa Constant || v isa ZExtInst) && continue
                fits_u32(n, v, mul) || continue
                position!(b, LLVM.before(mul))
                narrow = trunc!(b, v, n.i32, v.name * ".mw")
                mul.operands[i] = zext!(b, narrow, n.i64)
                rewritten!(n, mul, "extended operand $i from 32 bits")
            end
        end
    end

    return n.changed
end

function narrow_indices!(f::LLVM.Function, am::FunctionAnalysisManager, opts::NarrowingOptions)
    isdeclaration(f) && return false
    changed = @dispose builder=IRBuilder() begin
        narrow_indices!(Narrower(f, am, opts, builder))
    end
    # only instructions were added and removed
    return changed ? PreservedAnalyses(CFGAnalyses) : PreservedAnalyses(AllAnalyses)
end

"""
    NarrowIndicesPass(; options...)

A function pass, `narrow-idx` in a pipeline, that narrows 64-bit index arithmetic to 32 bits
where LLVM's analyses prove that the values fit (see the top of this file for a description,
and `NarrowingOptions`). The keyword arguments are the defaults of the options, which a
pipeline can override: `narrow-idx<no-compares;max-depth=8>`.
"""
NarrowIndicesPass(; defaults...) =
    FunctionPass("narrow-idx", narrow_indices!; analyses=true,
                 parse_options=str -> parse_options(NarrowingOptions, str; defaults...))

# run the narrowing pass on every function of a module, after deriving range information
# from branch conditions (which other passes may have exposed)
function narrow_indices!(mod::LLVM.Module, tm::LLVM.TargetMachine; kwargs...)
    @dispose pb=PassBuilder() begin
        pass = NarrowIndicesPass(; kwargs...)
        register!(pb, pass)
        add!(pb, FunctionPassManager()) do fpm
            add!(fpm, CorrelatedValuePropagationPass())
            add!(fpm, pass)
        end
        run!(pb, mod, tm)
    end
    return
end

end
