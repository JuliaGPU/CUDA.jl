# Warp Vote (B.13)

export vote_all, vote_any, vote_ballot

const all_asm = """{
    .reg .pred %p1;
    .reg .pred %p2;
    setp.ne.u32 %p1, \$1, 0;
    vote.all.pred %p2, %p1;
    selp.s32 \$0, 1, 0, %p2;
}"""

"""
    vote_all(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return non-zero if and only if
`predicate` evaluates to non-zero for all of them.
"""
@inline function vote_all(pred::Bool)
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

"""
    vote_any(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return non-zero if and only if
`predicate` evaluates to non-zero for any of them.
"""
@inline function vote_any(pred::Bool)
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

"""
    vote_ballot(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return an integer whose Nth bit
is set if and only if `predicate` evaluates to non-zero for the Nth thread of the warp and
the Nth thread is active.
"""
@inline function vote_ballot(pred::Bool)
    return Base.llvmcall(
        """%2 = call i32 asm sideeffect "$ballot_asm", "=r,r"(i32 %0)
           ret i32 %2""",
        UInt32, Tuple{Int32}, convert(Int32, pred))
end
