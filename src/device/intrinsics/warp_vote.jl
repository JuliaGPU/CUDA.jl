# Warp Vote (B.13)

export vote_all, vote_any, vote_ballot

"""
    vote_all(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return non-zero if and only if
`predicate` evaluates to non-zero for all of them.
"""
@inline function vote_all(pred::Bool)
    return @asmcall(
        """{
               .reg .pred %p1;
               .reg .pred %p2;
               setp.ne.u32 %p1, \$1, 0;
               vote.all.pred %p2, %p1;
               selp.s32 \$0, 1, 0, %p2;
           }""", "=r,r", true,
        Int32, Tuple{Int32}, convert(Int32, pred)) != Int32(0)
end

"""
    vote_any(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return non-zero if and only if
`predicate` evaluates to non-zero for any of them.
"""
@inline function vote_any(pred::Bool)
    return @asmcall(
        """{
               .reg .pred %p1;
               .reg .pred %p2;
               setp.ne.u32 %p1, \$1, 0;
               vote.any.pred %p2, %p1;
               selp.s32 \$0, 1, 0, %p2;
           }""", "=r,r", true,
        Int32, Tuple{Int32}, convert(Int32, pred)) != Int32(0)
end

"""
    vote_ballot(predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return an integer whose Nth bit
is set if and only if `predicate` evaluates to non-zero for the Nth thread of the warp and
the Nth thread is active.
"""
@inline function vote_ballot(pred::Bool)
    return @asmcall(
        """{
               .reg .pred %p1;
               setp.ne.u32 %p1, \$1, 0;
               vote.ballot.b32 \$0, %p1;
           }""", "=r,r", true,
        UInt32, Tuple{Int32}, convert(Int32, pred))
end
