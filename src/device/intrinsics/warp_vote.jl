# Warp Vote (B.13)

for mode in (:all, :any, :uni)
    fname = Symbol("vote_$mode")
    fname_sync = Symbol("vote_$(mode)_sync")
    @eval export $fname, $fname_sync

    intrinsic = "llvm.nvvm.vote.$mode"
    @eval @inline $fname(pred) =
        @typed_ccall($intrinsic, llvmcall, Bool, (Bool,), pred)

    # warp-synchronous
    intrinsic = "llvm.nvvm.vote.$mode.sync"
    @eval @inline $fname_sync(mask, pred) =
        @typed_ccall($intrinsic, llvmcall, Bool, (UInt32, Bool), mask, pred)
end

# ballot returns an integer, so we need to repeat the above
for mode in (:ballot, )
    fname = Symbol("vote_$mode")
    fname_sync = Symbol("vote_$(mode)_sync")
    @eval export $fname, $fname_sync

    intrinsic = "llvm.nvvm.vote.$mode"
    @eval @inline $fname(pred) =
        @typed_ccall($intrinsic, llvmcall, UInt32, (Bool,), pred)

    # warp-synchronous
    intrinsic = "llvm.nvvm.vote.$mode.sync"
    @eval @inline $fname_sync(mask, pred) =
        @typed_ccall($intrinsic, llvmcall, UInt32, (UInt32, Bool), mask, pred)
end


"""
    vote_all(predicate::Bool)
    vote_all_sync(mask::UInt32, predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return whether `predicate`
is true for all of them.
"""
vote_all
@doc (@doc vote_all) vote_all_sync

"""
    vote_any(predicate::Bool)
    vote_any_sync(mask::UInt32, predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return whether `predicate`
is true for any of them.
"""
vote_any
@doc (@doc vote_any) vote_any_sync

"""
    vote_uni(predicate::Bool)
    vote_uni_sync(mask::UInt32, predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return whether `predicate` is
the same for any of them.
"""
vote_uni
@doc (@doc vote_uni) vote_uni_sync

"""
    vote_ballot(predicate::Bool)
    vote_ballot_sync(mask::UInt32, predicate::Bool)

Evaluate `predicate` for all active threads of the warp and return an integer whose Nth bit
is set if and only if `predicate` is true for the Nth thread of the warp and the Nth thread
is active.
"""
vote_ballot
@doc (@doc vote_ballot) vote_ballot_sync
