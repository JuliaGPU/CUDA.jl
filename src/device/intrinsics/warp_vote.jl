# Warp Vote (B.13)

for mode in (:all, :any, :uni)
    fname = Symbol("vote_$mode")
    fname_sync = Symbol("vote_$(mode)_sync")
    @eval export $fname, $fname_sync

    intrinsic = "llvm.nvvm.vote.$mode"
    @eval begin
        # FIXME: ccall($intrinsic, llvmcall, $rettyp, (Bool,), pred)
        #        doesn't use i1 for Bool
        @inline $fname(pred) =
            Base.llvmcall($("""
                declare i1 @$intrinsic(i1)

                define i8 @entry(i8) #0 {
                    %predicate = icmp eq i8 %0, 1
                    %llvmbool = call i1 @$intrinsic(i1 %predicate)
                    %jlbool = zext i1 %llvmbool to i8
                    ret i8 %jlbool
                }

                attributes #0 = { alwaysinline }""", "entry"),
            Bool, Tuple{Bool}, pred)
    end

    # warp-synchronous
    intrinsic = "llvm.nvvm.vote.$mode.sync"
    @eval begin
        @inline $fname_sync(mask, pred) =
            Base.llvmcall($("""
                declare i1 @$intrinsic(i32, i1)

                define i8 @entry(i32 %mask, i8) #0 {
                    %predicate = icmp eq i8 %0, 1
                    %llvmbool = call i1 @$intrinsic(i32 %mask, i1 %predicate)
                    %jlbool = zext i1 %llvmbool to i8
                    ret i8 %jlbool
                }

                attributes #0 = { alwaysinline }""", "entry"),
            Bool, Tuple{UInt32, Bool}, mask, pred)
    end
end

# ballot returns an integer, so we need to repeat the above
for mode in (:ballot, )
    fname = Symbol("vote_$mode")
    fname_sync = Symbol("vote_$(mode)_sync")
    @eval export $fname, $fname_sync

    intrinsic = "llvm.nvvm.vote.$mode"
    @eval begin
        @inline $fname(pred) =
            Base.llvmcall($("""
                declare i32 @$intrinsic(i1)

                define i32 @entry(i8) #0 {
                    %predicate = icmp eq i8 %0, 1
                    %ret = call i32 @$intrinsic(i1 %predicate)
                    ret i32 %ret
                }

                attributes #0 = { alwaysinline }""", "entry"),
            UInt32, Tuple{Bool}, pred)
    end

    # warp-synchronous
    intrinsic = "llvm.nvvm.vote.$mode.sync"
    @eval begin
        @inline $fname_sync(mask, pred) =
            Base.llvmcall($("""
                declare i32 @$intrinsic(i32, i1)

                define i32 @entry(i32 %mask, i8) #0 {
                    %predicate = icmp eq i8 %0, 1
                    %ret = call i32 @$intrinsic(i32 %mask, i1 %predicate)
                    ret i32 %ret
                }

                attributes #0 = { alwaysinline }""", "entry"),
            UInt32, Tuple{UInt32, Bool}, mask, pred)
    end
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
