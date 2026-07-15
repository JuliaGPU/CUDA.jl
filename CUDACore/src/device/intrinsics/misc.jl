export clock, nanosleep
@public exit

"""
    exit()

Terminate a thread.
"""
exit() = @asmcall("exit;")

@device_functions begin
"""
    clock(UInt32)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle.
"""
clock(::Type{UInt32}) = ccall("llvm.nvvm.read.ptx.sreg.clock", llvmcall, UInt32, ())

"""
    clock(UInt64)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle.
"""
clock(::Type{UInt64}) = ccall("llvm.nvvm.read.ptx.sreg.clock64", llvmcall, UInt64, ())

end

"""
    nanosleep(t)

Puts a thread for a given amount `t`(in nanoseconds).

!!! note
    Requires compute capability 7.0.
"""
@inline function nanosleep(t::Unsigned)
    require_sm_70()
    @asmcall("nanosleep.u32 \$0;", "r", true,
             Cvoid, Tuple{UInt32}, convert(UInt32, t))
end
