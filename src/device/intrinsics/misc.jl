export clock, nanosleep

"""
    exit()

Terminate a thread.
"""
@device_function exit() = @asmcall("exit;")

"""
    clock(UInt32)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle.
"""
@device_function clock(::Type{UInt32}) = ccall("llvm.nvvm.read.ptx.sreg.clock", llvmcall, UInt32, ())

"""
    clock(UInt64)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle.
"""
@device_function clock(::Type{UInt64}) = ccall("llvm.nvvm.read.ptx.sreg.clock64", llvmcall, UInt64, ())


"""
    nanosleep(t)

Puts a thread for a given amount `t`(in nanoseconds).

!!! note
    Requires CUDA >= 10.0 and sm_6.2
"""
@device_function @inline function nanosleep(t::Unsigned)
    @asmcall("nanosleep.u32 \$0;", "r", true,
             Cvoid, Tuple{UInt32}, convert(UInt32, t))
end
