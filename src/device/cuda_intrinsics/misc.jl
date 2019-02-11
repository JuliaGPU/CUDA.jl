export clock, nanosleep

"""
    clock(UInt32)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle. 
"""
clock(::Type{UInt32}) = ccall("llvm.nvvm.read.ptx.sreg.clock", llvmcall, UInt32, ()) 

"""
    clock(UInt32)

Returns the value of a per-multiprocessor counter that is incremented every clock cycle. 
"""
clock(::Type{UInt64}) = ccall("llvm.nvvm.read.ptx.sreg.clock64", llvmcall, UInt64, ()) 


"""
    nanosleep(t)

Puts a thread for a given amount `t`(in nanoseconds).

!!! note
    Requires CUDA >= 10.0 and sm_6.2
"""
nanosleep

if cuda_driver_version >= v"10.0" && v"6.2" in ptx_support
    @inline function nanosleep(t::Unsigned)
        @asmcall("nanosleep.u32 \$0;", "r", true,
                 Cvoid, Tuple{UInt32}, convert(UInt32, t))
    end
else
    @inline function nanosleep(t::Unsigned)
	return nothing
    end
end
