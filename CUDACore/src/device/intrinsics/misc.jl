export clock, nanosleep
@public exit

"""
    exit()

Terminate a thread.
"""
@device_function exit() = @asmcall("exit;", "", true)

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
clock(::Type{UInt64}) = ccall("llvm.readcyclecounter", llvmcall, UInt64, ())

end

"""
    nanosleep(t)

Puts a thread for a given amount `t`(in nanoseconds).

!!! note
    Requires compute capability 7.0.
"""
@device_function @inline function nanosleep(t::Unsigned)
    require_sm_70()
    @asmcall("nanosleep.u32 \$0;", "r", true,
             Cvoid, Tuple{UInt32}, convert(UInt32, t))
end

## profiler counters

export prof_trigger

# Julia's LLVM may predate this intrinsic, so declare it with the attributes an older LLVM
# can't infer. `convergent` and `nomerge` keep calls on divergent paths apart: merging them
# would change how many times a warp triggers the event.
@device_function @llvmgenerated builder function pmevent_mask(::Val{mask})::Nothing where {mask}
    mod = current_module(builder)
    intr_name = "llvm.nvvm.pm.event.mask"
    intr_typ = LLVM.FunctionType(LLVM.VoidType(), [LLVM.Int16Type()])
    intr = get!(mod.functions, intr_name) do
        f = LLVM.Function(mod, intr_name, intr_typ)
        for kind in (:convergent, :nomerge, :nounwind)
            push!(f.function_attributes, EnumAttribute(kind))
        end
        f
    end
    call!(builder, intr_typ, intr, [ConstantInt(LLVM.Int16Type(), mask)])
    nothing
end

"""
    prof_trigger(counter)

Increment profiler counter `counter`, between 0 and 7, once per warp that executes this
call (the PTX `pmevent` instruction, or `__prof_trigger` in CUDA C). The counters can be
read with the `sm__inst_executed_op_pmtrig_<counter>_pred_on_any` metrics of
`CUDA.@metrics`, which require CUDA 13.5 or later.

A constant `counter` compiles to a single instruction; other values select it at run time.
"""
@device_function @inline function prof_trigger(counter::Integer)
    Base.Cartesian.@nif 9 i -> counter == i-1 i -> pmevent_mask(Val(Int16(1) << (i-1))) i ->
        throw(ArgumentError("profiler counter must be between 0 and 7"))
    return
end
