# device intrinsics for querying the compute SimpleVersion and PTX ISA version

export compute_capability, ptx_isa_version

for var in ["sm_major", "sm_minor", "ptx_major", "ptx_minor"]
    @eval @inline $(Symbol(var))() =
        Base.llvmcall(
            $("""@$var = external global i32
                 define i32 @entry() #0 {
                     %val = load i32, i32* @$var
                     ret i32 %val
                 }
                 attributes #0 = { alwaysinline }
            """, "entry"), UInt32, Tuple{})
end

@device_function @inline compute_capability() = SimpleVersion(sm_major(), sm_minor())
@device_function @inline ptx_isa_version() = SimpleVersion(ptx_major(), ptx_minor())

