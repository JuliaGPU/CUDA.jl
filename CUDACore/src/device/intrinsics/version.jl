# device intrinsics for querying the compute SimpleVersion and PTX ISA version

export compute_capability, ptx_isa_version, target_feature_set

for var in ["sm_major", "sm_minor", "sm_features", "ptx_major", "ptx_minor"]
    @eval @device_function @inline $(Symbol(var))() =
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

# Feature set encoded in the `.target` directive: one of `:baseline`, `:family`, `:arch`.
# (NVIDIA's PTX ISA reference: ".target specifies the set of features in the target
# architecture for which the current PTX code was generated.") GPUCompiler stamps the
# encoding in via the `sm_features` LLVM global, using `GPUCompiler.TargetFeatureSet`;
# the integer load + chained compare folds away after LLVM inlines the constant, so
# user code like `if target_feature_set() === :arch ... end` resolves to a single
# branch in the PTX output.
@device_function @inline function target_feature_set()
    f = sm_features()
    return f == UInt32(GPUCompiler.ArchFeatures)   ? :arch :
           f == UInt32(GPUCompiler.FamilyFeatures) ? :family : :baseline
end

