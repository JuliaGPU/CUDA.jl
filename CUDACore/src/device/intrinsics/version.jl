# device intrinsics for querying the compute SimpleVersion and PTX ISA version

export compute_capability, ptx_isa_version, target_feature_set

# Wire-format encoding of the feature set, stamped into the `sm_features` LLVM global.
@enum TargetFeatureSet::UInt32 begin
    BaselineFeatures = 0
    FamilyFeatures   = 1
    ArchFeatures     = 2
end

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

for cap in (sv"6.1", sv"7.0", sv"7.2", sv"8.0", sv"9.0")
    local requirement = Symbol("require_sm_", cap.major, cap.minor)
    local message = "requires compute capability $(cap.major).$(cap.minor) or higher"
    @eval @device_function @inline $requirement() =
        GPUCompiler.@static_assert compute_capability() >= $cap $message
end

# Feature set encoded in the `.target` directive: one of `:baseline`, `:family`, `:arch`.
# (NVIDIA's PTX ISA reference: ".target specifies the set of features in the target
# architecture for which the current PTX code was generated.") CUDA's compiler stamps the
# encoding in via the `sm_features` LLVM global, using `TargetFeatureSet`;
# the integer load + chained compare folds away after LLVM inlines the constant, so
# user code like `if target_feature_set() === :arch ... end` resolves to a single
# branch in the PTX output.
@device_function @inline function target_feature_set()
    f = sm_features()
    return f == UInt32(ArchFeatures)   ? :arch :
           f == UInt32(FamilyFeatures) ? :family : :baseline
end
