# compatibility of Julia, CUDA and LLVM

const lowest = v"0"
const highest = v"999"


# PTX compilation targets come in three feature-set flavors (carried on `SMVersion`),
# selected via the suffix on the `.target` directive (and the matching `--gpu-name`
# to ptxas):
#
#   - Baseline (no suffix, e.g. sm_90): the forward-compatible feature set. Code compiled
#     for sm_X runs on any sm_Y with Y >= X (onion model).
#   - Family (`f` suffix, e.g. sm_100f): a superset of Baseline. Same-major-family-portable;
#     code compiled for sm_100f runs on sm_100, sm_103, etc., but not across families.
#   - Architecture (`a` suffix, e.g. sm_90a): a superset of Family. Locked to one
#     exact CC; code compiled for sm_103a runs only on CC 10.3 devices.
#
# Which feature sets exist for a given CC, and which PTX ISA / LLVM versions ptxas / NVPTX
# require for them, is encoded directly in the keys of `ptx_cap_db` and `llvm_cap_db`
# below: an unsupported combination simply has no entry.


## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

between(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)


## devices supported by the CUDA toolkit

# Source:
# - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
# - ptxas |& grep -A 10 '\--gpu-name'
const cuda_cap_db = Dict(
    v"1.0"   => between(lowest, v"6.5"),
    v"1.1"   => between(lowest, v"6.5"),
    v"1.2"   => between(lowest, v"6.5"),
    v"1.3"   => between(lowest, v"6.5"),
    v"2.0"   => between(lowest, v"8.0"),
    v"2.1"   => between(lowest, v"8.0"),
    v"3.0"   => between(v"4.2", v"10.2"),
    v"3.2"   => between(v"6.0", v"10.2"),
    v"3.5"   => between(v"5.0", v"11.8"),
    v"3.7"   => between(v"6.5", v"11.8"),
    v"5.0"   => between(v"6.0", v"12.9"),
    v"5.2"   => between(v"7.0", v"12.9"),
    v"5.3"   => between(v"7.5", v"12.9"),
    v"6.0"   => between(v"8.0", v"12.9"),
    v"6.1"   => between(v"8.0", v"12.9"),
    v"6.2"   => between(v"8.0", v"12.9"),
    v"7.0"   => between(v"9.0", v"12.9"),
    v"7.2"   => between(v"9.2", v"12.9"),
    v"7.5"   => between(v"10.0", highest),
    v"8.0"   => between(v"11.0", highest),
    v"8.6"   => between(v"11.1", highest),
    v"8.7"   => between(v"11.4", highest),
    v"8.9"   => between(v"11.8", highest),
    v"9.0"   => between(v"11.8", highest),
    v"10.0"  => between(v"12.8", highest),
    v"10.3"  => between(v"12.8", highest),
    v"11.0"  => between(v"12.8", highest),
    v"12.0"  => between(v"12.8", highest),
    v"12.1"  => between(v"12.9", highest),
)

function cuda_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in cuda_cap_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## PTX ISAs supported by the CUDA toolkit

# Source: PTX ISA document, Release History table
const cuda_ptx_db = Dict(
    v"1.0" => between(v"1.0", highest),
    v"1.1" => between(v"1.1", highest),
    v"1.2" => between(v"2.0", highest),
    v"1.3" => between(v"2.1", highest),
    v"1.4" => between(v"2.2", highest),
    v"1.5" => between(v"2.2", highest),
    v"2.0" => between(v"3.0", highest),
    v"2.1" => between(v"3.1", highest),
    v"2.2" => between(v"3.2", highest),
    v"2.3" => between(v"4.2", highest),
    v"3.0" => between(v"4.1", highest),
    v"3.1" => between(v"5.0", highest),
    v"3.2" => between(v"5.5", highest),
    v"4.0" => between(v"6.0", highest),
    v"4.1" => between(v"6.5", highest),
    v"4.2" => between(v"7.0", highest),
    v"4.3" => between(v"7.5", highest),
    v"5.0" => between(v"8.0", highest),
    v"6.0" => between(v"9.0", highest),
    v"6.1" => between(v"9.1", highest),
    v"6.2" => between(v"9.2", highest),
    v"6.3" => between(v"10.0", highest),
    v"6.4" => between(v"10.1", highest),
    v"6.5" => between(v"10.2", highest),
    v"7.0" => between(v"11.0", highest),
    v"7.1" => between(v"11.1", highest),
    v"7.2" => between(v"11.2", highest),
    v"7.3" => between(v"11.3", highest),
    v"7.4" => between(v"11.4", highest),
    v"7.5" => between(v"11.5", highest),
    v"7.6" => between(v"11.6", highest),
    v"7.7" => between(v"11.7", highest),
    v"7.8" => between(v"11.8", highest),
    v"8.0" => between(v"12.0", highest),
    v"8.1" => between(v"12.1", highest),
    v"8.2" => between(v"12.2", highest),
    v"8.3" => between(v"12.3", highest),
    v"8.4" => between(v"12.4", highest),
    v"8.5" => between(v"12.5", highest),
    v"8.6" => between(v"12.7", highest),
    v"8.7" => between(v"12.8", highest),
    v"8.8" => between(v"12.9", highest),
    v"9.0" => between(v"13.0", highest),
    v"9.1" => between(v"13.1", highest),
    v"9.2" => between(v"13.2", highest),
)

function cuda_ptx_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in cuda_ptx_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## devices supported by each PTX ISA

# Source: PTX ISA document, Release History table. Architecture-specific (`*a`) variants
# were introduced at CC 9.0 / PTX 8.0; family-specific (`*f`) variants at CC 10.0 / PTX 8.8.
const ptx_cap_db = Dict{SMVersion, VersionRange}(
    sm"1.0"   => between(v"1.0", highest),
    sm"1.1"   => between(v"1.0", highest),
    sm"1.2"   => between(v"1.2", highest),
    sm"1.3"   => between(v"1.2", highest),
    sm"2.0"   => between(v"2.0", highest),
    sm"3.0"   => between(v"3.1", highest),
    sm"3.2"   => between(v"4.0", highest),
    sm"3.5"   => between(v"3.1", highest),
    sm"3.7"   => between(v"4.1", highest),
    sm"5.0"   => between(v"4.0", highest),
    sm"5.2"   => between(v"4.1", highest),
    sm"5.3"   => between(v"4.2", highest),
    sm"6.0"   => between(v"5.0", highest),
    sm"6.1"   => between(v"5.0", highest),
    sm"6.2"   => between(v"5.0", highest),
    sm"7.0"   => between(v"6.0", highest),
    sm"7.2"   => between(v"6.1", highest),
    sm"7.5"   => between(v"6.3", highest),
    sm"8.0"   => between(v"7.0", highest),
    sm"8.6"   => between(v"7.1", highest),
    sm"8.7"   => between(v"7.4", highest),
    sm"8.9"   => between(v"7.8", highest),
    sm"9.0"   => between(v"7.8", highest),
    sm"9.0a"  => between(v"8.0", highest),
    sm"10.0"  => between(v"8.6", highest),
    sm"10.0a" => between(v"8.6", highest),
    sm"10.0f" => between(v"8.8", highest),
    sm"10.1"  => between(v"8.6", highest),
    sm"10.1a" => between(v"8.6", highest),
    sm"10.1f" => between(v"8.8", highest),
    sm"10.3"  => between(v"8.8", highest),
    sm"10.3a" => between(v"8.8", highest),
    sm"10.3f" => between(v"8.8", highest),
    sm"12.0"  => between(v"8.7", highest),
    sm"12.0a" => between(v"8.7", highest),
    sm"12.0f" => between(v"8.8", highest),
    sm"12.1"  => between(v"8.8", highest),
    sm"12.1a" => between(v"8.8", highest),
    sm"12.1f" => between(v"8.8", highest),
)

# Set of `SMVersion`s (across all feature sets) whose ptxas floor is met by `ver`.
function ptx_cap_support(ver::VersionNumber)
    caps = Set{SMVersion}()
    for (cap, r) in ptx_cap_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end

# Baseline-only view, returned as `VersionNumber`s for use by the cap-clamp logic.
function ptx_baseline_caps(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap, r) in ptx_cap_db
        if cap.feature_set === :baseline && ver in r
            push!(caps, base_version(cap))
        end
    end
    return caps
end


## devices supported by the LLVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td. Each `def : Proc<"sm_NN[a|f]", ...>` shows up
# here as a separate entry; without an entry LLVM does not know the variant CPU name and
# constructing a TargetMachine with it would fall back to a generic subtarget.
const llvm_cap_db = Dict{SMVersion, VersionRange}(
    sm"2.0"   => between(v"3.2", highest),
    sm"2.1"   => between(v"3.2", highest),
    sm"3.0"   => between(v"3.2", highest),
    sm"3.2"   => between(v"3.7", highest),
    sm"3.5"   => between(v"3.2", highest),
    sm"3.7"   => between(v"3.7", highest),
    sm"5.0"   => between(v"3.5", highest),
    sm"5.2"   => between(v"3.7", highest),
    sm"5.3"   => between(v"3.7", highest),
    sm"6.0"   => between(v"3.9", highest),
    sm"6.1"   => between(v"3.9", highest),
    sm"6.2"   => between(v"3.9", highest),
    sm"7.0"   => between(v"6", highest),
    sm"7.2"   => between(v"7", highest),
    sm"7.5"   => between(v"8", highest),
    sm"8.0"   => between(v"11", highest),
    sm"8.6"   => between(v"13", highest),
    sm"8.7"   => between(v"16", highest),
    sm"8.9"   => between(v"16", highest),
    sm"9.0"   => between(v"16", highest),
    sm"9.0a"  => between(v"18", highest),
    sm"10.0"  => between(v"20", highest),
    sm"10.0a" => between(v"20", highest),
    sm"10.0f" => between(v"21", highest),
    sm"10.1"  => between(v"20", highest),
    sm"10.1a" => between(v"20", highest),
    sm"10.1f" => between(v"21", highest),
    sm"10.3"  => between(v"21", highest),
    sm"10.3a" => between(v"21", highest),
    sm"10.3f" => between(v"21", highest),
    sm"12.0"  => between(v"20", highest),
    sm"12.0a" => between(v"20", highest),
    sm"12.0f" => between(v"21", highest),
    sm"12.1"  => between(v"21", highest),
    sm"12.1a" => between(v"21", highest),
    sm"12.1f" => between(v"21", highest),
)

# Set of `SMVersion`s (across all feature sets) supported by LLVM `ver`.
function llvm_cap_support(ver::VersionNumber)
    caps = Set{SMVersion}()
    for (cap, r) in llvm_cap_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end

# Baseline-only view, returned as `VersionNumber`s for use by the cap-clamp logic.
function llvm_baseline_caps(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap, r) in llvm_cap_db
        if cap.feature_set === :baseline && ver in r
            push!(caps, base_version(cap))
        end
    end
    return caps
end


## PTX ISAs supported by the LVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td
const llvm_ptx_db = Dict(
    v"3.0" => between(v"3.2", v"3.5"),
    v"3.1" => between(v"3.2", v"3.5"),
    v"3.2" => between(v"3.5", highest),
    v"4.0" => between(v"3.5", highest),
    v"4.1" => between(v"3.7", highest),
    v"4.2" => between(v"3.7", highest),
    v"4.3" => between(v"3.9", highest),
    v"5.0" => between(v"3.9", highest),
    v"6.0" => between(v"6", highest),
    v"6.1" => between(v"7", highest),
    v"6.3" => between(v"8", highest),
    v"6.4" => between(v"9", highest),
    v"6.5" => between(v"11", highest),
    v"7.0" => between(v"11", highest),
    v"7.1" => between(v"13", highest),
    v"7.2" => between(v"13", highest),
    v"7.3" => between(v"14", highest),
    v"7.4" => between(v"14", highest),
    v"7.5" => between(v"14", highest),
    v"7.6" => between(v"16", highest),
    v"7.7" => between(v"16", highest),
    v"7.8" => between(v"16", highest),
    v"8.0" => between(v"17", highest),
    v"8.1" => between(v"17", highest),
    v"8.2" => between(v"18", highest),
    v"8.3" => between(v"18", highest),
    v"8.4" => between(v"19", highest),
    v"8.5" => between(v"19", highest),
    v"8.6" => between(v"20", highest),
    v"8.7" => between(v"20", highest),
    v"8.8" => between(v"21", highest),
    v"9.0" => between(v"22", highest),
    v"9.1" => between(v"24", highest),
)

function llvm_ptx_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in llvm_ptx_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## high-level functions that return target and isa support

function llvm_compat(version=LLVM.version())
    LLVM.InitializeNVPTXTarget()

    # the `.cap` field is used for the base-cap clamp in `_compiler_config`, so only
    # baseline entries are surfaced here. Variant support is queried point-wise via
    # `sm in llvm_cap_support(...)`.
    cap_support = sort(collect(llvm_baseline_caps(version)))
    ptx_support = sort(collect(llvm_ptx_support(version)))

    return (cap=cap_support, ptx=ptx_support)
end

function cuda_compat(runtime=runtime_version(), compiler=compiler_version())
    # we don't have to check the driver version, because it offers backwards compatbility
    # beyond the CUDA toolkit version (e.g. R580 for CUDA 13 still supports Volta as
    # deprecated in CUDA 13), and we don't have a reliable way to query the actual version
    # as NVML isn't available on all platforms. let's instead simply assume that unsupported
    # devices will not be exposed to the CUDA runtime and thus won't be visible to us.

    # the compiler and runtime are versioned independently (and either can come from a
    # local install), so we need to consider both:
    # - device caps are dropped when either ptxas can't emit for them or the runtime
    #   libraries drop them. take the intersection of both supported sets.
    # - PTX ISA availability is a property of ptxas; the runtime doesn't care which ISA
    #   compiled cubin came from.
    cap_support = sort(collect(intersect(cuda_cap_support(runtime),
                                         cuda_cap_support(compiler))))
    ptx_support = sort(collect(cuda_ptx_support(compiler)))

    return (cap=cap_support, ptx=ptx_support)
end

function ptx_compat(ptx)
    # Baseline view for the clamp; variant support is queried point-wise via
    # `sm in ptx_cap_support(...)`.
    return (cap=ptx_baseline_caps(ptx),)
end
