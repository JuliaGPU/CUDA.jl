# SMVersion: a PTX compilation target, identifying a CUDA compute capability together
# with its subtarget feature set.
#
# Constructed via the `sm"..."` string macro, which accepts the dotted form used in
# CUDA's documentation: `sm"10.3a"` is compute capability 10.3 with architecture-
# accelerated features. The bare form `sm"10.3"` is the default, forward-compatible
# feature set (the "onion model").
#
# See `lib/Target/NVPTX/NVPTX.td` in LLVM for the corresponding subtarget feature
# definitions, and CUDA's PTX ISA documentation under `.target` for the runtime
# compatibility implications:
#
#   :baseline (no suffix)   - forward-compatible (sm_X for any sm_Y >= X)
#   :family   ('f' suffix)  - same-major-family-portable
#   :arch     ('a' suffix)  - locked to one exact CC

export SMVersion, @sm_str

struct SMVersion
    major::Int
    minor::Int
    feature_set::Symbol

    function SMVersion(major::Integer, minor::Integer, feature_set::Symbol = :baseline)
        feature_set in (:baseline, :family, :arch) ||
            error("SMVersion feature_set must be one of :baseline, :family, :arch; got $(repr(feature_set))")
        return new(Int(major), Int(minor), feature_set)
    end
end

# Suffix on the LLVM CPU name / `.target` directive
suffix(sm::SMVersion) = sm.feature_set === :arch    ? "a" :
                        sm.feature_set === :family  ? "f" : ""

# LLVM CPU / PTX `.target` name (e.g. "sm_103a").
cpu_name(sm::SMVersion) = "sm_$(sm.major)$(sm.minor)$(suffix(sm))"

# Drop the feature set to recover the base compute-capability `VersionNumber`,
# usable against the version-keyed compatibility databases.
base_version(sm::SMVersion) = VersionNumber(sm.major, sm.minor)

# Would a cubin compiled for `sm` actually load and run on a device with capability
# `dev_cap`? Per NVIDIA's PTX ISA reference (.target directive):
#   - baseline: forward-compatible (onion model) -- any sm_X runs on sm_Y for Y >= X.
#   - family:   same architecture family (currently == same major) and forward-portable
#               within the family.
#   - arch:     locked to one exact CC; cubin only loads on devices with that exact cap.
function runs_on(sm::SMVersion, dev_cap::VersionNumber)
    if sm.feature_set === :arch
        return base_version(sm) == dev_cap
    elseif sm.feature_set === :family
        return sm.major == dev_cap.major && base_version(sm) <= dev_cap
    else  # :baseline
        return base_version(sm) <= dev_cap
    end
end


Base.show(io::IO, sm::SMVersion) = print(io, "sm\"", sm.major, ".", sm.minor, suffix(sm), "\"")

function _parse_sm(s::AbstractString)
    m = match(r"^(\d+)\.(\d+)([af]?)$", s)
    m === nothing && error("invalid sm version string: $(repr(s)); expected e.g. \"10.3\", \"10.3a\", or \"10.0f\"")
    major = parse(Int, m.captures[1])
    minor = parse(Int, m.captures[2])
    fs = m.captures[3] == "a" ? :arch :
         m.captures[3] == "f" ? :family : :baseline
    return SMVersion(major, minor, fs)
end

macro sm_str(s)
    return _parse_sm(s)
end
