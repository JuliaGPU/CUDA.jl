export SMVersion, @sm_str

"""
    SMVersion(major, minor, [feature_set])

A PTX compilation target, identifying a CUDA compute capability together with the
subtarget feature set selected by the suffix on its `.target` directive.

`feature_set` is one of:

- `:baseline` (no suffix, e.g. `sm_90`) — forward-compatible (the "onion model"):
  PTX compiled for `sm_X` runs on any `sm_Y` with `Y >= X`.
- `:family` (`f` suffix, e.g. `sm_100f`) — same-major-family-portable: PTX runs on
  any device in the same architecture family (currently == same major version) at
  or above this CC.
- `:arch` (`a` suffix, e.g. `sm_90a`) — locked to one exact CC: PTX runs only on
  devices with exactly this compute capability, but in exchange gets access to
  architecture-accelerated features.

See NVIDIA's PTX ISA reference under `.target` for the full compatibility rules,
and `lib/Target/NVPTX/NVPTX.td` in LLVM for the corresponding subtarget feature
definitions.

Public fields:
- `sm.major::Int`
- `sm.minor::Int`
- `sm.feature_set::Symbol`

See also [`@sm_str`](@ref) for an ergonomic string-macro constructor.

# Examples
```julia
julia> SMVersion(9, 0)            # baseline
sm"9.0"

julia> SMVersion(9, 0, :arch)
sm"9.0a"

julia> sm"10.0f" == SMVersion(10, 0, :family)
true
```
"""
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

function Base.parse(::Type{SMVersion}, s::AbstractString)
    m = match(r"^(\d+)\.(\d+)([af]?)$", s)
    m === nothing && error("invalid sm version string: $(repr(s)); expected e.g. \"10.3\", \"10.3a\", or \"10.0f\"")
    major = parse(Int, m.captures[1])
    minor = parse(Int, m.captures[2])
    fs = m.captures[3] == "a" ? :arch :
         m.captures[3] == "f" ? :family : :baseline
    return SMVersion(major, minor, fs)
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

"""
    @sm_str

String macro used to parse a string to an [`SMVersion`](@ref). Accepts the dotted form
used in NVIDIA's PTX ISA reference: `sm"9.0"` for baseline, `sm"9.0a"` for
architecture-accelerated, `sm"10.0f"` for family-specific.

# Examples
```julia
julia> sm"10.3a"
sm"10.3a"

julia> sm"10.0f" == SMVersion(10, 0, :family)
true
```
"""
macro sm_str(s)
    return :(Base.parse($SMVersion, $(esc(s))))
end
