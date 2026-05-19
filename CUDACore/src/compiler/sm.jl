export SMVersion, @sm_str

"""
    SMVersion(major, minor, [feature_set])
    SMVersion(s::AbstractString)
    SMVersion(v::VersionNumber)
    SMVersion(sm::SMVersion)

A PTX compilation target, identifying a CUDA compute capability together with the
subtarget feature set selected by the suffix on its `.target` directive. Printed and
parsed in NVIDIA's compact form -- `sm"90"` for compute capability 9.0, `sm"103a"`
for 10.3 architecture-accelerated, etc. -- to mirror the `.target sm_NN[a|f]`
notation in the PTX ISA reference and to distinguish visually from a device-level
[`VersionNumber`](@ref) like `v"9.0"`.

The single-argument constructors normalize various inputs to an `SMVersion`:

- `SMVersion(::AbstractString)` parses the compact form, with or without the `sm_`
  prefix (so e.g. `SMVersion("sm_103a")` and `SMVersion("103a")` both work).
- `SMVersion(::VersionNumber)` promotes a plain compute-capability version to a
  baseline `SMVersion` (`SMVersion(v"10.3") == SMVersion(10, 3, :baseline)`).
- `SMVersion(::SMVersion)` is the identity (idempotent).

This is what lets `@cuda arch=...` accept `v"10.3"`, `sm"103a"`, `"sm_103a"`, or
an already-constructed `SMVersion` interchangeably.

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
sm"90"

julia> SMVersion(9, 0, :arch)
sm"90a"

julia> sm"100f" == SMVersion(10, 0, :family)
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
    # Mirrors NVIDIA's `sm_NN[a|f]` notation: the last digit before the optional suffix
    # is the minor, everything before it is the major. Always one minor digit (NVIDIA
    # has never minted a CC with minor >= 10, and rolls the major over instead). The
    # optional `sm_` prefix is accepted so PTX-tool output / config strings can pass
    # straight through.
    raw = startswith(s, "sm_") ? SubString(s, 4) : s
    m = match(r"^(\d+)(\d)([af]?)$", raw)
    m === nothing && error("invalid sm version string: $(repr(s)); expected e.g. \"103\", \"sm_103a\", or \"100f\"")
    major = parse(Int, m.captures[1])
    minor = parse(Int, m.captures[2])
    fs = m.captures[3] == "a" ? :arch :
         m.captures[3] == "f" ? :family : :baseline
    return SMVersion(major, minor, fs)
end

# Single-argument constructor: the universal normalizer for accepting an `arch`/`cap`-like
# argument. Identity for SMVersion; baseline-promotes a plain VersionNumber; parses a
# string (with or without the `sm_` prefix).
SMVersion(sm::SMVersion) = sm
SMVersion(v::VersionNumber) = SMVersion(v.major, v.minor, :baseline)
SMVersion(s::AbstractString) = Base.parse(SMVersion, s)

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


Base.show(io::IO, sm::SMVersion) = print(io, "sm\"", sm.major, sm.minor, suffix(sm), "\"")

"""
    @sm_str

String macro used to parse a string to an [`SMVersion`](@ref). Accepts NVIDIA's
compact `sm_NN[a|f]` notation (with or without the `sm_` prefix): `sm"90"` for
baseline, `sm"90a"` for architecture-accelerated, `sm"100f"` for family-specific.
Equivalent to calling `SMVersion(str)`; parses at macro-expansion time, so the
resulting `SMVersion` is a compile-time constant in the surrounding expression.

# Examples
```julia
julia> sm"103a"
sm"103a"

julia> sm"100f" == SMVersion(10, 0, :family)
true
```
"""
macro sm_str(s); SMVersion(s); end
