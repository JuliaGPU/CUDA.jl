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
# require for them, is encoded directly in the keys of `ptx_sm_db` and `llvm_sm_db`
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


## devices supported by ptxas

# Source:
# - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
# - ptxas |& grep -A 10 '\--gpu-name'
const ptxas_cap_db = Dict(
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

function ptxas_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in ptxas_cap_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## PTX ISAs supported by ptxas

# Source: PTX ISA document, Release History table
const ptxas_ptx_db = Dict(
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
    v"9.3" => between(v"13.3", highest),
)

function ptxas_ptx_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in ptxas_ptx_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## devices supported by each PTX ISA

# Source: PTX ISA document, Release History table. Architecture-specific (`*a`) variants
# were introduced at CC 9.0 / PTX 8.0; family-specific (`*f`) variants at CC 10.0 / PTX 8.8.
const ptx_sm_db = Dict{SMVersion, VersionRange}(
    sm"10"   => between(v"1.0", highest),
    sm"11"   => between(v"1.0", highest),
    sm"12"   => between(v"1.2", highest),
    sm"13"   => between(v"1.2", highest),
    sm"20"   => between(v"2.0", highest),
    sm"30"   => between(v"3.1", highest),
    sm"32"   => between(v"4.0", highest),
    sm"35"   => between(v"3.1", highest),
    sm"37"   => between(v"4.1", highest),
    sm"50"   => between(v"4.0", highest),
    sm"52"   => between(v"4.1", highest),
    sm"53"   => between(v"4.2", highest),
    sm"60"   => between(v"5.0", highest),
    sm"61"   => between(v"5.0", highest),
    sm"62"   => between(v"5.0", highest),
    sm"70"   => between(v"6.0", highest),
    sm"72"   => between(v"6.1", highest),
    sm"75"   => between(v"6.3", highest),
    sm"80"   => between(v"7.0", highest),
    sm"86"   => between(v"7.1", highest),
    sm"87"   => between(v"7.4", highest),
    sm"89"   => between(v"7.8", highest),
    sm"90"   => between(v"7.8", highest),
    sm"90a"  => between(v"8.0", highest),
    sm"100"  => between(v"8.6", highest),
    sm"100a" => between(v"8.6", highest),
    sm"100f" => between(v"8.8", highest),
    sm"101"  => between(v"8.6", highest),
    sm"101a" => between(v"8.6", highest),
    sm"101f" => between(v"8.8", highest),
    sm"103"  => between(v"8.8", highest),
    sm"103a" => between(v"8.8", highest),
    sm"103f" => between(v"8.8", highest),
    sm"120"  => between(v"8.7", highest),
    sm"120a" => between(v"8.7", highest),
    sm"120f" => between(v"8.8", highest),
    sm"121"  => between(v"8.8", highest),
    sm"121a" => between(v"8.8", highest),
    sm"121f" => between(v"8.8", highest),
)

# Set of `SMVersion`s (across all feature sets) whose ptxas floor is met by `ver`.
function ptx_sm_support(ver::VersionNumber)
    caps = Set{SMVersion}()
    for (cap, r) in ptx_sm_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## devices supported by the LLVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td. Each `def : Proc<"sm_NN[a|f]", ...>` shows up
# here as a separate entry; without an entry LLVM does not know the variant CPU name and
# constructing a TargetMachine with it would fall back to a generic subtarget.
const llvm_sm_db = Dict{SMVersion, VersionRange}(
    sm"20"   => between(v"3.2", highest),
    sm"21"   => between(v"3.2", highest),
    sm"30"   => between(v"3.2", highest),
    sm"32"   => between(v"3.7", highest),
    sm"35"   => between(v"3.2", highest),
    sm"37"   => between(v"3.7", highest),
    sm"50"   => between(v"3.5", highest),
    sm"52"   => between(v"3.7", highest),
    sm"53"   => between(v"3.7", highest),
    sm"60"   => between(v"3.9", highest),
    sm"61"   => between(v"3.9", highest),
    sm"62"   => between(v"3.9", highest),
    sm"70"   => between(v"6", highest),
    sm"72"   => between(v"7", highest),
    sm"75"   => between(v"8", highest),
    sm"80"   => between(v"11", highest),
    sm"86"   => between(v"13", highest),
    sm"87"   => between(v"16", highest),
    sm"89"   => between(v"16", highest),
    sm"90"   => between(v"16", highest),
    sm"90a"  => between(v"18", highest),
    sm"100"  => between(v"20", highest),
    sm"100a" => between(v"20", highest),
    sm"100f" => between(v"21", highest),
    sm"101"  => between(v"20", highest),
    sm"101a" => between(v"20", highest),
    sm"101f" => between(v"21", highest),
    sm"103"  => between(v"21", highest),
    sm"103a" => between(v"21", highest),
    sm"103f" => between(v"21", highest),
    sm"120"  => between(v"20", highest),
    sm"120a" => between(v"20", highest),
    sm"120f" => between(v"21", highest),
    sm"121"  => between(v"21", highest),
    sm"121a" => between(v"21", highest),
    sm"121f" => between(v"21", highest),
)

# Set of `SMVersion`s (across all feature sets) supported by LLVM `ver`.
function llvm_sm_support(ver::VersionNumber)
    caps = Set{SMVersion}()
    for (cap, r) in llvm_sm_db
        if ver in r
            push!(caps, cap)
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
    v"9.1" => between(v"23", highest),
    v"9.2" => between(v"23", highest),
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

    # `.sm` is `Set{SMVersion}` (with variants); `.ptx` is `Set{VersionNumber}`.
    # `ptxas_compat()` returns `.cap` as `Set{VersionNumber}` because ptxas-level
    # support is per-CC -- the names track the value type.
    return (sm=llvm_sm_support(version),
            ptx=llvm_ptx_support(version))
end

function ptxas_compat(version=compiler_version())
    return (cap=ptxas_cap_support(version),
            ptx=ptxas_ptx_support(version))
end
