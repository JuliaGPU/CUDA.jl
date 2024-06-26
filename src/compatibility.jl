# compatibility of Julia, CUDA and LLVM

# NOTE: Target architectures with suffix “a”, such as sm_90a, include
# architecture-accelerated features that are supported on the specified architecture only,
# hence such targets do not follow the onion layer model. Therefore, PTX code generated for
# such targets cannot be run on later generation devices. Architecture-accelerated features
# can only be used with targets that support these features.

const lowest = v"0"
const highest = v"999"


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
    v"3.7"   => between(v"6.5", highest),
    v"5.0"   => between(v"6.0", highest),
    v"5.2"   => between(v"7.0", highest),
    v"5.3"   => between(v"7.5", highest),
    v"6.0"   => between(v"8.0", highest),
    v"6.1"   => between(v"8.0", highest),
    v"6.2"   => between(v"8.0", highest),
    v"7.0"   => between(v"9.0", highest),
    v"7.2"   => between(v"9.2", highest),
    v"7.5"   => between(v"10.0", highest),
    v"8.0"   => between(v"11.0", highest),
    v"8.6"   => between(v"11.1", highest),
    v"8.7"   => between(v"11.4", highest),
    v"8.9"   => between(v"11.8", highest),
    v"9.0"   => between(v"11.8", highest),
    #v"9.0a" => between(v"12.0", highest),
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

# Source: PTX ISA document, Release History table
const ptx_cap_db = Dict(
    v"1.0"   => between(v"1.0", highest),
    v"1.1"   => between(v"1.0", highest),
    v"1.2"   => between(v"1.2", highest),
    v"1.3"   => between(v"1.2", highest),
    v"2.0"   => between(v"2.0", highest),
    v"3.0"   => between(v"3.1", highest),
    v"3.2"   => between(v"4.0", highest),
    v"3.5"   => between(v"3.1", highest),
    v"3.7"   => between(v"4.1", highest),
    v"5.0"   => between(v"4.0", highest),
    v"5.2"   => between(v"4.1", highest),
    v"5.3"   => between(v"4.2", highest),
    v"6.0"   => between(v"5.0", highest),
    v"6.1"   => between(v"5.0", highest),
    v"6.2"   => between(v"5.0", highest),
    v"7.0"   => between(v"6.0", highest),
    v"7.2"   => between(v"6.1", highest),
    v"7.5"   => between(v"6.3", highest),
    v"8.0"   => between(v"7.0", highest),
    v"8.6"   => between(v"7.1", highest),
    v"8.7"   => between(v"7.4", highest),
    v"8.9"   => between(v"7.8", highest),
    v"9.0"   => between(v"7.8", highest),
    #v"9.0a" => between(v"8.0", highest)
)

function ptx_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in ptx_cap_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


## devices supported by the LLVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td
const llvm_cap_db = Dict(
    v"2.0"   => between(v"3.2", highest),
    v"2.1"   => between(v"3.2", highest),
    v"3.0"   => between(v"3.2", highest),
    v"3.2"   => between(v"3.7", highest),
    v"3.5"   => between(v"3.2", highest),
    v"3.7"   => between(v"3.7", highest),
    v"5.0"   => between(v"3.5", highest),
    v"5.2"   => between(v"3.7", highest),
    v"5.3"   => between(v"3.7", highest),
    v"6.0"   => between(v"3.9", highest),
    v"6.1"   => between(v"3.9", highest),
    v"6.2"   => between(v"3.9", highest),
    v"7.0"   => between(v"6", highest),
    v"7.2"   => between(v"7", highest),
    v"7.5"   => between(v"8", highest),
    v"8.0"   => between(v"11", highest),
    v"8.6"   => between(v"13", highest),
    v"8.7"   => between(v"16", highest),
    v"8.9"   => between(v"16", highest),
    v"9.0"   => between(v"16", highest),
    #v"9.0a" => between(v"18", highest),
)

function llvm_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in llvm_cap_db
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

    cap_support = sort(collect(llvm_cap_support(version)))
    ptx_support = sort(collect(llvm_ptx_support(version)))

    return (cap=cap_support, ptx=ptx_support)
end

function cuda_compat(driver=driver_version(), runtime=runtime_version())
    driver_cap_support = cuda_cap_support(driver)
    toolkit_cap_support = cuda_cap_support(runtime)
    cap_support = sort(collect(driver_cap_support ∩ toolkit_cap_support))

    driver_ptx_support = cuda_ptx_support(driver)
    toolkit_ptx_support = cuda_ptx_support(runtime)
    ptx_support = sort(collect(driver_ptx_support ∩ toolkit_ptx_support))

    return (cap=cap_support, ptx=ptx_support)
end

function ptx_compat(ptx)
    return (cap=ptx_cap_support(ptx),)
end
