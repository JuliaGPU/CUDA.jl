# compatibility of Julia, CUDA and LLVM

const lowest = v"0"
const highest = v"999"

join_capabilities(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")


## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

import Base.(:)
(:)(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)


## devices supported by the CUDA toolkit

# Source:
# - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
# - ptxas |& grep -A 10 '\--gpu-name'
const cuda_cap_db = Dict(
    v"1.0" => lowest:v"6.5",
    v"1.1" => lowest:v"6.5",
    v"1.2" => lowest:v"6.5",
    v"1.3" => lowest:v"6.5",
    v"2.0" => lowest:v"8.0",
    v"2.1" => lowest:v"8.0",
    v"3.0" => v"4.2":v"10.2",
    v"3.2" => v"6.0":v"10.2",
    v"3.5" => v"5.0":v"11.8",
    v"3.7" => v"6.5":highest,
    v"5.0" => v"6.0":highest,
    v"5.2" => v"7.0":highest,
    v"5.3" => v"7.5":highest,
    v"6.0" => v"8.0":highest,
    v"6.1" => v"8.0":highest,
    v"6.2" => v"8.0":highest,
    v"7.0" => v"9.0":highest,
    v"7.2" => v"9.2":highest,
    v"7.5" => v"10.0":highest,
    v"8.0" => v"11.0":highest,
    v"8.6" => v"11.1":highest,
    v"8.7" => v"11.5":highest,
    v"9.0" => v"11.8":highest,
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

# Source:
# - PTX ISA document, Release History table
const cuda_ptx_db = Dict(
    v"1.0" => v"1.0":highest,
    v"1.1" => v"1.1":highest,
    v"1.2" => v"2.0":highest,
    v"1.3" => v"2.1":highest,
    v"1.4" => v"2.2":highest,
    v"1.5" => v"2.2":highest,
    v"2.0" => v"3.0":highest,
    v"2.1" => v"3.1":highest,
    v"2.2" => v"3.2":highest,
    v"2.3" => v"4.2":highest,
    v"3.0" => v"4.1":highest,
    v"3.1" => v"5.0":highest,
    v"3.2" => v"5.5":highest,
    v"4.0" => v"6.0":highest,
    v"4.1" => v"6.5":highest,
    v"4.2" => v"7.0":highest,
    v"4.3" => v"7.5":highest,
    v"5.0" => v"8.0":highest,
    v"6.0" => v"9.0":highest,
    v"6.1" => v"9.1":highest,
    v"6.2" => v"9.2":highest,
    v"6.3" => v"10.0":highest,
    v"6.4" => v"10.1":highest,
    v"6.5" => v"10.2":highest,
    v"7.0" => v"11.0":highest,
    v"7.1" => v"11.1":highest,
    v"7.2" => v"11.2":highest,
    v"7.3" => v"11.3":highest,
    v"7.4" => v"11.4":highest,
    v"7.5" => v"11.5":highest,
    v"7.6" => v"11.6":highest,
    v"7.7" => v"11.7":highest,
    v"7.8" => v"11.8":highest,
    v"8.0" => v"12.0":highest,
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


## devices supported by the LLVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td
const llvm_cap_db = Dict(
    v"2.0" => v"3.2":highest,
    v"2.1" => v"3.2":highest,
    v"3.0" => v"3.2":highest,
    v"3.2" => v"3.7":highest,
    v"3.5" => v"3.2":highest,
    v"3.7" => v"3.7":highest,
    v"5.0" => v"3.5":highest,
    v"5.2" => v"3.7":highest,
    v"5.3" => v"3.7":highest,
    v"6.0" => v"3.9":highest,
    v"6.1" => v"3.9":highest,
    v"6.2" => v"3.9":highest,
    v"7.0" => v"6.0":highest,
    v"7.2" => v"7.0":highest,
    v"7.5" => v"8.0":highest,
    v"8.0" => v"11.0":highest,
    v"8.6" => v"13.0":highest,
    v"8.7" => v"16.0":highest,
    v"8.9" => v"16.0":highest,
    v"9.0" => v"16.0":highest,
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
    v"3.0" => v"3.2":v"3.5",
    v"3.1" => v"3.2":v"3.5",
    v"3.2" => v"3.5":highest,
    v"4.0" => v"3.5":highest,
    v"4.1" => v"3.7":highest,
    v"4.2" => v"3.7":highest,
    v"4.3" => v"3.9":highest,
    v"5.0" => v"3.9":highest,
    v"6.0" => v"6.0":highest,
    v"6.1" => v"7.0":highest,
    v"6.3" => v"8.0":highest,
    v"6.4" => v"9.0":highest,
    v"6.5" => v"11.0":highest,
    v"7.0" => v"11.0":highest,
    v"7.1" => v"13.0":highest,
    v"7.2" => v"13.0":highest,
    v"7.3" => v"14.0":highest,
    v"7.4" => v"14.0":highest,
    v"7.5" => v"14.0":highest,
    v"7.6" => v"16.0":highest,
    v"7.7" => v"16.0":highest,
    v"7.8" => v"16.0":highest,
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

function supported_toolchain()
    llvm_support = llvm_compat()
    cuda_support = cuda_compat()

    target_support = sort(collect(llvm_support.cap ∩ cuda_support.cap))
    isempty(target_support) && error("Your toolchain does not support any device capability")

    ptx_support = sort(collect(llvm_support.ptx ∩ cuda_support.ptx))
    isempty(ptx_support) && error("Your toolchain does not support any PTX ISA")

    return (cap=target_support, ptx=ptx_support)
end
