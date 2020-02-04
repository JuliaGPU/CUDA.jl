# compatibility of Julia, CUDA and LLVM

const lowest = v"0"
const highest = v"999"

# some version comparisons need to ignore part of the version number
strip_patch(ver) = VersionNumber(ver.major, ver.minor)
strip_minor(ver) = VersionNumber(ver.major)

verlist(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")


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
    v"3.0" => v"4.2":highest,
    v"3.2" => v"6.0":highest,
    v"3.5" => v"5.0":highest,
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
)

function cuda_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in cuda_cap_db
        if strip_patch(ver) in r
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
)

function cuda_ptx_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in cuda_ptx_db
        if strip_patch(ver) in r
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
)

function llvm_cap_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in llvm_cap_db
        if strip_patch(ver) in r
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
)

function llvm_ptx_support(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in llvm_ptx_db
        if strip_patch(ver) in r
            push!(caps, cap)
        end
    end
    return caps
end


## high-level functions that return target and isa support

function llvm_compat(version=LLVM.version())
    @debug("Using LLVM v$version")

    # https://github.com/JuliaGPU/CUDAnative.jl/issues/428
    if version >= v"8.0" && VERSION < v"1.3.0-DEV.547"
        error("LLVM 8.0 requires a newer version of Julia")
    end

    InitializeAllTargets()
    haskey(targets(), "nvptx") ||
        error("""
            Your LLVM does not support the NVPTX back-end.

            This is very strange; both the official binaries
            and an unmodified build should contain this back-end.""")

    cap_support = sort(collect(llvm_cap_support(version)))

    ptx_support = llvm_ptx_support(version)
    push!(ptx_support, v"6.0") # JuliaLang/julia#23817
    ptx_support = sort(collect(ptx_support))

    @debug("LLVM supports capabilities $(verlist(cap_support)) with PTX $(verlist(ptx_support))")
    return (cap=cap_support, ptx=ptx_support)
end

function cuda_compat(driver_version=CUDAdrv.version(), toolkit_version=CUDAnative.version())
    @debug("Using CUDA driver v$driver_version and toolkit v$toolkit_version")

    # the toolkit version as reported contains major.minor.patch,
    # but the version number returned by libcuda is only major.minor.
    toolkit_version = VersionNumber(toolkit_version.major, toolkit_version.minor)
    if toolkit_version > driver_version
        @warn("""CUDA $(toolkit_version.major).$(toolkit_version.minor) is not supported by
                 your driver (which supports up to $(driver_version.major).$(driver_version.minor))""")
    end

    driver_cap_support = cuda_cap_support(driver_version)
    toolkit_cap_support = cuda_cap_support(toolkit_version)
    cap_support = sort(collect(driver_cap_support ∩ toolkit_cap_support))

    driver_ptx_support = cuda_ptx_support(driver_version)
    toolkit_ptx_support = cuda_ptx_support(toolkit_version)
    ptx_support = sort(collect(driver_ptx_support ∩ toolkit_ptx_support))

    @debug("CUDA driver supports capabilities $(verlist(driver_cap_support)) with PTX $(verlist(driver_ptx_support))")
    @debug("CUDA toolkit supports capabilities $(verlist(toolkit_cap_support)) with PTX $(verlist(toolkit_ptx_support))")

    return (cap=cap_support, ptx=ptx_support)
end
