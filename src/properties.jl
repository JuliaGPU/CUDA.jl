# versions of the CUDA toolkit
const toolkits = [v"4.0", v"4.2", v"5.0", v"6.0", v"6.5", v"7.0", v"7.5", v"8.0", v"9.0"]


immutable VersionRange
    lowest::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v < r.upper)

Base.colon(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lowest:v) :
    v > r.upper ? (v:r.upper) : (v:v)

const lowest = v"0"
const highest = v"999"


# GCC compilers supported by the CUDA toolkit

# Source: CUDA/include/host_config.h
const cuda_gcc_db = Dict(
    v"5.5" => lowest:v"4.9-",   # (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)) && #error
    v"6.0" => lowest:v"4.9-",   # (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)) && #error
    v"6.5" => lowest:v"4.9-",   # (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)) && #error
    v"7.0" => lowest:v"4.10-",  # (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9)) && #error
    v"7.5" => lowest:v"4.10-",  # (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9)) && #error
    v"8.0" => lowest:v"6-",     # (__GNUC__ > 5)                                          && #error
    v"9.0" => lowest:v"7-"      # (__GNUC__ > 6)                                          && #error
)

function gcc_for_cuda(ver::VersionNumber)
    haskey(cuda_gcc_db, ver) || error("no support for CUDA $ver")
    return cuda_gcc_db[ver]
end


# devices supported by the CUDA toolkit

# Source:
# - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
# - ptxas |& grep -A 10 '\--gpu-name'
const dev_cuda_db = Dict(
    v"1.0" => lowest:v"7.0",
    v"1.1" => lowest:v"7.0",
    v"1.2" => lowest:v"7.0",
    v"1.3" => lowest:v"7.0",
    v"2.0" => lowest:v"9.0",
    v"2.1" => lowest:v"9.0",
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
    v"7.0" => v"9.0":highest
)

function devices_for_cuda(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in dev_cuda_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


# devices supported by the LLVM NVPTX back-end

# Source: LLVM/lib/Target/NVPTX/NVPTX.td
const dev_llvm_db = Dict(
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
    v"6.2" => v"3.9":highest
)

function devices_for_llvm(ver::VersionNumber)
    caps = Set{VersionNumber}()
    for (cap,r) in dev_llvm_db
        if ver in r
            push!(caps, cap)
        end
    end
    return caps
end


# other

shader(cap::VersionNumber) = "sm_$(cap.major)$(cap.minor)"
