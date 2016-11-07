# Discover the CUDA and LLVM toolchains, and figure out which devices we can compile for

using CUDAdrv
using LLVM

# non-exported utility functions
import CUDAdrv: debug, DEBUG, trace, TRACE

# helper methods for querying version DBs
# (tool::VersionNumber => devices::Vector{VersionNumber})
search(db, predicate) =
    Set(Base.Iterators.flatten(valvec for (key,valvec) in db if predicate(key)))

# figure out which devices this LLVM library supports
llvm_ver = LLVM.version()
debug("Using LLVM v$llvm_ver")
const llvm_db = [
    v"3.2" => [v"2.0", v"2.1", v"3.0", v"3.5"],
    v"3.5" => [v"5.0"],
    v"3.7" => [v"3.2", v"3.7", v"5.2", v"5.3"],
    v"3.9" => [v"6.0", v"6.1", v"6.2"]
]
llvm_ver = LLVM.version()
llvm_support = search(llvm_db, ver -> ver <= llvm_ver)
isempty(llvm_support) && error("LLVM library $llvm_ver does not support any compatible device")

# figure out which devices this CUDA toolkit supports
cuda_ver = CUDAdrv.version()
debug("Using CUDA v$cuda_ver")
const cuda_db = [
    v"4.0" => [v"2.0", v"2.1"],
    v"4.2" => [v"3.0"],
    v"5.0" => [v"3.5"],
    v"6.0" => [v"3.2", v"5.0"],
    v"6.5" => [v"3.7"],
    v"7.0" => [v"5.2"],
    v"7.5" => [v"5.3"],
    v"8.0" => [v"6.0", v"6.1", v"6.2"]
]
cuda_support = search(cuda_db, ver -> ver <= cuda_ver)
isempty(cuda_support) && error("CUDA toolkit $cuda_ver does not support any compatible device")

# check if Julia's LLVM version matches ours
jl_llvm_ver = VersionNumber(Base.libllvm_version)
debug("Using Julia v$jl_llvm_ver")
if jl_llvm_ver != llvm_ver
    error("LLVM library $llvm_ver incompatible with Julia's LLVM $jl_llvm_ver")
end

# write ext.jl
toolchain_caps = Vector{VersionNumber}()
append!(toolchain_caps, llvm_support ∩ cuda_support)
debug("Supported toolchains: $(join(toolchain_caps, ", "))")
open(joinpath(@__DIR__, "ext.jl"), "w") do fh
    write(fh, """
        # Toolchain properties
        const llvm_version = v"$llvm_ver"
        const cuda_version = v"$cuda_ver"

        const toolchain_caps = $toolchain_caps
        """)
end
