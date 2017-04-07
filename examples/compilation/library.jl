# EXCLUDE FROM TESTING

using CUDAdrv
using Compat

# Generate a temporary file with specific suffix
# NOTE: mkstemps is glibc 2.19+, so emulate its behavior
function mkstemps(suffix::AbstractString)
    base = tempname()
    filename = base * suffix

    # make sure the filename is unique
    i = 0
    while isfile(filename)
        i += 1
        filename = base * ".$i" * suffix
    end

    return (filename, Base.open(filename, "w"))
end

"Database of compute capabilities with matching shader model, and initial version of the CUDA
toolkit supporting this architecture."
const architectures = [
    # cap       SM          CUDA
    # NOTE: CUDA versions only checked starting with v4.0
    (v"1.0",    "sm_10",    v"4.0"),
    (v"1.1",    "sm_11",    v"4.0"),
    (v"1.2",    "sm_12",    v"4.0"),
    (v"1.3",    "sm_13",    v"4.0"),
    (v"2.0",    "sm_20",    v"4.0"),
    (v"2.1",    "sm_21",    v"4.0"),
    (v"3.0",    "sm_30",    v"4.2"),
    (v"3.2",    "sm_32",    v"6.0"),
    (v"3.5",    "sm_35",    v"5.0"),
    (v"3.7",    "sm_37",    v"6.5"),
    (v"5.0",    "sm_50",    v"6.0"),
    (v"5.2",    "sm_52",    v"7.0"),
    (v"5.3",    "sm_53",    v"7.5"),
    (v"6.0",    "sm_60",    v"8.0"),
    (v"6.1",    "sm_61",    v"8.0"),
    (v"6.2",    "sm_62",    v"8.0") ]

"Return the most recent supported architecture for a CUDA device"
function architecture(dev::CuDevice; cuda=nothing)
    cap = capability(dev)

    # Devices are compatible with code generated for lower compute models
    compat_architectures = filter(x -> x[1] <= cap, architectures)

    if cuda != nothing
        compat_architectures = filter(x -> cuda >= x[3], compat_architectures)
    end

    if length(compat_architectures) == 0
        error("No support for requested device or software (compute model <= $cap" * 
              cuda==nothing?"":", CUDA >= $cuda)")
    end

    return compat_architectures[length(compat_architectures)][2]
end

type Toolchain
    version::VersionNumber
    nvcc::String
    flags::Vector{String}
end

const toolchain = Ref{Toolchain}()
function discover_toolchain()
    # Check availability NVCC
    if haskey(ENV, "NVCC")
        nvcc = ENV["NVCC"]
    elseif haskey(ENV, "CUDA_PATH")
        nvcc = joinpath(ENV["CUDA_PATH"], "bin", "nvcc") * (is_windows() ? ".exe" : "")
    elseif !is_windows()
        try
            nvcc = chomp(readstring(pipeline(`which nvcc`, stderr=DevNull)))
        catch ex
            isa(ex, ErrorException) || rethrow(ex)
            rethrow(ErrorException("could not find NVCC; consider setting the NVCC environment variable or the CUDA_PATH environment variable"))
        end
    else
        throw(ErrorException("could not find NVCC; consider setting the NVCC environment variable or the CUDA_PATH environment variable"))
    end
    nvcc_ver = Nullable{VersionNumber}()
    for line in readlines(`$nvcc --version`)
        m = match(r"release ([0-9.]+)", line)
        if m != nothing
            nvcc_ver = Nullable(VersionNumber(m.captures[1]))
        end
    end
    if isnull(nvcc_ver)
        error("could not parse NVCC version info")
    end
    version = get(nvcc_ver)


    flags = [ "--compiler-bindir" ]

    # Collect possible hostcc executable names
    if !is_windows()
        # Determine host compiler version requirements
        # Source: "CUDA Getting Started Guide for Linux"
        const hostcc_support = [
            (v"5.0", v"4.6.4"),
            (v"5.5", v"4.7.3"),
            (v"6.0", v"4.8.1"),
            (v"6.5", v"4.8.2"),
            (v"7.0", v"4.9.2"),
            (v"7.5", v"4.9.2"),
            (v"8.0", v"5.3.1") ]
        if version < hostcc_support[1][1]
            error("no support for CUDA < $(hostcc_req[1][1])")
        end
        hostcc_maxver = Nullable{VersionNumber}()
        for i = 1:length(hostcc_support)
            if version == hostcc_support[i][1]
                hostcc_maxver = Nullable(hostcc_support[i][2])
                break
            end
        end
        if isnull(hostcc_maxver)
            error("unknown NVCC version $version")
        end
        hostcc_names = [ "gcc" ]
        for ver in [v"4.4" v"4.5" v"4.6" v"4.7" v"4.8" v"4.9"]
            push!(hostcc_names, "gcc-$(ver.major).$(ver.minor)")
            push!(hostcc_names, "gcc$(ver.major)$(ver.minor)")
        end

        # Check availability host compiler
        hostcc_possibilities = []
        for hostcc in hostcc_names
            hostcc_path = try
                chomp(readstring(pipeline(`which $hostcc`, stderr=DevNull)))
            catch ex
                isa(ex, ErrorException) || rethrow(ex)
                continue
            end

            verstring = chomp(readlines(`$hostcc_path --version`)[1])
            m = match(Regex("^$hostcc \\(.*\\) ([0-9.]+)"), verstring)
            if m == nothing
                warn("could not parse GCC version info (\"$verstring\"), skipping this compiler")
                continue
            end
            hostcc_ver = VersionNumber(m.captures[1])

            if hostcc_ver <= get(hostcc_maxver)
                push!(hostcc_possibilities, (hostcc_path, hostcc_ver))
            end
        end
        if length(hostcc_possibilities) == 0
            error("could not find a suitable host compiler (your NVCC $version needs GCC <= $(get(hostcc_maxver)))")
        end
        sort!(hostcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
        hostcc = hostcc_possibilities[1]

        push!(flags, hostcc[1])
    else
        vc_versions = ["VS140COMNTOOLS", "VS120COMNTOOLS", "VS110COMNTOOLS", "VS100COMNTOOLS"]
        vs_cmd_tools_dir = ENV[vc_versions[first(find(x -> haskey(ENV, x), vc_versions))]]
        hostccbin = joinpath(dirname(vs_cmd_tools_dir), "..", "..", "VC", Sys.WORD_SIZE == 64 ? "amd64" : "", "cl.exe")

        push!(flags, hostccbin)
    end

    # Determine compilation options
    if haskey(ENV, "ARCH")
        append!(flags, [ "--gpu-architecture", ENV["ARCH"] ])
    end

    global toolchain
    toolchain[] = Toolchain(version, nvcc, flags)
end
discover_toolchain()


macro compile(dev, kernel, code)
    kernel_name = string(kernel)
    containing_file = @__FILE__

    return Expr(:toplevel,
        Expr(:export,esc(kernel)),
        :($(esc(kernel)) = _compile($(esc(dev)), $kernel_name, $code, $containing_file)))
end

type CompileError <: Base.WrappedException
    message::String
    error
end

const builddir = joinpath(@__DIR__, ".cache")

function _compile(dev, kernel, code, containing_file)
    global toolchain

    # Get the target architecture
    arch = architecture(dev; cuda=toolchain[].version)

    if !isdir(builddir)
        println("Writing build artifacts to $builddir")
        mkpath(builddir)
    end

    # Check if we need to compile
    codehash = hex(hash(code))
    output = "$builddir/$(kernel)_$(codehash)-$(arch).ptx"
    if isfile(output)
        need_compile = (stat(containing_file).mtime > stat(output).mtime)
    else
        need_compile = true
    end

    # Compile the source, if necessary
    if need_compile
        # Write the source into a compilable file
        (source, io) = mkstemps(".cu")
        write(io, """
extern "C"
{
$code
}
""")
        close(io)

        compile_flags = vcat(toolchain[].flags, ["--gpu-architecture", arch])
        try
            # TODO: capture STDERR
            run(pipeline(`$(toolchain[].nvcc) $(compile_flags) -ptx -o $output $source`, stderr=DevNull))
        catch ex
            isa(ex, ErrorException) || rethrow(ex)
            rethrow(CompileError("compilation of kernel $kernel failed (typo in C++ source?)", ex))
        finally
            rm(source)
        end

        if !isfile(output)
            error("compilation of kernel $kernel failed (no output generated)")
        end
    end

    # Pass the module to the CUDA driver
    mod = try
        CuModuleFile(output)
    catch ex
        rethrow(CompileError("loading of kernel $kernel failed (invalid CUDA code?)", ex))
    end

    # Load the function pointer
    func = try
        CuFunction(mod, kernel)
    catch ex
        rethrow(CompileError("could not find kernel $kernel in the compiled binary (wrong function name?)", ex))
    end

    return func
end

function clean_cache()
    rm(builddir; recursive=true)
end
