#
# Set-up
#

# Check availability NVCC
nvcc = get(ENV, "NVCC", "nvcc")
nvcc_path = try
    chomp(readall(`which $nvcc` .> DevNull))
catch
    error("Could not find NVCC -- consider specifying with NVCC environment variable")
end
nvcc_ver = Nullable{VersionNumber}()
for line in readlines(`$(nvcc_path) --version`)
    m = match(r"release ([0-9.]+)", line)
    if m != nothing
        nvcc_ver = Nullable(VersionNumber(m.captures[1]))
    end
end
if isnull(nvcc_ver)
    error("Could not parse NVCC version info")
end

# Determine host compiler version requirements
const hostcc_support = [
    (v"5.0", v"4.6.4"),
    (v"5.5", v"4.7.2"),
    (v"6.0", v"4.8.1") ]
if get(nvcc_ver) < hostcc_support[1][1]
    error("No support for CUDA < $(hostcc_req[1][1])")
end
hostcc_maxver = Nullable{VersionNumber}()
for i = 1:length(hostcc_support)
    if get(nvcc_ver) == hostcc_support[i][1]
        hostcc_maxver = Nullable(hostcc_support[i][2])
        break
    end
end
if isnull(hostcc_maxver)
    error("Unknown NVCC version $(get(nvcc_ver))")
end

# Collect possible hostcc executable names
hostcc_names = [ "gcc" ]
for ver in [v"4.4" v"4.5" v"4.6" v"4.7" v"4.8" v"4.9"]
    push!(hostcc_names, "gcc-$(ver.major).$(ver.minor)")
    push!(hostcc_names, "gcc$(ver.major)$(ver.minor)")
end

# Check availability host compiler
hostcc_possibilities = []
for hostcc in hostcc_names
    hostcc_path = try
        chomp(readall(`which $hostcc` .> DevNull))
    catch
        continue
    end

    verstring = chomp(readlines(`$hostcc_path --version`)[1])
    m = match(Regex("^$hostcc \\(.*\\) ([0-9.]+)\$"), verstring)
    if m == nothing
        warn("Could not parse GCC version info (\"$verstring\")")
    end
    hostcc_ver = VersionNumber(m.captures[1])

    if hostcc_ver <= get(hostcc_maxver)
        push!(hostcc_possibilities, (hostcc_path, hostcc_ver))
    end
end
if length(hostcc_possibilities) == 0
    error("Could not find a suitable host compiler")
end
sort!(hostcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
hostcc = hostcc_possibilities[1]

# Determine compilation options
flags = [ "--compiler-bindir", hostcc[1] ]
if haskey(ENV, "ARCH")
    append!(flags, [ "--gpu-architecture", ENV["ARCH"] ])
end


#
# Compilation
#

using CUDA

scriptdir = dirname(Base.source_path())

# Generate a temporary file with specific suffix
function mkstemps(suffix::String)
    b = joinpath(tempdir(), "tmpXXXXXX$suffix")
    p = ccall(:mkstemps, Int32, (Ptr{UInt8}, Cint), b, length(suffix)) # modifies b
    return (b, fdio(p, true))
end

# Macro for C++ compilation
macro compile(kernel, code)
    kernel_name = string(kernel)
    :( _compile($kernel_name, $code) )
end
function _compile(kernel, code)
    (source, io) = mkstemps(".cu")
    write(io, """
extern "C"
{
$code
}
""")
    close(io)

    builddir = "$scriptdir/.build"
    if !isdir(builddir)
        mkdir(builddir)
    end

    output = "$builddir/$kernel.ptx"
    try
        run(`$(nvcc_path) $flags -ptx -o $output $source`)
    catch
        error("compilation of kernel $kernel failed (typo in C++ source?)")
    end

    mod = try
        CuModule(output)
    catch
        error("loading of kernel $kernel failed (invalid CUDA code?)")
    end
    func = try
        CuFunction(mod, kernel)
    catch
        error("could not find kernel $kernel in the compiled binary (wrong function name?)")
    end

    fname = symbol("reference_$kernel")
    @eval begin
        function $fname()
            $func
        end
    end
end

# Include all files
scriptdir = dirname(Base.source_path())
for file in readdir(scriptdir)
    m = match(r"^(.*)\.jl$", file)
    if m != nothing && m.captures[1] != "load"
        include(file)
    end
end
