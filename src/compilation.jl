export
    @compile

const initialized = Ref{Bool}()
const nvcc = Ref{ASCIIString}()
const flags = Ref{Array{ASCIIString, 1}}()

function discover_toolchain()
    # Check availability NVCC
    global nvcc
    if haskey(ENV, "NVCC")
        nvcc[] = get(ENV, "NVCC")
    else
        try
            nvcc[] = chomp(readall(pipeline(`which nvcc`, stderr=DevNull)))
        catch e
            error("could not find NVCC -- consider specifying with NVCC environment variable")
        end
    end
    nvcc_ver = Nullable{VersionNumber}()
    for line in readlines(`$(nvcc[]) --version`)
        m = match(r"release ([0-9.]+)", line)
        if m != nothing
            nvcc_ver = Nullable(VersionNumber(m.captures[1]))
        end
    end
    if isnull(nvcc_ver)
        error("could not parse NVCC version info")
    end

    # Determine host compiler version requirements
    # Source: CUDA Getting Started Guide for Linux
    const hostcc_support = [
        (v"5.0", v"4.6.4"),
        (v"5.5", v"4.7.3"),
        (v"6.0", v"4.8.1"),
        (v"6.5", v"4.8.2"),
        (v"7.0", v"4.9.2"),
        (v"7.5", v"4.9.2") ]
    if get(nvcc_ver) < hostcc_support[1][1]
        error("no support for CUDA < $(hostcc_req[1][1])")
    end
    hostcc_maxver = Nullable{VersionNumber}()
    for i = 1:length(hostcc_support)
        if get(nvcc_ver) == hostcc_support[i][1]
            hostcc_maxver = Nullable(hostcc_support[i][2])
            break
        end
    end
    if isnull(hostcc_maxver)
        error("unknown NVCC version $(get(nvcc_ver))")
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
            chomp(readall(pipeline(`which $hostcc`, stderr=DevNull)))
        catch
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
        error("could not find a suitable host compiler (your NVCC $(get(nvcc_ver)) needs GCC <= $(get(hostcc_maxver)))")
    end
    sort!(hostcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
    hostcc = hostcc_possibilities[1]

    # Determine compilation options
    global flags
    flags[] = [ "--compiler-bindir", hostcc[1] ]
    if haskey(ENV, "ARCH")
        append!(flags[], [ "--gpu-architecture", ENV["ARCH"] ])
    end
end


macro compile(dev, kernel, code)
    global initialized
    if !initialized[]
        discover_toolchain()
        initialized[] = true
    end

    kernel_name = string(kernel)
    containing_file = Base.source_path()

    @gensym func
    return quote
        const $func = _compile($(esc(dev)), $kernel_name, $code, $containing_file)

        function $(esc(kernel))()
            return $func
        end
    end
end

function _compile(dev, kernel, code, containing_file)
    # Get the target architecture
    if haskey(ENV, "CUDA_FORCE_GPU_ARCH")
        arch = ENV["CUDA_FORCE_GPU_ARCH"]
    else
        arch = architecture(dev)
    end

    # Write the source into a compilable file
    (source, io) = mkstemps(".cu")
    write(io, """
extern "C"
{
$code
}
""")
    close(io)

    # Manage the build directory
    scriptdir = dirname(containing_file)
    builddir = "$scriptdir/.build"
    if !isdir(builddir)
        mkdir(builddir)
    end

    # Check if we need to compile
    output = "$builddir/$kernel-$arch.ptx"
    if isfile(output)
        need_compile = (stat(containing_file).mtime > stat(output).mtime)
    else
        need_compile = true
    end

    # Compile the source, if necessary
    if need_compile
        global flags
        compile_flags = vcat(flags[], ["--gpu-architecture", arch])

        global nvcc
        try
            run(`$(nvcc[]) $(compile_flags) -ptx -o $output $source`)
        catch
            error("compilation of kernel $kernel failed (typo in C++ source?)")
        end

        if !isfile(output)
            error("compilation of kernel $kernel failed (no output generated)")
        end
    end
    rm(source)

    # Pass the module to the CUDA driver
    mod = try
        CuModule(output)
    catch
        error("loading of kernel $kernel failed (invalid CUDA code?)")
    end

    # Load the function pointer
    func = try
        CuFunction(mod, kernel)
    catch
        error("could not find kernel $kernel in the compiled binary (wrong function name?)")
    end

    return func
end
