const ci_cache = GPUCompiler.CodeCache()

const DeviceProperties = @NamedTuple{cap::VersionNumber, ptx::VersionNumber,
                                     exitable::Bool, debuginfo::Bool, unreachable::Bool}
const __device_properties = LazyInitialized{Vector{DeviceProperties}}()
function device_properties(dev)
    props = get!(__device_properties) do
        toolchain = supported_toolchain()

        # NOTE: this doesn't initialize any context, so we can pre-compute for all devices
        val = Vector{DeviceProperties}(undef, ndevices())
        for dev in devices()
            # select the highest capability that is supported by both the toolchain and device
            caps = filter(toolchain_cap -> toolchain_cap <= capability(dev), toolchain.cap)
            isempty(caps) &&
                error("Your $(name(dev)) GPU with capability v$(capability(dev)) is not supported by the available toolchain")
            cap = maximum(caps)

            # select the PTX ISA we assume to be available
            # (we actually only need 6.2, but NVPTX doesn't support that)
            ptx = v"6.3"

            # we need to take care emitting LLVM instructions like `unreachable`, which
            # may result in thread-divergent control flow that older `ptxas` doesn't like.
            # see e.g. JuliaGPU/CUDAnative.jl#4
            unreachable = true
            if cap < v"7" || toolkit_release() < v"11.3"
                unreachable = false
            end

            # there have been issues with emitting PTX `exit` instead of `trap` as well,
            # see e.g. JuliaGPU/CUDA.jl#431 and NVIDIA bug #3231266 (but since switching
            # to the toolkit's `ptxas` that specific machine/GPU now _requires_ exit...)
            exitable = true
            if cap < v"7"
                exitable = false
            end

            # NVIDIA bug #3305774: ptxas segfaults with our debug info, fixed in 11.5.1
            debuginfo = toolkit_release() >= v"11.5" # we don't track patch versions...

            val[deviceid(dev)+1] =
                (; cap, ptx, exitable, debuginfo, unreachable)
        end
        val
    end
    @inbounds(props[deviceid(dev)+1])
end

function CUDACompilerTarget(dev::CuDevice; kwargs...)
    PTXCompilerTarget(; device_properties(dev)..., kwargs...)
end

struct CUDACompilerParams <: AbstractCompilerParams end

CUDACompilerJob = CompilerJob{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(@nospecialize(job::CUDACompilerJob)) = CUDA

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(@nospecialize(job::CUDACompilerJob), fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

function GPUCompiler.link_libraries!(@nospecialize(job::CUDACompilerJob), mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    invoke(GPUCompiler.link_libraries!,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(mod), typeof(undefined_fns)},
           job, mod, undefined_fns)
    link_libdevice!(mod, job.target.cap, undefined_fns)
end

GPUCompiler.ci_cache(@nospecialize(job::CUDACompilerJob)) = ci_cache

GPUCompiler.method_table(@nospecialize(job::CUDACompilerJob)) = method_table

GPUCompiler.kernel_state_type(job::CUDACompilerJob) = KernelState
