const ci_cache = GPUCompiler.CodeCache()

const DeviceProperties = @NamedTuple{cap::VersionNumber, ptx::VersionNumber,
                                     exitable::Bool, debuginfo::Bool, unreachable::Bool}
const __device_properties = LazyInitialized{Vector{DeviceProperties}}()
function device_properties(dev)
    props = get!(__device_properties) do
        # NOTE: this doesn't initialize any context, so we can pre-compute for all devices
        val = Vector{DeviceProperties}(undef, ndevices())
        for dev in devices()
            cap = supported_capability(capability(dev))
            ptx = v"6.3"    # we only need 6.2, but NVPTX doesn't support that

            # we need to take care emitting LLVM instructions like `unreachable`, which
            # may result in thread-divergent control flow that older `ptxas` doesn't like.
            # see e.g. JuliaGPU/CUDAnative.jl#4
            unreachable = true
            if cap < v"7" || toolkit_version() < v"11.3"
                unreachable = false
            end

            # there have been issues with emitting PTX `exit` instead of `trap` as well,
            # see e.g. JuliaGPU/CUDA.jl#431 and NVIDIA bug #3231266 (but since switching
            # to the toolkit's `ptxas` that specific machine/GPU now _requires_ exit...)
            exitable = true
            if cap < v"7"
                exitable = false
            end

            debuginfo = getenv("JULIA_CUDA_DEBUG_INFO", true)

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

PTXCompiler = Compiler{PTXCompilerTarget}
CUDACompiler = Compiler{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(job::CUDACompiler) = CUDA

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(job::CUDACompiler, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{PTXCompiler, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

function GPUCompiler.link_libraries!(compiler::CUDACompiler, source::FunctionSpec,
                                     mod::LLVM.Module, undefined_fns::Vector{String})
    invoke(GPUCompiler.link_libraries!,
           Tuple{PTXCompiler, typeof(source), typeof(mod), typeof(undefined_fns)},
           compiler, source, mod, undefined_fns)
    link_libdevice!(mod, compiler.target.cap, undefined_fns)
end

GPUCompiler.ci_cache(job::CUDACompiler) = ci_cache

GPUCompiler.method_table(job::CUDACompiler) = method_table
