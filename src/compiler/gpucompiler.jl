const ci_cache = GPUCompiler.CodeCache()

const __device_properties_lock = ReentrantLock()
const __device_properties = @NamedTuple{cap::VersionNumber, ptx::VersionNumber,
                                        exitable::Bool, debuginfo::Bool, unreachable::Bool}[]
function device_properties(dev)
    @lock __device_properties_lock  begin
        if isempty(__device_properties)
            resize!(__device_properties, ndevices())

            # determine compilation properties of each device
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

                debuginfo = true

                __device_properties[deviceid(dev)+1] =
                    (; cap, ptx, exitable, debuginfo, unreachable)
            end
        end
        @inbounds __device_properties[deviceid(dev)+1]
    end
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
