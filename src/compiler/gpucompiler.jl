const ci_cache = GPUCompiler.CodeCache()

const __device_properties_lock = ReentrantLock()
const __device_properties = @NamedTuple{cap::VersionNumber, ptx::VersionNumber, exitable::Bool, debuginfo::Bool}[]
function device_properties(dev)
    @lock __device_properties_lock  begin
        if isempty(__device_properties)
            resize!(__device_properties, ndevices())

            # determine compilation properties of each device
            for dev in devices()
                cap = supported_capability(capability(dev))
                ptx = v"6.3"    # we only need 6.2, but NVPTX doesn't support that

                exitable = true
                if cap < v"7"
                    # JuliaGPU/CUDAnative.jl#4
                    # ptxas for old compute capabilities has a bug where it messes up the
                    # synchronization stack in the presence of shared memory and thread-divergent exit.
                    exitable = false
                end
                if !has_nvml() || NVML.driver_version() < v"460"
                    # JuliaGPU/CUDA.jl#431
                    # TODO: tighten this conditional
                    exitable = false
                end

                debuginfo = false

                __device_properties[deviceid(dev)+1] = (; cap, ptx, exitable, debuginfo)
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
