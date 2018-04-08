# NOTE: all kernel function definitions are prefixed with @eval to force toplevel definition,
#       avoiding boxing as seen in https://github.com/JuliaLang/julia/issues/18077#issuecomment-255215304

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        let fname = tempname()
            try
                ret = nothing
                open(fname, "w") do fout
                    redirect_stdout(fout) do
                        ret = $(esc(ex))
                    end
                end
                ret, read(fname, String)
            finally
                rm(fname, force=true)
            end
        end
    end
end

# Run some code on-device, returning captured standard output
macro on_device(ex)
    @gensym kernel
    esc(quote
        let
            @eval function $kernel()
                $ex
            end

            @cuda $kernel()
            synchronize()
        end
    end)
end

function julia_cmd(cmd)
    return `
        $(Base.julia_cmd())
        --color=$(Base.have_color ? "yes" : "no")
        --compiled-modules=$(Base.JLOptions().use_compiled_modules != 0 ? "yes" : "no")
        --history-file=no
        --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
        --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
        $cmd
    `
end

# helper function for sinking a value to prevent the callee from getting optimized away
@eval @inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@eval @inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

# a lightweight CUDA array type for testing purposes
## ctor & finalizer
mutable struct CuTestArray{T,N}
    buf::Mem.Buffer
    shape::NTuple{N,Int}
    function CuTestArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        len = prod(shape)
        buf = Mem.alloc(len*sizeof(T))

        obj = new{T,N}(buf, shape)
        finalizer(unsafe_free!, obj)
        return obj
    end
end
function unsafe_free!(a::CuTestArray)
    CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
end
## memory copy operations
function CuTestArray(src::Array{T,N}) where {T,N}
    dst = CuTestArray{T,N}(size(src))
    Mem.upload!(dst.buf, pointer(src), length(src) * sizeof(T))
    return dst
end
function Base.Array(src::CuTestArray{T,N}) where {T,N}
    dst = Array{T,N}(undef, src.shape)
    Mem.download!(pointer(dst), src.buf, prod(src.shape) * sizeof(T))
    return dst
end
## conversions
function CUDAnative.cudaconvert(a::CuTestArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, a.buf)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, devptr)
end
