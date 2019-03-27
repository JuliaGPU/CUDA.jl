# @test_throw, with additional testing for the exception message
macro test_throws_message(f, typ, ex...)
    quote
        msg = ""
        @test_throws $(esc(typ)) try
            $(esc(ex...))
        catch err
            msg = sprint(showerror, err)
            rethrow()
        end

        if !$(esc(f))(msg)
            # @test should return its result, but doesn't
            @error "Failed to validate error message\n$msg"
        end
        @test $(esc(f))(msg)
    end
end

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))
                end
            end
            ret, read(fname, String)
        end
    end
end

# Run some code on-device, returning captured standard output
macro on_device(ex)
    quote
        let
            function kernel()
                $(esc(ex))
                return
            end

            @cuda kernel()
            synchronize()
        end
    end
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

function julia_script(code, args=``)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    script = "using CUDAnative; import CUDAdrv; $code"
    out = Pipe()
    err = Pipe()
    cmd = `$(Base.julia_cmd()) -e $script`
    if Base.JLOptions().project != C_NULL
        # --project isn't preserved by julia_cmd()
        cmd = `$cmd --project=$(unsafe_string(Base.JLOptions().project))`
    end
    cmd = `$cmd $args`
    proc = run(pipeline(cmd, stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc.exitcode, read(out, String), read(err, String)
end

# a lightweight CUDA array type for testing purposes
## ctor & finalizer
mutable struct CuTestArray{T,N}
    buf::Mem.DeviceBuffer
    shape::NTuple{N,Int}
    function CuTestArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        len = prod(shape)
        buf = Mem.alloc(Mem.Device, len*sizeof(T))

        obj = new{T,N}(buf, shape)
        finalizer(unsafe_free!, obj)
        return obj
    end
end
function unsafe_free!(a::CuTestArray)
    CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
end
Base.cconvert(::Type{<:CuPtr}, x::CuTestArray) = x.buf
## memory copy operations
function CuTestArray(src::Array{T,N}) where {T,N}
    dst = CuTestArray{T,N}(size(src))
    Mem.copy!(dst.buf, pointer(src), length(src) * sizeof(T))
    return dst
end
function Base.Array(src::CuTestArray{T,N}) where {T,N}
    dst = Array{T,N}(undef, src.shape)
    Mem.copy!(pointer(dst), src.buf, prod(src.shape) * sizeof(T))
    return dst
end
## conversions
using Adapt
function Adapt.adapt_storage(::CUDAnative.Adaptor, a::CuTestArray{T,N}) where {T,N}
    ptr = convert(CuPtr{T}, a.buf)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, devptr)
end
