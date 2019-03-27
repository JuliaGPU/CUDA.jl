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

# variant on @test_throws that checks the CuError error code
macro test_throws_cuerror(kind, ex)
    # generate a test only returning CuError if it is the correct one
    test = quote
        try
            $(esc(ex))
        catch err
            isa(err, CuError) || rethrow()
            err == $kind      || rethrow(ErrorException(string("Wrong CuError kind: ", err, " instead of ", $kind)))
            rethrow()
        end
    end

    # now re-use @test_throws (which ties into @testset, etc)
    quote
        @test_throws CuError $test
    end
end

mutable struct NoThrowTestSet <: Test.AbstractTestSet
    results::Vector
    NoThrowTestSet(desc) = new([])
end
Test.record(ts::NoThrowTestSet, t::Test.Result) = (push!(ts.results, t); t)
Test.finish(ts::NoThrowTestSet) = ts.results
fails = @testset NoThrowTestSet begin
    # OK
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_UNKNOWN)
    # Fail, wrong CuError
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN throw(CUDAdrv.ERROR_INVALID_VALUE)
    # Fail, wrong Exception
    @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN error()
end
@test isa(fails[1], Test.Pass)
@test isa(fails[2], Test.Fail)
@test isa(fails[3], Test.Fail)

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
