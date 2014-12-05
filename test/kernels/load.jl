using CUDA

julia = realpath("/proc/$(getpid())/exe")
scriptdir = dirname(Base.source_path())

# TODO: put the CUDA and native kernels in the same file


#
# CUDA kernels
#

cuda_kernels = [ "vadd" ]
for kernel in cuda_kernels
    fname = symbol("cuda_$kernel")
    ptx = "$scriptdir/$kernel.ptx"
    @eval begin
        function $fname()
            mod = CuModule($ptx)
            CuFunction(mod, $kernel)
        end
    end
end
run(`$julia $scriptdir/Makefile.jl $cuda_kernels`)


#
# Native kernels
#

include("native.jl")
