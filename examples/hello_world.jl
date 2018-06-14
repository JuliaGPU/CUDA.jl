using CUDAdrv, CUDAnative

if Sys.iswindows()
	hello_world() =
   	@cuprintf("Greetings from block %lld, thread %lld!\n", Int64(blockIdx().x), Int64(threadIdx().x))
else
	hello_world() =
    	@cuprintf("Greetings from block %ld, thread %ld!\n", Int64(blockIdx().x), Int64(threadIdx().x))
end
@cuda blocks=2 threads=2 hello_world()
synchronize()
