# Deprecated in v6.6: the devices cuBLASXt uses are not CUDACore's devices().
@deprecate devices!(devs::Vector{CuDevice}) xt_devices!(devs) false
@deprecate devices() xt_devices() false
@deprecate ndevices() xt_ndevices() false

# Deprecated in v6.6: the batched LAPACK routines that take device pointers are internal;
# use the methods that take arrays.
@deprecate getrf_batched!(n, ptrs::CuVector{<:CuPtr}, lda, pivot, info) getrf_batched_ptrs!(n, ptrs, lda, pivot, info) false
@deprecate getrs_batched!(trans::Char, n, nrhs, Aptrs::CuVector{<:CuPtr}, lda, pivotArray::CuPtr, Bptrs::CuVector{<:CuPtr}, ldb) getrs_batched_ptrs!(trans, n, nrhs, Aptrs, lda, pivotArray, Bptrs, ldb) false
@deprecate getri_batched!(n, Aptrs::CuVector{<:CuPtr}, lda, Cptrs::CuVector{<:CuPtr}, ldc, pivotArray::CuArray{Cint}) getri_batched_ptrs!(n, Aptrs, lda, Cptrs, ldc, pivotArray) false
