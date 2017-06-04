# Performance

## Object arguments

When passing a rich object like a `CuArray` to a GPU kernel, there's a memory allocation and
copy happening behind the scenes. This means that every kernel call is synchronizing, which
can easily kill performance in the case of fine-grained kernels.

Although this issue will probably get fixed in the future, a workaround for now is to ensure
all arguments are `bitstype` (ie. declared as primitive `bitstype` types, not to be confused
with the `isbits` property). Specific to arrays, you can access and pass the underlying
device pointer by means of the `ptr` field of `CuArray` objects, in addition to the size of
the array:

```julia
function inc_slow(a)
    a[threadIdx().x] += 1

    return nothing
end

@cuda (1,3) inc_slow(d_a)                       # implicit alloc & memcpy


function inc_fast(a_ptr, a_len)
    a = CuDeviceArray(a_len, a_ptr)
    a[threadIdx().x] += 1

    return nothing
end

@cuda (1,3) inc_fast(pointer(d_a), length(d_a)) # no implicit memory ops
```
