#=
MIT License

Copyright (c) 2022 Takafumi Arakaki <aka.tkf@gmail.com> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

module AtomixExt

# TODO: respect ordering

using Atomix: Atomix, IndexableRef
using CUDA: CUDA, CuDeviceArray

const CuIndexableRef{Indexable<:CuDeviceArray} = IndexableRef{Indexable}

function Atomix.get(ref::CuIndexableRef, order)
    error("not implemented")
end

function Atomix.set!(ref::CuIndexableRef, v, order)
    error("not implemented")
end

@inline function Atomix.replace!(
    ref::CuIndexableRef,
    expected,
    desired,
    success_ordering,
    failure_ordering,
)
    ptr = Atomix.pointer(ref)
    expected = convert(eltype(ref), expected)
    desired = convert(eltype(ref), desired)
    begin
        old = CUDA.atomic_cas!(ptr, expected, desired)
    end
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(ref::CuIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    begin
        old = if op === (+)
            CUDA.atomic_add!(ptr, x)
        elseif op === (-)
            CUDA.atomic_sub!(ptr, x)
        elseif op === (&)
            CUDA.atomic_and!(ptr, x)
        elseif op === (|)
            CUDA.atomic_or!(ptr, x)
        elseif op === xor
            CUDA.atomic_xor!(ptr, x)
        elseif op === min
            CUDA.atomic_min!(ptr, x)
        elseif op === max
            CUDA.atomic_max!(ptr, x)
        else
            error("not implemented")
        end
    end
    return old => op(old, x)
end

end  # module AtomixCUDA
