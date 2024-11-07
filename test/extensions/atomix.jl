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

using Atomix
using Test
using CUDA

function cuda(f)
    function g()
        f()
        nothing
    end
    CUDA.@cuda g()
end

@testset "cas" begin
    idx = (
        data = 1,
        cas1_ok = 2,
        cas2_ok = 3,
        # ...
    )
    @assert minimum(idx) >= 1
    @assert maximum(idx) == length(idx)

    A = CUDA.zeros(Int, length(idx))
    cuda() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            (old, success) = Atomix.replace!(ref, 0, 42)
            A[idx.cas1_ok] = old == 0 && success
            (old, success) = Atomix.replace!(ref, 0, 43)
            A[idx.cas2_ok] = old == 42 && !success
        end
    end
    @test collect(A) == [42, 1, 1]
end

@testset "inc" begin
    @testset "core" begin
        A = CUDA.CuVector(1:3)
        cuda() do
            GC.@preserve A begin
                ref = Atomix.IndexableRef(A, (1,))
                pre, post = Atomix.modify!(ref, +, 1)
                A[2] = pre
                A[3] = post
            end
        end
        @test collect(A) == [2, 1, 2]
    end

    @testset "sugar" begin
        A = CUDA.ones(Int, 3)
        cuda() do
            GC.@preserve A begin
                @atomic A[begin] += 1
            end
        end
        @test collect(A) == [2, 1, 1]
    end
end
