@testset "essentials" begin
    @on_device hostcall(identity, Nothing, Tuple{Nothing}, nothing)
    @on_device @hostcall identity(nothing)
    @on_device @hostcall identity(nothing)::Nothing
end

saved = []
function save(args...)
    push!(saved, args...)
    return
end

@testset "argument passing" begin
    # no args
    @on_device @hostcall identity(nothing)
    CUDA.hostcall_synchronize()
    @test isempty(saved)

    # 1 primitive arg
    @on_device @hostcall save(threadIdx().x)::Nothing
    CUDA.hostcall_synchronize()
    @test saved == [1]
    empty!(saved)

    # multiple primitive args
    @on_device @hostcall save(threadIdx().x, blockIdx().x)::Nothing
    CUDA.hostcall_synchronize()
    @test saved == [1, 1]
    empty!(saved)

    # isbits args
    @on_device @hostcall save((threadIdx().x, blockIdx().x))::Nothing
    CUDA.hostcall_synchronize()
    @test saved == [(1, 1)]
    empty!(saved)
end

@testset "return values" begin
    # primitive
    @on_device @hostcall save(@hostcall +(threadIdx().x, 1))::Nothing
    CUDA.hostcall_synchronize()
    @test saved == [2]
    empty!(saved)

    # isbits
    @on_device @hostcall save(@hostcall tuple(threadIdx().x, blockIdx().x))::Nothing
    CUDA.hostcall_synchronize()
    @test saved == [(1, 1)]
    empty!(saved)
end
