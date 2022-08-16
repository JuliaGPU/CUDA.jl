using JET

@static if VERSION >= v"1.7"

# XXX: JET's test failures only render during record
#      https://github.com/aviatesk/JET.jl/blob/c2c42e03994fe80dc03a4568c42e43311b581b15/src/JET.jl#L1564
old_print_setting = Test.TESTSET_PRINT_ENABLE[]
Test.TESTSET_PRINT_ENABLE[] = true

# array allocation
@test_call CuArray{Int}(undef, 1)
@test_call cu([1])

# kernel compilation
@test_call target_modules=(CUDA,) cufunction(identity, Tuple{Nothing})

# array op
@test_call target_modules=(CUDA,) cu([1]) * 2

Test.TESTSET_PRINT_ENABLE[] = old_print_setting

end
