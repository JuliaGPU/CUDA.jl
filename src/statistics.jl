using Statistics

Statistics._var(A::CuArray, corrected::Bool, mean, dims) =
    sum((A .- something(mean, Statistics.mean(A, dims=dims))).^2, dims=dims)/(prod(size(A)[[dims...]])-corrected)

Statistics._var(A::CuArray, corrected::Bool, mean, ::Colon) =
    sum((A .- something(mean, Statistics.mean(A))).^2)/(length(A)-corrected)

Statistics._std(A::CuArray, corrected::Bool, mean, dims) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=dims))

Statistics._std(A::CuArray, corrected::Bool, mean, ::Colon) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=:))

# Revert https://github.com/JuliaLang/Statistics.jl/pull/25
Statistics._mean(A::CuArray, ::Colon)    = sum(A) / length(A)
Statistics._mean(f, A::CuArray, ::Colon) = sum(f, A) / length(A)
Statistics._mean(A::CuArray, dims)    = mean!(Base.reducedim_init(t -> t/2, +, A, dims), A)
Statistics._mean(f, A::CuArray, dims) = sum(f, A, dims=dims) / mapreduce(i -> size(A, i), *, unique(dims); init=1)
