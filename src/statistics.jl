import Statistics

Statistics._var(A::CuArray, corrected::Bool, mean, dims) =
    sum((A .- something(mean, Statistics.mean(A, dims=dims))).^2, dims=dims)/(prod(size(A)[[dims...]])-corrected)

Statistics._var(A::CuArray, corrected::Bool, mean, ::Colon) =
    sum((A .- something(mean, Statistics.mean(A))).^2)/(length(A)-corrected)

Statistics._std(A::CuArray, corrected::Bool, mean, dims) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=dims))

Statistics._std(A::CuArray, corrected::Bool, mean, ::Colon) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=:))
