
module CURAND

export create_generator,
       destroy_generator,
       # generator types
       CURAND_RNG_TEST,
       CURAND_RNG_PSEUDO_DEFAULT,
       CURAND_RNG_PSEUDO_XORWOW,
       CURAND_RNG_PSEUDO_MRG32K3A,
       CURAND_RNG_PSEUDO_MTGP32,
       CURAND_RNG_PSEUDO_MT19937,
       CURAND_RNG_PSEUDO_PHILOX4_32_10,
       CURAND_RNG_QUASI_DEFAULT,
       CURAND_RNG_QUASI_SOBOL32,
       CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
       CURAND_RNG_QUASI_SOBOL64,
       CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
       # high-level API
       curand,
       curandn,
       curand_logn,
       curand_poisson,
       # wrappers
       get_version,
       set_pseudo_random_generator_seed,
       set_generator_offset,
       set_generator_ordering,
       set_quasi_random_generator_dimensions,
       generate,
       generate_long_long,
       generate_uniform,
       generate_uniform_double,
       generate_normal,
       generate_normal_double,
       generate_log_normal,
       generate_log_normal_double,
       create_poisson_distribtion,
       destroy_distribution,
       generate_poisson,
       generate_seeds


include("core.jl")

end # module
