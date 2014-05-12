# Support for cuda math intrinsics


# Trigonometric
sin(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_sinf( float %0 )
										  ret float %2""", Float32, (Float32,), x)
sin(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_sin( double %0 )
										  ret double %2""", Float64, (Float64,), x)

cos(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_cosf( float %0 )
										  ret float %2""", Float32, (Float32,), x)
cos(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_cos( double %0 )
										  ret double %2""", Float64, (Float64,), x)

# Rounding
floor(x::Float32) = Base.llvmcall(false, """%2 = call float @__nv_floorf( float %0 )
											ret float %2""", Float32, (Float32,), x)
floor(x::Float64) = Base.llvmcall(false, """%2 = call double @__nv_floor( double %0 )
											ret double %2""", Float64, (Float64,), x)