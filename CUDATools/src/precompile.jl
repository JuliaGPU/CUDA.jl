# @profile infrastructure
precompile(Tuple{typeof(Profile.detect_cupti)})
precompile(Tuple{typeof(Profile.profile_internally), Function})
precompile(Tuple{typeof(Profile.capture), CUPTI.ActivityConfig})
