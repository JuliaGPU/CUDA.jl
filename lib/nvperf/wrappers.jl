function initialize()
    params = Ref(NVPW_InitializeHost_Params(NVPW_InitializeHost_Params_STRUCT_SIZE, C_NULL))
    NVPW_InitializeHost(params)
end

function supported_chips()
    params = Ref(NVPW_GetSupportedChipNames_Params(
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE,
        C_NULL, C_NULL, 0))
    NVPW_GetSupportedChipNames(params)

    names = String[]
    for i in params[].numChipNames
        push!(names, Base.unsafe_string(Base.unsafe_load(params[].ppChipNames, i)))
    end
    return names
end
