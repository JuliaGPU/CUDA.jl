
@static if is_windows()    
    # location of cudart64_xx.dll or cudart32_xx.dll have to be in PATH env var
    # ex: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin
    # (by default, it is done by CUDA toolkit installer)
    
    const dllnames = (WORD_SIZE==64) ?
        ["curand64_70", "curand64_65", "curand64_60", "curand64_55", "curand64_50", "curand64_50_35"] :
        ["curand32_70", "curand32_65", "curand32_60", "curand32_55", "curand32_50", "curand32_50_35"]
    const libcurand = Libdl.find_library(dllnames, [""])
else
    # linux or mac
    const libcurand = Libdl.find_library(["libcurand"],
                                         ["/usr/local/cuda/lib", "/usr/local/cuda/lib64"])
end

if isempty(libcurand)
    error("CURAND library cannot be found")
end
