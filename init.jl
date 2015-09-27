
const libcurand = Libdl.find_library(["libcurand"],
                               ["/usr/local/cuda/lib", "/usr/local/cuda/lib64"])
if isempty(libcurand)
    error("CURAND library cannot be found")
end
