# CUDA module management

immutable CuModule
	handle::Ptr{Void}

	function CuModule(filename::ASCIIString)
		a = Array(Ptr{Void}, 1)
		@cucall(:cuModuleLoad, (Ptr{Ptr{Void}}, Ptr{Cchar}), a, filename)
		new(a[1])
	end
end

function unload(md::CuModule)
	@cucall(:cuModuleUnload, (Ptr{Void},), md.handle)
end


immutable CuFunction
	handle::Ptr{Void}

	function CuFunction(md::CuModule, name::ASCIIString)
		a = Array(Ptr{Void}, 1)
		@cucall(:cuModuleGetFunction, (Ptr{Ptr{Void}}, Ptr{Void}, Ptr{Cchar}), 
			a, md.handle, name)
		new(a[1])
	end
end


