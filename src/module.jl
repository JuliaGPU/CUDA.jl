# CUDA module management

immutable CuModule
	handle::Ptr{Void}

	function CuModule(mod::ASCIIString)
		a = Array(Ptr{Void}, 1)
		is_data = true
		try
		  is_data = !ispath(mod)
		catch
		  is_data = true
		end
		call = is_data ? (:cuModuleLoadData) : (:cuModuleLoad)
		@cucall(call, (Ptr{Ptr{Void}}, Ptr{Cchar}), a, mod)
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


