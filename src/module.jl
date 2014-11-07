# CUDA module management

immutable CuModule
	handle::Ptr{Void}

	function CuModule(mod::ASCIIString)
		a = Ptr{Void}[0]
		is_data = true
		try
		  is_data = !ispath(mod)
		catch
		  is_data = true
		end
		fname = is_data ? (:cuModuleLoadData) : (:cuModuleLoad)
		@cucall(fname, (Ptr{Ptr{Void}}, Ptr{Cchar}), a, mod)
		new(a[1])
	end
end

function unload(md::CuModule)
	@cucall(:cuModuleUnload, (Ptr{Void},), md.handle)
end


immutable CuFunction
	handle::Ptr{Void}

	function CuFunction(md::CuModule, name::ASCIIString)
		a = Ptr{Void}[0]
		@cucall(:cuModuleGetFunction, (Ptr{Ptr{Void}}, Ptr{Void}, Ptr{Cchar}), 
			a, md.handle, name)
		new(a[1])
	end
end


# TODO: parametric type given knowledge about device type?
immutable CuGlobal{T}
	pointer::CuPtr
	nbytes::Cssize_t

	function CuGlobal(md::CuModule, name::ASCIIString)
		a = CuPtr[0]
		b = Cssize_t[0]
		@cucall(:cuModuleGetGlobal, (Ptr{CuPtr}, Ptr{Cssize_t}, Ptr{Void}, Ptr{Cchar}), 
			a, b, md.handle, name)
		@assert b[1] == sizeof(T)
		new(a[1], b[1])
	end
end

eltype{T}(var::CuGlobal{T}) = T

function get{T}(var::CuGlobal{T})
	a = T[0]
	@cucall(:cuMemcpyDtoH, (Ptr{Void}, CuPtr, Csize_t), a, var.pointer, var.nbytes)
	return a[1]
end

function set{T}(var::CuGlobal{T}, val::T)
	a = T[0]
	a[1] = val
	@cucall(:cuMemcpyHtoD, (CuPtr, Ptr{Void}, Csize_t), var.pointer, a, var.nbytes)
end
