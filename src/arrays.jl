# Arrays on GPU

immutable DevicePtr
	p::Cuint
end


type GArray{T}
	ptr::DevicePtr
	len::Int
	nbytes::Int
end



