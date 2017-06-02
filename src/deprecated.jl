# Deprecated functionality

export devcount, list_devices

@deprecate devcount() length(devices())
@deprecate list_devices() map(println, devices())
