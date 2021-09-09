using CUDA

function test(x)
  println("This is a hostcall from thread $x")
  x+1
end

function kernel()
  rv = hostcall(test, Int, Tuple{Int}, threadIdx().x)
  @cuprintln("Hostcall returned $rv")
  return
end

function main()
  @cuda threads=16 kernel()
  synchronize()
  return
end

isinteractive() || main()
