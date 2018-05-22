Set-PSDebug -Trace 1
$ErrorActionPreference = "Stop"


# Julia

$julia_installers = @{}
$julia_installers.Add('0.6',     'https://julialang-s3.julialang.org/bin/winnt/x64/0.6/julia-0.6-latest-win64.exe')
$julia_installers.Add('nightly', 'https://julialangnightlies-s3.julialang.org/bin/winnt/x64/julia-latest-win64.exe')

$julia_installer = $julia_installers.Get_Item($env:JULIA)
Start-FileDownload $julia_installer -FileName julia.exe

.\julia.exe /S /D=C:\julia


# CUDA

$cuda_installers = @{}
$cuda_installers.Add('8.0', 'https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe')
$cuda_installers.Add('9.0', 'https://developer.nvidia.com/compute/cuda/9.0/prod/local_installers/cuda_9.0.176_windows-exe')
$cuda_installers.Add('9.1', 'https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows')
$cuda_installers.Add('9.2', 'https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.88_windows')

$cuda_installer = $cuda_installers.Get_Item($env:CUDA)
Start-FileDownload $cuda_installer -FileName cuda.exe

.\cuda.exe -s "compiler_$env:CUDA" "cudart_$env:CUDA"
