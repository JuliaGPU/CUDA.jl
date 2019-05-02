$ErrorActionPreference = "Stop"

[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

$cuda_installers = @{}
$cuda_installers.Add('8.0', 'https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe')
$cuda_installers.Add('9.0', 'https://developer.nvidia.com/compute/cuda/9.0/prod/local_installers/cuda_9.0.176_windows-exe')
$cuda_installers.Add('9.1', 'https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows')
$cuda_installers.Add('9.2', 'https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.88_windows')
$cuda_installers.Add('10.0', 'https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10')
$cuda_installers.Add('10.1', 'https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.96_win10.exe')

$cuda_installer = $cuda_installers.Get_Item($env:cuda_version)
Start-FileDownload $cuda_installer -FileName cuda.exe

# Source: NVIDIA CUDA Installation Guide for Microsoft Windows
$cuda_components = @{}
$cuda_components.Add('8.0', @("compiler_8.0", "cudart_8.0", "visual_studio_integration_8.0"))
$cuda_components.Add('9.0', @("compiler_9.0", "cudart_9.0", "visual_studio_integration_9.0"))
$cuda_components.Add('9.1', @("nvcc_9.1", "cudart_9.1", "visual_studio_integration_9.1"))
$cuda_components.Add('9.2', @("nvcc_9.2", "cudart_9.2", "visual_studio_integration_9.2"))
$cuda_components.Add('10.0', @("nvcc_10.0", "cudart_10.0", "visual_studio_integration_10.0"))
$cuda_components.Add('10.1', @("nvcc_10.1", "cudart_10.1", "visual_studio_integration_10.1"))

$components = $cuda_components.Get_Item($env:cuda_version)
.\cuda.exe -s @components | Out-Null
