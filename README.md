# `netlist_example` #

The file `netlist_example.cu` contains an example program demonstrating how
`thrust::reduce_by_key` can be used to sum the `x`-positions and `y`-positions
of all blocks connected to each net.  The goal of the `netlist_example` program
is to demonstrate basic interfacing with the [`thrust`][thrust] library (e.g., copying
data to the GPU from host memory, calling `thrust` operations such as `copy_n`
and `reduce_by_key`).  Note that more usage details regarding the `thrust`
library can be found in the [quick start guide][thrust-quick-start] and in the
examples included in the `thrust` source code (available on the [`thrust`
project page][thrust]).


### Compilation ###

A program that uses [`thrust`][thrust] device vectors may be compiled to target
one of several available device [backends][thrust-device-backends]. Note that
the [NVIDIA CUDA Toolkit][cuda-toolkit] includes `thrust`, which is a
header-only library.

After installing the [NVIDIA CUDA Toolkit][cuda-toolkit], to target
CUDA-capable GPU devices, the `netlist_example` program may be compiled using:

    nvcc netlist_example.cu -o netlist_example-cuda -O3 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

Alternatively, the `netlist_example` program may be compiled to run entirely on
the CPU, utilizing only host memory by using the following command:

    nvcc netlist_example.cu -o netlist_example-cpp -O3 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP

Note that the `netlist_example` program may be compiled to run entirely on the
CPU without installing the CUDA Toolkit.  This can be helpful when developing
in an environment without a CUDA GPU installed.  To compile the
`netlist_example` code using `g++` without installing the CUDA Toolkit:

 1. Download the `thrust` header library from the [`thrust` project page][thrust].
 2. Rename the `netlist_example.cu` file to `netlist_example.cpp`, since `g++`
    does not recognize files with the `.cu` extension.
 3. Run the following command to compile using the host C++ backend:

      g++ netlist_example.cpp -o netlist_example-cpp -O3 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP

[thrust-device-backends]: https://github.com/thrust/thrust/wiki/Device-Backends
[cuda-toolkit]: https://developer.nvidia.com/cuda-downloads
[thrust]: http://thrust.github.io/
[thrust-quick-start]: https://github.com/thrust/thrust/wiki/Quick-Start-Guide
