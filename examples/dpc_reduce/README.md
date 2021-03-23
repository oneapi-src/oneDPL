# DPC Reduce sample

This example demonstrates how to do reduction by using the CPU in serial mode, 
the CPU in parallel mode (using TBB), the GPU using direct DPC++ coding, the 
GPU using multiple steps with DPC++ Library algorithms transform and reduce, 
and then finally using the DPC++ Library transform_reduce algorithm.  

All the different modes use a simple calculation for Pi.   It is a well known 
mathematical formula that if you integrate from 0 to 1 over the function, 
(4.0 / (1+x*x) )dx the answer is pi.   One can approximate this integral 
by summing up the area of a large number of rectangles over this same range.  

Each of the different function calculates pi by breaking the range into many 
tiny rectangles and then summing up the results. 

The parallel computations are performed using oneTBB and oneAPI DPC++ library 
(oneDPL).

| Optimized for                   | Description                               |
|---------------------------------|-------------------------------------------|
| OS                              | Linux Ubuntu 18.04                        |
| Hardware                        | Skylake with GEN9 or newer                |
| Software                        | Intel® oneAPI DPC++ Compiler (beta)       |
| What you will learn             | different strategies for reduction        |
| Time to complete                | At most 5 minutes                         |

## License

This code sample is licensed under MIT license.

## How to build

```bash
# To this point you should have
# - Data Parallel C++ Library installed and
# - environment variables set up to use it

mkdir build && cd build  # execute in this directory

```

### Linux

```bash
CXX=dpcpp cmake ..
cmake --build .  # or "make"
cmake --build . --target run  # or "make run"
```

### Windows

```bash
cmake -G "MinGW Makefiles" -DCMAKE_MAKE_PROGRAM=gmake -DCMAKE_CXX_COMPILER=dpcpp-cl ..
cmake --build .  # or "gmake"
cmake --build . --target run  # or "gmake run"
```
