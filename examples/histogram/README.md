# Histogram

This example demonstrates how to create dense and sparse histograms using dpstd APIs.

The example creates an input array of 1000 elements with random values between 0 and 9. We make sure that one of the values never occurs in the array, i.e. one bin will have 0 value. For the dense histogram all the bins are stored, whereas for the sparse algorithm only non-zero bins are stored.

The computations are performed using Intel® oneAPI DPC++ library (oneDPL).

| Optimized for                   | Description                                                                        |
|---------------------------------|------------------------------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                                                 |
| Hardware                        | Skylake with GEN9 or newer                                                         |
| Software                        | Intel® oneAPI DPC++ Compiler (beta)                                                |
| What you will learn             | How to use algorithms that are a part of the DPC++ Library.                        |
| Time to complete                | At most 5 minutes                                                                  |

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

## Example of Output

Input:
1 1 8 1 8 6 1 0 1 5 5 2 2 8 1 2 1 1 1 6 2 1 1 8 3 6 6 2 2 1 1 8 1 0 0 0 2 2 7 6 5 1 6 1 1 6 1 5 1 0 0 1 1 1 0 5 5 0 7 0 1 6 0 5 7 0 3 0 0 0 0 6 0 2 5 5 6 6 8 7 6 6 8 8 7 7 2 2 0 7 2 2 5 2 7 1 3 0 1 1 0 1 7 2 0 1 5 1 7 0 8 3 1 5 0 6 1 0 8 2 7 2 1 1 1 3 2 5 1 2 5 1 6 3 3 1 3 8 0 1 1 8 2 0 2 0 1 2 0 2 1 8 1 6 0 6 7 1 1 8 3 6 0 7 7 1 6 1 7 6 1 8 3 3 6 3 1 2 7 2 1 0 1 8 7 0 5 5 1 1 3 2 1 3 7 0 3 2 1 1 8 0 1 0 2 5 3 6 7 0 6 2 0 8 8 5 6 3 0 5 7 3 5 0 0 3 7 7 5 6 7 2 7 8 0 0 2 3 0 1 3 1 1 2 7 1 5 1 0 3 7 2 0 3 0 0 6 7 5 0 5 3 0 3 0 0 1 3 2 5 2 3 6 3 5 5 2 0 7 6 3 6 7 6 0 7 6 5 6 0 3 0 2 1 1 0 2 2 1 1 7 3 8 2 5 2 7 7 2 1 3 2 1 1 1 8 6 5 2 3 3 6 1 5 8 2 1 1 2 5 2 0 7 3 3 3 3 8 8 0 1 2 8 2 3 7 0 8 1 2 2 1 6 2 8 5 1 3 5 7 8 0 5 2 1 8 7 0 6 7 8 7 7 5 8 0 3 8 8 2 8 1 7 2 1 6 0 0 7 3 2 2 1 7 0 2 5 7 5 2 3 1 0 2 1 6 2 2 3 1 5 3 0 3 5 0 7 3 1 5 7 6 7 8 2 7 0 7 2 5 7 5 0 6 5 8 3 7 0 7 6 5 8 5 6 2 5 2 5 0 5 1 1 3 1 6 0 8 3 0 0 1 7 2 5 2 0 7 2 0 3 7 3 0 3 0 2 6 0 7 6 5 0 1 8 8 5 8 7 8 1 0 8 0 2 2 2 2 0 2 0 3 0 3 3 3 3 3 7 3 2 0 6 0 3 0 8 0 1 1 6 3 1 3 1 0 6 3 7 1 5 7 8 6 0 0 7 1 1 6 3 2 8 0 2 3 0 1 1 6 3 5 7 7 0 8 2 1 0 7 8 5 2 5 0 0 6 6 5 8 3 8 1 2 7 5 3 2 1 0 8 7 8 1 3 8 1 3 3 1 2 0 5 1 6 3 6 1 0 2 7 3 0 8 1 7 2 5 7 6 8 5 2 7 0 5 6 2 8 7 1 8 7 2 3 2 8 0 3 8 1 1 1 1 7 5 6 0 8 2 6 7 7 8 5 8 2 2 8 2 7 0 1 6 3 5 8 2 3 1 1 2 0 2 3 8 5 7 8 5 1 1 1 8 1 7 5 0 7 1 0 6 3 5 1 6 8 0 6 1 8 7 5 0 8 7 6 2 5 5 5 6 7 7 1 0 5 0 2 3 3 6 0 1 0 1 8 7 0 5 8 6 3 2 2 0 0 1 3 6 5 8 1 3 2 5 1 0 6 3 0 7 7 2 2 8 2 1 1 2 6 3 6 7 5 2 8 6 3 0 1 8 6 0 1 2 6 0 0 1 2 2 8 0 5 1 6 7 0 1 7 6 1 2 2 8 6 8 5 8 8 1 5 1 1 6 6 8 7 6 0 0 0 6 7 3 5 5 8 5 2 6 2 7 8 3 6 1 2 0 1 2 1 6 6 6 2 1 6 7 5 0 5 3 2 3 6 7 6 5 2 2 0 1 0 7 7 6 0 8 1 1 1 8 7 5 3 7 1 0 5 0 3 1 2 5 5 8 1 0 3 5 0 1 8 0 6 0 0 6 3 8 5 2 5 1 5 0 2 0 7 6 8 1 7 1 0 1 0 6 0 1 0 0 1 8 1 7 2 3 3 5 1 8 6 6 1 2 2 2 3 1 8 2 2 6 3 7 6 1 2 6 1 2 6 2 0 5 0 2 7 3 5 8 3 2 3 1 5 6 6 6 7 3 8 0 8 0 5 5 8 5 0 0 6 2 0 6 8 1 6 6 2 0 3 5 3 2 8 6 1 3 3 8 7 0 7 6 7 1 0 6 7 0 5 0 0 5 8 1
Dense Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (4, 0) (5, 105) (6, 110) (7, 108) (8, 102) ]
Sparse Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (5, 105) (6, 110) (7, 108) (8, 102) ]
