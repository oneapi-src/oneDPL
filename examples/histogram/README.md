# Histogram

This example demonstrates a histogram that groups numbers together and provides the count of a particular number in the input. In this example we are using oneDPL APIs to offload the computation to the selected device.

| Optimized for                   | Description                                                                    |
|---------------------------------|--------------------------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                                             |
| Hardware                        | Skylake with GEN9 or newer                                                     |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL) |
| Time to complete                | At most 1 minute                                                               |

## Purpose
This example creates both dense and sparse histograms using oneDPL APIs, on an input array of 1000 elements with values chosen randomly berween 0 and 9. To differentiate between sparse and dense histogram, we make sure that one of the values never occurs in the input array, i.e. one bin will always have 0 value.

For the dense histogram all the bins(including the zero-size bins) are stored, whereas for the sparse algorithm only non-zero sized bins are stored.

## License

This code example is licensed under MIT license.

## Building the histogram program for CPU and GPU

### On a Linux* System
Perform the following steps:

1. Source Intel&reg; oneAPI DPC++/C++ Compiler and oneDPL

2. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

3. Run the program:
```
    $ make run
```

4. Clean the program using:
```
    $ make clean
```

## Running the program

You can modify the histogram from within src/main.cpp. The functions sparse_histogram() and dense_histogram() can be reused for any set of input values.

#### Example of Output

```
Input:
6 5 5 1 2 0 3 8 7 3 8 8 0 5 7 5 7 1 1 2 7 7 5 6 6 2 7 3 0 7 5 6 1 2 5 1 6 0 0 2 1 7 1 5 3 0 8 0 2 0 2 7 8 6 2 3 8 1 2 8 8 1 1 7 1 0 8 7 7 6 0 8 0 8 5 6 6 0 6 8 3 6 6 0 3 0 1 1 8 5 0 7 1 2 3 5 1 2 1 8 6 8 8 2 8 2 8 5 1 3 3 7 0 7 7 1 7 6 5 0 3 3 0 5 6 3 2 7 3 3 1 1 3 3 3 0 5 0 5 8 3 6 6 3 5 3 7 3 0 1 6 1 5 6 7 0 1 7 5 1 1 0 5 2 2 6 0 7 6 1 6 0 1 2 1 6 3 0 8 3 0 5 3 5 2 1 3 1 8 8 6 7 6 0 8 8 7 6 5 2 1 0 1 3 2 5 0 5 3 8 7 3 2 8 6 5 0 0 6 6 6 1 2 2 0 8 1 5 6 6 7 7 7 8 8 7 2 0 2 6 6 0 7 7 8 5 3 6 3 7 1 8 7 1 1 7 1 2 3 7 7 1 3 3 8 3 1 1 1 3 5 5 1 1 3 7 7 2 1 1 3 3 7 8 5 8 6 2 0 7 0 7 6 3 1 5 1 2 5 5 2 1 0 5 3 3 1 8 8 6 1 0 8 6 8 1 6 3 8 6 1 6 2 7 0 3 2 5 3 7 8 7 6 8 2 1 1 6 0 7 1 8 7 0 6 0 2 1 7 8 5 6 5 7 2 6 8 1 0 2 2 6 8 0 2 1 8 5 7 8 3 6 6 8 7 1 3 0 2 8 0 7 3 3 5 5 0 2 1 7 1 1 5 3 8 0 1 7 3 0 5 6 7 2 5 3 1 6 1 3 3 8 1 6 3 0 0 3 6 8 0 1 8 5 3 7 3 7 0 6 6 0 1 2 0 2 5 1 8 6 2 2 0 1 6 7 5 6 8 0 6 8 8 3 2 2 1 5 8 5 0 3 3 8 5 1 1 8 0 0 3 2 0 7 1 6 5 6 2 5 2 6 2 2 0 7 0 1 1 3 5 0 2 6 6 0 6 8 8 6 8 3 7 0 1 8 6 7 1 6 1 6 3 5 1 3 1 1 5 1 5 8 8 1 5 5 1 0 1 8 7 2 2 3 2 1 0 6 6 2 0 7 2 7 2 1 2 1 5 5 2 0 2 1 1 5 7 2 6 0 1 0 2 1 5 1 3 5 2 7 8 0 6 8 2 6 7 2 7 2 7 8 2 0 7 3 1 3 3 1 2 3 3 6 2 8 2 5 2 2 1 1 6 8 7 8 3 6 0 1 8 7 7 1 6 6 2 1 0 5 0 0 8 3 0 0 2 2 7 2 2 0 1 8 6 2 5 0 8 5 0 5 2 7 6 6 2 8 5 3 5 5 5 2 8 3 6 8 3 3 1 5 3 5 1 1 8 8 1 5 1 8 2 2 7 6 1 0 2 1 1 7 0 5 0 6 8 5 5 3 8 7 8 2 1 2 1 7 8 1 1 3 0 2 6 5 1 5 2 5 1 5 3 1 8 2 6 8 7 2 0 6 7 6 6 7 6 8 5 5 7 0 7 5 1 2 2 1 0 6 7 1 0 2 1 0 2 7 6 0 7 1 6 2 1 1 2 8 7 7 0 5 6 0 2 8 0 2 3 3 8 2 2 8 2 3 6 0 8 3 1 5 7 8 0 7 0 0 2 6 5 8 2 2 7 2 2 2 2 3 5 1 3 8 1 3 0 5 7 0 0 0 5 5 6 3 1 5 3 5 2 8 3 2 0 1 3 0 1 7 3 7 0 5 0 8 8 2 2 7 2 2 5 7 6 3 1 7 6 2 2 8 0 5 8 0 6 2 0 7 1 1 3 8 6 5 5 6 1 7 2 3 1 7 0 5 8 8 1 5 2 3 2 2 8 2 2 3 1 0 0 3 1 1 0 8 7 5 3 6 2 5 0 3 1 7 6 1 7 7 1 0 2 7 0 1 7 0 3 2 0 1 2 8 2 0 7 7 8 8 1 1 0 3 2 6 1 8 5 8 6 7 6 8 5 2 8 3 1 0 1 2 1 8 2 1 1 7 0 7 7 2 6 0 5 6 1 5 5 0 2 1 8 7 7 2 2 1 2 7 2 8 0 3 5 2 3 2 8 3 1 6 3 7 6 7 5 2 3 8
Dense Histogram:
[(0, 128) (1, 157) (2, 143) (3, 111) (4, 0) (5, 110) (6, 113) (7, 118) (8, 120) ]
Sparse Histogram:
[(0, 128) (1, 157) (2, 143) (3, 111) (5, 110) (6, 113) (7, 118) (8, 120) ]
```
