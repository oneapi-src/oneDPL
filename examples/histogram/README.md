# Histogram

This example demonstrates a histogram that groups numbers together and provides the count of a particular number in the input.
In this example we are using oneDPL APIs to offload the computation to the selected device.

| Optimized for                   | Description                                                                    |
|---------------------------------|--------------------------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                                             |
| Hardware                        | Skylake with GEN9 or newer                                                     |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL) |
| Time to complete                | At most 1 minute                                                               |

## Purpose
This example creates both dense and sparse histograms using oneDPL APIs, on an input array of 1000 elements with values chosen randomly between 0 and 9.
To differentiate between sparse and dense histogram, we make sure that one of the values never occurs in the input array, i.e. one bin will always have 0 value.

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
success for Dense Histogram:
[(0, 151) (1, 158) (2, 135) (3, 127) (4, 0) (5, 105) (6, 126) (7, 97) (8, 101) ]
success for Sparse Histogram:
[(0, 151) (1, 158) (2, 135) (3, 127) (5, 105) (6, 126) (7, 97) (8, 101) ]
```
