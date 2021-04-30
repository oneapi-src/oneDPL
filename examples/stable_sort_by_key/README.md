# Stable sort by key

Stable sort by key is a sorting operation when sorting of 2 sequences (keys and values) only keys are compared but both keys and values are swapped.

| Optimized for                   | Description                                                                    |
|---------------------------------|--------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                           |
| Hardware                        | Skylake with GEN9 or newer                                                     |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL) |
| Time to complete                | At most 1 minute                                                               |

## Purpose

The example models stable sorting by key: during the sorting of 2 sequences (keys and values) only keys are compared but both keys and values are swapped.
It fills two buffers (one of the buffer is filled using `counting_iterator`) and then sorts them both using `zip_iterator`.

The example demonstrates how to use `counting_iterator` and `zip_iterator` using oneDPL.
* `counting_iterator` helps to fill the sequence with the numbers zero through `n` using std::copy.
* `zip_iterator` provides the ability to iterate over several sequences simultaneously.

## License

This code example is licensed under MIT license.

## Building the 'Stable sort by key' Program for CPU and GPU

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

## Running the Program
### Example of Output

```
success
Run on Intel(R) Gen9
```
