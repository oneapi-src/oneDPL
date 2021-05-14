# Gamma Correction
Gamma correction is a nonlinear operation used to encode and decode the luminance of each pixel of an image.
This example demonstrates how to use oneDPL to facilitate offload to devices.

| Optimized for                   | Description                                                                                                                          |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                                                                                 |
| Hardware                        | Skylake with GEN9 or newer                                                                                                           |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL); Intel&reg; oneAPI Threading Building Blocks (oneTBB) |
| Time to complete                | At most 1 minute                                                                                                                     |

## Purpose

Gamma correction uses nonlinear operations to encode and decode the luminance of each pixel of an image.
See https://en.wikipedia.org/wiki/Gamma_correction for more information.
It does so by creating a fractal image in memory and performs gamma correction on it with `gamma=2`.

|Original image | After applying gamma correction |
|---|---|
|<img src="images/original.bmp">|<img src="images/gamma.bmp">|

## License

This code example is licensed under MIT license.

## Building the 'Gamma Correction' Program for CPU and GPU

### On a Linux* System
Perform the following steps:

1. Source Intel&reg; oneAPI DPC++/C++ Compiler, oneTBB and oneDPL

2. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake .. # or $ cmake -DCMAKE_CXX_FLAGS=-DBUILD_FOR_HOST .. # to run on Host
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
### Example of Output

The output of the example application is a BMP image with corrected luminance. Original image is created by the program.
```
success
Run on Intel(R) Gen9
Original image is in the fractal_original.bmp file
Image after applying gamma correction on the device is in the fractal_gamma.bmp file
```
