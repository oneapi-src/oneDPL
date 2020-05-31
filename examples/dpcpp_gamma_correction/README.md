# Gamma Correction sample

This example demonstrates gamma correction - a nonlinear operation used to encode and decode the luminance of each image pixel. See https://en.wikipedia.org/wiki/Gamma_correction for more information.

The example creates a fractal image in memory and performs gamma correction on it. The output of the example application is a BMP image with corrected luminance.

The computations are performed using DPC++ backend of Parallel STL.


| Optimized for                   | Description                                                     |
|---------------------------------|-----------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                              |
| Hardware                        | SKL with GEN9 or newer                                          |
| Software                        | Intel Data Parallel C++ Compiler beta                           |
| What you will learn             | How to offoad the computation to GPU using Intel DPC++ Compiler |
| Time to complete                | 5 minutes                                                       |

## License

This code sample is licensed under MIT license.

## How to build

```bash
# To this point you should have
# - TBB and Parallel STL installed and
# - environment variables set up to use them

# Configure, build and run the example
mkdir build && cd build  # execute in this directory
CXX=clang++ cmake ..
cmake --build .  # or "make"
cmake --build . --target run  # or "make run"
```

You can also pass

- options during the configuration:
    ```bash
    CXX=clang++ cmake ..
    cmake --build .  # or "make"
    cmake --build . --target run  # or "make run"
    ```

