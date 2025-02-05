#include <iostream>
#include <limits>
#include <complex>

#include <sycl/sycl.hpp>

#define CALL_STD_ACOS 1

constexpr bool
is_fast_math_switched_on()
{
#if defined(__FAST_MATH__)
    return true;
#else
    return false;
#endif
}

template <typename T>
int
run_test();

template <typename T>
static constexpr float infinity_val = std::numeric_limits<T>::infinity();

int
main()
{
    static_assert(!is_fast_math_switched_on(),
                  "Tests of std::complex are not compatible with -ffast-math compiler option.");

    std::cout << "Run test on host...";
    run_test<float>();
    std::cout << "done." << std::endl;

    try
    {
        sycl::queue deviceQueue;

        if (deviceQueue.get_device().has(sycl::aspect::fp64))
        {
            std::cout << "Run test on device with double support...";
            deviceQueue.submit([&](sycl::handler& cgh) {
                cgh.single_task<class Kernel0>([&]() {

                    // !!!!! ATTENTION: no Kernel code in the case if fp64 is supported !!!!!

                    //run_test<float>();
                    //run_test<double>();
                });
            });
        }
        else
        {
            std::cout << "Run test on device without double support...";
            deviceQueue.submit([&](sycl::handler& cgh) {
                cgh.single_task<class Kernel1>([&]() [[sycl::device_has()]] {
                    run_test<float>();
                });
            });
        }
        deviceQueue.wait_and_throw();

        std::cout << "done." << std::endl;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception occurred: " << exc.what() <<  std::endl;
        return 1;
    }

    return 0;
}

template <typename T>
int
run_test()
{
    std::complex<T> val_c(T{-2.0}, T{0.0});

    auto res = val_c + val_c;

#if CALL_STD_ACOS
    res = std::acos(val_c);
#endif

    return 0;
}
