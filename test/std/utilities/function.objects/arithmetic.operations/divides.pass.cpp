#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelDividesTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);

    int div_array[2] = {10, 5};
    cl::sycl::range<1> numOfItems2{2};
    cl::sycl::buffer<cl::sycl::cl_int, 1> div_buffer(div_array, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        auto div_access = div_buffer.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelDividesTest>([=]() {
            typedef s::divides<int> Fint;
            const Fint f1 = Fint();
            ret_access[0] = (f1(36, 4) == 9);
            ret_access[0] &= (f1(div_access[0], div_access[1]) == 2);

            typedef s::divides<float> Ffloat;
            const Ffloat f2 = Ffloat();
            ret_access[0] &= (f2(18, 4.0f) == 4.5f);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{
    kernel_test();
    return 0;
}
