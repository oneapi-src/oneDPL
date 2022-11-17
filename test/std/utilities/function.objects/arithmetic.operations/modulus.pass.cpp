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

class KernelModulusTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelModulusTest>([=]() {
            typedef s::modulus<int> F;
            const F f = F();
            ret_access[0] = (f(36, 8) == 4);

            typedef s::modulus<long> FL;
            const FL f1 = FL();
            ret_access[0] &= (f1(36L, 8) == 4);

            ret_access[0] &= (f1(43L, 7L) == 1);
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
