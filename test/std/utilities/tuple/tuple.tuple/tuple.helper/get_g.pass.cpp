// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                int j = 1;
                const int k = 2;
                s::tuple<int, int&, const int&> a(0, j, k);
                const s::tuple<int, int&, const int&> b(1, j, k);
                ret_access[0] = (s::get<0>(a) == 0 && s::get<1>(a) == 1 && s::get<2>(a) == 2);
                s::get<0>(a) = 3;
                s::get<1>(a) = 4;
                ret_access[0] &= (s::get<0>(a) == 3 && s::get<1>(a) == 4);
                ret_access[0] &= (j == 4);
                s::get<1>(b) = 5;
                ret_access[0] &= (s::get<0>(b) == 1 && s::get<1>(b) == 5 && s::get<2>(b) == 2);
                ret_access[0] &= (j == 5);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
}
