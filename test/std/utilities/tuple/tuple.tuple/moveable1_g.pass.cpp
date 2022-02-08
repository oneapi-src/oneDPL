#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItem{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                s::tuple<int, float> a(1, 2.f), b;
                b = s::move(a);
                ret_access[0] &= (s::get<0>(b) == 1 && s::get<1>(b) == 2.f);
                ret_access[0] &= (s::get<0>(a) == 1 && s::get<1>(a) == 2.f);

                s::tuple<int, double> c(s::move(b));
                ret_access[0] &= (s::get<0>(c) == 1 && s::get<1>(c) == 2.f);
                ret_access[0] &= (s::get<0>(b) == 1 && s::get<1>(b) == 2.f);
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
    return 0;
}
