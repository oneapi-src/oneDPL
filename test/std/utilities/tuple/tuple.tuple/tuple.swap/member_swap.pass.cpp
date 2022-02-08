#include "oneapi_std_test_config.h"
#include "MoveOnly.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <functional>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelMemberSwapTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelMemberSwapTest>([=]() {
            {
                typedef s::tuple<> T;
                T t0;
                T t1;
                t0.swap(t1);
            }

            {
                typedef s::tuple<MoveOnly> T;
                T t0(MoveOnly(0));
                T t1(MoveOnly(1));
                t0.swap(t1);
                ret_access[0] = (s::get<0>(t0) == 1 && s::get<0>(t1) == 0);
            }

            {
                typedef s::tuple<MoveOnly, MoveOnly> T;
                T t0(MoveOnly(0), MoveOnly(1));
                T t1(MoveOnly(2), MoveOnly(3));
                t0.swap(t1);
                ret_access[0] &= (s::get<0>(t0) == 2 && s::get<1>(t0) == 3 && s::get<0>(t1) == 0 && s::get<1>(t1) == 1);
            }

            {
                typedef s::tuple<MoveOnly, MoveOnly, MoveOnly> T;
                T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
                T t1(MoveOnly(3), MoveOnly(4), MoveOnly(5));
                t0.swap(t1);
                ret_access[0] &= (s::get<0>(t0) == 3 && s::get<1>(t0) == 4 && s::get<2>(t0) == 5 &&
                                  s::get<0>(t1) == 0 && s::get<1>(t1) == 1 && s::get<2>(t1) == 2);
            }
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
