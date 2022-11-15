// This code is used to test
// template <class... Types>
// void swap(tuple<Types...>&x, tuple<Types...>& y);

#include "oneapi_std_test_config.h"
#include "MoveOnly.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelNonMemberSwapTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelNonMemberSwapTest>([=]() {
            {
                typedef s::tuple<> T;
                T t0;
                T t1;
                swap(t0, t1);
            }

            {
                typedef s::tuple<MoveOnly> T;
                T t0(MoveOnly(0));
                T t1(MoveOnly(1));
                swap(t0, t1);
                ret_access[0] = (s::get<0>(t0) == 1 && s::get<0>(t1) == 0);
            }

            {
                typedef s::tuple<MoveOnly, MoveOnly> T;
                T t0(MoveOnly(0), MoveOnly(1));
                T t1(MoveOnly(2), MoveOnly(3));
                swap(t0, t1);
                ret_access[0] &= (s::get<0>(t0) == 2 && s::get<1>(t0) == 3 && s::get<0>(t1) == 0 && s::get<1>(t1) == 1);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
