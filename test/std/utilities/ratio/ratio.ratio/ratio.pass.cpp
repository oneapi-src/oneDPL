// test ratio:  The static data members num and den shall have the common
//    divisor of the absolute values of N and D:

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(ratio)
namespace s = oneapi_cpp_ns;
#else
#    include <ratio>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <long long N, long long D, long long eN, long long eD, class KernelName>
cl::sycl::cl_bool
test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> item1{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName>([=]() {
                static_assert((s::ratio<N, D>::num == eN), "");
                static_assert((s::ratio<N, D>::den == eD), "");
                ret_acc[0] = true;
            });
        });
    }
    return ret;
}

class T1;
class T2;
class T3;
class T4;
class T5;
class T6;
class T7;
class T8;
class T9;
class T10;
class T11;
class T12;
class T13;
class T14;
class T15;
class T16;
class T17;
class T18;
class T19;
class T20;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = test<1, 1, 1, 1, T1>();
    ret &= test<1, 10, 1, 10, T2>();
    ret &= test<10, 10, 1, 1, T3>();
    ret &= test<10, 1, 10, 1, T4>();
    ret &= test<12, 4, 3, 1, T5>();
    ret &= test<12, -4, -3, 1, T6>();
    ret &= test<-12, 4, -3, 1, T7>();
    ret &= test<-12, -4, 3, 1, T8>();
    ret &= test<4, 12, 1, 3, T9>();
    ret &= test<4, -12, -1, 3, T10>();
    ret &= test<-4, 12, -1, 3, T11>();
    ret &= test<-4, -12, 1, 3, T12>();
    ret &= test<222, 333, 2, 3, T13>();
    ret &= test<222, -333, -2, 3, T14>();
    ret &= test<-222, 333, -2, 3, T15>();
    ret &= test<-222, -333, 2, 3, T16>();
    ret &= test<0x7FFFFFFFFFFFFFFFLL, 127, 72624976668147841LL, 1, T17>();
    ret &= test<-0x7FFFFFFFFFFFFFFFLL, 127, -72624976668147841LL, 1, T18>();
    ret &= test<0x7FFFFFFFFFFFFFFFLL, -127, -72624976668147841LL, 1, T19>();
    ret &= test<-0x7FFFFFFFFFFFFFFFLL, -127, 72624976668147841LL, 1, T20>();
    if (ret)
        std::cout << "pass" << std::endl;
    else
        std::cout << "fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
