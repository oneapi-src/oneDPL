// test ratio_less

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
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class Rat1, class Rat2, bool result, class KernelName>
sycl::cl_bool
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName>([=]() {
                static_assert((result == s::ratio_less<Rat1, Rat2>::value), "");
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

sycl::cl_bool
kernel_test()
{
    auto ret = true;
    {
        typedef s::ratio<1, 1> R1;
        typedef s::ratio<1, 1> R2;
        ret &= test<R1, R2, false, T1>();
    }
    {
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T2>();
    }
    {
        typedef s::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef s::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T3>();
    }
    {
        typedef s::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef s::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, false, T4>();
    }
    {
        typedef s::ratio<1, 1> R1;
        typedef s::ratio<1, -1> R2;
        ret &= test<R1, R2, false, T5>();
    }
    {
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef s::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T6>();
    }
    {
        typedef s::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, true, T7>();
    }
    {
        typedef s::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef s::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, false, T8>();
    }
    {
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef s::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R2;
        ret &= test<R1, R2, true, T9>();
    }
    {
        typedef s::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        ret &= test<R1, R2, false, T10>();
    }
    {
        typedef s::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef s::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        ret &= test<R1, R2, true, T11>();
    }
    {
        typedef s::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef s::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL> R2;
        ret &= test<R1, R2, true, T12>();
    }
    {
        typedef s::ratio<641981, 1339063> R1;
        typedef s::ratio<1291640, 2694141LL> R2;
        ret &= test<R1, R2, false, T13>();
    }
    {
        typedef s::ratio<1291640, 2694141LL> R1;
        typedef s::ratio<641981, 1339063> R2;
        ret &= test<R1, R2, true, T14>();
    }

    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
