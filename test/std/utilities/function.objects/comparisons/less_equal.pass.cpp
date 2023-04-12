#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class KernelLessEqualTest;

void
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelLessEqualTest>([=]() {
            typedef s::less_equal<int> F;
            const F f = F();
            static_assert((s::is_same<int, F::first_argument_type>::value), "");
            static_assert((s::is_same<int, F::second_argument_type>::value), "");
            static_assert((s::is_same<bool, F::result_type>::value), "");
            ret_access[0] = (f(36, 36));
            ret_access[0] &= (!f(36, 6));
            ret_access[0] &= (f(6, 36));

            typedef s::less_equal<float> Fd;
            const Fd f2 = Fd();
            ret_access[0] &= (!f2(36, 6.0f));
            ret_access[0] &= (!f2(36.0f, 6));
            ret_access[0] &= (f2(6, 36.0f));
            ret_access[0] &= (f2(6.0f, 36));
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
