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

class KernelBitNotTest;

void
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelBitNotTest>([=]() {
            typedef s::bit_not<int> F;
            const F f = F();
            static_assert((s::is_same<F::argument_type, int>::value), "");
            static_assert((s::is_same<F::result_type, int>::value), "");
            ret_access[0] = ((f(0xEA95) & 0xFFFF) == 0x156A);
            ret_access[0] &= ((f(0x58D3) & 0xFFFF) == 0xA72C);
            ret_access[0] &= ((f(0) & 0xFFFF) == 0xFFFF);
            ret_access[0] &= ((f(0xFFFF) & 0xFFFF) == 0);

            typedef s::bit_not<long> F2;
            const F2 f2 = F2();
            ret_access[0] &= ((f2(0xEA95L) & 0xFFFF) == 0x156A);
            ret_access[0] &= ((f2(0x58D3L) & 0xFFFF) == 0xA72C);
            ret_access[0] &= ((f2(0L) & 0xFFFF) == 0xFFFF);
            ret_access[0] &= ((f2(0xFFFFL) & 0xFFFF) == 0);
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
