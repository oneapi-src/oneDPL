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
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelBitNotTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
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
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
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
