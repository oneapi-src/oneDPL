// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

sycl::cl_bool
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = true;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                ret_access[0] &= (s::tuple_size<const s::tuple<>>::value == 0);
                ret_access[0] &= (s::tuple_size<volatile s::tuple<int>>::value == 1);
                ret_access[0] &= (s::tuple_size<const volatile s::tuple<void>>::value == 1);

                typedef s::tuple<int, const int&, void> test_tuple1;
                ret_access[0] &= (s::tuple_size<const test_tuple1>::value == 3);
                ret_access[0] &= (s::tuple_size<const volatile test_tuple1>::value == 3);
                ret_access[0] &= (s::tuple_size<volatile s::tuple<s::tuple<void>>>::value == 1);
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
