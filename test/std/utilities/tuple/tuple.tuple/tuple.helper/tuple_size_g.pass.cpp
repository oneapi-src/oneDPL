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

void
kernel_test()
{
    sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                static_assert(s::tuple_size<s::tuple<>>::value == 0, "");
                static_assert(s::tuple_size<s::tuple<int>>::value == 1, "");
                static_assert(s::tuple_size<s::tuple<void>>::value == 1, "");
                typedef s::tuple<int, const int&, void> test_tuple1;
                static_assert(s::tuple_size<test_tuple1>::value == 3, "");
                static_assert(s::tuple_size<s::tuple<s::tuple<void>>>::value == 1, "");

                static_assert(s::tuple_size<const s::tuple<>>::value == 0, "");
                static_assert(s::tuple_size<const s::tuple<int>>::value == 1, "");
                static_assert(s::tuple_size<const s::tuple<void>>::value == 1, "");
                static_assert(s::tuple_size<const test_tuple1>::value == 3, "");
                static_assert(s::tuple_size<const s::tuple<s::tuple<void>>>::value == 1, "");

                static_assert(s::tuple_size<volatile s::tuple<>>::value == 0, "");
                static_assert(s::tuple_size<volatile s::tuple<int>>::value == 1, "");
                static_assert(s::tuple_size<volatile s::tuple<void>>::value == 1, "");
                static_assert(s::tuple_size<volatile test_tuple1>::value == 3, "");
                static_assert(s::tuple_size<volatile s::tuple<s::tuple<void>>>::value == 1, "");

                static_assert(s::tuple_size<const volatile s::tuple<>>::value == 0, "");
                static_assert(s::tuple_size<const volatile s::tuple<int>>::value == 1, "");
                static_assert(s::tuple_size<const volatile s::tuple<void>>::value == 1, "");
                static_assert(s::tuple_size<const volatile test_tuple1>::value == 3, "");
                static_assert(s::tuple_size<const volatile s::tuple<s::tuple<void>>>::value == 1, "");
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    std::cout << "pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
