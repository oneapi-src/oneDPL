#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    using s::array;
    using s::tuple_element;
    // This relies on the fact that <utility> includes <type_traits>:
    using s::is_same;

    const size_t len = 3;
    typedef array<int, len> array_type;

    static_assert(is_same<tuple_element<0, array_type>::type, int>::value, "");
    static_assert(is_same<tuple_element<1, array_type>::type, int>::value, "");
    static_assert(is_same<tuple_element<2, array_type>::type, int>::value, "");

    static_assert(is_same<tuple_element<0, const array_type>::type, const int>::value, "");
    static_assert(is_same<tuple_element<1, const array_type>::type, const int>::value, "");
    static_assert(is_same<tuple_element<2, const array_type>::type, const int>::value, "");

    static_assert(is_same<tuple_element<0, volatile array_type>::type, volatile int>::value, "");
    static_assert(is_same<tuple_element<1, volatile array_type>::type, volatile int>::value, "");
    static_assert((is_same<tuple_element<2, volatile array_type>::type, volatile int>::value == true), "");

    static_assert(is_same<tuple_element<0, const volatile array_type>::type, const volatile int>::value, "");
    static_assert(is_same<tuple_element<1, const volatile array_type>::type, const volatile int>::value, "");
    static_assert(is_same<tuple_element<2, const volatile array_type>::type, const volatile int>::value, "");
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
