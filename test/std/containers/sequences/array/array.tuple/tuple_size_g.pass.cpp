#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    using s::array;
    using s::tuple_size;
    // This relies on the fact that <utility> includes <type_traits>:
    using s::is_same;

    {
        const size_t len = 5;
        typedef array<int, len> array_type;
        static_assert(tuple_size<array_type>::value == 5, "");
        static_assert(tuple_size<const array_type>::value == 5, "");
        static_assert(tuple_size<volatile array_type>::value == 5, "");
        static_assert(tuple_size<const volatile array_type>::value == 5, "");
    }

    {
        const size_t len = 0;
        typedef array<float, len> array_type;
        static_assert(tuple_size<array_type>::value == 0, "");
        static_assert(tuple_size<const array_type>::value == 0, "");
        static_assert(tuple_size<volatile array_type>::value == 0, "");
        static_assert(tuple_size<const volatile array_type>::value == 0, "");
    }
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

    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
