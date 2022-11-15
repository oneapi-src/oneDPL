// type_traits

// aligned_storage
//
//  Issue 3034 added:
//  The member typedef type shall be a trivial standard-layout type.

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    {
        typedef s::aligned_storage<10, 1>::type T1;
#if TEST_STD_VER >= 14
        ASSERT_SAME_TYPE(T1, s::aligned_storage_t<10, 1>);
#endif

#if TEST_STD_VER <= 17
        static_assert(s::is_pod<T1>::value, "");
#endif
        static_assert(s::is_trivial<T1>::value, "");
        static_assert(s::is_standard_layout<T1>::value, "");
        static_assert(s::alignment_of<T1>::value == 1, "");
        static_assert(sizeof(T1) == 10, "");
    }
    {
        typedef s::aligned_storage<10, 2>::type T1;
#if TEST_STD_VER >= 14
        ASSERT_SAME_TYPE(T1, s::aligned_storage_t<10, 2>);
#endif

#if TEST_STD_VER <= 17
        static_assert(s::is_pod<T1>::value, "");
#endif
        static_assert(s::is_trivial<T1>::value, "");
        static_assert(s::is_standard_layout<T1>::value, "");
        static_assert(s::alignment_of<T1>::value == 2, "");
        static_assert(sizeof(T1) == 10, "");
    }
    {
        typedef s::aligned_storage<10, 4>::type T1;
#if TEST_STD_VER >= 14
        ASSERT_SAME_TYPE(T1, s::aligned_storage_t<10, 4>);
#endif

#if TEST_STD_VER <= 17
        static_assert(s::is_pod<T1>::value, "");
#endif
        static_assert(s::is_trivial<T1>::value, "");
        static_assert(s::is_standard_layout<T1>::value, "");
        static_assert(s::alignment_of<T1>::value == 4, "");
        static_assert(sizeof(T1) == 12, "");
    }
    {
        typedef s::aligned_storage<10, 8>::type T1;
#if TEST_STD_VER >= 14
        ASSERT_SAME_TYPE(T1, s::aligned_storage_t<10, 8>);
#endif

#if TEST_STD_VER <= 17
        static_assert(s::is_pod<T1>::value, "");
#endif
        static_assert(s::is_trivial<T1>::value, "");
        static_assert(s::is_standard_layout<T1>::value, "");
        static_assert(s::alignment_of<T1>::value == 8, "");
        static_assert(sizeof(T1) == 16, "");
    }
    return true;
}
class KernelTest;

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
