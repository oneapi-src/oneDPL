//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// aligned_union<size_t Len, class ...Types>

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

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            {
                typedef s::aligned_union<10, char>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, char>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 1, "");
                static_assert(sizeof(T1) == 10, "");
            }
            {
                typedef s::aligned_union<10, short>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, short>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 2, "");
                static_assert(sizeof(T1) == 10, "");
            }
            {
                typedef s::aligned_union<10, int>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, int>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 4, "");
                static_assert(sizeof(T1) == 12, "");
            }
            {
                typedef s::aligned_union<10, short, char>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, short, char>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 2, "");
                static_assert(sizeof(T1) == 10, "");
            }
            {
                typedef s::aligned_union<10, char, short>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, char, short>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 2, "");
                static_assert(sizeof(T1) == 10, "");
            }
            {
                typedef s::aligned_union<2, int, char, short>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<2, int, char, short>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 4, "");
                static_assert(sizeof(T1) == 4, "");
            }
            {
                typedef s::aligned_union<2, char, int, short>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<2, char, int, short>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 4, "");
                static_assert(sizeof(T1) == 4, "");
            }
            {
                typedef s::aligned_union<2, char, short, int>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<2, char, short, int>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 4, "");
                static_assert(sizeof(T1) == 4, "");
            }
        });
    });
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            {
                typedef s::aligned_union<10, double>::type T1;
#if TEST_STD_VER > 11
                ASSERT_SAME_TYPE(T1, s::aligned_union_t<10, double>);
#endif
                static_assert(s::is_trivial<T1>::value, "");
                static_assert(s::is_standard_layout<T1>::value, "");
                static_assert(s::alignment_of<T1>::value == 8, "");
                static_assert(sizeof(T1) == 16, "");
            }
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
