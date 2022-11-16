//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// add_rvalue_reference
// If T names a referenceable type then the member typedef type
//   shall name T&&; otherwise, type shall name T.

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

template <class T, class U>
void
test_add_rvalue_reference()
{
    ASSERT_SAME_TYPE(U, typename s::add_rvalue_reference<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, s::add_rvalue_reference_t<T>);
#endif
}

template <class F>
void
test_function0()
{
    ASSERT_SAME_TYPE(F&&, typename s::add_rvalue_reference<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F&&, s::add_rvalue_reference_t<F>);
#endif
}

template <class F>
void
test_function1()
{
    ASSERT_SAME_TYPE(F, typename s::add_rvalue_reference<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F, s::add_rvalue_reference_t<F>);
#endif
}

struct Foo
{
};

cl::sycl::cl_bool
kernel_test()
{
    test_add_rvalue_reference<void, void>();
    test_add_rvalue_reference<int, int&&>();
    test_add_rvalue_reference<int[3], int(&&)[3]>();
    test_add_rvalue_reference<int&, int&>();
    test_add_rvalue_reference<const int&, const int&>();
    test_add_rvalue_reference<int*, int*&&>();
    test_add_rvalue_reference<const int*, const int*&&>();
    test_add_rvalue_reference<Foo, Foo&&>();

    //  LWG 2101 specifically talks about add_rvalue_reference and functions.
    //  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
    test_function0<void()>();
    test_function1<void() const>();
    test_function1<void()&>();
    test_function1<void() &&>();
    test_function1<void() const&>();
    test_function1<void() const&&>();

    //  But a cv- or ref-qualified member function *is* "a referenceable type"
    test_function0<void (Foo::*)()>();
    test_function0<void (Foo::*)() const>();
    test_function0<void (Foo::*)()&>();
    test_function0<void (Foo::*)() &&>();
    test_function0<void (Foo::*)() const&>();
    test_function0<void (Foo::*)() const&&>();

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
            cgh.single_task<class KernelNot1Test>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
