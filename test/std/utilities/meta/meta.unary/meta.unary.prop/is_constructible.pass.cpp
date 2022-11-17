//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class T, class... Args>
//   struct is_constructible;

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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct A
{
    explicit A(int);
    A(int, double);
    A(int, long, double);

  private:
    A(char);
};

struct Base
{
};
struct Derived : public Base
{
};

struct PrivateDtor
{
    PrivateDtor(int) {}

  private:
    ~PrivateDtor() {}
};

struct S
{
    template <class T>
    explicit operator T() const;
};

template <class To>
struct ImplicitTo
{
    operator To();
};

template <class To>
struct ExplicitTo
{
    explicit operator To();
};

template <class T>
void
test_is_constructible()
{
    static_assert((s::is_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert(s::is_constructible_v<T>, "");
#endif
}

template <class T, class A0>
void
test_is_constructible()
{
    static_assert((s::is_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
void
test_is_constructible()
{
    static_assert((s::is_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_constructible_v<T, A0, A1>), "");
#endif
}

template <class T, class A0, class A1, class A2>
void
test_is_constructible()
{
    static_assert((s::is_constructible<T, A0, A1, A2>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_constructible_v<T, A0, A1, A2>), "");
#endif
}

template <class T>
void
test_is_not_constructible()
{
    static_assert((!s::is_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void
test_is_not_constructible()
{
    static_assert((!s::is_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_constructible_v<T, A0>), "");
#endif
}

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            typedef Base B;
            typedef Derived D;

            test_is_constructible<int>();
            test_is_constructible<int, const int>();
            test_is_constructible<int&, int&>();

            test_is_not_constructible<int, void()>();
            test_is_not_constructible<int, void (&)()>();
            test_is_not_constructible<int, void() const>();
            test_is_not_constructible<int&, void>();
            test_is_not_constructible<int&, void()>();
            test_is_not_constructible<int&, void() const>();
            test_is_not_constructible<int&, void (&)()>();

            test_is_not_constructible<void>();
            test_is_not_constructible<const void>(); // LWG 2738
            test_is_not_constructible<volatile void>();
            test_is_not_constructible<const volatile void>();
            test_is_not_constructible<int&>();
            test_is_constructible<int, S>();
            test_is_not_constructible<int&, S>();

            test_is_constructible<void (&)(), void (&)()>();
            test_is_constructible<void (&)(), void()>();
            test_is_constructible<void(&&)(), void(&&)()>();
            test_is_constructible<void(&&)(), void()>();
            test_is_constructible<void(&&)(), void (&)()>();

            test_is_constructible<int const&, int>();
            test_is_constructible<int const&, int&&>();

            test_is_constructible<void (&)(), void(&&)()>();

            test_is_not_constructible<int&, int>();
            test_is_not_constructible<int&, int const&>();
            test_is_not_constructible<int&, int&&>();

            test_is_constructible<int&&, int>();
            test_is_constructible<int&&, int&&>();
            test_is_not_constructible<int&&, int&>();
            test_is_not_constructible<int&&, int const&&>();

            test_is_constructible<Base, Derived>();
            test_is_constructible<Base&, Derived&>();
            test_is_not_constructible<Derived&, Base&>();
            test_is_constructible<Base const&, Derived const&>();
            test_is_not_constructible<Derived const&, Base const&>();
            test_is_not_constructible<Derived const&, Base>();

            test_is_constructible<Base&&, Derived>();
            test_is_constructible<Base&&, Derived&&>();
            test_is_not_constructible<Derived&&, Base&&>();
            test_is_not_constructible<Derived&&, Base>();

            // test that T must also be destructible
            test_is_constructible<PrivateDtor&, PrivateDtor&>();
            test_is_not_constructible<PrivateDtor, int>();

            test_is_not_constructible<void() const, void() const>();
            test_is_not_constructible<void() const, void*>();

            test_is_constructible<int&, ImplicitTo<int&>>();
            test_is_constructible<const int&, ImplicitTo<int&&>>();
            test_is_constructible<int&&, ImplicitTo<int&&>>();
            test_is_constructible<const int&, ImplicitTo<int>>();

            test_is_not_constructible<B&&, B&>();
            test_is_not_constructible<B&&, D&>();
            test_is_constructible<B&&, ImplicitTo<D&&>>();
            test_is_constructible<B&&, ImplicitTo<D&&>&>();
            test_is_constructible<const int&, ImplicitTo<int&>&>();
            test_is_constructible<const int&, ImplicitTo<int&>>();
            test_is_constructible<const int&, ExplicitTo<int&>&>();
            test_is_constructible<const int&, ExplicitTo<int&>>();

            test_is_constructible<const int&, ExplicitTo<int&>&>();
            test_is_constructible<const int&, ExplicitTo<int&>>();
            test_is_constructible<int&, ExplicitTo<int&>>();
            test_is_constructible<const int&, ExplicitTo<int&&>>();

            test_is_constructible<int&, ExplicitTo<int&>>();

            static_assert(s::is_constructible<int&&, ExplicitTo<int&&>>::value, "");
        });
    });
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            test_is_constructible<A, int>();
            test_is_constructible<A, int, double>();
            test_is_constructible<A, int, long, double>();

            test_is_not_constructible<A>();
            test_is_not_constructible<A, char>();
            test_is_not_constructible<A, void>();
            test_is_constructible<int&&, double&>();
            test_is_constructible<int&&, double&>();
            test_is_not_constructible<const int&, ExplicitTo<double&&>>();
            test_is_not_constructible<int&&, ExplicitTo<double&&>>();
        });
    });
}
int
main(int, char**)
{
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    std::cout << "Pass" << std::endl;
    return 0;
}
