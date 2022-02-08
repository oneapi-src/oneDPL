//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// common_type

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <functional>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct E
{
};

template <class T>
struct X
{
    explicit X(T const&) {}
};

template <class T>
struct S
{
    explicit S(T const&) {}
};

namespace std
{
template <typename T>
struct common_type<T, ::S<T>>
{
    typedef S<T> type;
};

template <class T>
struct common_type<::S<T>, T>
{
    typedef S<T> type;
};

//  P0548
template <class T>
struct common_type<::S<T>, ::S<T>>
{
    typedef S<T> type;
};

template <>
struct common_type<::S<long>, long>
{
};
template <>
struct common_type<long, ::S<long>>
{
};
template <>
struct common_type<::X<double>, ::X<double>>
{
};
} // namespace std

template <class>
struct VoidT
{
    typedef void type;
};

template <class Tp>
struct always_bool_imp
{
    using type = bool;
};
template <class Tp>
using always_bool = typename always_bool_imp<Tp>::type;

template <class... Args>
constexpr auto
no_common_type_imp(int) -> always_bool<typename s::common_type<Args...>::type>
{
    return false;
}

template <class... Args>
constexpr bool
no_common_type_imp(long)
{
    return true;
}

template <class... Args>
using no_common_type = s::integral_constant<bool, no_common_type_imp<Args...>(0)>;

template <class T1, class T2>
struct TernaryOp
{
    static_assert((s::is_same<typename s::decay<T1>::type, T1>::value), "must be same");
    static_assert((s::is_same<typename s::decay<T2>::type, T2>::value), "must be same");
    typedef typename s::decay<decltype(false ? s::declval<T1>() : s::declval<T2>())>::type type;
};

// -- If sizeof...(T) is zero, there shall be no member type.
void
test_bullet_one()
{
    static_assert(no_common_type<>::value, "");
}

// If sizeof...(T) is one, let T0 denote the sole type constituting the pack T.
// The member typedef-name type shall denote the same type as decay_t<T0>.
void
test_bullet_two()
{
    static_assert((s::is_same<s::common_type<void>::type, void>::value), "");
    static_assert((s::is_same<s::common_type<int>::type, int>::value), "");
    static_assert((s::is_same<s::common_type<int const>::type, int>::value), "");
    static_assert((s::is_same<s::common_type<int volatile[]>::type, int volatile*>::value), "");
}

// (3.4)
// -- If sizeof...(T) is greater than two, let T1, T2, and R, respectively,
// denote the first, second, and (pack of) remaining types constituting T.
// Let C denote the same type, if any, as common_type_t<T1, T2>. If there is
// such a type C, the member typedef-name type shall denote the
// same type, if any, as common_type_t<C, R...>. Otherwise, there shall be
// no member type.
void
test_bullet_four()
{
    { // test that there is no ::type member
        static_assert((no_common_type<int, E>::value), "");
        static_assert((no_common_type<int, int, E>::value), "");
        static_assert((no_common_type<int, int, E, int>::value), "");
        static_assert((no_common_type<int, int, int, E>::value), "");
    }
}

// The example code specified in Note B for common_type
namespace note_b_example
{

typedef bool (&PF1)();
typedef short (*PF2)(long);

struct S
{
    operator PF2() const;
    double
    operator()(char, int&);
    void
    fn(long) const;
    char data;
};

typedef void (S::*PMF)(long) const;
typedef char S::*PMD;

using s::is_same;
using s::result_of;
} // namespace note_b_example

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            static_assert((s::is_same<s::common_type<int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<char>::type, char>::value), "");
#if TEST_STD_VER > 11
            static_assert((s::is_same<s::common_type_t<int>, int>::value), "");
            static_assert((s::is_same<s::common_type_t<char>, char>::value), "");
#endif

            static_assert((s::is_same<s::common_type<int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<volatile int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const volatile int>::type, int>::value), "");

            static_assert((s::is_same<s::common_type<int, int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<int, const int>::type, int>::value), "");

            static_assert((s::is_same<s::common_type<long, const int>::type, long>::value), "");
            static_assert((s::is_same<s::common_type<const long, int>::type, long>::value), "");
            static_assert((s::is_same<s::common_type<long, volatile int>::type, long>::value), "");
            static_assert((s::is_same<s::common_type<volatile long, int>::type, long>::value), "");
            static_assert((s::is_same<s::common_type<const long, const int>::type, long>::value), "");

            static_assert((s::is_same<s::common_type<short, char>::type, int>::value), "");
#if TEST_STD_VER > 11
            static_assert((s::is_same<s::common_type_t<short, char>, int>::value), "");
#endif

            static_assert((s::is_same<s::common_type<unsigned, char, long long>::type, long long>::value), "");
#if TEST_STD_VER > 11
            static_assert((s::is_same<s::common_type_t<unsigned, char, long long>, long long>::value), "");
#endif

            static_assert((s::is_same<s::common_type<void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<const void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<volatile void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<const volatile void>::type, void>::value), "");

            static_assert((s::is_same<s::common_type<void, const void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<const void, void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<void, volatile void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<volatile void, void>::type, void>::value), "");
            static_assert((s::is_same<s::common_type<const void, const void>::type, void>::value), "");

            static_assert((s::is_same<s::common_type<int, S<int>>::type, S<int>>::value), "");
            static_assert((s::is_same<s::common_type<int, S<int>, S<int>>::type, S<int>>::value), "");
            static_assert((s::is_same<s::common_type<int, int, S<int>>::type, S<int>>::value), "");

            test_bullet_one();
            test_bullet_two();

            //  P0548
            static_assert((s::is_same<s::common_type<S<int>>::type, S<int>>::value), "");
            static_assert((s::is_same<s::common_type<S<int>, S<int>>::type, S<int>>::value), "");

            static_assert((s::is_same<s::common_type<int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<volatile int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const volatile int>::type, int>::value), "");

            static_assert((s::is_same<s::common_type<int, int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const int, int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<int, const int>::type, int>::value), "");
            static_assert((s::is_same<s::common_type<const int, const int>::type, int>::value), "");

            // Test that we're really variadic in C++11
            static_assert(s::is_same<s::common_type<int, int, int, int, int, int, int, int>::type, int>::value, "");
        });
    });
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            static_assert((s::is_same<s::common_type<double, char>::type, double>::value), "");
#if TEST_STD_VER > 11
            static_assert((s::is_same<s::common_type_t<double, char>, double>::value), "");
#endif

            static_assert((s::is_same<s::common_type<double, char, long long>::type, double>::value), "");
#if TEST_STD_VER > 11
            static_assert((s::is_same<s::common_type_t<double, char, long long>, double>::value), "");
#endif
        });
    });
}

int
main()
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
