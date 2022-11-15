//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class U> optional<T>& operator=(U&& v);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

template <class T, class Arg = T, bool Expect = true>
void
assert_assignable()
{
    static_assert(s::is_assignable<optional<T>&, Arg>::value == Expect, "");
    static_assert(!s::is_assignable<const optional<T>&, Arg>::value, "");
}

struct MismatchType
{
    explicit MismatchType(int) {}
    explicit MismatchType(char*) {}
    explicit MismatchType(int*) = delete;
    MismatchType&
    operator=(int)
    {
        return *this;
    }
    MismatchType&
    operator=(int*)
    {
        return *this;
    }
    MismatchType&
    operator=(char*) = delete;
};

struct FromOptionalType
{
    using Opt = s::optional<FromOptionalType>;
    FromOptionalType() = default;
    FromOptionalType(FromOptionalType const&) = delete;
    template <class Dummy = void>
    constexpr FromOptionalType(Opt&)
    {
        Dummy::BARK;
    }
    template <class Dummy = void>
    constexpr FromOptionalType&
    operator=(Opt&)
    {
        Dummy::BARK;
        return *this;
    }
};

void
test_sfinae()
{
    cl::sycl::queue q;
    cl::sycl::range<1> numOfItems1{1};
    {

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                assert_assignable<int>();
                assert_assignable<int, int&>();
                assert_assignable<int, int const&>();
                // Mismatch type
                assert_assignable<MismatchType, int>();
                assert_assignable<MismatchType, int*, false>();
                assert_assignable<MismatchType, char*, false>();
                // Type constructible from optional
                assert_assignable<FromOptionalType, s::optional<FromOptionalType>&, false>();
            });
        });
    }
}

template <class KernelTest, class T, class Value = int>
bool
test_with_type()
{

    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                { // to empty
                    optional<T> opt;
                    opt = Value(3);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(3));
                }
                { // to existing
                    optional<T> opt(Value(42));
                    opt = Value(3);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(3));
                }
                { // test const
                    optional<T> opt(Value(42));
                    const T t(Value(3));
                    opt = t;
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(3));
                }
                { // test default argument
                    optional<T> opt;
                    opt = {Value(1)};
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(1));
                }
                { // test default argument
                    optional<T> opt(Value(42));
                    opt = {};
                    ret_access[0] &= (static_cast<bool>(opt) == false);
                }
            });
        });
    }
    return ret;
}

enum MyEnum
{
    Zero,
    One,
    Two,
    Three,
    FortyTwo = 42
};

class KernelTest1;
class KernelTest2;
class KernelTest3;

int
main(int, char**)
{
    test_sfinae();
    // Test with various scalar types
    auto ret = test_with_type<KernelTest1, int>();
    ret &= test_with_type<KernelTest2, MyEnum, MyEnum>();
    ret &= test_with_type<KernelTest3, int, MyEnum>();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;

    return 0;
}
