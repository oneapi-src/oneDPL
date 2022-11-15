// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelTupleSizeIncompleteTest;

template <class T, size_t Size = sizeof(s::tuple_size<T>)>
constexpr bool
is_complete(int)
{
    static_assert(Size > 0, "");
    return true;
}
template <class>
constexpr bool
is_complete(long)
{
    return false;
}
template <class T>
constexpr bool
is_complete()
{
    return is_complete<T>(0);
}

template <class T, size_t Size = sizeof(s::tuple_size<T>)>
constexpr bool
is_complete_runtime(int)
{
    return (Size > 0);
}
template <class>
constexpr bool
is_complete_runtime(long)
{
    return false;
}
template <class T>
constexpr bool
is_complete_runtime()
{
    return is_complete<T>(0);
}

struct Dummy1
{
};
struct Dummy2
{
};

namespace std
{
template <>
struct tuple_size<Dummy1> : public integral_constant<size_t, 0>
{
};
} // namespace std

template <class T>
void __attribute__((always_inline)) test_complete_compile()
{
    static_assert(is_complete<T>(), "");
    static_assert(is_complete<const T>(), "");
    static_assert(is_complete<volatile T>(), "");
    static_assert(is_complete<const volatile T>(), "");
}

template <class T>
bool __attribute__((always_inline)) test_complete_runtime()
{
    bool ret;
    ret = is_complete<T>();
    ret &= is_complete<const T>();
    ret &= is_complete<volatile T>();
    ret &= is_complete<const volatile T>();
    return ret;
}

template <class T>
bool __attribute__((always_inline)) test_incomplete()
{
    bool ret;
    ret = !is_complete_runtime<T>();
    ret &= !is_complete_runtime<const T>();
    ret &= !is_complete_runtime<volatile T>();
    ret &= !is_complete_runtime<const volatile T>();
    return ret;
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleSizeIncompleteTest>([=]() {
            // Compile time check

            test_complete_compile<s::tuple<>>();
            test_complete_compile<s::tuple<int&>>();
            test_complete_compile<s::tuple<int&&, int&, void*>>();
            test_complete_compile<s::pair<int, long>>();
            test_complete_compile<s::array<int, 5>>();
            test_complete_compile<Dummy1>();

            ret_access[0] = test_complete_runtime<s::tuple<>>();
            ret_access[0] &= test_complete_runtime<s::tuple<int&>>();
            ret_access[0] &= test_complete_runtime<s::tuple<int&&, int&, void*>>();
            ret_access[0] &= test_complete_runtime<s::pair<int, long>>();
            ret_access[0] &= test_complete_runtime<s::array<int, 5>>();
            ret_access[0] &= test_complete_runtime<Dummy1>();
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
