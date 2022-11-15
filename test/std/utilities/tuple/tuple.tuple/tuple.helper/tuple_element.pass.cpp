// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T, s::size_t N, class U>
void __attribute__((always_inline)) test_compile()
{
    static_assert((s::is_same<typename s::tuple_element<N, T>::type, U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<N, const T>::type, const U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<N, volatile T>::type, volatile U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
}

template <class T, s::size_t N, class U>
bool __attribute__((always_inline)) test_runtime()
{
    bool ret = (s::is_same<typename s::tuple_element<N, T>::type, U>::value);
    ret &= (s::is_same<typename s::tuple_element<N, const T>::type, const U>::value);
    ret &= (s::is_same<typename s::tuple_element<N, volatile T>::type, volatile U>::value);
    ret &= (s::is_same<typename s::tuple_element<N, const volatile T>::type, const volatile U>::value);

    return ret;
}

class KernelTupleElementTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleElementTest>([=]() {
            test_compile<s::tuple<int>, 0, int>();
            test_compile<s::tuple<char, int>, 0, char>();
            test_compile<s::tuple<char, int>, 1, int>();
            test_compile<s::tuple<int*, char, int>, 0, int*>();
            test_compile<s::tuple<int*, char, int>, 1, char>();
            test_compile<s::tuple<int*, char, int>, 2, int>();

            // Runtime test

            ret_access[0] = test_runtime<s::tuple<int>, 0, int>();
            ret_access[0] &= test_runtime<s::tuple<char, int>, 0, char>();
            ret_access[0] &= test_runtime<s::tuple<char, int>, 1, int>();
            ret_access[0] &= test_runtime<s::tuple<int*, char, int>, 0, int*>();
            ret_access[0] &= test_runtime<s::tuple<int*, char, int>, 1, char>();
            ret_access[0] &= test_runtime<s::tuple<int*, char, int>, 2, int>();
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
