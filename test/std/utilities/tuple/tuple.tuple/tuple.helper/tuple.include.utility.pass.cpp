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
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
#    include <cstddef>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelTupleIncludeUtilityTest;

template <class T, s::size_t N, class U, size_t idx>
void __attribute__((always_inline)) test()
{
    static_assert((s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<T>>::value), "");
    static_assert((s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<const T>>::value), "");
    static_assert((s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<volatile T>>::value), "");
    static_assert((s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<const volatile T>>::value), "");
    static_assert((s::is_same<typename s::tuple_element<idx, T>::type, U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<idx, const T>::type, const U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<idx, volatile T>::type, volatile U>::value), "");
    static_assert((s::is_same<typename s::tuple_element<idx, const volatile T>::type, const volatile U>::value), "");
}

template <class T, s::size_t N, class U, size_t idx>
bool __attribute__((always_inline)) test_runtime()
{
    bool ret = (s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<T>>::value);
    ret &= (s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<const T>>::value);
    ret &= (s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<volatile T>>::value);
    ret &= (s::is_base_of<s::integral_constant<s::size_t, N>, s::tuple_size<const volatile T>>::value);
    ret &= (s::is_same<typename s::tuple_element<idx, T>::type, U>::value);
    ret &= (s::is_same<typename s::tuple_element<idx, const T>::type, const U>::value);
    ret &= (s::is_same<typename s::tuple_element<idx, volatile T>::type, volatile U>::value);
    ret &= (s::is_same<typename s::tuple_element<idx, const volatile T>::type, const volatile U>::value);

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
        cgh.single_task<class KernelTupleIncludeUtilityTest>([=]() {
            // Compile time check

            test<s::pair<int, int>, 2, int, 0>();
            test<s::pair<int, int>, 2, int, 1>();
            test<s::pair<const int, int>, 2, int, 1>();
            test<s::pair<int, volatile int>, 2, volatile int, 1>();
            test<s::pair<char*, int>, 2, char*, 0>();
            test<s::pair<char*, int>, 2, int, 1>();

            ret_access[0] = test_runtime<s::pair<int, int>, 2, int, 0>();
            ret_access[0] &= test_runtime<s::pair<int, int>, 2, int, 1>();
            ret_access[0] &= test_runtime<s::pair<const int, int>, 2, int, 1>();
            ret_access[0] &= test_runtime<s::pair<int, volatile int>, 2, volatile int, 1>();
            ret_access[0] &= test_runtime<s::pair<char*, int>, 2, char*, 0>();
            ret_access[0] &= test_runtime<s::pair<char*, int>, 2, int, 1>();
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{
    kernel_test();
    return 0;
}
