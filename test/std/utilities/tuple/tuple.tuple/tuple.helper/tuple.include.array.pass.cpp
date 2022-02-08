// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };
//
//  LWG #2212 says that tuple_size and tuple_element must be
//     available after including <utility>

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <array>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelTupleArrayTest;

template <class T, s::size_t N, class U, size_t idx>
void __attribute__((always_inline)) test_compile()
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
        cgh.single_task<class KernelTupleArrayTest>([=]() {
            // Compile time check
            test_compile<s::array<int, 5>, 5, int, 0>();
            test_compile<s::array<int, 5>, 5, int, 1>();
            test_compile<s::array<const char*, 4>, 4, const char*, 3>();
            test_compile<s::array<volatile int, 4>, 4, volatile int, 3>();
            test_compile<s::array<char*, 3>, 3, char*, 1>();
            test_compile<s::array<char*, 3>, 3, char*, 2>();

            //Runtime check

            ret_access[0] = test_runtime<s::array<int, 5>, 5, int, 0>();
            ret_access[0] &= test_runtime<s::array<int, 5>, 5, int, 1>();
            ret_access[0] &= test_runtime<s::array<const char*, 4>, 4, const char*, 3>();
            ret_access[0] &= test_runtime<s::array<volatile int, 4>, 4, volatile int, 3>();
            ret_access[0] &= test_runtime<s::array<char*, 3>, 3, char*, 1>();
            ret_access[0] &= test_runtime<s::array<char*, 3>, 3, char*, 2>();
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
