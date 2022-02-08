// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T, class = decltype(s::tuple_size<T>::value)>
constexpr bool
has_value(int)
{
    return true;
}
template <class>
constexpr bool
has_value(long)
{
    return false;
}
template <class T>
constexpr bool
has_value()
{
    return has_value<T>(0);
}

struct Dummy
{
};

class KernelTupleSizeValueSfinaeTest;

template <class T, s::size_t N>
void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleSizeValueSfinaeTest>([=]() {
            ret_access[0] = has_value<s::tuple<int> const>();
            ret_access[0] &= has_value<s::pair<int, long> volatile>();
            ret_access[0] &= !has_value<int>();
            ret_access[0] &= !has_value<const int>();
            ret_access[0] &= !has_value<volatile void>();
            ret_access[0] &= !has_value<const volatile s::tuple<int>&>();
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
    kernel_test<std::tuple<>, 0>();
    return 0;
}
