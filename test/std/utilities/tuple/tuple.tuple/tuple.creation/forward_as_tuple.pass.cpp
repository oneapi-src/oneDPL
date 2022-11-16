#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(functional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <functional>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
// This code is used to test std::forward_tuple_as
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class Tuple>
void __attribute__((always_inline)) test0(const Tuple&)
{
    static_assert(s::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
bool
test1a(const Tuple& t) __attribute__((always_inline));
template <class Tuple>
bool
test1a(const Tuple& t)
{
    static_assert(s::tuple_size<Tuple>::value == 1, "");
    static_assert(s::is_same<typename s::tuple_element<0, Tuple>::type, int&&>::value, "");
    return (s::get<0>(t) == 1);
}

template <class Tuple>
bool __attribute__((always_inline)) test1b(const Tuple& t)
{
    static_assert(s::tuple_size<Tuple>::value == 1, "");
    static_assert(s::is_same<typename s::tuple_element<0, Tuple>::type, int&>::value, "");
    return (s::get<0>(t) == 2);
}

template <class Tuple>
bool __attribute__((always_inline)) test2a(const Tuple& t)
{
    static_assert(s::tuple_size<Tuple>::value == 2, "");
    static_assert(s::is_same<typename s::tuple_element<0, Tuple>::type, float&>::value, "");
    static_assert(s::is_same<typename s::tuple_element<1, Tuple>::type, char&>::value, "");
    return (s::get<0>(t) == 2.5f && s::get<1>(t) == 'a');
}

template <class Tuple>
constexpr int
test3(const Tuple&)
{
    return s::tuple_size<Tuple>::value;
}
class KernelForwardAsTupleTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelForwardAsTupleTest>([=]() {
            int i = 2;
            test0(s::forward_as_tuple());
            ret_access[0] = test1a(s::forward_as_tuple(1));
            ret_access[0] &= test1b(s::forward_as_tuple(i));

            float j = 2.5f;
            char c = 'a';
            ret_access[0] &= test2a(s::forward_as_tuple(j, c));

            ret_access[0] &= (test3(s::forward_as_tuple(j, c)) == 2);
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
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
