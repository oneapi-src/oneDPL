// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename T1, typename T2>
bool
test()
{

    bool result = false;
    {
        typedef T1 Exp1;
        typedef T2 Exp2;
        typedef s::pair<T1, T2> P;
        static_assert((s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value), "");

        result = (s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value);
        result &= (s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value);
    }
    {
        typedef T1 const Exp1;
        typedef T2 const Exp2;
        typedef s::pair<T1, T2> const P;
        static_assert((s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value), "");

        result &= (s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value);
        result &= (s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value);
    }
    {
        typedef T1 volatile Exp1;
        typedef T2 volatile Exp2;
        typedef s::pair<T1, T2> volatile P;
        static_assert((s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value), "");
        result &= (s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value);
        result &= (s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value);
    }
    {
        typedef T1 const volatile Exp1;
        typedef T2 const volatile Exp2;
        typedef s::pair<T1, T2> const volatile P;
        static_assert((s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value), "");

        result &= (s::is_same<typename s::tuple_element<0, P>::type, Exp1>::value);
        result &= (s::is_same<typename s::tuple_element<1, P>::type, Exp2>::value);
    }

    return result;
}

class KernelPairTest1;
class KernelPairTest2;

template <typename T1, typename T2, typename KC>
void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<KC>([=]() { ret_access[0] = test<T1, T2>(); });
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

    kernel_test<int, short, KernelPairTest1>();
    kernel_test<int*, char, KernelPairTest2>();
    return 0;
}
