// <utility>

// template <class T1, class T2> struct pair

// tuple_size<pair<T1, T2> >::value

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelPairTest;
void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef std::pair<int, short> P1;
                static_assert((std::tuple_size<P1>::value == 2), "");
                ret_access[0] = (std::tuple_size<P1>::value == 2);
            }

            {
                typedef std::pair<int, short> const P1;
                static_assert((std::tuple_size<P1>::value == 2), "");
                ret_access[0] &= (std::tuple_size<P1>::value == 2);
            }

            {
                typedef std::pair<int, short> volatile P1;
                static_assert((std::tuple_size<P1>::value == 2), "");
                ret_access[0] &= (std::tuple_size<P1>::value == 2);
            }

            {
                typedef std::pair<int, short> const volatile P1;
                ret_access[0] &= (std::tuple_size<P1>::value == 2);
            }
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
