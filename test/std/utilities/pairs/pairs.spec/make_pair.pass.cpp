// <utility>

// template <class T1, class T2> pair<V1, V2> make_pair(T1&&, T2&&);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

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
            typedef s::pair<int, short> P1;
            P1 p1 = s::make_pair(3, static_cast<short>(4));
            ret_access[0] = (p1.first == 3);
            ret_access[0] &= (p1.second == 4);

            int elem1 = 3;
            typedef s::pair<int*, short> P2;
            P2 p2 = s::make_pair(&elem1, static_cast<short>(4));
            ret_access[0] &= (*p2.first == 3);
            ret_access[0] &= (p2.second == 4);

            P2 p2_null = s::make_pair(nullptr, static_cast<short>(4));
            ret_access[0] &= (p2_null.first == nullptr);
            ret_access[0] &= (p2_null.second == 4);
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
