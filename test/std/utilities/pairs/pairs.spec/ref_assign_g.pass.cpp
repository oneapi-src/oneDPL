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

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

sycl::cl_bool
kernel_test1()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                typedef s::pair<int&, int> pair_type;
                int i = 1;
                int j = 2;
                pair_type p(i, 3);
                const pair_type q(j, 4);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == j);
            });
        });
    }
    return ret;
}

sycl::cl_bool
kernel_test2()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                typedef s::pair<int, int&> pair_type;
                int i = 1;
                int j = 2;
                pair_type p(3, i);
                const pair_type q(4, j);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == j);
            });
        });
    }
    return ret;
}
sycl::cl_bool
kernel_test3()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest3>([=]() {
                typedef s::pair<int&, int&> pair_type;
                int i = 1;
                int j = 2;
                int k = 3;
                int l = 4;
                pair_type p(i, j);
                const pair_type q(k, l);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == k);
                ret_acc[0] &= (j == l);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test1() && kernel_test2() && kernel_test3();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
    return 0;
}
