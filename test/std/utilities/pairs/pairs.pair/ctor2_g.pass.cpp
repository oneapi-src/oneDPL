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

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test1()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> numOfItem{1};
    s::pair<s::pair<int, int>, int> p;
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<sycl::cl_bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<decltype(p), 1> buffer3(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto check_acc = buffer2.get_access<sycl_write>(cgh);
            auto acc1 = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == s::pair<int, int>());
                check_acc[0] &= (acc1[0].second == 0);
                if (check_acc[0])
                {
                    static_assert(sizeof(acc1[0]) == (3 * sizeof(int)), "assertion fail");
                    ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
                    ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first);
                }
            });
        });
    }
    // check data after executing kernel function
    check &= (p.first == s::pair<int, int>());
    check &= (p.second == 0);
    if (!check)
        return false;
    return ret;
}
struct empty
{
};

cl::sycl::cl_bool
kernel_test2()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    s::pair<s::pair<empty, empty>, empty> p;

    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<decltype(p), 1> buffer2(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc1 = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                static_assert(sizeof(acc1[0]) == (3 * sizeof(empty)), "assertion fail");
                ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
            });
        });
    }
    return ret;
}

cl::sycl::cl_bool
kernel_test3()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    typedef s::pair<int, int> int_pair;
    typedef s::pair<int_pair, int_pair> int_pair_pair;
    s::pair<int_pair_pair, int_pair_pair> p;

    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<decltype(p), 1> buffer2(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto acc1 = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest3>([=]() {
                static_assert(sizeof(int_pair_pair) == (2 * sizeof(int_pair)), "nested");
                static_assert(sizeof(acc1[0]) == (2 * sizeof(int_pair_pair)), "nested again");
                ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
                ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first);
                ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first.first);
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
