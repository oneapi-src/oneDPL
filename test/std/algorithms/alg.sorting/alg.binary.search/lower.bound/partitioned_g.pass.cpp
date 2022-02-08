#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
#    include <functional>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct X
{
    int val;

    bool
    odd() const
    {
        return val % 2;
    }

    // Partitioned so that all odd values come before even values:
    bool
    operator<(const X& x) const
    {
        return this->odd() && !x.odd();
    }
    bool
    operator!=(const X& x) const
    {
        return this->val != x.val;
    }
};

bool
kernel_test1(cl::sycl::queue& deviceQueue)
{
    // Test with range that is partitioned, but not sorted.
    X seq[] = {1, 3, 5, 7, 1, 6, 4, 2};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};

    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<bool, 1> buffer2(&check, item1);
        cl::sycl::buffer<X, 1> buffer3(seq, itemN);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                X arr[] = {1, 3, 5, 7, 1, 6, 4, 2};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], arr, N);
                if (check_access[0])
                {
                    test_container<X, forward_iterator_wrapper> c(&access[0], &access[0] + N);
                    auto part1 = s::lower_bound(c.begin(), c.end(), X{2});
                    ret_access[0] = (part1 != c.end());
                    ret_access[0] &= (part1->val == 6);
                    auto part2 = s::lower_bound(c.begin(), c.end(), X{2}, s::less<X>{});
                    ret_access[0] &= (part2 != c.end());
                    ret_access[0] &= (part2->val == 6);

                    auto part3 = s::lower_bound(c.begin(), c.end(), X{9});
                    ret_access[0] &= (part3 != c.end());
                    ret_access[0] &= (part3->val == 1);
                    auto part4 = s::lower_bound(c.begin(), c.end(), X{9}, s::less<X>{});
                    ret_access[0] &= (part4 != c.end());
                    ret_access[0] &= (part4->val == 1);
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(tmp, seq, N);
    if (!check)
        return false;
    return ret;
}

struct Y
{
    double val;

    // Not irreflexive, so not a strict weak order.
    bool
    operator<(const Y& y) const
    {
        return val < int(y.val);
    }
    bool
    operator!=(const Y& y) const
    {
        return val != (y.val);
    }
};

bool
kernel_test2(cl::sycl::queue& deviceQueue)
{
    Y seq[] = {-0.1, 1.2, 5.0, 5.2, 5.1, 5.9, 5.5, 6.0};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{8};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<bool, 1> buffer2(&check, item1);
        cl::sycl::buffer<Y, 1> buffer3(seq, itemN);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                Y arr[] = {-0.1, 1.2, 5.0, 5.2, 5.1, 5.9, 5.5, 6.0};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], arr, N);
                if (check_access[0])
                {
                    test_container<Y, forward_iterator_wrapper> c(&access[0], &access[0] + N);
                    auto part1 = std::lower_bound(c.begin(), c.end(), Y{5.5});
                    ret_access[0] = (part1 != c.end());
                    ret_access[0] &= (part1->val == 5.0);
                    auto part2 = std::lower_bound(c.begin(), c.end(), Y{5.5}, std::less<Y>{});
                    ret_access[0] &= (part2 != c.end());
                    ret_access[0] &= (part2->val == 5.0);

                    auto part3 = std::lower_bound(c.begin(), c.end(), Y{1.0});
                    ret_access[0] &= (part3 != c.end());
                    ret_access[0] &= (part3->val == 1.2);
                    auto part4 = std::lower_bound(c.begin(), c.end(), Y{1.0}, std::less<Y>{});
                    ret_access[0] &= (part4 != c.end());
                    ret_access[0] &= (part4->val == 1.2);
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(tmp, seq, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    cl::sycl::queue deviceQueue;
    auto ret = kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        ret &= kernel_test2(deviceQueue);
    }
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
}
