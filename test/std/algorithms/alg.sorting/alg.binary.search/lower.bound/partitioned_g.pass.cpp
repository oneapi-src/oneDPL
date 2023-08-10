#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include <oneapi/dpl/algorithm>
#    include <oneapi/dpl/functional>
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
#    include <functional>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

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
kernel_test1(sycl::queue& deviceQueue)
{
    // Test with range that is partitioned, but not sorted.
    X seq[] = {1, 3, 5, 7, 1, 6, 4, 2};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};

    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<X, 1> buffer3(seq, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
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
kernel_test2(sycl::queue& deviceQueue)
{
    Y seq[] = {-0.1, 1.2, 5.0, 5.2, 5.1, 5.9, 5.5, 6.0};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{8};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<Y, 1> buffer3(seq, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
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
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    auto ret = kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        ret &= kernel_test2(deviceQueue);
    }
    TestUtils::exit_on_error(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
