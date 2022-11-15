#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct X
{
    int val;
    X(int v) : val(v){};
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

cl::sycl::cl_bool
kernel_test()
{
    // Test with range that is partitioned, but not sorted.
    cl::sycl::queue deviceQueue;
    X seq[] = {1, 3, 5, 7, 1, 6, 4};
    auto tmp = seq;
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool check = false;
    const int N = sizeof(seq) / sizeof(seq[0]);
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, itemN);
        cl::sycl::buffer<X, 1> buffer3(seq, itemN);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                X tmp[] = {1, 3, 5, 7, 1, 6, 4};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], &tmp[0], N);

                if (check_access[0])
                {
                    test_container<X, forward_iterator_wrapper> c(&access[0], &access[0] + N);

                    ret_access[0] = s::binary_search(c.begin(), c.end(), X{2});
                    ret_access[0] &= s::binary_search(c.begin(), c.end(), X{2}, s::less<X>{});

                    ret_access[0] &= s::binary_search(c.begin(), c.end(), X{9});
                    ret_access[0] &= s::binary_search(c.begin(), c.end(), X{9}, s::less<X>{});

                    ret_access[0] &= !(s::binary_search(&access[0], &access[0] + 5, X{2}));
                    ret_access[0] &= !(s::binary_search(&access[0], &access[0] + 5, X{2}, s::less<X>{}));
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(seq, tmp, N);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
