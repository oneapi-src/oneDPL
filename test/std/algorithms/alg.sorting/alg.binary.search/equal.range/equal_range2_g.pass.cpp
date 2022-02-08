#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

// A comparison, equalivalent to std::greater<int> without the
// dependency on <functional>.
struct gt
{
    bool
    operator()(const int& x, const int& y) const
    {
        return x > y;
    }
};

// Each test performs general-case, bookend, not-found condition,
// and predicate functional checks.

// equal_range, with and without comparison predicate
cl::sycl::cl_bool
kernel_test()
{
    using s::equal_range;
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool check = false;

    typedef std::pair<const int*, const int*> Ipair;
    const int A[] = {1, 2, 3, 3, 3, 5, 8};
    const int C[] = {8, 5, 3, 3, 3, 2, 1};
    auto A1 = A, C1 = C;
    const int N = sizeof(A) / sizeof(int);
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};

    const int first = A[0];
    const int last = A[N - 1];

    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, item1);
        cl::sycl::buffer<int, 1> buffer3(A, itemN);
        cl::sycl::buffer<int, 1> buffer4(C, itemN);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access1 = buffer3.get_access<sycl_write>(cgh);
            auto access2 = buffer4.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                const int A1[] = {1, 2, 3, 3, 3, 5, 8};
                const int C1[] = {8, 5, 3, 3, 3, 2, 1};
                // check if there is change after data transfer
                check_access[0] = checkData(&access1[0], A1, N);
                check_access[0] &= checkData(&access2[0], C1, N);
                if (check_access[0])
                {
                    Ipair p = equal_range(&access1[0], &access1[0] + N, 3);
                    ret_access[0] = (p.first == &access1[0] + 2);
                    ret_access[0] &= (p.second == &access1[0] + 5);

                    Ipair q = equal_range(&access1[0], &access1[0] + N, first);
                    ret_access[0] &= (q.first == &access1[0] + 0);
                    ret_access[0] &= (q.second == &access1[0] + 1);

                    Ipair r = equal_range(&access1[0], &access1[0] + N, last);
                    ret_access[0] &= (r.first == &access1[0] + N - 1);
                    ret_access[0] &= (r.second == &access1[0] + N);

                    Ipair s = equal_range(&access1[0], &access1[0] + N, 4);
                    ret_access[0] &= (s.first == &access1[0] + 5);
                    ret_access[0] &= (s.second == &access1[0] + 5);

                    Ipair t = equal_range(&access2[0], &access2[0] + N, 3, gt());
                    ret_access[0] &= (t.first == &access2[0] + 2);
                    ret_access[0] &= (t.second == &access2[0] + 5);

                    Ipair u = equal_range(&access2[0], &access2[0] + N, first, gt());
                    ret_access[0] &= (u.first == &access2[0] + N - 1);
                    ret_access[0] &= (u.second == &access2[0] + N);

                    Ipair v = equal_range(&access2[0], &access2[0] + N, last, gt());
                    ret_access[0] &= (v.first == &access2[0] + 0);
                    ret_access[0] &= (v.second == &access2[0] + 1);

                    Ipair w = equal_range(&access2[0], &access2[0] + N, 4, gt());
                    ret_access[0] &= (w.first == &access2[0] + 2);
                    ret_access[0] &= (w.second == &access2[0] + 2);
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(A, A1, N);
    check &= checkData(C, C1, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test();
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
