#include "oneapi_std_test_config.h"
#include "checkData.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
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

// binary_search, with and without comparison predicate
cl::sycl::cl_bool
kernel_test()
{
    using s::binary_search;
    cl::sycl::queue deviceQueue;

    const int A[] = {1, 2, 3, 3, 3, 5, 8};
    const int C[] = {8, 5, 3, 3, 3, 2, 1};
    auto A1 = A, C1 = C;
    const int N = sizeof(A) / sizeof(int);
    const int first = A[0];
    const int last = A[N - 1];
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool check = false; // for checking data transfer
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};

    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<int, 1> buffer2(A, itemN);
        cl::sycl::buffer<int, 1> buffer3(C, itemN);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer4(&check, item1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto access2 = buffer2.get_access<sycl_write>(cgh);
            auto access3 = buffer3.get_access<sycl_write>(cgh);
            auto check_access = buffer4.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const int A1[] = {1, 2, 3, 3, 3, 5, 8};
                const int C1[] = {8, 5, 3, 3, 3, 2, 1};
                // check if there is change after data transfer
                check_access[0] = checkData(&access2[0], A1, N);
                check_access[0] &= checkData(&access3[0], C1, N);

                if (check_access[0])
                {
                    ret_access[0] = (binary_search(&access2[0], &access2[0] + N, 5));
                    ret_access[0] &= (binary_search(&access2[0], &access2[0] + N, first));
                    ret_access[0] &= (binary_search(&access2[0], &access2[0] + N, last));
                    ret_access[0] &= (!binary_search(&access2[0], &access2[0] + N, 4));

                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, 5, gt()));
                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, first, gt()));
                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, last, gt()));
                    ret_access[0] &= (!binary_search(&access3[0], &access3[0] + N, 4, gt()));
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
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
