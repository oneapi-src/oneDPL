#include "oneapi_std_test_config.h"
#include "testsuite_iterators.h"
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

using s::upper_bound;

typedef test_container<int, forward_iterator_wrapper> Container;

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1};
    auto tmp = array;
    const int N = sizeof(array) / sizeof(array[0]);
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool check = false;
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&check, item1);
        cl::sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int arr[] = {0, 0, 0, 0, 1, 1, 1, 1};
                // check if there is change after data transfer
                check_access[0] = checkData(&access[0], arr, N);
                auto ret = true;
                if (check_access[0])
                {
                    auto ret = true;
                    for (int i = 0; i < 5; ++i)
                    {
                        for (int j = 4; j < 7; ++j)
                        {
                            Container con(&access[0] + i, &access[0] + j);
                            ret &= (upper_bound(con.begin(), con.end(), 0).ptr == &access[0] + 4);
                        }
                    }
                    ret_access[0] = ret;
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check = checkData(tmp, array, N);
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
