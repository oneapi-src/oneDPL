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

using s::binary_search;

typedef test_container<int, forward_iterator_wrapper> Container;

cl::sycl::cl_bool
kernel_test1()
{
    cl::sycl::queue deviceQueue;
    int array[] = {0};
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool transferCheck = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&transferCheck, numOfItems);
        cl::sycl::buffer<int, 1> buffer3(array, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto acc_arr = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                // check if there is change after data transfer
                check_access[0] = (acc_arr[0] == 0);
                if (check_access[0])
                {
                    Container con(&acc_arr[0], &acc_arr[0]);
                    ret_access[0] = (!binary_search(con.begin(), con.end(), 1));
                }
            });
        });
    }
    // check if there is change after executing kernel function
    transferCheck &= (array[0] == 0);
    if (!transferCheck)
        return false;
    return ret;
}

cl::sycl::cl_bool
kernel_test2()
{
    cl::sycl::queue deviceQueue;
    int array[] = {0, 2, 4, 6, 8};
    int tmp[] = {0, 2, 4, 6, 8};
    const int N = sizeof(array) / sizeof(array[0]);
    cl::sycl::cl_bool ret = false;
    cl::sycl::cl_bool check = false;
    cl::sycl::range<1> item1{1};
    cl::sycl::range<1> itemN{N};
    {
        cl::sycl::buffer<int, 1> buffer1(array, itemN);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer2(&ret, item1);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer3(&check, item1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto access1 = buffer1.get_access<sycl_write>(cgh);
            auto ret_access = buffer2.get_access<sycl_write>(cgh);
            auto check_access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                int tmp[] = {0, 2, 4, 6, 8};
                // check if there is change after data transfer
                check_access[0] = checkData(&access1[0], tmp, N);
                if (check_access[0])
                {
                    Container con(&access1[0], &access1[0] + N);
                    ret_access[0] = (binary_search(con.begin(), con.end(), 0));

                    for (int i = 2; i < 10; i += 2)
                    {
                        ret_access[0] &= (binary_search(con.begin(), con.end(), i));
                    }

                    for (int i = -1; i < 11; i += 2)
                    {
                        ret_access[0] &= (!binary_search(con.begin(), con.end(), i));
                    }
                }
            });
        });
    }
    // check if there is change after executing kernel function
    check &= checkData(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test1();
    ret &= kernel_test2();
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
