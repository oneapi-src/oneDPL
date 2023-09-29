#include "oneapi_std_test_config.h"

#include _ONEAPI_STD_TEST_HEADER(algorithm)

#include <iostream>

#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"

namespace test_ns = _ONEAPI_TEST_NAMESPACE;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using test_ns::binary_search;

typedef test_container<int, forward_iterator_wrapper> Container;

bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0};
    bool ret = false;
    bool transferCheck = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        sycl::buffer<bool, 1> buffer2(&transferCheck, numOfItems);
        sycl::buffer<int, 1> buffer3(array, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto acc_arr = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                // check if there is change after data transfer
                check_access[0] = (acc_arr[0] == 0);
                if (check_access[0])
                {
                    Container con(acc_arr.get_pointer().get(), acc_arr.get_pointer().get());
                    ret_access[0] = (!binary_search(con.begin(), con.end(), 1));
                }
            });
        }).wait();
    }
    // check if there is change after executing kernel function
    transferCheck &= (array[0] == 0);
    if (!transferCheck)
        return false;
    return ret;
}

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 2, 4, 6, 8};
    int tmp[] = {0, 2, 4, 6, 8};
    const int N = sizeof(array) / sizeof(array[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(array, itemN);
        sycl::buffer<bool, 1> buffer2(&ret, item1);
        sycl::buffer<bool, 1> buffer3(&check, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto access1 = buffer1.get_access<sycl_write>(cgh);
            auto ret_access = buffer2.get_access<sycl_write>(cgh);
            auto check_access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                int tmp[] = {0, 2, 4, 6, 8};
                // check if there is change after data transfer
                check_access[0] = check_data(access1.get_pointer().get(), tmp, N);
                if (check_access[0])
                {
                    Container con(access1.get_pointer().get(), access1.get_pointer().get() + N);
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
        }).wait();
    }
    // check if there is change after executing kernel function
    check &= check_data(tmp, array, N);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test1();
    ret &= kernel_test2();
    EXPECT_TRUE(ret, "");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
