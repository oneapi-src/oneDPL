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

cl::sycl::cl_bool
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> numOfItem{1};
    s::pair<int, int> a(1, 1), b(2, 2);
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<sycl::cl_bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<decltype(a), 1> buffer3(&a, numOfItem);
        sycl::buffer<decltype(b), 1> buffer4(&b, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto check_acc = buffer2.get_access<sycl_write>(cgh);
            auto acc1 = buffer3.get_access<sycl_write>(cgh);
            auto acc2 = buffer4.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == 1);
                check_acc[0] &= (acc1[0].second == 1);
                check_acc[0] &= (acc2[0].first == 2);
                check_acc[0] &= (acc2[0].second == 2);
                if (check_acc[0])
                {
                    acc1[0] = s::move(acc2[0]);
                    s::pair<int, int> c(s::move(acc1[0]));
                    ret_acc[0] = (c.first == 2 && c.second == 2);
                    ret_acc[0] &= (acc1[0].first == 2 && acc1[0].second == 2);
                }
            });
        });
    }
    // check data after executing kernel functio
    check &= (a.first == 2 && a.second == 2);
    check &= (b.first == 2 && b.second == 2);
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
