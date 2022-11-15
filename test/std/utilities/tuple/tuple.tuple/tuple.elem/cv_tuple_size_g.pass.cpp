// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                ret_access[0] &= (s::tuple_size<const s::tuple<>>::value == 0);
                ret_access[0] &= (s::tuple_size<volatile s::tuple<int>>::value == 1);
                ret_access[0] &= (s::tuple_size<const volatile s::tuple<void>>::value == 1);

                typedef s::tuple<int, const int&, void> test_tuple1;
                ret_access[0] &= (s::tuple_size<const test_tuple1>::value == 3);
                ret_access[0] &= (s::tuple_size<const volatile test_tuple1>::value == 3);
                ret_access[0] &= (s::tuple_size<volatile s::tuple<s::tuple<void>>>::value == 1);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    if (ret)
        std::cout << "pass" << std::endl;
    else
        std::cout << "fail" << std::endl;
    return 0;
}
