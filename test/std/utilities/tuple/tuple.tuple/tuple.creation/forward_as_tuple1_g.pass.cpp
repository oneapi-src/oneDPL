// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test1(cl::sycl::queue& deviceQueue)
{
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                s::forward_as_tuple();

                ret_access[0] &= (s::get<0>(s::forward_as_tuple(-1)) == -1);
                ret_access[0] &= ((s::is_same<decltype(s::forward_as_tuple(-1)), s::tuple<int&&>>::value));

                const int i1 = 1;
                const int i2 = 2;
                const float d1 = 4.0f;
                auto t1 = s::forward_as_tuple(i1, i2, d1);
                ret_access[0] &= ((s::is_same<decltype(t1), s::tuple<const int&, const int&, const float&>>::value));
                ret_access[0] &= (s::get<0>(t1) == i1);
                ret_access[0] &= (s::get<1>(t1) == i2);
                ret_access[0] &= (s::get<2>(t1) == d1);

                typedef const int a_type1[3];
                a_type1 a1 = {-1, 1, 2};
                auto t2 = s::forward_as_tuple(a1);
                ret_access[0] &= ((s::is_same<decltype(t2), s::tuple<a_type1&>>::value));
                ret_access[0] &= (s::get<0>(t2)[0] == a1[0]);
                ret_access[0] &= (s::get<0>(t2)[1] == a1[1]);
                ret_access[0] &= (s::get<0>(t2)[2] == a1[2]);

                typedef int a_type2[2];
                a_type2 a2 = {2, -2};
                volatile int i4 = 1;
                auto t3 = s::forward_as_tuple(a2, i4);
                ret_access[0] &= ((s::is_same<decltype(t3), s::tuple<a_type2&, volatile int&>>::value));
                ret_access[0] &= (s::get<0>(t3)[0] == a2[0]);
                ret_access[0] &= (s::get<0>(t3)[1] == a2[1]);
                ret_access[0] &= (s::get<1>(t3) == i4);
            });
        });
    }
    return ret;
}

cl::sycl::cl_bool
kernel_test2(cl::sycl::queue& deviceQueue)
{
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                const int i1 = 1;
                const int i2 = 2;
                const double d1 = 4.0;
                auto t1 = s::forward_as_tuple(i1, i2, d1);
                ret_access[0] &= ((s::is_same<decltype(t1), s::tuple<const int&, const int&, const double&>>::value));
                ret_access[0] &= (s::get<0>(t1) == i1);
                ret_access[0] &= (s::get<1>(t1) == i2);
                ret_access[0] &= (s::get<2>(t1) == d1);
            });
        });
    }
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
    return 0;
}
