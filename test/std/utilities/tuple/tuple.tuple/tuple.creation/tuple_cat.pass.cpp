#include "oneapi_std_test_config.h"
#include "MoveOnly.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
#    include <array>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelTupleCatTest1;
class KernelTupleCatTest2;

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleCatTest1>([=]() {
            {
                s::tuple<> t;
                t = s::tuple_cat();
            }

            {
                s::tuple<> t1;
                s::tuple<> t2 = s::tuple_cat(t1);
                ((void)t2); // Prevent unused warning
            }

            {
                s::tuple<> t = s::tuple_cat(s::tuple<>());
                ((void)t); // Prevent unused warning
            }

            {
                s::tuple<> t = s::tuple_cat(s::array<int, 0>());
                ((void)t); // Prevent unused warning
            }

            {
                s::tuple<int> t1(1);
                s::tuple<int> t = s::tuple_cat(t1);
                ret_access[0] &= (s::get<0>(t) == 1);
            }

            {
                constexpr s::tuple<> t = s::tuple_cat();
                ((void)t); // Prevent unused warning
            }

            {
                constexpr s::tuple<> t1;
                constexpr s::tuple<> t2 = s::tuple_cat(t1);
                ((void)t2); // Prevent unused warning
            }

            {
                constexpr s::tuple<> t = s::tuple_cat(s::tuple<>());
                ((void)t); // Prevent unused warning
            }

            {
                constexpr s::tuple<> t = s::tuple_cat(s::array<int, 0>());
                ((void)t); // Prevent unused warning
            }

            {
                constexpr s::tuple<int> t1(1);
                constexpr s::tuple<int> t = s::tuple_cat(t1);
                static_assert(s::get<0>(t) == 1, "");
            }

            {
                constexpr s::tuple<int> t1(1);
                constexpr s::tuple<int, int> t = s::tuple_cat(t1, t1);
                static_assert(s::get<0>(t) == 1, "");
                static_assert(s::get<1>(t) == 1, "");
            }

            {
                s::tuple<int, MoveOnly> t = s::tuple_cat(s::tuple<int, MoveOnly>(1, 2));
                ret_access[0] &= (s::get<0>(t) == 1 && s::get<1>(t) == 2);
            }

            {
                s::tuple<int, int, int> t = s::tuple_cat(s::array<int, 3>());
                ret_access[0] &= (s::get<0>(t) == 0 && s::get<1>(t) == 0 && s::get<2>(t) == 0);
            }

            {
                s::tuple<int, MoveOnly> t = s::tuple_cat(s::pair<int, MoveOnly>(2, 1));
                ret_access[0] &= (s::get<0>(t) == 2 && s::get<1>(t) == 1);
            }

            {
                s::tuple<> t1;
                s::tuple<> t2;
                s::tuple<> t3 = s::tuple_cat(t1, t2);
                ((void)t3); // Prevent unused warning
            }

            {
                s::tuple<> t1;
                s::tuple<int> t2(2);
                s::tuple<int> t3 = s::tuple_cat(t1, t2);
                ret_access[0] &= (s::get<0>(t3) == 2);
            }

            {
                s::tuple<> t1;
                s::tuple<int> t2(2);
                s::tuple<int> t3 = s::tuple_cat(t2, t1);
                ret_access[0] &= (s::get<0>(t3) == 2);
            }

            {
                s::tuple<int*> t1;
                s::tuple<int> t2(2);
                s::tuple<int*, int> t3 = s::tuple_cat(t1, t2);
                ret_access[0] &= (s::get<0>(t3) == nullptr && s::get<1>(t3) == 2);
            }

            {
                s::tuple<int*> t1;
                s::tuple<int> t2(2);
                s::tuple<int, int*> t3 = s::tuple_cat(t2, t1);
                ret_access[0] &= (s::get<0>(t3) == 2 && s::get<1>(t3) == nullptr);
            }

            {
                s::tuple<MoveOnly, MoveOnly> t1(1, 2);
                s::tuple<int*, MoveOnly> t2(nullptr, 4);
                s::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 = s::tuple_cat(s::move(t1), s::move(t2));
                ret_access[0] &=
                    (s::get<0>(t3) == 1 && s::get<1>(t3) == 2 && s::get<2>(t3) == nullptr && s::get<3>(t3) == 4);
            }

            {
                s::tuple<MoveOnly, MoveOnly> t1(1, 2);
                s::tuple<int*, MoveOnly> t2(nullptr, 4);
                s::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 = s::tuple_cat(s::tuple<>(), s::move(t1), s::move(t2));
                ret_access[0] &=
                    (s::get<0>(t3) == 1 && s::get<1>(t3) == 2 && s::get<2>(t3) == nullptr && s::get<3>(t3) == 4);
            }

            {
                s::tuple<MoveOnly, MoveOnly> t1(1, 2);
                s::tuple<int*, MoveOnly> t2(nullptr, 4);
                s::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 = s::tuple_cat(s::move(t1), s::tuple<>(), s::move(t2));
                ret_access[0] &=
                    (s::get<0>(t3) == 1 && s::get<1>(t3) == 2 && s::get<2>(t3) == nullptr && s::get<3>(t3) == 4);
            }

            {
                s::tuple<MoveOnly, MoveOnly> t1(1, 2);
                s::tuple<int*, MoveOnly> t2(nullptr, 4);
                s::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 = s::tuple_cat(s::move(t1), s::move(t2), s::tuple<>());
                ret_access[0] &=
                    (s::get<0>(t3) == 1 && s::get<1>(t3) == 2 && s::get<2>(t3) == nullptr && s::get<3>(t3) == 4);
            }

            {
                s::tuple<MoveOnly, MoveOnly> t1(1, 2);
                s::tuple<int*, MoveOnly> t2(nullptr, 4);
                s::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                    s::tuple_cat(s::move(t1), s::move(t2), s::tuple<int>(5));
                ret_access[0] &= (s::get<0>(t3) == 1 && s::get<1>(t3) == 2 && s::get<2>(t3) == nullptr &&
                                  s::get<3>(t3) == 4 && s::get<4>(t3) == 5);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTupleCatTest>([=]() {
            {
                s::tuple<int*> t1;
                s::tuple<int, double> t2(2, 3.5);
                s::tuple<int*, int, double> t3 = s::tuple_cat(t1, t2);
                ret_access[0] &= (s::get<0>(t3) == nullptr && s::get<1>(t3) == 2 && s::get<2>(t3) == 3.5);
            }

            {
                s::tuple<int*> t1;
                s::tuple<int, double> t2(2, 3.5);
                s::tuple<int, double, int*> t3 = s::tuple_cat(t2, t1);
                ret_access[0] &= (s::get<0>(t3) == 2 && s::get<1>(t3) == 3.5 && s::get<2>(t3) == nullptr);
            }

            {
                s::tuple<int*, MoveOnly> t1(nullptr, 1);
                s::tuple<int, double> t2(2, 3.5);
                s::tuple<int*, MoveOnly, int, double> t3 = s::tuple_cat(s::move(t1), t2);
                ret_access[0] &=
                    (s::get<0>(t3) == nullptr && s::get<1>(t3) == 1 && s::get<2>(t3) == 2 && s::get<3>(t3) == 3.5);
            }

            {
                s::tuple<int*, MoveOnly> t1(nullptr, 1);
                s::tuple<int, double> t2(2, 3.5);
                s::tuple<int, double, int*, MoveOnly> t3 = s::tuple_cat(t2, s::move(t1));
                ret_access[0] &=
                    (s::get<0>(t3) == 2 && s::get<1>(t3) == 3.5 && s::get<2>(t3) == nullptr && s::get<3>(t3) == 1);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    return 0;
}
