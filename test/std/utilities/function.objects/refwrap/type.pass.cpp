#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class C
{
};

class KernelTypePassTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTypePassTest>([=]() {
            // Static assert check...

            static_assert((s::is_same<s::reference_wrapper<C>::type, C>::value), "");
            static_assert((s::is_same<s::reference_wrapper<void()>::type, void()>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int*(float*)>::type, int*(float*)>::value), "");
            static_assert((s::is_same<s::reference_wrapper<void (*)()>::type, void (*)()>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int* (*)(float*)>::type, int* (*)(float*)>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int* (C::*)(float*)>::type, int* (C::*)(float*)>::value),
                          "");
            static_assert((s::is_same<s::reference_wrapper<int (C::*)(float*) const volatile>::type,
                                      int (C::*)(float*) const volatile>::value),
                          "");
            // Runtime check...

            ret_access[0] = s::is_same<s::reference_wrapper<C>::type, C>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<void()>::type, void()>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int*(float*)>::type, int*(float*)>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<void (*)()>::type, void (*)()>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int* (*)(float*)>::type, int* (*)(float*)>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int* (C::*)(float*)>::type, int* (C::*)(float*)>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int (C::*)(float*) const volatile>::type,
                                        int (C::*)(float*) const volatile>::value;
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
    kernel_test();
    return 0;
}
