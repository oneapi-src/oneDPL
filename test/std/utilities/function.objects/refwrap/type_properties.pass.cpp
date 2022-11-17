// <functional>

// reference_wrapper

// Test that reference wrapper meets the requirements of CopyConstructible and
// CopyAssignable, and TriviallyCopyable (starting in C++14).

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

class MoveOnly
{
    MoveOnly(const MoveOnly&);
    MoveOnly&
    operator=(const MoveOnly&);

    int data_;

  public:
    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x) : data_(x.data_) { x.data_ = 0; }
    MoveOnly&
    operator=(MoveOnly&& x)
    {
        data_ = x.data_;
        x.data_ = 0;
        return *this;
    }

    int
    get() const
    {
        return data_;
    }
};

template <class T>
class KernelTypePropertiesPassTest;

template <class T>
void
kernel_test(cl::sycl::queue& deviceQueue)
{
    typedef s::reference_wrapper<T> Wrap;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTypePropertiesPassTest<T>>([=]()
        {
            // Static assert check...
            static_assert(s::is_copy_constructible<Wrap>::value, "");
            static_assert(s::is_copy_assignable<Wrap>::value, "");
            // Runtime check...
            ret_access[0] = s::is_copy_constructible<Wrap>::value;
            ret_access[0] &= s::is_copy_assignable<Wrap>::value;
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
    kernel_test<int>(deviceQueue);
    kernel_test<MoveOnly>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test<double>(deviceQueue);
    }
    return 0;
}
