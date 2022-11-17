#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct Poison
{
    Poison(Poison&&) = delete;
};

struct ThrowingCopy
{
    ThrowingCopy(const ThrowingCopy&);
    ThrowingCopy&
    operator=(const ThrowingCopy&);
};

cl::sycl::cl_bool
kernel_test()
{
    static_assert(!s::is_copy_constructible<Poison>::value, "");
    static_assert(!s::is_move_constructible<Poison>::value, "");
    static_assert(!s::is_copy_assignable<Poison>::value, "");
    static_assert(!s::is_move_assignable<Poison>::value, "");
    static_assert(!s::is_copy_constructible<s::pair<int, Poison>>::value, "");
    static_assert(!s::is_move_constructible<s::pair<int, Poison>>::value, "");
    static_assert(!s::is_copy_assignable<s::pair<int, Poison>>::value, "");
    static_assert(!s::is_move_assignable<s::pair<int, Poison>>::value, "");
    static_assert(!s::is_constructible<s::pair<int, Poison>&, s::pair<char, Poison>&>::value, "");
    static_assert(!s::is_assignable<s::pair<int, Poison>&, s::pair<char, Poison>&>::value, "");
    static_assert(!s::is_constructible<s::pair<int, Poison>&, s::pair<char, Poison>>::value, "");
    static_assert(!s::is_assignable<s::pair<int, Poison>&, s::pair<char, Poison>>::value, "");
    static_assert(!s::is_copy_constructible<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    static_assert(s::is_move_constructible<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    static_assert(!s::is_nothrow_move_constructible<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    static_assert(!s::is_copy_assignable<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    static_assert(s::is_move_assignable<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    static_assert(!s::is_nothrow_move_assignable<s::pair<ThrowingCopy, std::unique_ptr<int>>>::value, "");
    return true;
}

int
main()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
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
