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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

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

sycl::cl_bool
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
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
