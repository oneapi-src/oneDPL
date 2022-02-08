// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct NonAssignable
{
    NonAssignable&
    operator=(NonAssignable const&) = delete;
    NonAssignable&
    operator=(NonAssignable&&) = delete;
};
struct CopyAssignable
{
    CopyAssignable() = default;
    CopyAssignable(CopyAssignable const&) = default;
    CopyAssignable&
    operator=(CopyAssignable const&) = default;
    CopyAssignable&
    operator=(CopyAssignable&&) = delete;
};
struct MoveAssignable
{
    MoveAssignable() = default;
    MoveAssignable&
    operator=(MoveAssignable const&) = delete;
    MoveAssignable&
    operator=(MoveAssignable&&) = default;
};

class KernelPairTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef s::pair<CopyAssignable, short> P;
                const P p1(CopyAssignable(), 4);
                P p2;
                p2 = p1;
                ret_access[0] = (p2.second == 4);
            }

            {
                using P = s::pair<int&, int&&>;
                int x = 42;
                int y = 101;
                int x2 = -1;
                int y2 = 300;
                P p1(x, s::move(y));
                P p2(x2, s::move(y2));
                p1 = p2;
                ret_access[0] &= (p1.first == x2);
                ret_access[0] &= (p1.second == y2);
            }

            {
                using P = s::pair<int, NonAssignable>;
                static_assert(!s::is_copy_assignable<P>::value, "");
            }

            {
                using P = s::pair<int, MoveAssignable>;
                static_assert(!s::is_copy_assignable<P>::value, "");
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
    kernel_test();
    return 0;
}
