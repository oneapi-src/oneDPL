//===----------------------------------------------------------------------===//
//
// template<class E>
// class initializer_list
// {
// public:
//     typedef E        value_type;
//     typedef const E& reference;
//     typedef const E& const_reference;
//     typedef size_t   size_type;
//
//     typedef const E* iterator;
//     typedef const E* const_iterator;
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>
#include <initializer_list>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
};
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    bool ret = true;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                static_assert((s::is_same<std::initializer_list<A>::value_type, A>::value), "");
                static_assert((s::is_same<std::initializer_list<A>::reference, const A&>::value), "");
                static_assert((s::is_same<std::initializer_list<A>::const_reference, const A&>::value), "");
                static_assert((s::is_same<std::initializer_list<A>::size_type, s::size_t>::value), "");
                static_assert((s::is_same<std::initializer_list<A>::iterator, const A*>::value), "");
                static_assert((s::is_same<std::initializer_list<A>::const_iterator, const A*>::value), "");
                acc[0] = true;
            });
        });
    }

    if (ret)

        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
