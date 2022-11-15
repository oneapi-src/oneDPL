// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   typename tuple_element<I, tuple<Types...> >::type&
//   get(tuple<Types...>& t);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct Empty
{
};

struct S
{
    s::tuple<int, Empty> a;
    int k;
    Empty e;
    constexpr S() : a{1, Empty{}}, k(s::get<0>(a)), e(s::get<1>(a)) {}
};

constexpr s::tuple<int, int>
getP()
{
    return {3, 4};
}

class KernelGetNonConstTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelGetNonConstTest>([=]() {
            {
                typedef s::tuple<int> T;
                T t(3);
                ret_access[0] = (s::get<0>(t) == 3);
                s::get<0>(t) = 2;
                ret_access[0] &= (s::get<0>(t) == 2);
            }

            { // get on an rvalue tuple
                static_assert(s::get<0>(s::make_tuple(0.0f, 1, 2.0, 3L)) == 0, "");
                static_assert(s::get<1>(s::make_tuple(0.0f, 1, 2.0, 3L)) == 1, "");
                static_assert(s::get<2>(s::make_tuple(0.0f, 1, 2.0, 3L)) == 2, "");
                static_assert(s::get<3>(s::make_tuple(0.0f, 1, 2.0, 3L)) == 3, "");
                static_assert(S().k == 1, "");
                static_assert(s::get<1>(getP()) == 4, "");
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
