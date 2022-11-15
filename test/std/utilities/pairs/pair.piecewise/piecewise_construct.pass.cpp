// <utility>

// template <class T1, class T2> struct pair

// struct piecewise_construct_t { };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <functional>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class A
{
    int i_;
    char c_;

  public:
    A(int i, char c) : i_(i), c_(c) {}
    int
    get_i() const
    {
        return i_;
    }
    char
    get_c() const
    {
        return c_;
    }
};

class B
{
    float f_;
    unsigned u1_;
    unsigned u2_;

  public:
    B(float f, unsigned u1, unsigned u2) : f_(f), u1_(u1), u2_(u2) {}
    float
    get_f() const
    {
        return f_;
    }
    unsigned
    get_u1() const
    {
        return u1_;
    }
    unsigned
    get_u2() const
    {
        return u2_;
    }
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
            s::pair<A, B> p(s::piecewise_construct, s::make_tuple(4, 'a'), s::make_tuple(3.5f, 6u, 2u));
            ret_access[0] = (p.first.get_i() == 4);
            ret_access[0] &= (p.first.get_c() == 'a');
            ret_access[0] &= (p.second.get_f() == 3.5f);
            ret_access[0] &= (p.second.get_u1() == 6u);
            ret_access[0] &= (p.second.get_u2() == 2u);
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
