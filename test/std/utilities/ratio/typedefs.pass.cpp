#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(ratio)
namespace s = oneapi_cpp_ns;
#else
#    include <ratio>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

sycl::cl_bool
test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                static_assert(std::atto::num == 1 && std::atto::den == 1000000000000000000ULL, "");
                static_assert(std::femto::num == 1 && std::femto::den == 1000000000000000ULL, "");
                static_assert(std::pico::num == 1 && std::pico::den == 1000000000000ULL, "");
                static_assert(std::nano::num == 1 && std::nano::den == 1000000000ULL, "");
                static_assert(std::micro::num == 1 && std::micro::den == 1000000ULL, "");
                static_assert(std::milli::num == 1 && std::milli::den == 1000ULL, "");
                static_assert(std::centi::num == 1 && std::centi::den == 100ULL, "");
                static_assert(std::deci::num == 1 && std::deci::den == 10ULL, "");
                static_assert(std::deca::num == 10ULL && std::deca::den == 1, "");
                static_assert(std::hecto::num == 100ULL && std::hecto::den == 1, "");
                static_assert(std::kilo::num == 1000ULL && std::kilo::den == 1, "");
                static_assert(std::mega::num == 1000000ULL && std::mega::den == 1, "");
                static_assert(std::giga::num == 1000000000ULL && std::giga::den == 1, "");
                static_assert(std::tera::num == 1000000000000ULL && std::tera::den == 1, "");
                static_assert(std::peta::num == 1000000000000000ULL && std::peta::den == 1, "");
                static_assert(std::exa::num == 1000000000000000000ULL && std::exa::den == 1, "");
                ret_acc[0] = true;
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
