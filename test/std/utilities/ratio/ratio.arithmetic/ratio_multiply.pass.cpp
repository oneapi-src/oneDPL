// test ratio_multiply

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
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                ret_acc[0] = true;
                {
                    typedef s::ratio<1, 1> R1;
                    typedef s::ratio<1, 1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 1 && R::den == 1, "");
                }
                {
                    typedef s::ratio<1, 2> R1;
                    typedef s::ratio<1, 1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 1 && R::den == 2, "");
                }
                {
                    typedef s::ratio<-1, 2> R1;
                    typedef s::ratio<1, 1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2, "");
                }
                {
                    typedef s::ratio<1, -2> R1;
                    typedef s::ratio<1, 1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2, "");
                }
                {
                    typedef s::ratio<1, 2> R1;
                    typedef s::ratio<-1, 1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2, "");
                }
                {
                    typedef s::ratio<1, 2> R1;
                    typedef s::ratio<1, -1> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == -1 && R::den == 2, "");
                }
                {
                    typedef s::ratio<56987354, 467584654> R1;
                    typedef s::ratio<544668, 22145> R2;
                    typedef s::ratio_multiply<R1, R2>::type R;
                    static_assert(R::num == 15519594064236LL && R::den == 5177331081415LL, "");
                }
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
