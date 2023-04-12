#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <type_traits>
#    include <iterator>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;
constexpr sycl::access::mode sycl_read_write = sycl::access::mode::read_write;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    bool ret = true;
    {
        sycl::queue deviceQueue = TestUtils::get_test_queue();
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf1.get_access<sycl_read_write>(cgh);

            cgh.single_task<class KernelIteratorTest1>([=]() {
                {
                    typedef s::array<int, 5> C;
                    C c;
                    C::iterator i;
                    i = c.begin();
                    C::const_iterator j;
                    j = c.cbegin();
                    ret_acc[0] &= (i == j);
                }
                {
                    typedef s::array<int, 0> C;
                    C c;
                    C::iterator i;
                    i = c.begin();
                    C::const_iterator j;
                    j = c.cbegin();
                    ret_acc[0] &= (i == j);
                }
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
