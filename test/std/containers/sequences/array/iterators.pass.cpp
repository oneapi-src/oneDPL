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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

int
main(int, char**)
{
    bool ret = true;
    {
        cl::sycl::queue deviceQueue;
        cl::sycl::range<1> numOfItems{1};
        cl::sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](cl::sycl::handler& cgh) {
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

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
