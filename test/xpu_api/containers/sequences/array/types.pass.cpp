#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename T>
class KernelTest1;
template <typename T>
class KernelTest2;

template <class C>
bool
test_iterators()
{
    cl::sycl::cl_bool ret = true;
    {
        cl::sycl::queue deviceQueue;
        cl::sycl::range<1> numOfItems{1};
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buf1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest1<C>>([=]() {
                typedef s::iterator_traits<typename C::iterator> ItT;
                typedef s::iterator_traits<typename C::const_iterator> CItT;

                ret_acc[0] &=
                    (s::is_same<typename ItT::iterator_category, s::random_access_iterator_tag>::value == true);

                ret_acc[0] &= (s::is_same<typename ItT::value_type, typename C::value_type>::value == true);

                ret_acc[0] &= (s::is_same<typename ItT::reference, typename C::reference>::value == true);

                ret_acc[0] &= (s::is_same<typename ItT::pointer, typename C::pointer>::value == true);
                ret_acc[0] &= (s::is_same<typename ItT::difference_type, typename C::difference_type>::value == true);

                ret_acc[0] &=
                    (s::is_same<typename CItT::iterator_category, typename s::random_access_iterator_tag>::value ==
                     true);
                ret_acc[0] &= (s::is_same<typename CItT::value_type, typename C::value_type>::value == true);
                ret_acc[0] &= (s::is_same<typename CItT::reference, typename C::const_reference>::value == true);

                ret_acc[0] &= (s::is_same<typename CItT::pointer, typename C::const_pointer>::value == true);

                ret_acc[0] &= (s::is_same<typename CItT::difference_type, typename C::difference_type>::value == true);
            });
        });
    }
    return ret;
}

template <typename T>
bool
kernel_test()
{
    bool ret = true;
    {
        cl::sycl::queue deviceQueue;
        cl::sycl::range<1> numOfItems{1};
        cl::sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ptr1 = buf1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest2<T>>([=]() {
                typedef s::array<T, 10> C;
                ptr1[0] = (s::is_same<typename C::reference, T&>::value == true);
                ptr1[0] &= (s::is_same<typename C::const_reference, const T&>::value == true);
                ptr1[0] &= (s::is_same<typename C::pointer, T*>::value == true);
                ptr1[0] &= (s::is_same<typename C::const_pointer, const T*>::value == true);
                ptr1[0] &= (s::is_same<typename C::size_type, s::size_t>::value == true);
                ptr1[0] &= (s::is_same<typename C::difference_type, s::ptrdiff_t>::value == true);
                ptr1[0] &=
                    (s::is_same<typename C::reverse_iterator, s::reverse_iterator<typename C::iterator>>::value ==
                     true);
                ptr1[0] &= (s::is_same<typename C::const_reverse_iterator,
                                       s::reverse_iterator<typename C::const_iterator>>::value == true);
                ptr1[0] &= (s::is_signed<typename C::difference_type>::value == true);
                ptr1[0] &= (s::is_unsigned<typename C::size_type>::value == true);
                ptr1[0] &=
                    (s::is_same<typename C::difference_type,
                                typename s::iterator_traits<typename C::iterator>::difference_type>::value == true);
                ptr1[0] &=
                    (s::is_same<typename C::difference_type,
                                typename s::iterator_traits<typename C::const_iterator>::difference_type>::value ==
                     true);
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = true;
    typedef s::array<float, 10> C;
    ret &= test_iterators<C>();
    ret &= kernel_test<int*>();
    ret &= kernel_test<float>();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
