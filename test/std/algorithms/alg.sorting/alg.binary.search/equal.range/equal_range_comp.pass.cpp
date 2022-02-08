
#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "checkData.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
namespace s = std;
#endif

// <algorithm>

// template<ForwardIterator Iter, class T, CopyConstructible Compare>
//   constexpr pair<Iter, Iter>   // constexpr after c++17
//   equal_range(Iter first, Iter last, const T& value, Compare comp);

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

//Revert the change when GPU runtime fix their issue
template <class Iter, class T>
bool test(Iter first, Iter last, const T& value)
{
    std::pair<Iter, Iter> i = s::equal_range(first, last, value, s::greater<int>());

    for (Iter j = first; j != i.first; ++j)
        if (!(s::greater<int>()(*j, value)))
            return false;
    /*for (Iter j = i.first; j != last; ++j)
        if (s::greater<int>()(*j, value))
            return false;
    */
    for (Iter j = first; j != i.second; ++j)
        if (s::greater<int>()(value, *j))
            return false;
    /*for (Iter j = i.second; j != last; ++j)
        if (!s::greater<int>()(value, *j))
            return false;
    */
    return true;
}

class KernelEqualRangeTest1;
class KernelEqualRangeTest2;
class KernelEqualRangeTest3;
class KernelEqualRangeTest4;

template <typename Iter, typename KC>
void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    const unsigned N = 1000;
    const int M = 10;
    int host_vbuf[N];
    int x = 0;
    for (size_t i = 0; i < N; ++i)
    {
        host_vbuf[i] = x;
        if (++x == M)
            x = 0;
    }

    std::sort(host_vbuf, host_vbuf + N, s::greater<int>());
    cl::sycl::range<1> host_buffer_sz{N};
    cl::sycl::buffer<cl::sycl::cl_int, 1> host_data_buffer(host_vbuf, host_buffer_sz);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        auto data_access = host_data_buffer.get_access<sycl_read>(cgh);
        cgh.single_task<KC>([=]() {
            int device_vbuf[N];
            for (size_t i = 0; i < N; i++)
                device_vbuf[i] = data_access[i];
            ret_access[0] = test(Iter(device_vbuf), Iter(device_vbuf + N), 0);
            for (int x = 1; x <= M; ++x)
                ret_access[0] &= test(Iter(device_vbuf), Iter(device_vbuf + N), x);

            // Simple local array testing
            /* int d[] = {3, 2, 1, 0};
	for (int* e = d; e < d+4; ++e)
	    for (int y = -1; y <= 4; ++y)
	        ret_access[0] &= test(d, e, y);*/
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

    // kernel_test<forward_iterator<const int*>, KernelEqualRangeTest1>();
    // kernel_test<bidirectional_iterator<const int*>, KernelEqualRangeTest2>();
    // kernel_test<random_access_iterator<const int*>, KernelEqualRangeTest3>();
    kernel_test<const int*, KernelEqualRangeTest4>();
    return 0;
}
