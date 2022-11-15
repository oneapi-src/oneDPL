// <algorithm>

// template<ForwardIterator Iter, class T, CopyConstructible Compare>
//   constexpr bool      // constexpr after C++17
//   binary_search(Iter first, Iter last, const T& value, Compare comp);

#include "oneapi_std_test_config.h"
#include "test_iterators.h"
#include "checkData.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(algorithm)
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <algorithm>
#    include <iterator>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class Iter, class T>
bool
test(Iter first, Iter last, const T& value, bool x)
{
    return (s::binary_search(first, last, value, s::greater<int>()) == x);
}

class KernelBSearchTest1;
class KernelBSearchTest2;
class KernelBSearchTest3;
class KernelBSearchTest4;

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
            ret_access[0] = test(Iter(device_vbuf), Iter(device_vbuf + N), 0, true);

            for (int x = 1; x < M; ++x)
                ret_access[0] &= test(Iter(device_vbuf), Iter(device_vbuf + N), x, true);

            ret_access[0] &= test(Iter(device_vbuf), Iter(device_vbuf + N), -1, false);
            ret_access[0] &= test(Iter(device_vbuf), Iter(device_vbuf + N), M, false);
            // Simple local array testing

            /* int d[] = {6, 4, 2, 0};
            for (int* e = d; e <= d+4; ++e)
               for (int y = -1; y <= 7; ++y)
                  ret_access[0] &= test(d, e, y, (y % 2 == 0) && e != d && (-2*(e-d) + 8 <= y));
            */
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
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test<forward_iterator<const int*>, KernelBSearchTest1>();
    kernel_test<bidirectional_iterator<const int*>, KernelBSearchTest2>();
    // kernel_test<random_access_iterator<const int*>, KernelBSearchTest3>();
    kernel_test<const int*, KernelBSearchTest4>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
