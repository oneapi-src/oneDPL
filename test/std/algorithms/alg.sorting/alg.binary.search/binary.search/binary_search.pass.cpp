// <algorithm>

// template<ForwardIterator Iter, class T>
//   constexpr bool      // constexpr after C++17
//   binary_search(Iter first, Iter last, const T& value);

#include "oneapi_std_test_config.h"
#include "test_iterators.h"
#include "checkData.h"

#include <iostream>

#include _ONEAPI_STD_TEST_HEADER(algorithm)
#include _ONEAPI_STD_TEST_HEADER(iterator)
namespace test_ns = _ONEAPI_TEST_NAMESPACE;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class Iter, class T>
bool
test(Iter first, Iter last, const T& value, bool x)
{
    return (test_ns::binary_search(first, last, value) == x);
}

class KernelBSearchTest1;
class KernelBSearchTest2;
class KernelBSearchTest3;
class KernelBSearchTest4;

template <typename Iter, typename KC>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    const unsigned N = 1000;
    const int M = 10;
    int host_vbuf[N];
    for (size_t i = 0; i < N; ++i)
    {
        host_vbuf[i] = i % M;
    }

    std::sort(host_vbuf, host_vbuf + N);
    sycl::range<1> host_buffer_sz{N};
    sycl::buffer<sycl::cl_int, 1> host_data_buffer(host_vbuf, host_buffer_sz);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        auto data_access = host_data_buffer.get_access<sycl_read>(cgh);
        cgh.single_task<KC>([=]() {
            int device_vbuf[N];
            for (size_t i = 0; i < N; i++)
                device_vbuf[i] = data_access[i];
            ret_access[0] = test(device_vbuf, device_vbuf + N, 0, true);

            for (int x = 1; x < M; ++x)
                ret_access[0] &= test(device_vbuf, device_vbuf + N, x, true);

            ret_access[0] &= test(device_vbuf, device_vbuf + N, -1, false);
            ret_access[0] &= test(device_vbuf, device_vbuf + N, M, false);
        });
    }).wait();

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exit_on_error(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test<forward_iterator<const int*>, KernelBSearchTest1>();
    kernel_test<bidirectional_iterator<const int*>, KernelBSearchTest2>();
    kernel_test<random_access_iterator<const int*>, KernelBSearchTest3>();
    kernel_test<const int*, KernelBSearchTest4>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
