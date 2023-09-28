#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "checkData.h"
#include "test_macros.h"

#include <iostream>

#include _ONEAPI_STD_TEST_HEADER(algorithm)
#include _ONEAPI_STD_TEST_HEADER(iterator)
namespace s = _ONEAPI_TEST_NAMESPACE;

// <algorithm>

// template<ForwardIterator Iter, class T>
//   constexpr Iter    // constexpr after c++17
//   upper_bound(Iter first, Iter last, const T& value);

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class Iter, class T>
bool
test(Iter first, Iter last, const T& value)
{
    Iter i = s::upper_bound(first, last, value);
    for (Iter j = first; j != i; ++j)
        if ((value < *j))
        {
            return false;
        }
    for (Iter j = i; j != last; ++j)
        if (!(value < *j))
        {
            return false;
        }

    return true;
}

class KernelLowerBoundTest1;
class KernelLowerBoundTest2;
class KernelLowerBoundTest3;
class KernelLowerBoundTest4;

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
            ret_access[0] = test(device_vbuf, device_vbuf + N, 0);

            for (int x = 1; x <= M; ++x)
                ret_access[0] &= test(device_vbuf, device_vbuf + N, x);
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
    kernel_test<forward_iterator<const int*>, KernelLowerBoundTest1>();
    kernel_test<bidirectional_iterator<const int*>, KernelLowerBoundTest2>();
    kernel_test<random_access_iterator<const int*>, KernelLowerBoundTest3>();
    kernel_test<const int*, KernelLowerBoundTest4>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
