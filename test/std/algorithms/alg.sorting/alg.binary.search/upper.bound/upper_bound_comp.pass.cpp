#include "oneapi_std_test_config.h"

#include _ONEAPI_STD_TEST_HEADER(algorithm)
#include _ONEAPI_STD_TEST_HEADER(iterator)
#include _ONEAPI_STD_TEST_HEADER(functional)

#include <iostream>

#include "test_iterators.h"
#include "checkData.h"
#include "test_macros.h"
#include "support/sycl_alloc_utils.h"

namespace test_ns = _ONEAPI_TEST_NAMESPACE;

// <algorithm>

// template<ForwardIterator Iter, class T, Predicate<auto, T, Iter::value_type> Compare>
//   constexpr Iter    // constexpr after c++17
//   upper_bound(Iter first, Iter last, const T& value, Compare comp);

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class Iter, class T>
bool __attribute__((always_inline)) test(Iter first, Iter last, const T& value)
{
    Iter i = test_ns::upper_bound(first, last, value, test_ns::greater<int>());
    for (Iter j = first; j != i; ++j)
        if ((test_ns::greater<int>()(value, *j)))
        {
            return false;
        }
    for (Iter j = i; j != last; ++j)
        if (!test_ns::greater<int>()(value, *j))
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

    std::sort(host_vbuf, host_vbuf + N, test_ns::greater<int>());

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> dt_helper(deviceQueue, host_vbuf, N);

    deviceQueue.submit([&](sycl::handler& cgh) {
        int* device_vbuf = dt_helper.get_data();
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<KC>([=]() {
            ret_access[0] = test(device_vbuf, device_vbuf + N, 0);
            for (int x = 1; x <= M; ++x)
                ret_access[0] &= test(device_vbuf, device_vbuf + N, x);
        });
    }).wait();

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "");
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
