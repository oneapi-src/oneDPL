#define _GLIBCXX_USE_TBB_PAR_BACKEND 0 // libstdc++10

#include <cassert>

#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <algorithm>
#include <vector>
#include <iostream>

#include <numeric> // std::inclusive_scan, exclusive_scan
#include <functional>

int
main()
{
    std::vector<int> v{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    sycl::queue syclQue(sycl::gpu_selector_v);
    int* dev_v = sycl::malloc_device<int>(10, syclQue);
    syclQue.memcpy(dev_v, v.data(), 10 * sizeof(int));

    // Inclusive scan
    std::vector<int> gold_std_incl_result_host(10);
    std::inclusive_scan(v.begin(), v.end(), gold_std_incl_result_host.begin());

    int* incl_result_dev = sycl::malloc_device<int>(10, syclQue);
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(syclQue), dev_v, dev_v + 10,
                                incl_result_dev); //, oneapi::dpl::maximum<int>() );
    int* incl_result_host = new int[10];
    syclQue.memcpy(incl_result_host, incl_result_dev, 10 * sizeof(int)).wait();

    for (int i = 0; i < 10; i++)
    {
        //    assert(("inclusive_scan failed", gold_std_incl_result_host[i] == incl_result_host[i]));
        assert(gold_std_incl_result_host[i] == incl_result_host[i]);
    }
    sycl::free(incl_result_dev, syclQue);

    // Exclusive scan
    std::vector<int> gold_std_excl_result_host(10);
    std::exclusive_scan(v.begin(), v.end(), gold_std_excl_result_host.begin(), 0);

    int* excl_result_dev = sycl::malloc_device<int>(10, syclQue);
    oneapi::dpl::exclusive_scan(oneapi::dpl::execution::make_device_policy(syclQue), dev_v, dev_v + 10, excl_result_dev,
                                0); //, oneapi::dpl::maximum<int>() );
    int* excl_result_host = new int[10];
    syclQue.memcpy(excl_result_host, excl_result_dev, 10 * sizeof(int)).wait();

    for (int i = 0; i < 10; i++)
    {
        //assert("exclusive_scan failed", gold_std_excl_result_host[i] == excl_result_host[i]);
        assert(gold_std_excl_result_host[i] == excl_result_host[i]);
    }
    sycl::free(excl_result_dev, syclQue);
}
