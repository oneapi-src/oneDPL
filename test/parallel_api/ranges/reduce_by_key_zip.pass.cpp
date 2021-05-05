#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>

#include "support/test_config.h"

#include <oneapi/dpl/ranges>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <functional>
#include <iostream>
#include <vector>

#if 1 //range based API

int32_t
main()
{
    sycl::queue q(sycl::default_selector{});

    const int n = 9, n_res = 5;
    int a[n] =  {11, 11, 21, 20, 21, 21, 21, 37, 37};
    int a1[n] = {11, 11, 20, 20, 20, 21, 21, 37, 37};
    int b[n] = {0,  1,  2,  3,  4,  5,  6,  7,  8};

    sycl::buffer<int> A(a, sycl::range<1>(n));
    sycl::buffer<int> A1(a1, sycl::range<1>(n));
    sycl::buffer<int> B(b, sycl::range<1>(n));
    sycl::buffer<int> C(n);           // output keys
    sycl::buffer<int> C1(n);           // output keys
    sycl::buffer<int> D(n);           // output values

    using namespace oneapi::dpl::experimental::ranges;

    auto res = reduce_by_key(oneapi::dpl::execution::make_device_policy(q),
        zip_view(views::all_read(A), views::all_read(A1)), 
        views::all_read(B), zip_view(views::all_write(C), views::all_write(C1)), views::all_write(D));

    auto res_keys1 = views::host_all(C);
    auto res_keys2 = views::host_all(C1);
    auto res_values = views::host_all(D);

    ::std::cout << "Output: " << std::endl;
    for(int i = 0; i <  res; ++i)
        ::std::cout << "{" << res_keys1[i] << ", " << res_keys2[i] << "}: " << res_values[i] << std::endl;
    ::std::cout << ::std::endl;
}

#else
int main()
{
    sycl::queue q(sycl::gpu_selector{});

    std::vector<int> keys1{11, 11, 21, 20, 21, 21, 21, 37, 37};
    std::vector<int> keys2{11, 11, 20, 20, 20, 21, 21, 37, 37};
    std::vector<int> values{0,  1,  2,  3,  4,  5,  6,  7,  8};
    std::vector<int> output_keys1(keys1.size());
    std::vector<int> output_keys2(keys2.size());    
    std::vector<int> output_values(values.size());

    int* d_keys1         = sycl::malloc_device<int>(9, q);
    int* d_keys2         = sycl::malloc_device<int>(9, q);
    int* d_values        = sycl::malloc_device<int>(9, q);
    int* d_output_keys1  = sycl::malloc_device<int>(9, q);
    int* d_output_keys2  = sycl::malloc_device<int>(9, q);
    int* d_output_values = sycl::malloc_device<int>(9, q);

    q.memcpy(d_keys1, keys1.data(), sizeof(int)*9);
    q.memcpy(d_keys2, keys2.data(), sizeof(int)*9);
    q.memcpy(d_values, values.data(), sizeof(int)*9);

    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + 9, d_keys2 + 9);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);

    auto new_last = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q),
        begin_keys_in, end_keys_in, d_values, begin_keys_out, d_output_values);

    q.memcpy(output_keys1.data(), d_output_keys1, sizeof(int)*9);
    q.memcpy(output_keys2.data(), d_output_keys2, sizeof(int)*9);    
    q.memcpy(output_values.data(), d_output_values, sizeof(int)*9);
    q.wait();

    // Expected output
    // {11, 11}: 1
    // {21, 20}: 2
    // {20, 20}: 3
    // {21, 20}: 4
    // {21, 21}: 11
    // {37, 37}: 15
    for(int i=0; i<9; i++) {
      std::cout << "{" << output_keys1[i] << ", " << output_keys2[i] << "}: " << output_values[i] << std::endl;
    }
}
#endif
