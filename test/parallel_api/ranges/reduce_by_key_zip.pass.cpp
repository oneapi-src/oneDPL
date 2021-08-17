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

    auto res = reduce_by_segment(oneapi::dpl::execution::make_device_policy(q),
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
