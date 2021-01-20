/*
 *  Copyright (c) 2020 Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include "async.hpp"

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

/*
 The goal of this example is to provide opportunity for asynchronous execution of algorithms
 where the executing thread in main will not have to immediately stop and wait on a result.

 The example uses two input vectors, x and y, and creates an additional vector z through processing.
 The dependence graph of the algorithm calls is:

    x       y
    |       |
    V       V
    iota(x)     fill(y)
    |       |
    V       V
    transform(x)    transform(y)
    |     \ /
    |      V   V
    |   z = transform(x, y, sum_components)
    V       |
  alpha = reduce(x) |
    |       |
    V       V
  beta = transform_reduce(z, scale_components(alpha))
*/

//using oneapi::dpl::begin;
//using oneapi::dpl::end;
//using oneapi::dpl::execution::make_device_policy;

int main() {
    using namespace oneapi;
    const int n = 100;
    {
        sycl::queue q;
        sycl::buffer<int> x{n};//std::vector<int> x(n);
        sycl::buffer<int> y{n};//std::vector<int> y(n);

        auto my_policy = dpl::execution::make_device_policy(q);
        auto res_1a = dpl::experimental::copy_async(my_policy, dpl::counting_iterator<int>(0), dpl::counting_iterator<int>(n), dpl::begin(x));
        //std::iota(x.begin(), x.end(), 0); // x = [0..n]
        auto res_1b = dpl::experimental::fill_async(my_policy, dpl::begin(y), dpl::end(y), 7); // y = [7..7]

        auto res_2a = dpl::experimental::for_each_async(my_policy, dpl::begin(x), dpl::end(x), [](auto& e) { ++e; }, res_1a); // x = [1..n]
        auto res_2b = dpl::experimental::transform_async(my_policy, dpl::begin(y), dpl::end(y), dpl::begin(y), [](const auto& e) { return e / 2; }, res_1b); // y = [3..3]

        sycl::buffer<int> z{n};//std::vector<int> z(n);
        auto res_3 = dpl::experimental::transform_async(my_policy, dpl::begin(x), dpl::end(x), dpl::begin(y), dpl::begin(z), std::plus<int>(), res_2a, res_2b); // z = [4..n+3]

        auto res_4 = dpl::experimental::reduce_async(my_policy, dpl::begin(x), dpl::end(x), 0, std::plus<int>(), res_2a); // alpha = n*(n+1)/2
        auto alpha = res_4.get();

        //auto new_transform_iterator = oneapi::dpl::make_transform_iterator(dpl::begin(z), [=](auto x) { return alpha*x; });
        // beta = (n*(n+1)/2) * ((n+3)*(n+4)/2 - 6)
#if 0
        auto beta = std::transform_reduce(my_policy, dpl::begin(z), dpl::end(z), 0, std::plus<int>(), [=](auto e){ return alpha*e; });
#else
        auto r_beta = dpl::experimental::transform_reduce_async(my_policy, dpl::begin(z), dpl::end(z), 0, std::plus<int>(), [=](auto e){ return alpha*e; });
        auto beta = r_beta.get();
#endif


        if (beta == (n*(n+1)/2) * ((n+3)*(n+4)/2 - 6))
            std::cout << "done\n";
        else
            std::cout << "FAIL: expected " << (n*(n+1)/2) * ((n+3)*(n+4)/2 - 6) << " actual " << beta << "\n";
    }
    return 0;
}

