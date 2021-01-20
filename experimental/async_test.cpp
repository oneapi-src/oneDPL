//
//  async_reduce.cpp
//  
//
//  Created by Pablo Reble (Intel) on 2/3/20.
//

// Compile with xmain: dpcpp -fsycl async_reduce.cpp -std=c++11 -I ../parallelstl/include -I ../parallelstl/include/dpstd/pstl/hetero/dpcpp -I [add TBB include path]

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include "async.hpp"

using namespace oneapi;

int main() {
    
    const int N = 7;

    int a[N] = {0,2,4,6,8,10,12};
    {    
        sycl::buffer<int> b{a,sycl::range<1>{N}};
    
        sycl::queue q;
        auto my_policy = dpl::execution::make_device_policy(q);

        auto result1 = dpl::experimental::for_each_async(my_policy, dpl::begin(b), dpl::end(b), [](int &n){ n++; });

        auto result = dpl::experimental::reduce_async(my_policy, dpl::begin(b), dpl::end(b), 1, std::plus<int>(), result1);

        auto result2 = dpl::experimental::sort_async(my_policy, dpl::begin(b), dpl::end(b), result);

        // Test different signature of sort algorithm:
        auto result3 = dpl::experimental::sort_async(my_policy, dpl::begin(b), dpl::end(b), std::greater<int>(), result2);
    
        //oneapi::dpl::async::wait_for_all(result1,result,result2);
        result3.wait();

        auto x = result.get();
        if (x == 1+0+2+4+6+8+10+12+7)
            std::cout << "done\n";
        else
            std::cout << "FAIL: expected " << (1+0+2+4+6+8+10+12+7) << " actual " << x << "\n";    
    }

    return 0;
}

