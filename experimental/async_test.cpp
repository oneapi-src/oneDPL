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

int main() {
    
    const int N = 7;

    int a[N] = {0,2,4,6,8,10,12};
    {    
        sycl::buffer<int> b{a,sycl::range<1>{N}};
    
        sycl::queue q;

#if 0
        std::for_each(oneapi::dpl::execution::make_device_policy<class algo1>(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), [](int &n){ n++; });
        sycl::event result1;
#else
        auto result1 = oneapi::dpl::for_each_async(oneapi::dpl::execution::make_device_policy<class algo1>(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), [](int &n){ n++; });
#endif

        auto result = oneapi::dpl::reduce_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), 1, std::plus<int>(), result1);

        auto result2 = oneapi::dpl::sort_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), result);

        // Test different signature of sort algorithm:
        auto result3 = oneapi::dpl::sort_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), std::greater<int>(), result2);
    
        //oneapi::dpl::async::wait_for_all(result1,result,result2);
        result3.wait();
    
        std::cout << "Result: " << result.data() << std::endl;
        std::cout << "Expected: 50\n";
    }

    return 0;
}

