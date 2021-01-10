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

        auto result1 = oneapi::dpl::async::for_each(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), [](int &n){ n++; });

        auto result = oneapi::dpl::reduce_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), 1, std::plus<int>(), result1);

        auto result2 = oneapi::dpl::sort_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), result);

        auto result3 = oneapi::dpl::sort_async(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), std::greater<int>(), result2);
    
        //oneapi::dpl::async::wait_for_all(result1,result,result2);
        //oneapi::dpl::async::wait_for_all(result1.get_event(),result.get_event(),result2.get_event());
        //result.wait();
        oneapi::dpl::async::wait_for_all(result3);
    
        std::cout << "" << result.data() << std::endl;
    }

    return 0;
}

