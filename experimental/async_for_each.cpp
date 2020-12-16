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
        sycl::buffer<int,1> b{a,sycl::range<1>{N}};
    
        sycl::queue q;

        auto result = oneapi::dpl::async::for_each(oneapi::dpl::execution::make_device_policy(q), oneapi::dpl::begin(b), oneapi::dpl::end(b), [](int &n){ n++; });
    
        result.wait();
    
        auto data = b.get_access<sycl::access::mode::read>();
        
        for(int i=0; i<N; ++i)
            std::cout << "" << data[i] << ", ";
        std::cout << std::endl;
    }

    return 0;
}

