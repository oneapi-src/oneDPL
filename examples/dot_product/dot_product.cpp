/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

/*
    This file contains the implementation of dot product based on std::transform_reduce
*/

#include <vector>
#include <random>
#include <iostream>
#include <pstl/algorithm>
#include <pstl/numeric>
#include <pstl/execution>

double random_number_generator() {
    // usage of thread local random engines allows running the generator in concurrent mode
    thread_local static std::default_random_engine rd;
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(rd);
}

int main(int argc, char* argv[]) {

    const size_t size = 10000000;

    std::vector<double> v1(size), v2(size);

    //initialize vectors with random numbers
    std::generate(pstl::execution::par, v1.begin(), v1.end(), random_number_generator);
    std::generate(pstl::execution::par, v2.begin(), v2.end(), random_number_generator);

    //the dot product calculation
    double res = std::transform_reduce(pstl::execution::par_unseq, v1.cbegin(), v1.cend(), v2.cbegin(), .0,
        std::plus<double>(), std::multiplies<double>());

    std::cout << "The dot product is: " << res << std::endl;

    return 0;
}
