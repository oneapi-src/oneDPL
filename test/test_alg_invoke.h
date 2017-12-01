/*
    Copyright (c) 2017 Intel Corporation

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

// This header is included *twice* by test.cpp, once with USE_POLICY defined and once without.

#ifdef USE_POLICY    
template<class ExecutionPolicy>
    void run(ExecutionPolicy policy) 
    #define P policy,
#else
    void run_nat()
    #define P
#endif
    {
        using namespace std;
        for( int k=0; k<2; ++k )
            switch( kind ) {
            default: assert(0);
            case AnyOf: boolActual[k] = any_of(P input[k].begin(),input[k].end(),[](T x) {return x<0;}); break;
            case AllOf: boolActual[k] = all_of(P input[k].begin(),input[k].end(),[](T x) {return x>=0;}); break;
            case NoneOf: boolActual[k] = none_of(P input[k].begin(),input[k].end(),[](T x) {return x<0;}); break;
            case ForEach: for_each(P input[k].begin(),input[k].end(),[&](T& x){output[k].corr(input[k],x)=x*x;}); break;
            case ForEachN: std::experimental::parallel::for_each_n(P input[k].begin(),input[k].size(),[&](T& x){output[k].corr(input[k],x)=x*x;}); break;
            }
    }
#undef P
