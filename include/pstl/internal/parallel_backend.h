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

#ifndef __PSTL_parallel_backend_H
#define __PSTL_parallel_backend_H

#if __PSTL_PAR_BACKEND_TBB
    #include "parallel_backend_tbb.h"
#else
    __PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif /* __PSTL_parallel_backend_H */
