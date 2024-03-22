// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class Category, class T, class Distance = ptrdiff_t,
//          class Pointer = T*, class Reference = T&>
// struct iterator
// {
//   typedef T         value_type;
//   typedef Distance  difference_type;
//   typedef Pointer   pointer;
//   typedef Reference reference;
//   typedef Category  iterator_category;
// };

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/utils.h"

struct A
{
};

template <class T>
class IteratorTest;

template <class T>
void
kernelTest()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<IteratorTest<T>>([=]() {
            {
                typedef dpl::iterator<dpl::forward_iterator_tag, T> It;
                static_assert(dpl::is_same<typename It::value_type, T>::value);
                static_assert(dpl::is_same<typename It::difference_type, dpl::ptrdiff_t>::value);
                static_assert(dpl::is_same<typename It::pointer, T*>::value);
                static_assert(dpl::is_same<typename It::reference, T&>::value);
                static_assert(dpl::is_same<typename It::iterator_category, dpl::forward_iterator_tag>::value);
            }
            {
                typedef dpl::iterator<dpl::bidirectional_iterator_tag, T, short> It;
                static_assert(dpl::is_same<typename It::value_type, T>::value);
                static_assert(dpl::is_same<typename It::difference_type, short>::value);
                static_assert(dpl::is_same<typename It::pointer, T*>::value);
                static_assert(dpl::is_same<typename It::reference, T&>::value);
                static_assert(dpl::is_same<typename It::iterator_category, dpl::bidirectional_iterator_tag>::value);
            }
            {
                typedef dpl::iterator<dpl::random_access_iterator_tag, T, int, const T*> It;
                static_assert(dpl::is_same<typename It::value_type, T>::value);
                static_assert(dpl::is_same<typename It::difference_type, int>::value);
                static_assert(dpl::is_same<typename It::pointer, const T*>::value);
                static_assert(dpl::is_same<typename It::reference, T&>::value);
                static_assert(dpl::is_same<typename It::iterator_category, dpl::random_access_iterator_tag>::value);
            }
            {
                typedef dpl::iterator<dpl::input_iterator_tag, T, long, const T*, const T&> It;
                static_assert(dpl::is_same<typename It::value_type, T>::value);
                static_assert(dpl::is_same<typename It::difference_type, long>::value);
                static_assert(dpl::is_same<typename It::pointer, const T*>::value);
                static_assert(dpl::is_same<typename It::reference, const T&>::value);
                static_assert(dpl::is_same<typename It::iterator_category, dpl::input_iterator_tag>::value);
            }
        });
    });
}

int
main()
{
    kernelTest<A>();

    return TestUtils::done();
}
