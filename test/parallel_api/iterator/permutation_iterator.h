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

#ifndef _PERMUTATION_ITERATOR_H
#define _PERMUTATION_ITERATOR_H

#include "support/utils_const.h"
#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#endif // TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_test_base.h"

#include <type_traits>
#include <vector>
#include <iterator>

/// struct perm_it_index_tags - describe indexes of permutation iterator
struct perm_it_index_tags
{
    // Index of permutation iterator is based on counting iterator
    struct counting { };

    // Index of permutation iterator is based on host iterator
    struct host { };

#if TEST_DPCPP_BACKEND_PRESENT
    // Index of permutation iterator is based on USM shared/device memory
    struct usm_shared { };
    struct usm_device { };
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Index of permutation iterator is based on transform iterator
    struct transform_iterator { };
};

////////////////////////////////////////////////////////////////////////////////
/**
 * @param template typename TSourceIterator - source iterator type
 * @param typename TSourceDataSize - type of source data size
 * @param typename PermItIndexTag - tag permutation iterator base kind:
 *          - perm_it_index_tags::counting;
 *          - perm_it_index_tags::host;
 *          - perm_it_index_tags::usm_shared;
 *          - perm_it_index_tags::usm_device;
 *          - perm_it_index_tags::transform_iterator.
 */
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator_data
{
    TSourceIterator itSource;                // First source iterator
    const TSourceDataSize src_data_size = 0; // Source data items count
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize, typename PermItIndexTag>
struct test_through_permutation_iterator
{
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags::counting>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename TStepFunctor, typename Operand>
    void
    operator()(TStepFunctor stepOp, Operand op)
    {
        auto indexes_begin = dpl::counting_iterator<TSourceDataSize>(0);

        op(dpl::make_permutation_iterator(data.itSource, indexes_begin),
           dpl::make_permutation_iterator(data.itSource, indexes_begin) + data.src_data_size);
    }
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags::host>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename TStepFunctor, typename Operand>
    void
    operator()(TStepFunctor stepOp, Operand op)
    {
        ::std::vector<TSourceDataSize> indexes;

        for (TSourceDataSize perm_idx_step = 1; perm_idx_step < data.src_data_size; perm_idx_step = stepOp(perm_idx_step))
        {
            const TSourceDataSize idx_size = data.src_data_size / perm_idx_step;
            indexes.resize(idx_size);
            for (TSourceDataSize idx = 0, val = 0; idx < idx_size; ++idx, val += perm_idx_step)
                indexes[idx] = val;

            op(dpl::make_permutation_iterator(data.itSource, indexes.begin()),
               dpl::make_permutation_iterator(data.itSource, indexes.begin()) + indexes.size());
        }
    }
};

#if TEST_DPCPP_BACKEND_PRESENT

template <sycl::usm::alloc alloc_type, typename TSourceIterator, typename TSourceDataSize, typename TStepFunctor, typename Operand>
void
call_op_impl_usm(
    TStepFunctor stepOp, Operand op,
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data)
{
    using TestBaseData = TestUtils::test_base_data_usm<alloc_type, TSourceDataSize>;

    TestBaseData test_base_data(TestUtils::get_test_queue(), {{TestUtils::max_n, TestUtils::inout1_offset}});
    TSourceDataSize* itIndexStart = test_base_data.get_start_from(TestUtils::UDTKind::eKeys);

    std::vector<TSourceDataSize> indexes;

    for (TSourceDataSize perm_idx_step = 1; perm_idx_step < data.src_data_size; perm_idx_step = stepOp(perm_idx_step))
    {
        const TSourceDataSize idx_size = data.src_data_size / perm_idx_step;
        indexes.resize(idx_size);
        for (TSourceDataSize idx = 0, val = 0; idx < idx_size; ++idx, val += perm_idx_step)
            indexes[idx] = val;

        test_base_data.update_data(TestUtils::UDTKind::eKeys, indexes.data(), indexes.data() + indexes.size());

        TSourceDataSize* itIndexStart = test_base_data.get_start_from(TestUtils::UDTKind::eKeys);
        op(dpl::make_permutation_iterator(data.itSource, itIndexStart),
           dpl::make_permutation_iterator(data.itSource, itIndexStart) + indexes.size());
    }
}

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags::usm_shared>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename TStepFunctor, typename Operand>
    void
    operator()(TStepFunctor stepOp, Operand op)
    {
        call_op_impl_usm<sycl::usm::alloc::shared, TSourceIterator, TSourceDataSize>(stepOp, op, data);
    }
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags::usm_device>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename TStepFunctor, typename Operand>
    void
    operator()(TStepFunctor stepOp, Operand op)
    {
        call_op_impl_usm<sycl::usm::alloc::device, TSourceIterator, TSourceDataSize>(stepOp, op, data);
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags::transform_iterator>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename TStepFunctor, typename Operand>
    void
    operator()(TStepFunctor stepOp, Operand op)
    {
        using ValueType = typename ::std::iterator_traits<TSourceIterator>::value_type;

        auto no_transformation = [](ValueType val) { return val; };

        op(dpl::make_transform_iterator(data.itSource, no_transformation),
           dpl::make_transform_iterator(data.itSource, no_transformation) + data.src_data_size);
    }
};

////////////////////////////////////////////////////////////////////////////////
struct multiply_index_by_two
{
    template <typename Index>
    Index
    operator()(const Index& i) const
    {
        return i * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////
// struct test_through_permutation_iterator_mibt - implementation of the
// test_through_permutation_iterator with automatically pass of
// multiply_index_by_two instance into test_through_permutation_iterator::operator()
template <typename TSourceIterator, typename TSourceDataSize, typename PermItIndexTag,
          typename TStepFunctor = multiply_index_by_two>
struct test_through_permutation_iterator_mibt
    : test_through_permutation_iterator<TSourceIterator, TSourceDataSize, PermItIndexTag>
{
    using base = test_through_permutation_iterator<TSourceIterator, TSourceDataSize, PermItIndexTag>;

    /**
     * Constructor
     * 
     * @param TSourceIterator itSource - first source iterator
     * @param const TSourceDataSize src_data_size - source data items count
     */
    test_through_permutation_iterator_mibt(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : base(itSource, src_data_size)
    {
    }

    template <typename Operand>
    void operator()(Operand op)
    {
        base::operator()(TStepFunctor{}, op);
    }
};

#endif // _PERMUTATION_ITERATOR_H
