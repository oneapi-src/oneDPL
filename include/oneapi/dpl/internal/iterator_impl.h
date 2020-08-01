// -*- C++ -*-
//===-- iterator_impl.h -------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#ifndef _DPSTD_iterator_impl_H
#define _DPSTD_iterator_impl_H

namespace oneapi
{
namespace dpl
{
namespace internal
{

// Helper struct to extract sycl_iterator types needed to construct accessors
template <typename Iterator>
struct extract_accessor
{
    using accessor_type = Iterator;

    static accessor_type
    get(Iterator& i)
    {
        return i;
    }
};

#if _PSTL_BACKEND_SYCL
// Specialization for sycl_iterator to provide access to its component types needed to
// construct the accessor type
template <cl::sycl::access::mode Mode, typename T, typename Allocator>
struct extract_accessor<dpstd::__internal::sycl_iterator<Mode, T, Allocator>>
{
    static constexpr cl::sycl::access::mode mode = Mode;
    static constexpr int dim = 1;
    using buffer_type = cl::sycl::buffer<T, dim, Allocator>;
    using accessor_type = cl::sycl::accessor<T, dim, mode, cl::sycl::access::target::host_buffer>;

    static accessor_type
    get(dpstd::__internal::sycl_iterator<Mode, T, Allocator>& iter)
    {
        return iter.get_buffer().template get_access<mode>();
    }
};
#endif
} // namespace internal


class discard_iterator
{
  public:
    typedef std::ptrdiff_t difference_type;
    typedef decltype(std::ignore) value_type;
    typedef void* pointer;
    typedef value_type reference;
    typedef std::random_access_iterator_tag iterator_category;
    using is_passed_directly = std::true_type;

    discard_iterator() : __my_position_() {}
    explicit discard_iterator(difference_type __init) : __my_position_(__init) {}

    auto operator*() const -> decltype(std::ignore) { return std::ignore; }
    auto operator[](difference_type) const -> decltype(std::ignore) { return std::ignore; }

    constexpr bool
    operator==(const discard_iterator& __it) const
    {
        return __my_position_ - __it.__my_position_ == 0;
    }
    constexpr bool
    operator!=(const discard_iterator& __it) const
    {
        return !(*this == __it);
    }

    bool
    operator<(const discard_iterator& __it) const
    {
        return __my_position_ - __it.__my_position_ < 0;
    }
    bool
    operator>(const discard_iterator& __it) const
    {
        return __my_position_ - __it.__my_position_ > 0;
    }

    difference_type
    operator-(const discard_iterator& __it) const
    {
        return __my_position_ - __it.__my_position_;
    }

    discard_iterator&
    operator++()
    {
        ++__my_position_;
        return *this;
    }
    discard_iterator&
    operator--()
    {
        --__my_position_;
        return *this;
    }
    discard_iterator
    operator++(int)
    {
        discard_iterator __it(__my_position_);
        ++__my_position_;
        return __it;
    }
    discard_iterator
    operator--(int)
    {
        discard_iterator __it(__my_position_);
        --__my_position_;
        return __it;
    }
    discard_iterator&
    operator+=(difference_type __forward)
    {
        __my_position_ += __forward;
        return *this;
    }
    discard_iterator&
    operator-=(difference_type __backward)
    {
        __my_position_ -= __backward;
        return *this;
    }

    discard_iterator
    operator+(difference_type __forward) const
    {
        return discard_iterator(__my_position_ + __forward);
    }
    discard_iterator
    operator-(difference_type __backward) const
    {
        return discard_iterator(__my_position_ - __backward);
    }

  private:
    difference_type __my_position_;
};

template <typename SourceIterator, typename IndexMap>
class permutation_iterator
{
  private:
    using source_accessor_extractor = internal::extract_accessor<SourceIterator>;
    using map_accessor_extractor    = internal::extract_accessor<IndexMap>;

    // constructor used by operator+ and operator-
    permutation_iterator(const SourceIterator& input1, const IndexMap& input2,
                         const typename source_accessor_extractor::accessor_type source,
                         const typename map_accessor_extractor::accessor_type map,
                         std::size_t index)
        : my_source_it(input1), my_index_map(input2), my_source(source), my_map(map), my_index(index)
    {
    }

  public:
    typedef typename std::iterator_traits<SourceIterator>::difference_type difference_type;
    typedef typename std::iterator_traits<SourceIterator>::value_type value_type;
    typedef typename std::iterator_traits<SourceIterator>::pointer pointer;
    typedef typename std::iterator_traits<SourceIterator>::reference reference;
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::true_type is_permutation;

    permutation_iterator(const SourceIterator& input1, const IndexMap& input2, std::size_t index = 0)
        : my_source_it(input1), my_index_map(input2),
          my_source(source_accessor_extractor::get(my_source_it)),
          my_map(map_accessor_extractor::get(my_index_map)), my_index(index)
    {
    }

    SourceIterator&
    get_source_iterator()
    {
        return my_source_it;
    }

    IndexMap&
    get_map_iterator()
    {
        return my_index_map;
    }

    reference operator*() const { return my_source[difference_type(my_map[my_index])];
    }

    reference operator[](difference_type i) const { return *(*this + i); }

    permutation_iterator&
    operator++()
    {
        ++my_index;
        return *this;
    }

    permutation_iterator
    operator++(int)
    {
        permutation_iterator it(*this);
        ++(*this);
        return it;
    }

    permutation_iterator&
    operator--()
    {
        --my_index;
        return *this;
    }

    permutation_iterator
    operator--(int)
    {
        permutation_iterator it(*this);
        --(*this);
        return it;
    }

    permutation_iterator
    operator+(difference_type forward) const
    {
        return permutation_iterator(my_source_it, my_index_map, my_source, my_map, my_index + forward);
    }

    permutation_iterator
    operator-(difference_type backward)
    {
        return permutation_iterator(my_source_it, my_index_map, my_source, my_map, my_index - backward);
    }

    permutation_iterator&
    operator+=(difference_type forward)
    {
        my_index += forward;
        return *this;
    }

    permutation_iterator&
    operator-=(difference_type forward)
    {
        my_index -= forward;
        return *this;
    }

    difference_type
    operator-(const permutation_iterator& it) const
    {
        return my_index - it.my_index;
    }

    bool
    operator==(const permutation_iterator& it) const
    {
        return *this - it == 0;
    }
    bool
    operator!=(const permutation_iterator& it) const
    {
        return !(*this == it);
    }
    bool
    operator<(const permutation_iterator& it) const
    {
        return *this - it < 0;
    }
    bool
    operator>(const permutation_iterator& it) const
    {
        return it < *this;
    }
    bool
    operator<=(const permutation_iterator& it) const
    {
        return !(*this > it);
    }
    bool
    operator>=(const permutation_iterator& it) const
    {
        return !(*this < it);
    }

private:
    SourceIterator my_source_it;
    IndexMap my_index_map;
    typename source_accessor_extractor::accessor_type my_source;
    typename map_accessor_extractor::accessor_type my_map;
    std::size_t my_index;
};

template <typename SourceIterator, typename IndexMap>
permutation_iterator<SourceIterator, IndexMap>
make_permutation_iterator( SourceIterator source, IndexMap map )
{
    return permutation_iterator<SourceIterator, IndexMap>(source, map);
}

} // end namespace dpl
} // end namespace oneapi

namespace dpstd
{
using oneapi::dpl::discard_iterator;
using oneapi::dpl::permutation_iterator;
using oneapi::dpl::make_permutation_iterator;
} // end namespace dpstd
#endif /* __DPSTD_iterator_impl_H */
