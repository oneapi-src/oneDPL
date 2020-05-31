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

namespace dpstd {

namespace internal {

    // Helper struct to extract sycl_iterator types needed to construct accessors
    template<typename Iterator>
    struct extract_accessor
    {
        using accessor_type = Iterator;

        static accessor_type get(Iterator& i) { return i; }
    };

#if _PSTL_BACKEND_SYCL
    // Specialization for sycl_iterator to provide access to its component types needed to
    // construct the accessor type
    template<cl::sycl::access::mode Mode, typename T, typename Allocator>
    struct extract_accessor<__internal::sycl_iterator<Mode, T, Allocator>>
    {
        static constexpr cl::sycl::access::mode mode = Mode;
        static constexpr int dim = 1;
        using buffer_type = cl::sycl::buffer<T, dim, Allocator>;
        using accessor_type = cl::sycl::accessor<T, dim, mode, cl::sycl::access::target::host_buffer>;

        static accessor_type get(__internal::sycl_iterator<Mode, T, Allocator>& iter)
        { return iter.get_buffer().template get_access<mode>(); }
    };
#endif

    template< typename Iter1, typename Iter2 >
    class permutation_iterator {
    private:
        using source_accessor_extractor = extract_accessor<Iter1>;
        using map_accessor_extractor    = extract_accessor<Iter2>;

        // constructor used by operator+ and operator-
        permutation_iterator(const Iter1& input1, const Iter2& input2,
                             const typename source_accessor_extractor::accessor_type source,
                             const typename map_accessor_extractor::accessor_type map,
                             std::size_t index)
            : my_source_it(input1), my_map_it(input2), my_source(source), my_map(map), my_index(index)
        { }

    public:
        typedef typename std::iterator_traits<Iter1>::difference_type difference_type;
        typedef typename std::iterator_traits<Iter1>::value_type value_type;
        typedef typename std::iterator_traits<Iter1>::pointer pointer;
        typedef typename std::iterator_traits<Iter1>::reference reference;
        typedef std::random_access_iterator_tag iterator_category;

        permutation_iterator(const Iter1& input1, const Iter2& input2, std::size_t index = 0)
            : my_source_it(input1), my_map_it(input2),
              my_source(source_accessor_extractor::get(my_source_it)),
              my_map(map_accessor_extractor::get(my_map_it)), my_index(index)
        { }

        Iter1& get_source_iterator()
        {
            return my_source_it;
        }

        Iter2& get_map_iterator()
        {
            return my_map_it;
        }

        reference operator*() const
        {
            return my_source[difference_type(my_map[my_index])];
        }

        reference operator[](difference_type i) const
        {
            return *(*this + i);
        }

        permutation_iterator& operator++()
        {
            ++my_index;
            return *this;
        }

        permutation_iterator operator++(int)
        {
            permutation_iterator it(*this);
            ++(*this);
            return it;
        }

        permutation_iterator& operator--()
        {
            --my_index;
            return *this;
        }

        permutation_iterator operator--(int)
        {
            permutation_iterator it(*this);
            --(*this);
            return it;
        }

        permutation_iterator operator+(difference_type forward) const
        {
            return permutation_iterator(my_source_it, my_map_it, my_source, my_map, my_index + forward);
        }

        permutation_iterator operator-(difference_type backward)
        {
            return permutation_iterator(my_source_it, my_map_it, my_source, my_map, my_index - backward);
        }

        permutation_iterator& operator+=(difference_type forward)
        {
            my_index += forward;
            return *this;
        }

        permutation_iterator& operator-=(difference_type forward)
        {
            my_index -= forward;
            return *this;
        }

        difference_type operator-(const permutation_iterator& it) const
        {
            return my_index - it.my_index;
        }

        bool operator==(const permutation_iterator& it) const { return *this - it == 0; }
        bool operator!=(const permutation_iterator& it) const { return !(*this == it); }
        bool operator<(const permutation_iterator& it) const { return *this - it < 0; }
        bool operator>(const permutation_iterator& it) const { return it < *this; }
        bool operator<=(const permutation_iterator& it) const { return !(*this > it); }
        bool operator>=(const permutation_iterator& it) const { return !(*this < it); }

    private:
        Iter1 my_source_it;
        Iter2 my_map_it;
        typename source_accessor_extractor::accessor_type my_source;
        typename map_accessor_extractor::accessor_type my_map;
        std::size_t my_index;
    };

    template< typename Iter1, typename Iter2 >
    permutation_iterator<Iter1, Iter2> make_permutation_iterator( Iter1 source, Iter2 map ) {
        return permutation_iterator<Iter1, Iter2>(source, map);
    }
} // namespace internal

} // namespace dpstd

#endif /* __DPSTD_iterator_impl_H */
