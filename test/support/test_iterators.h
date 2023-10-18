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

#ifndef _TEST_ITERATORS_H
#define _TEST_ITERATORS_H

#include <iterator>
#include <cstddef>

#define DELETE_FUNCTION = delete
#define TEST_CONSTEXPR_CXX14 constexpr

template <class It>
class output_iterator
{
    It it_;

    template <class U>
    friend class output_iterator;

  public:
    typedef ::std::output_iterator_tag iterator_category;
    typedef void value_type;
    typedef typename ::std::iterator_traits<It>::difference_type difference_type;
    typedef typename ::std::iterator_traits<It>::pointer pointer;
    typedef typename ::std::iterator_traits<It>::reference reference;

    It
    base() const
    {
        return it_;
    }

    output_iterator() {}
    explicit output_iterator(It it) : it_(it) {}
    template <class U>
    output_iterator(const output_iterator<U>& u) : it_(u.it_)
    {
    }

    reference operator*() const { return *it_; }

    output_iterator&
    operator++()
    {
        ++it_;
        return *this;
    }
    output_iterator
    operator++(int)
    {
        output_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class It>
class input_iterator
{
    It it_;

    template <class U>
    friend class input_iterator;

  public:
    typedef ::std::input_iterator_tag iterator_category;
    typedef typename ::std::iterator_traits<It>::value_type value_type;
    typedef typename ::std::iterator_traits<It>::difference_type difference_type;
    typedef typename ::std::iterator_traits<It>::pointer pointer;
    typedef typename ::std::iterator_traits<It>::reference reference;

    TEST_CONSTEXPR_CXX14 It
    base() const
    {
        return it_;
    }

    TEST_CONSTEXPR_CXX14 input_iterator() : it_() {}
    explicit TEST_CONSTEXPR_CXX14 input_iterator(It it) : it_(it) {}
    template <class U>
    TEST_CONSTEXPR_CXX14 input_iterator(const input_iterator<U>& u) : it_(u.it_)
    {
    }

    TEST_CONSTEXPR_CXX14 reference operator*() const { return *it_; }
    TEST_CONSTEXPR_CXX14 pointer operator->() const 
    {
        if constexpr (::std::is_pointer_v<It>)
        {
            return it_;
        }
        else
        {
            return it_.operator->();
        }
    }

    TEST_CONSTEXPR_CXX14 input_iterator&
    operator++()
    {
        ++it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 input_iterator
    operator++(int)
    {
        input_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    friend TEST_CONSTEXPR_CXX14 bool
    operator==(const input_iterator& x, const input_iterator& y)
    {
        return x.it_ == y.it_;
    }
    friend TEST_CONSTEXPR_CXX14 bool
    operator!=(const input_iterator& x, const input_iterator& y)
    {
        return !(x == y);
    }

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator==(const input_iterator<T>& x, const input_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator!=(const input_iterator<T>& x, const input_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class forward_iterator
{
    It it_;

    template <class U>
    friend class forward_iterator;

  public:
    typedef ::std::forward_iterator_tag iterator_category;
    typedef typename ::std::iterator_traits<It>::value_type value_type;
    typedef typename ::std::iterator_traits<It>::difference_type difference_type;
    typedef typename ::std::iterator_traits<It>::pointer pointer;
    typedef typename ::std::iterator_traits<It>::reference reference;

    TEST_CONSTEXPR_CXX14 It
    base() const
    {
        return it_;
    }

    TEST_CONSTEXPR_CXX14 forward_iterator() : it_() {}
    explicit TEST_CONSTEXPR_CXX14 forward_iterator(It it) : it_(it) {}
    template <class U>
    TEST_CONSTEXPR_CXX14 forward_iterator(const forward_iterator<U>& u) : it_(u.it_)
    {
    }

    TEST_CONSTEXPR_CXX14 reference operator*() const { return *it_; }
    TEST_CONSTEXPR_CXX14 pointer operator->() const 
    {
        if constexpr (::std::is_pointer_v<It>)
        {
            return it_;
        }
        else
        {
            return it_.operator->();
        }
    }

    TEST_CONSTEXPR_CXX14 forward_iterator&
    operator++()
    {
        ++it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 forward_iterator
    operator++(int)
    {
        forward_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    friend TEST_CONSTEXPR_CXX14 bool
    operator==(const forward_iterator& x, const forward_iterator& y)
    {
        return x.it_ == y.it_;
    }
    friend TEST_CONSTEXPR_CXX14 bool
    operator!=(const forward_iterator& x, const forward_iterator& y)
    {
        return !(x == y);
    }

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator==(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator!=(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class bidirectional_iterator
{
    It it_;

    template <class U>
    friend class bidirectional_iterator;

  public:
    typedef ::std::bidirectional_iterator_tag iterator_category;
    typedef typename ::std::iterator_traits<It>::value_type value_type;
    typedef typename ::std::iterator_traits<It>::difference_type difference_type;
    typedef typename ::std::iterator_traits<It>::pointer pointer;
    typedef typename ::std::iterator_traits<It>::reference reference;

    TEST_CONSTEXPR_CXX14 It
    base() const
    {
        return it_;
    }

    TEST_CONSTEXPR_CXX14 bidirectional_iterator() : it_() {}
    explicit TEST_CONSTEXPR_CXX14 bidirectional_iterator(It it) : it_(it) {}
    template <class U>
    TEST_CONSTEXPR_CXX14 bidirectional_iterator(const bidirectional_iterator<U>& u) : it_(u.it_)
    {
    }

    TEST_CONSTEXPR_CXX14 reference operator*() const { return *it_; }
    TEST_CONSTEXPR_CXX14 pointer operator->() const 
    {
        if constexpr (::std::is_pointer_v<It>)
        {
            return it_;
        }
        else
        {
            return it_.operator->();
        }
    }

    TEST_CONSTEXPR_CXX14 bidirectional_iterator&
    operator++()
    {
        ++it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 bidirectional_iterator
    operator++(int)
    {
        bidirectional_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    TEST_CONSTEXPR_CXX14 bidirectional_iterator&
    operator--()
    {
        --it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 bidirectional_iterator
    operator--(int)
    {
        bidirectional_iterator tmp(*this);
        --(*this);
        return tmp;
    }

    friend TEST_CONSTEXPR_CXX14 bool operator==(const bidirectional_iterator& x, const bidirectional_iterator& y)
        {return x.it_ == y.it_;}
    friend TEST_CONSTEXPR_CXX14 bool operator!=(const bidirectional_iterator& x, const bidirectional_iterator& y)
        {return !(x == y);}

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator==(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator!=(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class random_access_iterator
{
    It it_;

    template <class U>
    friend class random_access_iterator;

  public:
    typedef ::std::random_access_iterator_tag iterator_category;
    typedef typename ::std::iterator_traits<It>::value_type value_type;
    typedef typename ::std::iterator_traits<It>::difference_type difference_type;
    typedef typename ::std::iterator_traits<It>::pointer pointer;
    typedef typename ::std::iterator_traits<It>::reference reference;

    TEST_CONSTEXPR_CXX14 It
    base() const
    {
        return it_;
    }

    TEST_CONSTEXPR_CXX14 random_access_iterator() : it_() {}
    explicit TEST_CONSTEXPR_CXX14 random_access_iterator(It it) : it_(it) {}
    template <class U>
    TEST_CONSTEXPR_CXX14 random_access_iterator(const random_access_iterator<U>& u) : it_(u.it_)
    {
    }

    TEST_CONSTEXPR_CXX14 reference operator*() const { return *it_; }
    TEST_CONSTEXPR_CXX14 pointer operator->() const 
    {
        if constexpr (::std::is_pointer_v<It>)
        {
            return it_;
        }
        else
        {
            return it_.operator->();
        }
    }

    TEST_CONSTEXPR_CXX14 random_access_iterator&
    operator++()
    {
        ++it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 random_access_iterator
    operator++(int)
    {
        random_access_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    TEST_CONSTEXPR_CXX14 random_access_iterator&
    operator--()
    {
        --it_;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 random_access_iterator
    operator--(int)
    {
        random_access_iterator tmp(*this);
        --(*this);
        return tmp;
    }

    TEST_CONSTEXPR_CXX14 random_access_iterator&
    operator+=(difference_type n)
    {
        it_ += n;
        return *this;
    }
    TEST_CONSTEXPR_CXX14 random_access_iterator
    operator+(difference_type n) const
    {
        random_access_iterator tmp(*this);
        tmp += n;
        return tmp;
    }
    friend TEST_CONSTEXPR_CXX14 random_access_iterator
    operator+(difference_type n, random_access_iterator x)
    {
        x += n;
        return x;
    }
    TEST_CONSTEXPR_CXX14 random_access_iterator&
    operator-=(difference_type n)
    {
        return *this += -n;
    }
    TEST_CONSTEXPR_CXX14 random_access_iterator
    operator-(difference_type n) const
    {
        random_access_iterator tmp(*this);
        tmp -= n;
        return tmp;
    }

    TEST_CONSTEXPR_CXX14 reference operator[](difference_type n) const { return it_[n]; }

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator==(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator!=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x == y);
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator<(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() < y.base();
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator<=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(y < x);
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator>(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return y < x;
}

template <class T, class U>
inline bool TEST_CONSTEXPR_CXX14
operator>=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x < y);
}

template <class T, class U>
inline typename ::std::iterator_traits<T>::difference_type TEST_CONSTEXPR_CXX14
operator-(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() - y.base();
}

template <class Iter>
inline TEST_CONSTEXPR_CXX14 Iter
base(output_iterator<Iter> i)
{
    return i.base();
}

template <class Iter>
inline TEST_CONSTEXPR_CXX14 Iter
base(input_iterator<Iter> i)
{
    return i.base();
}

template <class Iter>
inline TEST_CONSTEXPR_CXX14 Iter
base(forward_iterator<Iter> i)
{
    return i.base();
}

template <class Iter>
inline TEST_CONSTEXPR_CXX14 Iter
base(bidirectional_iterator<Iter> i)
{
    return i.base();
}

template <class Iter>
inline TEST_CONSTEXPR_CXX14 Iter
base(random_access_iterator<Iter> i)
{
    return i.base();
}

template <class Iter> // everything else
inline TEST_CONSTEXPR_CXX14 Iter
base(Iter i)
{
    return i;
}

#undef DELETE_FUNCTION

#endif // _TEST_ITERATORS_H
