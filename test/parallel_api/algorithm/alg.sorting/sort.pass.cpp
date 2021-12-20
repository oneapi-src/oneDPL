// -*- C++ -*-
//===-- sort.pass.cpp -----------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"
#endif


#if !defined(_PSTL_TEST_SORT) && !defined(_PSTL_TEST_STABLE_SORT)
#define _PSTL_TEST_SORT
#define _PSTL_TEST_STABLE_SORT
#endif

// Testing with and without predicate may be useful due to different implementations, e.g. merge-sort and radix-sort
#if !defined(_PSTL_TEST_WITH_PREDICATE) && !defined(_PSTL_TEST_WITHOUT_PREDICATE)
#define _PSTL_TEST_WITH_PREDICATE
#define _PSTL_TEST_WITHOUT_PREDICATE
#endif

using namespace TestUtils;
#define _CRT_SECURE_NO_WARNINGS

#include <atomic>

static bool Stable;

//! Number of extant keys
static ::std::atomic<std::int32_t> KeyCount;

//! One more than highest index in array to be sorted.
static std::uint32_t LastIndex;

//! Keeping Equal() static and a friend of ParanoidKey class (C++, paragraphs 3.5/7.1.1)
class ParanoidKey;
#if !TEST_DPCPP_BACKEND_PRESENT
static bool
Equal(const ParanoidKey& x, const ParanoidKey& y);
#endif

//! A key to be sorted, with lots of checking.
class ParanoidKey
{
    //! Value used by comparator
    std::int32_t value;
    //! Original position or special value (Empty or Dead)
    std::int32_t index;
    //! Special value used to mark object without a comparable value, e.g. after being moved from.
    static const std::int32_t Empty = -1;
    //! Special value used to mark destroyed objects.
    static const std::int32_t Dead = -2;
    // True if key object has comparable value
    bool
    isLive() const
    {
        return (std::uint32_t)(index) < LastIndex;
    }
    // True if key object has been constructed.
    bool
    isConstructed() const
    {
        return isLive() || index == Empty;
    }

  public:
    ParanoidKey()
    {
        ++KeyCount;
        index = Empty;
        value = Empty;
    }
    ParanoidKey(const ParanoidKey& k) : value(k.value), index(k.index)
    {
        EXPECT_TRUE(k.isLive(), "source for copy-constructor is dead");
        ++KeyCount;
    }
    ~ParanoidKey()
    {
        EXPECT_TRUE(isConstructed(), "double destruction");
        index = Dead;
        --KeyCount;
    }
    ParanoidKey&
    operator=(const ParanoidKey& k)
    {
        EXPECT_TRUE(k.isLive(), "source for copy-assignment is dead");
        EXPECT_TRUE(isConstructed(), "destination for copy-assignment is dead");
        value = k.value;
        index = k.index;
        return *this;
    }
    ParanoidKey(std::int32_t index, std::int32_t value, OddTag) : value(value), index(index) {}
    ParanoidKey(ParanoidKey&& k) : value(k.value), index(k.index)
    {
        EXPECT_TRUE(k.isConstructed(), "source for move-construction is dead");
// ::std::stable_sort() fails in move semantics on paranoid test before VS2015
#if !defined(_MSC_VER) || _MSC_VER >= 1900
        k.index = Empty;
#endif
        ++KeyCount;
    }
    ParanoidKey&
    operator=(ParanoidKey&& k)
    {
        EXPECT_TRUE(k.isConstructed(), "source for move-assignment is dead");
        EXPECT_TRUE(isConstructed(), "destination for move-assignment is dead");
        value = k.value;
        index = k.index;
// ::std::stable_sort() fails in move semantics on paranoid test before VS2015
#if !defined(_MSC_VER) || _MSC_VER >= 1900
        k.index = Empty;
#endif
        return *this;
    }
    friend class KeyCompare;
    friend bool
    Equal(const ParanoidKey& x, const ParanoidKey& y);
};

class KeyCompare
{
    enum statusType
    {
        //! Special value used to mark defined object.
        Live = 0xabcd,
        //! Special value used to mark destroyed objects.
        Dead = -1
    } status;

  public:
    KeyCompare(OddTag) : status(Live) {}
    ~KeyCompare() { status = Dead; }
    bool
    operator()(const ParanoidKey& j, const ParanoidKey& k) const
    {
        EXPECT_TRUE(status == Live, "key comparison object not defined");
        EXPECT_TRUE(j.isLive(), "first key to operator() is not live");
        EXPECT_TRUE(k.isLive(), "second key to operator() is not live");
        return j.value < k.value;
    }
};

// Equal is equality comparison used for checking result of sort against expected result.
#if !TEST_DPCPP_BACKEND_PRESENT
static bool
Equal(const ParanoidKey& x, const ParanoidKey& y)
{
    return (x.value == y.value && !Stable) || (x.index == y.index);
}
#endif

static bool
Equal(float32_t x, float32_t y)
{
    return x == y;
}

static bool
Equal(std::int32_t x, std::int32_t y)
{
    return x == y;
}

struct run_sort
{
    template <typename ...Args>
    void operator()(Args... args)
    {
       if (Stable)
            std::stable_sort(args...);
        else
            std::sort(args...);
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
struct run_sort_device
{
    template <typename Policy, typename RandomAccessIterator, typename ...Compare>
    void operator()(Policy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare... compare)
    {
        // allocate USM device memory and copying data to USM device memory
       using value_type = typename ::std::iterator_traits<RandomAccessIterator>::value_type;
       usm_data_transfer<sycl::usm::alloc::device, value_type> device_data(exec.queue(), first, last);

       //run tested algorithm
       run_sort()(exec, device_data.begin(), device_data.end(), compare...);

       //copy data back to the host
       device_data.retrieve_data(first);
    }
};
#endif //TEST_DPCPP_BACKEND_PRESENT 

template <typename T, typename Algo>
struct sort_body
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename ...Compare>
    typename ::std::enable_if<is_base_of_iterator_category<::std::random_access_iterator_tag, InputIterator>::value,
                            void>::type
    operator()(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, InputIterator first, InputIterator /* last */, Size n, Compare... compare)
    {
        //keep the original data
        std::copy_n(first, n, tmp_first);        

	//run a tested algorithm
        std::int32_t count0 = KeyCount;
        Algo()(exec, tmp_first + 1, tmp_last - 1, compare...);
        
        //generate exam(expected) and check result
        std::copy_n(first, n, expected_first);
        run_sort()(expected_first + 1, expected_last - 1, compare...);
        for (size_t i = 0; i < n; ++i, ++expected_first, ++tmp_first)
        {
            // Check that expected[i] is equal to tmp[i]
            EXPECT_TRUE(Equal(*expected_first, *tmp_first), "wrong result from sort without predicate");
        }
        std::int32_t count1 = KeyCount;
        EXPECT_EQ(count0, count1, "key cleanup error");
    }
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename ...Compare>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, InputIterator>::value,
                            void>::type
    operator()(Policy&& /* exec */, OutputIterator /* tmp_first */, OutputIterator /* tmp_last */, OutputIterator2 /* expected_first */,
               OutputIterator2 /* expected_last */, InputIterator /* first */, InputIterator /* last */, Size /* n */, Compare... /* compare */)
    {
    }
};

template <bool Condition, typename F, typename ...Args>
typename std::enable_if<Condition, void>::type
call_if(F f, Args... args) { f(args...); }

template <bool Condition, typename F, typename ...Args>
typename std::enable_if<!Condition, void>::type
call_if(F f, Args... args) {}

template <typename T, typename Compare, typename Convert>
void
test_sort(Compare compare, Convert convert)
{
    for (size_t n = 0; n < 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        LastIndex = n + 2;
        // The rand()%(2*n+1) encourages generation of some duplicates.
        // Sequence is padded with an extra element at front and back, to detect overwrite bugs.
        Sequence<T> in(n + 2, [=](size_t k) { return convert(k, rand() % (2 * n + 1)); });
        Sequence<T> expected(in);
        Sequence<T> tmp(in);
#ifdef _PSTL_TEST_WITHOUT_PREDICATE
        call_if<can_use_default_less_operator<T>::value>(
            invoke_on_all_policies<0>{}, sort_body<T, run_sort>(), tmp.begin(), tmp.end(), expected.begin(),
                                         expected.end(), in.begin(), in.end(), in.size());
#endif
#ifdef _PSTL_TEST_WITH_PREDICATE
        invoke_on_all_policies<1>()(sort_body<T, run_sort>(), tmp.begin(), tmp.end(), expected.begin(),
                                    expected.end(), in.begin(), in.end(), in.size(), compare);
#endif

#if TEST_DPCPP_BACKEND_PRESENT && _PSTL_SYCL_TEST_USM
        invoke_on_all_policies<2>()(sort_body<T, run_sort_device>(), tmp.begin(), tmp.end(), expected.begin(),
                                    expected.end(), in.begin(), in.end(), in.size(), compare);
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
#ifdef _PSTL_TEST_SORT
        sort(exec, iter, iter, non_const(::std::less<T>()));
#endif
#ifdef _PSTL_TEST_STABLE_SORT
        stable_sort(exec, iter, iter, non_const(::std::less<T>()));
#endif
    }
};

int
main()
{
    ::std::srand(42);
    std::int32_t start = 0;
    std::int32_t end = 2;
#ifndef _PSTL_TEST_SORT
    start = 1;
#endif
#ifndef _PSTL_TEST_STABLE_SORT
    end = 1;
#endif
    for (std::int32_t kind = start; kind < end; ++kind)
    {
        Stable = kind != 0;

#if !TEST_DPCPP_BACKEND_PRESENT
        // ParanoidKey has atomic increment in ctors. It's not allowed in kernel
        test_sort<ParanoidKey>(KeyCompare(OddTag()),
                               [](size_t k, size_t val) { return ParanoidKey(k, val, OddTag()); });
#endif

#if !ONEDPL_FPGA_DEVICE
        test_sort<float32_t>([](float32_t x, float32_t y) { return x < y; },
                             [](size_t, size_t val) { return float32_t(val); });
#endif
        test_sort<std::int32_t>(
            [](std::int32_t x, std::int32_t y) { return x > y; }, // Reversed so accidental use of < will be detected.
            [](size_t, size_t val) { return std::int32_t(val); });
    }

#if !ONEDPL_FPGA_DEVICE
    test_algo_basic_single<std::int32_t>(run_for_rnd<test_non_const<std::int32_t>>());
#endif

    return done();
}
