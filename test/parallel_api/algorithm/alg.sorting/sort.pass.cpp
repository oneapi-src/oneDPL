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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if !defined(_PSTL_TEST_SORT) && !defined(_PSTL_TEST_STABLE_SORT)
#define _PSTL_TEST_SORT
#define _PSTL_TEST_STABLE_SORT
#endif // !defined(_PSTL_TEST_SORT) && !defined(_PSTL_TEST_STABLE_SORT)

// Testing with and without predicate may be useful due to different implementations, e.g. merge-sort and radix-sort
#if !defined(_PSTL_TEST_WITH_PREDICATE) && !defined(_PSTL_TEST_WITHOUT_PREDICATE)
#define _PSTL_TEST_WITH_PREDICATE
#define _PSTL_TEST_WITHOUT_PREDICATE
#endif // !defined(_PSTL_TEST_WITH_PREDICATE) && !defined(_PSTL_TEST_WITHOUT_PREDICATE)

#define _CRT_SECURE_NO_WARNINGS

#include <atomic>

#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

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
#endif // !TEST_DPCPP_BACKEND_PRESENT

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
    ParanoidKey(std::int32_t index, std::int32_t value, TestUtils::OddTag) : value(value), index(index) {}
    ParanoidKey(ParanoidKey&& k) : value(k.value), index(k.index)
    {
        EXPECT_TRUE(k.isConstructed(), "source for move-construction is dead");
// ::std::stable_sort() fails in move semantics on paranoid test before VS2015
#if !defined(_MSC_VER) || _MSC_VER >= 1900
        k.index = Empty;
#endif // !defined(_MSC_VER) || _MSC_VER >= 1900
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
#endif // !defined(_MSC_VER) || _MSC_VER >= 1900
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
    KeyCompare(TestUtils::OddTag) : status(Live) {}
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
#endif // !TEST_DPCPP_BACKEND_PRESENT

static bool
Equal(TestUtils::float32_t x, TestUtils::float32_t y)
{
    return x == y;
}

static bool
Equal(std::int32_t x, std::int32_t y)
{
    return x == y;
}

template <typename T, typename Compare>
bool check_by_predicate(T t1, T t2, Compare c)
{
    return !c(t2, t1);
}

template <typename T>
bool check_by_predicate(T t1, T t2)
{
    return !(t2 < t1);
}

template <typename InputIterator, typename OutputIterator1, typename OutputIterator2, typename Size>
void
copy_data(InputIterator first, OutputIterator1 expected_first, OutputIterator1 expected_last, OutputIterator2 tmp_first,
          Size n)
{
    ::std::copy_n(first, n, expected_first);
    ::std::copy_n(first, n, tmp_first);
}

template <typename ...Params>
void
sort_data(Params&& ...params)
{
    if (Stable)
        ::std::stable_sort(::std::forward<Params>(params)...);
    else
        ::std::sort(::std::forward<Params>(params)...);
}

template <typename OutputIterator1, typename OutputIterator2, typename Size, typename... Compare>
void
check_results(OutputIterator1 expected_first, OutputIterator2 tmp_first, Size n, const char* error_msg, Compare... compare)
{
    auto pred = *tmp_first;
    for (size_t i = 0; i < n; ++i, ++expected_first, ++tmp_first)
    {
        // Check that expected[i] is equal to tmp[i]
        EXPECT_TRUE(Equal(*expected_first, *tmp_first), error_msg);
        // Compare with the previous element using the predicate
        if (i > 1 && i < n-1) // first and last were not sorted
        {
            EXPECT_TRUE(check_by_predicate(pred, *tmp_first, compare...), error_msg);
        }
        pred = *tmp_first;
    }
}

#if TEST_DPCPP_BACKEND_PRESENT
#if _PSTL_SYCL_TEST_USM
template <sycl::usm::alloc alloc_type, typename Policy, typename InputIterator, typename OutputIterator,
            typename OutputIterator2, typename Size, typename ...Compare>
void
test_usm(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
         OutputIterator2 expected_last, InputIterator first, InputIterator /* last */, Size n, Compare... compare)
{
    // Prepare data for sort algorithm
    copy_data(first, expected_first, expected_last, tmp_first, n);
    sort_data(expected_first + 1, expected_last - 1, compare...);

    using _ValueType = typename std::iterator_traits<OutputIterator>::value_type;

    auto queue = exec.queue();

    // allocate USM memory and copying data to USM shared/device memory
    const auto _it_from = tmp_first + 1;
    const auto _it_to = tmp_last - 1;
    TestUtils::usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, _it_from, _it_to);
    auto sortingData = dt_helper.get_data();

    const std::int32_t count0 = KeyCount;

    // Call sort algorithm on prepared data
    const auto _size = _it_to - _it_from;
    sort_data(::std::forward<Policy>(exec), sortingData, sortingData + _size, compare...);

    // check result
    dt_helper.retrieve_data(_it_from);

    check_results(expected_first, tmp_first, n, "wrong result from sort without predicate #2", compare...);

    const std::int32_t count1 = KeyCount;
    EXPECT_EQ(count0, count1, "key cleanup error");
}
#endif // _PSTL_SYCL_TEST_USM
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
          typename... Compare>
oneapi::dpl::__internal::__enable_if_host_execution_policy<Policy, void>
run_test(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
         OutputIterator2 expected_last, InputIterator first, InputIterator /*last*/, Size n, Compare ...compare)
{
    // Prepare data for sort algorithm
    copy_data(first, expected_first, expected_last, tmp_first, n);
    sort_data(expected_first + 1, expected_last - 1, compare...);

    // Call sort algorithm on prepared data
    const std::int32_t count0 = KeyCount;
    sort_data(::std::forward<Policy>(exec), tmp_first + 1, tmp_last - 1, compare...);

    check_results(expected_first, tmp_first, n, "wrong result from sort without predicate #1", compare...);

    const std::int32_t count1 = KeyCount;
    EXPECT_EQ(count0, count1, "key cleanup error");
}

#if TEST_DPCPP_BACKEND_PRESENT
#if _PSTL_SYCL_TEST_USM
template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
          typename... Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<Policy, void>
run_test(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
            OutputIterator2 expected_last, InputIterator first, InputIterator last, Size n, Compare ...compare)
{
    // Run tests for USM shared memory (external testing for USM shared memory, once already covered in sycl_iterator.pass.cpp)
    test_usm<sycl::usm::alloc::shared>(::std::forward<Policy>(exec), tmp_first, tmp_last, expected_first, expected_last,
                                       first, last, n, compare...);
    // Run tests for USM device memory
    test_usm<sycl::usm::alloc::device>(::std::forward<Policy>(exec), tmp_first, tmp_last, expected_first, expected_last,
                                       first, last, n, compare...);
}
#endif // _PSTL_SYCL_TEST_USM
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename T>
struct test_sort_op
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename ...Compare>
    typename ::std::enable_if<TestUtils::is_base_of_iterator_category<::std::random_access_iterator_tag, InputIterator>::value
                          && (TestUtils::can_use_default_less_operator<T>::value || sizeof...(Compare) > 0),
                              void>::type
    operator()(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, InputIterator first, InputIterator last, Size n, Compare ...compare)
    {
        run_test(::std::forward<Policy>(exec), tmp_first, tmp_last, expected_first, expected_last, first, last, n,
                 compare...);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename ...Compare>
    typename ::std::enable_if<!TestUtils::is_base_of_iterator_category<::std::random_access_iterator_tag, InputIterator>::value
                          || !(TestUtils::can_use_default_less_operator<T>::value || sizeof...(Compare) > 0),
                              void>::type
    operator()(Policy&& /* exec */, OutputIterator /* tmp_first */, OutputIterator /* tmp_last */,
               OutputIterator2 /* expected_first */, OutputIterator2 /* expected_last */, InputIterator /* first */,
               InputIterator /* last */, Size /* n */, Compare .../*compare*/)
    {
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
#    if __SYCL_UNNAMED_LAMBDA__
template <typename T, typename Convert>
void
test_default_name_gen(Convert convert, size_t n)
{
    LastIndex = n + 2;
    // The rand()%(2*n+1) encourages generation of some duplicates.
    // Sequence is padded with an extra element at front and back, to detect overwrite bugs.
    TestUtils::Sequence<T> in(n + 2, [=](size_t k) { return convert(k, rand() % (2 * n + 1)); });
    TestUtils::Sequence<T> expected(in);
    TestUtils::Sequence<T> tmp(in);
    auto my_policy = TestUtils::make_device_policy(TestUtils::get_test_queue());
    
    TestUtils::iterator_invoker<::std::random_access_iterator_tag, /*IsReverse*/ ::std::false_type>()(
                my_policy, test_sort_op<T>(), tmp.begin(), tmp.end(), expected.begin(), expected.end(), in.begin(), in.end(),
                    in.size(), ::std::greater<void>());
    TestUtils::iterator_invoker<::std::random_access_iterator_tag, /*IsReverse*/ ::std::false_type>()(
                my_policy, test_sort_op<T>(), tmp.begin(), tmp.end(), expected.begin(), expected.end(), in.begin(), in.end(),
                    in.size(), ::std::less<void>());
                    
}
#    endif //__SYCL_UNNAMED_LAMBDA__
#endif //TEST_DPCPP_BACKEND_PRESENT


template <::std::size_t CallNumber, typename T, typename Compare, typename Convert>
void
test_sort(Compare compare, Convert convert)
{
    for (size_t n = 0; n < 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        LastIndex = n + 2;
        // The rand()%(2*n+1) encourages generation of some duplicates.
        // Sequence is padded with an extra element at front and back, to detect overwrite bugs.
        TestUtils::Sequence<T> in(n + 2, [=](size_t k) { return convert(k, rand() % (2 * n + 1)); });
        TestUtils::Sequence<T> expected(in);
        TestUtils::Sequence<T> tmp(in);
#ifdef _PSTL_TEST_WITHOUT_PREDICATE
        TestUtils::invoke_on_all_policies<CallNumber>()(test_sort_op<T>(), tmp.begin(), tmp.end(), expected.begin(),
                                                        expected.end(), in.begin(), in.end(), in.size());
#endif // _PSTL_TEST_WITHOUT_PREDICATE
#ifdef _PSTL_TEST_WITH_PREDICATE
        TestUtils::invoke_on_all_policies<CallNumber + 1>()(test_sort_op<T>(), tmp.begin(), tmp.end(), expected.begin(),
                                                            expected.end(), in.begin(), in.end(), in.size(), compare);
#endif // _PSTL_TEST_WITH_PREDICATE
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
        ::std::sort(::std::forward<Policy>(exec), iter, iter, TestUtils::non_const(::std::less<T>()));
#endif // _PSTL_TEST_SORT
#ifdef _PSTL_TEST_STABLE_SORT
        ::std::stable_sort(::std::forward<Policy>(exec), iter, iter, TestUtils::non_const(::std::less<T>()));
#endif // _PSTL_TEST_STABLE_SORT
    }
};

struct NonConstCmp
{
    template<typename T, typename U>
    bool operator()(T x, U y)
    {
        return x < y;
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
#endif // #ifndef _PSTL_TEST_SORT
#ifndef _PSTL_TEST_STABLE_SORT
    end = 1;
#endif // _PSTL_TEST_STABLE_SORT
    for (std::int32_t kind = start; kind < end; ++kind)
    {
        Stable = kind != 0;

#if !TEST_DPCPP_BACKEND_PRESENT
        // ParanoidKey has atomic increment in ctors. It's not allowed in kernel
        test_sort<0, ParanoidKey>(KeyCompare(TestUtils::OddTag()),
                                  [](size_t k, size_t val) { return ParanoidKey(k, val, TestUtils::OddTag()); });
#endif // !TEST_DPCPP_BACKEND_PRESENT

#if !ONEDPL_FPGA_DEVICE
        test_sort<10, TestUtils::float32_t>([](TestUtils::float32_t x, TestUtils::float32_t y) { return x < y; },
                                            [](size_t k, size_t val)
                                            { return TestUtils::float32_t(val) * (k % 2 ? 1 : -1); });

        test_sort<20, unsigned char>([](unsigned char x, unsigned char y)
                                     { return x > y; }, // Reversed so accidental use of < will be detected.
                                     [](size_t k, size_t val) { return (unsigned char)val; });

        test_sort<30, unsigned char>(NonConstCmp{}, [](size_t k, size_t val) { return (unsigned char)val; });

#endif // !ONEDPL_FPGA_DEVICE
        test_sort<40, std::int32_t>([](std::int32_t x, std::int32_t y)
                                    { return x > y; }, // Reversed so accidental use of < will be detected.
                                    [](size_t k, size_t val) { return std::int32_t(val) * (k % 2 ? 1 : -1); });

        test_sort<50, std::int16_t>(
            std::greater<std::int16_t>(),
            [](size_t k, size_t val) {
            return std::int16_t(val) * (k % 2 ? 1 : -1); });
    }

#if TEST_DPCPP_BACKEND_PRESENT
#    ifdef _PSTL_TEST_WITH_PREDICATE
#        if __SYCL_UNNAMED_LAMBDA__
    // testing potentially clashing naming for radix sort descending / ascending with minimal timing impact
    test_default_name_gen<std::int32_t>([](size_t, size_t val) { return std::int32_t(val); }, 10);
#        endif //__SYCL_UNNAMED_LAMBDA__
#    endif     //_PSTL_TEST_WITH_PREDICATE
#endif         //TEST_DPCPP_BACKEND_PRESENT

#if !ONEDPL_FPGA_DEVICE
    TestUtils::test_algo_basic_single<int32_t>(TestUtils::run_for_rnd<test_non_const<int32_t>>());
#endif // !ONEDPL_FPGA_DEVICE

    return TestUtils::done();
}
