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
#ifndef _UTILS_TEST_BASE_H
#define _UTILS_TEST_BASE_H

#include <memory>
#include <vector>

#include "utils_const.h"
#include "utils_sequence.h"
#include "utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

namespace TestUtils
{
////////////////////////////////////////////////////////////////////////////////
/// enum UDTKind - describe test source data kinds
enum class UDTKind
{
    eKeys = 0,  // Keys
    eVals,      // Values
    eRes        // Results
};

template <typename TEnum>
auto
enum_val_to_index(TEnum enumVal)
    -> decltype(static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal))
{
    return static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal);
}

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data - test source data base class
template <typename TestValueType>
struct test_base_data
{
    /// Check that host buffering is required
    /**
     * @return bool - true, if host buffering of test data is required, false - otherwise
     */
    virtual bool host_buffering_required() const = 0;

    /// Get test data
    /**
     * @param UDTKind kind - test data kind
     * @return TestValueType* - pointer to test data.
     *      ATTENTION: return nullptr, if host buffering is required.
     * @see host_buffering_required
     */
    virtual TestValueType* get_data(UDTKind kind) = 0;

    /// Retrieve data
    /**
     * Retrieve data from source test data to host buffer
     *
     * @param UDTKind kind - test data kind
     * @param TestValueType* __it_from - pointer to begin of host buffer
     * @param TestValueType* __it_to - pointer to end of host buffer
     */
    virtual void retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) = 0;

    /// Update data
    /**
     * Update data from host buffer data to test source data
     *
     * @param UDTKind kind - test data kind
     * @param TestValueType* __it_from - pointer to begin of host buffer
     * @param TestValueType* __it_to - pointer to end of host buffer
     */
    virtual void update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) = 0;
};

#if TEST_DPCPP_BACKEND_PRESENT
////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_usm -  test source data for USM shared/device memory
template <sycl::usm::alloc alloc_type, typename TestValueType>
struct test_base_data_usm : test_base_data<TestValueType>
{
    struct Data
    {
        usm_data_transfer<alloc_type, TestValueType> src_data_usm;      // USM data transfer helper
        ::std::size_t                                offset = 0;        // Offset in USM buffer

        template<typename _Size>
        Data(sycl::queue __q, _Size __sz, ::std::size_t __offset)
            : src_data_usm(__q, __sz + __offset)
            , offset(__offset)
        {
        }
        

        TestValueType* get_start_from()
        {
            return src_data_usm.get_data() + offset;
        }

        /// Retrieve data from USM shared/device memory
        /**
         * @param _Iterator __it - start iterator
         * @param TDiff __objects_count - retrieving items couunt
         */
        template<typename _Iterator, typename TDiff>
        void retrieve_data(_Iterator __it, TDiff __objects_count)
        {
            src_data_usm.retrieve_data(__it, offset, __objects_count);
        }

        /// Update data in USM shared/device memory
        /**
         * @param _Iterator __it - start iterator
         * @param TDiff __objects_count - updating items couunt
         */
        template<typename _Iterator, typename TDiff>
        void update_data(_Iterator __it, TDiff __objects_count)
        {
            src_data_usm.update_data(__it, offset, __objects_count);
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 1 item for test1buffer;
                                //  - 2 items for test2buffers;
                                //  - 3 items for test3buffers

    struct InitParam
    {
        ::std::size_t size   = 0;   // Source test data size
        ::std::size_t offset = 0;   // Offset from test data
    };

    test_base_data_usm(sycl::queue __q, ::std::initializer_list<InitParam> init);

    TestValueType* get_start_from(::std::size_t index);

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Retrieve data
    virtual void retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;

    // Update data
    virtual void update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_buffer - test source data for SYCL buffer
template <typename TestValueType>
struct test_base_data_buffer : test_base_data<TestValueType>
{
    using Data = sycl::buffer<TestValueType, 1>;

    struct Data
    {
        using TSourceData = sycl::buffer<TestValueType, 1>;

        TSourceData   src_data_buf;     // SYCL buffer
        ::std::size_t offset = 0;       // Offset in SYCL buffer

        template<typename _Size>
        Data(_Size __sz, ::std::size_t __offset)
            : src_data_buf(sycl::range<1>(__sz + __offset))
            , offset(__offset)
        {
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 1 item for test1buffer;
                                //  - 2 items for test2buffers;
                                //  - 3 items for test3buffers

    test_base_data_buffer(::std::initializer_list<::std::size_t> size_list);

    sycl::buffer<TestValueType, 1>& get_buffer(::std::size_t index);

    auto get_start_from(::std::size_t index)
        -> decltype(oneapi::dpl::begin(data.at(index)));

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Retrieve data
    virtual void retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;

    // Update data
    virtual void update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_sequence -  test source data for sequence (based on std::vector)
template <typename TestValueType>
struct test_base_data_sequence : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = Sequence<TestValueType>;

        TSourceData   src_data_seq;     // Sequence
        ::std::size_t offset = 0;       // Offset in sequence

        Data(::std::size_t size, ::std::size_t __offset)
            : src_data_seq(size)
            , offset(__offset)
        {
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 3 items for test_algo_three_sequences

    test_base_data_sequence(::std::initializer_list<Data> init);

    auto get_start_from(::std::size_t index)
        -> decltype(data.at(index).src_data_seq.begin());

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Retrieve data
    virtual void retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;

    // Update data
    virtual void update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to) override;
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base - base class for new tests
template <typename TestValueType>
struct test_base
{
    test_base_data<TestValueType>& base_data_ref;

    test_base(test_base_data<TestValueType>& _base_data_ref);

    /// Check that host buffering is required
    /**
     * @return bool - true, if host buffering of test data is required, false - otherwise
     */
    bool host_buffering_required() const;

    /// class TestDataTransfer - copy test data from/to source test data storage
    /// to/from local buffer for modification processing.
    template <UDTKind kind, typename Size>
    class TestDataTransfer
    {
    public:

        using HostData = std::vector<TestValueType>;
        using Iterator = typename HostData::iterator;

        /// Constructor
        /**
         * @param test_base& _test_base - reference to test base class
         * @param Size _count - count of objects in source test storage
         */
        TestDataTransfer(test_base& _test_base, Size _count);

        /// Get pointer to internal data buffer
        /**
         * @return TestValueType* - pointer to internal data buffer
         */
        TestValueType* get();

        /// Retrieve data
        /**
         * Method copy data from test source data storage (USM shared/device buffer, SYCL buffer)
         * to internal buffer.
         */
        void retrieve_data();

        /// Update data
        /**
         * Method copy data from internal buffer to test source data storage.
         * 
         * @param Size count - count of items to copy, if 0 - copy all data.
         */
        void update_data(Size count = 0);

    protected:

        test_base& __test_base;     // Test base class ref
        bool       __host_buffering_required = false;
        HostData   __host_buffer;   // Local test data buffer
        const Size __count = 0;     // Count of items in test data
    };
};

namespace
{
void retrieve_data()
{
}

void update_data()
{
}
};

/// Copy data from source test data storage into local buffers
template <typename TTestDataTransfer, typename... Args>
void retrieve_data(TTestDataTransfer& helper, Args&& ...args)
{
    helper.retrieve_data();
    retrieve_data(::std::forward<Args>(args)...);
}

/// Copy data from local buffers into source test data storage
template <typename TTestDataTransfer, typename... Args>
void update_data(TTestDataTransfer& helper, Args&& ...args)
{
    helper.update_data();
    update_data(::std::forward<Args>(args)...);
}

// 1) define class as
//      template <typename TestValueType>
//      struct TestClassName : TestUtils::test_base<TestValueType>
// 2) define class as
//      struct TestClassName
#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST(TestClassName)                                                  \
    template <typename TestValueType>                                               \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST(TestClassName)                                                  \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TestValueType, typename TemplateParams>                      \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TemplateParams>                                              \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)                                                                    \
    TestClassName(test_base_data<TestValueType>& _test_base_data)                                                 \
        : TestUtils::test_base<TestValueType>(_test_base_data)                                                    \
    {                                                                                                             \
    }                                                                                                             \
                                                                                                                  \
    template <UDTKind kind, typename Size>                                                                        \
    using TestDataTransfer = typename TestUtils::test_base<TestValueType>::template TestDataTransfer<kind, Size>; \
                                                                                                                  \
    using UsedValueType = TestValueType;
#else
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData& data)
{
    return TestName(data);
}

template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<!::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData&)
{
    return TestName();
}

//--------------------------------------------------------------------------------------------------------------------//
// Used with algorithms that have two input sequences and one output sequences
template <typename T, typename TestName>
//typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, void>::type
void
test_algo_three_sequences()
{
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        using TestBaseData = test_base_data_sequence<T>;

        TestBaseData test_base_data({ { max_n, inout1_offset },
                                      { max_n, inout2_offset },
                                      { max_n, inout3_offset } });

        // create iterators
        auto inout1_offset_first = test_base_data.get_start_from(0);
        auto inout2_offset_first = test_base_data.get_start_from(1);
        auto inout3_offset_first = test_base_data.get_start_from(2);

        invoke_on_all_host_policies()(create_test_obj<T, TestName>(test_base_data),
                                      inout1_offset_first, inout1_offset_first + n,
                                      inout2_offset_first, inout2_offset_first + n,
                                      inout3_offset_first, inout3_offset_first + n,
                                      n);
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestName>
typename ::std::enable_if<
    ::std::is_base_of<test_base<typename TestName::UsedValueType>, TestName>::value,
    void>::type
test_algo_three_sequences()
{
    test_algo_three_sequences<typename TestName::UsedValueType, TestName>();
}

}; // namespace TestUtils

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type, typename TestValueType>
TestUtils::test_base_data_usm<alloc_type, TestValueType>::test_base_data_usm(
    sycl::queue __q, ::std::initializer_list<InitParam> init)
{
    for (auto& initParam : init)
        data.emplace_back(__q, initParam.size, initParam.offset);
}

//--------------------------------------------------------------------------------------------------------------------//
template <sycl::usm::alloc alloc_type, typename TestValueType>
TestValueType*
TestUtils::test_base_data_usm<alloc_type, TestValueType>::get_start_from(::std::size_t index)
{
    auto& data_item = data.at(index);
    return data_item.get_start_from();
}

//--------------------------------------------------------------------------------------------------------------------//
template <sycl::usm::alloc alloc_type, typename TestValueType>
bool
TestUtils::test_base_data_usm<alloc_type, TestValueType>::host_buffering_required() const
{
    return alloc_type != sycl::usm::alloc::shared;
}

//--------------------------------------------------------------------------------------------------------------------//
template <sycl::usm::alloc alloc_type, typename TestValueType>
TestValueType*
TestUtils::test_base_data_usm<alloc_type, TestValueType>::get_data(UDTKind kind)
{
    if (host_buffering_required())
        return nullptr;

    return get_start_from(enum_val_to_index(kind));
}

//--------------------------------------------------------------------------------------------------------------------//
template <sycl::usm::alloc alloc_type, typename TestValueType>
void
TestUtils::test_base_data_usm<alloc_type, TestValueType>::retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to)
{
    assert(alloc_type == sycl::usm::alloc::device);

    auto& data_item = data.at(enum_val_to_index(kind));
    data_item.retrieve_data(__it_from, __it_to - __it_from);
}

//--------------------------------------------------------------------------------------------------------------------//
template <sycl::usm::alloc alloc_type, typename TestValueType>
void
TestUtils::test_base_data_usm<alloc_type, TestValueType>::update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to)
{
    assert(alloc_type == sycl::usm::alloc::device);

    auto& data_item = data.at(enum_val_to_index(kind));
    data_item.update_data(__it_from, __it_to - __it_from);
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestUtils::test_base_data_buffer<TestValueType>::test_base_data_buffer(
    ::std::initializer_list<::std::size_t> size_list)
{
    for (auto& size : size_list)
        data.emplace_back(sycl::range<1>(size));
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
sycl::buffer<TestValueType, 1>&
TestUtils::test_base_data_buffer<TestValueType>::get_buffer(::std::size_t index)
{
    return data.at(index);
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
auto
TestUtils::test_base_data_buffer<TestValueType>::get_start_from(::std::size_t index)
-> decltype(oneapi::dpl::begin(data.at(index)))
{
    return oneapi::dpl::begin(data.at(index).src_data_buf) + data.at(index).offset;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_buffer<TestValueType>::host_buffering_required() const
{
    return true;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestValueType*
TestUtils::test_base_data_buffer<TestValueType>::get_data(UDTKind /*kind*/)
{
    return nullptr;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_buffer<TestValueType>::retrieve_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to)
{
    auto& data_item = data.at(enum_val_to_index(kind));
    auto acc = data_item.template get_access<sycl::access::mode::read_write>();

    auto __index = 0;
    for (auto __it = __it_from; __it != __it_to; ++__it, ++__index)
    {
        *__it = acc[__index];
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_buffer<TestValueType>::update_data(UDTKind kind, TestValueType* __it_from, TestValueType* __it_to)
{
    auto& data_item = data.at(enum_val_to_index(kind));
    auto acc = data_item.template get_access<sycl::access::mode::read_write>();

    auto __index = 0;
    for (auto __it = __it_from; __it != __it_to; ++__it, ++__index)
    {
        acc[__index] = *__it;
    }
}
#endif //  TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestUtils::test_base_data_sequence<TestValueType>::test_base_data_sequence(::std::initializer_list<Data> init)
    : data(init)
{
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
auto
TestUtils::test_base_data_sequence<TestValueType>::get_start_from(::std::size_t index)
-> decltype(data.at(index).src_data_seq.begin())
{
    return data.at(index).src_data_seq.begin() + data.at(index).offset;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_sequence<TestValueType>::host_buffering_required() const
{
    return false;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestValueType*
TestUtils::test_base_data_sequence<TestValueType>::get_data(UDTKind kind)
{
    auto& data_item = data.at(enum_val_to_index(kind));
    return data_item.src_data_seq.data();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_sequence<TestValueType>::retrieve_data(
    UDTKind /*kind*/, TestValueType* /*__it_from*/, TestValueType* /*__it_to*/)
{
    // No action required here
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_sequence<TestValueType>::update_data(
    UDTKind /*kind*/, TestValueType* /*__it_from*/, TestValueType* /*__it_to*/)
{
    // No action required here
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestUtils::test_base<TestValueType>::test_base(test_base_data<TestValueType>& _base_data_ref)
    : base_data_ref(_base_data_ref)
{
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base<TestValueType>::host_buffering_required() const
{
    return base_data_ref.host_buffering_required();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
template <TestUtils::UDTKind kind, typename Size>
TestUtils::test_base<TestValueType>::TestDataTransfer<kind, Size>::TestDataTransfer(test_base& _test_base, Size _count)
    : __test_base(_test_base)
    , __host_buffering_required(_test_base.host_buffering_required())
    , __host_buffer(__host_buffering_required ? _count : 0)
    , __count(_count)
{
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
template <TestUtils::UDTKind kind, typename Size>
TestValueType*
TestUtils::test_base<TestValueType>::TestDataTransfer<kind, Size>::get()
{
    if (__host_buffering_required)
        return __host_buffer.data();

    return __test_base.base_data_ref.get_data(kind);
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
template <TestUtils::UDTKind kind, typename Size>
void
TestUtils::test_base<TestValueType>::TestDataTransfer<kind, Size>::retrieve_data()
{
    if (__host_buffering_required)
    {
        __test_base.base_data_ref.retrieve_data(kind,
            __host_buffer.data(),
            __host_buffer.data() + __host_buffer.size());
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
template <TestUtils::UDTKind kind, typename Size>
void
TestUtils::test_base<TestValueType>::TestDataTransfer<kind, Size>::update_data(Size count /*= 0*/)
{
    assert(count <= __count);

    if (__host_buffering_required)
    {
        if (count == 0)
            count = __count;

        __test_base.base_data_ref.update_data(kind,
            __host_buffer.data(),
            __host_buffer.data() + count);
    }
}

//--------------------------------------------------------------------------------------------------------------------//

#endif // _UTILS_TEST_BASE_H
