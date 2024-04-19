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

#ifndef _UNINITIALIZED_CUSTOM_DATA_STRUCT_H
#define _UNINITIALIZED_CUSTOM_DATA_STRUCT_H

#include <type_traits>

struct NonAssignableTriviallyCopyConstructible
{
    NonAssignableTriviallyCopyConstructible() = default;
    NonAssignableTriviallyCopyConstructible(NonAssignableTriviallyCopyConstructible&&) = delete;
    NonAssignableTriviallyCopyConstructible(const NonAssignableTriviallyCopyConstructible&) = default;

    NonAssignableTriviallyCopyConstructible&
    operator=(const NonAssignableTriviallyCopyConstructible&) = delete;
    NonAssignableTriviallyCopyConstructible&
    operator=(NonAssignableTriviallyCopyConstructible&&) = delete;

    bool
    operator==(const NonAssignableTriviallyCopyConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const NonAssignableTriviallyCopyConstructible& other) const
    {
        return false;
    }
};
static_assert(std::is_copy_constructible_v<NonAssignableTriviallyCopyConstructible>);

struct NonAssignableTriviallyMoveConstructible
{
    NonAssignableTriviallyMoveConstructible() = default;
    NonAssignableTriviallyMoveConstructible(NonAssignableTriviallyMoveConstructible&&) = default;
    NonAssignableTriviallyMoveConstructible(const NonAssignableTriviallyMoveConstructible&) = delete;

    NonAssignableTriviallyMoveConstructible&
    operator=(const NonAssignableTriviallyMoveConstructible&) = delete;
    NonAssignableTriviallyMoveConstructible&
    operator=(NonAssignableTriviallyMoveConstructible&&) = delete;

    bool
    operator==(const NonAssignableTriviallyMoveConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const NonAssignableTriviallyMoveConstructible& other) const
    {
        return false;
    }
};
static_assert(std::is_move_constructible_v<NonAssignableTriviallyMoveConstructible>);

#endif // _UNINITIALIZED_CUSTOM_DATA_STRUCT_H
