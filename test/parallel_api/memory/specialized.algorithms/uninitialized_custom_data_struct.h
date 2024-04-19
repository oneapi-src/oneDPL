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

struct NonAssignableCopyConstructible
{
    NonAssignableCopyConstructible() = default;
    NonAssignableCopyConstructible(NonAssignableCopyConstructible&&) = delete;
    NonAssignableCopyConstructible(const NonAssignableCopyConstructible&) = default;

    NonAssignableCopyConstructible&
    operator=(const NonAssignableCopyConstructible&) = delete;
    NonAssignableCopyConstructible&
    operator=(NonAssignableCopyConstructible&&) = delete;

    bool
    operator==(const NonAssignableCopyConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const NonAssignableCopyConstructible& other) const
    {
        return false;
    }
};
static_assert(std::is_copy_constructible_v<NonAssignableCopyConstructible>);

struct NonAssignableMoveConstructible
{
    NonAssignableMoveConstructible() = default;
    NonAssignableMoveConstructible(NonAssignableMoveConstructible&&) = default;
    NonAssignableMoveConstructible(const NonAssignableMoveConstructible&) = delete;

    NonAssignableMoveConstructible&
    operator=(const NonAssignableMoveConstructible&) = delete;
    NonAssignableMoveConstructible&
    operator=(NonAssignableMoveConstructible&&) = delete;

    bool
    operator==(const NonAssignableMoveConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const NonAssignableMoveConstructible& other) const
    {
        return false;
    }
};
static_assert(std::is_move_constructible_v<NonAssignableMoveConstructible>);

#endif // _UNINITIALIZED_CUSTOM_DATA_STRUCT_H
