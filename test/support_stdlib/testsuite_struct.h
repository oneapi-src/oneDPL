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

#ifndef _TESTSUITE_STRUCT_H
#define _TESTSUITE_STRUCT_H

struct NoexceptMoveConsClass {
  NoexceptMoveConsClass(NoexceptMoveConsClass &&) noexcept;
  NoexceptMoveConsClass &operator=(NoexceptMoveConsClass &&) = default;
};
struct NoexceptMoveAssignClass {
  NoexceptMoveAssignClass(NoexceptMoveAssignClass &&) = default;
  NoexceptMoveAssignClass &operator=(NoexceptMoveAssignClass &&) noexcept;
};


struct NoexceptMoveConsNoexceptMoveAssignClass {
  NoexceptMoveConsNoexceptMoveAssignClass(
      NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept;

  NoexceptMoveConsNoexceptMoveAssignClass &
  operator=(NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept;
};

#endif
