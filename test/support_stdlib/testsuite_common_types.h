// -*- C++ -*-
// typelist for the C++ library testsuite.
//
// Copyright (C) 2005-2019 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING3.  If not see
// <http://www.gnu.org/licenses/>.
//

#ifndef _TESTSUITE_COMMON_TYPES_H
#define _TESTSUITE_COMMON_TYPES_H 1

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

struct constexpr_comparison_operators {
  template <typename _Tp> void operator()() {
    static_assert(!(_Tp() < _Tp()), "less");
    static_assert(_Tp() <= _Tp(), "leq");
    static_assert(!(_Tp() > _Tp()), "more");
    static_assert(_Tp() >= _Tp(), "meq");
    static_assert(_Tp() == _Tp(), "eq");
    static_assert(!(_Tp() != _Tp()), "ne");
  }
};

// Generator to test default constructor.
struct constexpr_default_constructible {
  template <typename _Tp, bool _IsLitp = std::is_literal_type<_Tp>::value>
  struct _Concept;

  // NB: _Tp must be a literal type.
  // Have to have user-defined default ctor for this to work,
  // or implicit default ctor must initialize all members.
  template <typename _Tp> struct _Concept<_Tp, true> {
    void __constraint() { constexpr _Tp __obj; }
  };

  // Non-literal type, declare local static and verify no
  // constructors generated for _Tp within the translation unit.
  template <typename _Tp> struct _Concept<_Tp, false> {
    void __constraint() { static _Tp __obj; }
  };

  template <typename _Tp> void operator()() {
    _Concept<_Tp> c;
    c.__constraint();
  }
};

struct constexpr_single_value_constructible {
  template <typename _Ttesttype, typename _Tvaluetype,
            bool _IsLitp = std::is_literal_type<_Ttesttype>::value>
  struct _Concept;

  // NB: _Tvaluetype and _Ttesttype must be literal types.
  // Additional constraint on _Tvaluetype needed.  Either assume
  // user-defined default ctor as per
  // constexpr_default_constructible and provide no initializer,
  // provide an initializer, or assume empty-list init-able. Choose
  // the latter.
  template <typename _Ttesttype, typename _Tvaluetype>
  struct _Concept<_Ttesttype, _Tvaluetype, true> {
    void __constraint() {
      constexpr _Tvaluetype __v{};
      constexpr _Ttesttype __obj(__v);
    }
  };
  template <typename _Ttesttype, typename _Tvaluetype>
  struct _Concept<_Ttesttype, _Tvaluetype, false> {
    void __constraint() {
      const _Tvaluetype __v{};
      static _Ttesttype __obj(__v);
    }
  };

  template <typename _Ttesttype, typename _Tvaluetype> void operator()() {
    _Concept<_Ttesttype, _Tvaluetype> c;
    c.__constraint();
  }
};


#endif
