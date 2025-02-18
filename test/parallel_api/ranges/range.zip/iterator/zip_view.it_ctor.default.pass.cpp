//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() = default;

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <ranges>
#include <tuple>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

struct PODIter {
  int i; // deliberately uninitialised

  using iterator_category = std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = std::intptr_t;

  constexpr int operator*() const { return i; }

  constexpr PODIter& operator++() { return *this; }
  constexpr void operator++(int) {}

  friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
};

struct IterDefaultCtrView : std::ranges::view_base {
  PODIter begin() const;
  PODIter end() const;
};

struct IterNoDefaultCtrView : std::ranges::view_base {
  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class... Views>
using zip_iter = std::ranges::iterator_t<dpl_ranges::zip_view<Views...>>;

static_assert(!std::default_initializable<zip_iter<IterNoDefaultCtrView>>);
static_assert(!std::default_initializable<zip_iter<IterNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(!std::default_initializable<zip_iter<IterNoDefaultCtrView, IterNoDefaultCtrView>>);
static_assert(std::default_initializable<zip_iter<IterDefaultCtrView>>);
static_assert(std::default_initializable<zip_iter<IterDefaultCtrView, IterDefaultCtrView>>);

void test() {
  using ZipIter = zip_iter<IterDefaultCtrView>;
  {
    ZipIter iter;
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }

  {
    ZipIter iter = {};
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }
}

#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
