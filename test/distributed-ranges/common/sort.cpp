// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// TODO: add sort tests with ISHMEM, currently doesn't compile
using T = int;
using DV = xhp::distributed_vector<T>;
using LV = std::vector<T>;

void test_sort(LV v, auto func) {
  auto size = v.size();
  DV d_v(size);

  for (std::size_t idx = 0; idx < size; idx++) {
    d_v[idx] = v[idx];
  }
  barrier();

  std::sort(v.begin(), v.end(), func);
  xhp::sort(d_v, func);

  EXPECT_TRUE(equal(v, d_v));
}

void test_sort2s(LV v) {
  test_sort(v, std::less<T>());
  test_sort(v, std::greater<T>());
}

void test_sort_randomvec(std::size_t size, std::size_t bound = 100) {
  LV l_v = generate_random<T>(size, bound);
  test_sort2s(l_v);
}

TEST(Sort, Random_1) { test_sort_randomvec(1); }

TEST(Sort, Random_CommSize_m1) { test_sort_randomvec(comm_size - 1); }

TEST(Sort, Random_CommSize_m1_sq) {
  test_sort_randomvec((comm_size - 1) * (comm_size - 1));
}

TEST(Sort, Random_dist_small) { test_sort_randomvec(17); }

TEST(Sort, Random_dist_med) { test_sort_randomvec(123); }

TEST(Sort, AllSame) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST(Sort, AllSameButOneMid) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1});
}

TEST(Sort, AllSameButOneEnd) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9});
}

TEST(Sort, AllSameButOneSmaller) {
  test_sort2s({5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, AllSameButOneBigger) {
  test_sort2s({5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, AllSameButOneSBeg) {
  test_sort2s({5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}
TEST(Sort, AllSameButOneBBeg) {
  test_sort2s({5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, MostSame) { test_sort2s({1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 9, 1}); }

TEST(Sort, Pyramid) { test_sort2s({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}); }

TEST(Sort, RevPyramid) { test_sort2s({6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6}); }

TEST(Sort, Wave) { test_sort2s({1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}); }

TEST(Sort, LongSorted) {
  LV v(100000);
  rng::iota(v, 1);
  test_sort2s(v);

  rng::reverse(v);
  test_sort2s(v);
}
