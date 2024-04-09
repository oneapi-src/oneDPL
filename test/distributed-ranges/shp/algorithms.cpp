// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::device_allocator<T>>;
using V = std::vector<T>;

TEST(ShpTests, InclusiveScan_aligned) {
  std::size_t n = 100;

  // With execution Policy
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(shp::par_unseq, v, v);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), v.begin());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }

  // Without execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(v, v);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(v.begin(), v.end(), v.begin());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }
}

TEST(ShpTests, DISABLED_InclusiveScan_nonaligned) {
  std::size_t n = 100;

  // With execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    shp::distributed_vector<int, shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(shp::par_unseq, v, o, std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    shp::inclusive_scan(shp::par_unseq, v, o, std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                            std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                            std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }

  // Without execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    shp::distributed_vector<int, shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(v, o, std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    shp::inclusive_scan(v, o, std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    shp::inclusive_scan(v.begin(), v.end(), o.begin(), std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    shp::inclusive_scan(v.begin(), v.end(), o.begin(), std::multiplies<>(),
                            12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }
}

TEST(ShpTests, ExclusiveScan_aligned) {
  std::size_t n = 100;

  // With execution Policy
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(shp::par_unseq, v, v, int(0));

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(shp::par_unseq, v.begin(), v.end(), v.begin(),
                            int(0));

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }

  // Without execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(v, v, int(0));

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(v.begin(), v.end(), v.begin(), int(0));

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }
}

TEST(ShpTests, DISABLED_ExclusiveScan_nonaligned) {
  std::size_t n = 100;

  // With execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    shp::distributed_vector<int, shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(shp::par_unseq, v, o, int(0), std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), 12,
                        std::multiplies<>());
    shp::exclusive_scan(shp::par_unseq, v, o, 12, std::multiplies<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                            int(0), std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(12),
                        std::multiplies<>());
    shp::exclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                            int(12), std::multiplies<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }

  // Without execution policies
  {
    shp::distributed_vector<int, shp::device_allocator<int>> v(n);
    shp::distributed_vector<int, shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(12));
    shp::exclusive_scan(v, o, int(12), std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), 12,
                        std::multiplies<>());
    shp::exclusive_scan(v, o, 12, std::multiplies<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), int(0));
    shp::exclusive_scan(v.begin(), v.end(), o.begin(), int(0),
                            std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    shp::copy(lv.begin(), lv.end(), v.begin());

    std::exclusive_scan(lv.begin(), lv.end(), lv.begin(), 12,
                        std::multiplies<>());
    shp::exclusive_scan(v.begin(), v.end(), o.begin(), 12,
                            std::multiplies<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }
}

TEST(ShpTests, Sort) {
  std::vector<std::size_t> sizes = {1, 2, 4, 100};

  for (std::size_t n : sizes) {
    std::vector<T> l_v = generate_random<T>(n, 100);

    shp::distributed_vector<T> d_v(n);

    shp::copy(l_v.begin(), l_v.end(), d_v.begin());

    std::sort(l_v.begin(), l_v.end());
    shp::sort(d_v);

    std::vector<T> d_v_l(n);

    shp::copy(d_v.begin(), d_v.end(), d_v_l.begin());

    for (std::size_t i = 0; i < l_v.size(); i++) {
      EXPECT_EQ(l_v[i], d_v_l[i]);
    }
  }
}
