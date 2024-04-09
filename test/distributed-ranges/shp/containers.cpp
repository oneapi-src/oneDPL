// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "containers.hpp"

TYPED_TEST_SUITE(DistributedVectorTest, AllocatorTypes);

TYPED_TEST(DistributedVectorTest, is_random_access_range) {
  static_assert(rng::random_access_range<typename TestFixture::LocalVec>);
  static_assert(rng::random_access_range<const typename TestFixture::LocalVec>);
  static_assert(
      rng::random_access_range<const typename TestFixture::LocalVec &>);
  static_assert(rng::random_access_range<typename TestFixture::DistVec>);
  static_assert(rng::random_access_range<const typename TestFixture::DistVec>);
  static_assert(
      rng::random_access_range<const typename TestFixture::DistVec &>);
}

TYPED_TEST(DistributedVectorTest,
           segments_begins_where_its_creating_iterator_points_to) {
  typename TestFixture::DistVec dv(10);
  std::iota(dv.begin(), dv.end(), 20);

  auto second = dv.begin() + 2;
  EXPECT_EQ(second[0], dr::ranges::segments(second)[0][0]);
}

TYPED_TEST(DistributedVectorTest, fill_constructor) {
  EXPECT_TRUE(equal(typename TestFixture::DistVec(10, 1),
                    typename TestFixture::LocalVec(10, 1)));
}

TYPED_TEST(DistributedVectorTest, fill_constructor_large) {
  const typename TestFixture::DistVec v(12345, 17);
  EXPECT_EQ(v[0], 17);
  EXPECT_EQ(v[1111], 17);
  EXPECT_EQ(v[2222], 17);
  EXPECT_EQ(v[3333], 17);
  EXPECT_EQ(v[4444], 17);
  EXPECT_EQ(v[5555], 17);
  EXPECT_EQ(v[6666], 17);
  EXPECT_EQ(v[7777], 17);
  EXPECT_EQ(v[8888], 17);
  EXPECT_EQ(v[9999], 17);
  EXPECT_EQ(v[11111], 17);
  EXPECT_EQ(v[12222], 17);
  EXPECT_EQ(v[12344], 17);
}

TYPED_TEST(DistributedVectorTest, fill_constructor_one_item) {
  EXPECT_TRUE(equal(typename TestFixture::DistVec(1, 77),
                    typename TestFixture::LocalVec(1, 77)));
}
TYPED_TEST(DistributedVectorTest, fill_constructor_no_items) {
  EXPECT_TRUE(
      equal(typename TestFixture::DistVec(), typename TestFixture::LocalVec()));
}

TYPED_TEST(DistributedVectorTest, initializer_list) {
  EXPECT_TRUE(
      equal(typename TestFixture::DistVec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            typename TestFixture::LocalVec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(DistributedVectorTest, initializer_list_constructor_one_item) {
  EXPECT_TRUE(equal(typename TestFixture::DistVec{0},
                    typename TestFixture::LocalVec{0}));
}

TYPED_TEST(DistributedVectorTest, initializer_list_constructor_no_items) {
  EXPECT_TRUE(
      equal(typename TestFixture::DistVec{}, typename TestFixture::LocalVec{}));
}

TYPED_TEST(DistributedVectorTest, Iterator) {
  const int n = 10;
  typename TestFixture::DistVec dv_a(n);
  typename TestFixture::LocalVec v_a(n);

  std::iota(dv_a.begin(), dv_a.end(), 20);
  std::iota(v_a.begin(), v_a.end(), 20);

  EXPECT_TRUE(std::equal(v_a.begin(), v_a.end(), dv_a.begin()));
}

TYPED_TEST(DistributedVectorTest, Resize) {
  std::size_t size = 100;
  typename TestFixture::DistVec dv(size);
  dr::shp::iota(dv.begin(), dv.end(), 20);

  typename TestFixture::LocalVec v(size);
  std::iota(v.begin(), v.end(), 20);

  dv.resize(size * 2);
  v.resize(size * 2);
  EXPECT_EQ(dv, v);

  dv.resize(size);
  v.resize(size);
  EXPECT_EQ(dv, v);

  dv.resize(size * 2, 12);
  v.resize(size * 2, 12);
  EXPECT_EQ(dv, v);
}

template <typename AllocT> class DeviceVectorTest : public testing::Test {
public:
  using DeviceVec = dr::shp::device_vector<typename AllocT::value_type, AllocT>;
};

TYPED_TEST_SUITE(DeviceVectorTest, AllocatorTypes);

TYPED_TEST(DeviceVectorTest, is_remote_contiguous_range) {
  static_assert(dr::remote_contiguous_range<typename TestFixture::DeviceVec>);
}
