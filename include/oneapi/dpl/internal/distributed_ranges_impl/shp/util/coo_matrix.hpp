// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/containers/matrix_entry.hpp>
#include <memory>
#include <vector>

namespace dr::shp {

namespace __detail {

template <typename T, typename I, typename Allocator = std::allocator<T>>
class coo_matrix {
public:
  using value_type = dr::shp::matrix_entry<T, I>;
  using scalar_type = T;
  using index_type = I;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using allocator_type = Allocator;

  using key_type = dr::index<I>;
  using map_type = T;

  using backend_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<value_type>;
  using backend_type = std::vector<value_type, backend_allocator_type>;

  using iterator = typename backend_type::iterator;
  using const_iterator = typename backend_type::const_iterator;

  using reference = dr::shp::matrix_ref<T, I>;
  using const_reference = dr::shp::matrix_ref<std::add_const_t<T>, I>;

  using scalar_reference = T &;

  coo_matrix(dr::index<I> shape) : shape_(shape) {}

  dr::index<I> shape() const noexcept { return shape_; }

  size_type size() const noexcept { return tuples_.size(); }

  void reserve(size_type new_cap) { tuples_.reserve(new_cap); }

  iterator begin() noexcept { return tuples_.begin(); }

  const_iterator begin() const noexcept { return tuples_.begin(); }

  iterator end() noexcept { return tuples_.end(); }

  const_iterator end() const noexcept { return tuples_.end(); }

  template <typename InputIt> void insert(InputIt first, InputIt last) {
    for (auto iter = first; iter != last; ++iter) {
      insert(*iter);
    }
  }

  template <typename InputIt> void push_back(InputIt first, InputIt last) {
    for (auto iter = first; iter != last; ++iter) {
      push_back(*iter);
    }
  }

  void push_back(const value_type &value) { tuples_.push_back(value); }

  template <typename InputIt> void assign_tuples(InputIt first, InputIt last) {
    tuples_.assign(first, last);
  }

  std::pair<iterator, bool> insert(value_type &&value) {
    auto &&[insert_index, insert_value] = value;
    for (auto iter = begin(); iter != end(); ++iter) {
      auto &&[index, v] = *iter;
      if (index == insert_index) {
        return {iter, false};
      }
    }
    tuples_.push_back(value);
    return {--tuples_.end(), true};
  }

  std::pair<iterator, bool> insert(const value_type &value) {
    auto &&[insert_index, insert_value] = value;
    for (auto iter = begin(); iter != end(); ++iter) {
      auto &&[index, v] = *iter;
      if (index == insert_index) {
        return {iter, false};
      }
    }
    tuples_.push_back(value);
    return {--tuples_.end(), true};
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(key_type k, M &&obj) {
    for (auto iter = begin(); iter != end(); ++iter) {
      auto &&[index, v] = *iter;
      if (index == k) {
        v = std::forward<M>(obj);
        return {iter, false};
      }
    }
    tuples_.push_back({k, std::forward<M>(obj)});
    return {--tuples_.end(), true};
  }

  iterator find(key_type key) noexcept {
    return std::find_if(begin(), end(), [&](auto &&v) {
      auto &&[i, v_] = v;
      return i == key;
    });
  }

  const_iterator find(key_type key) const noexcept {
    return std::find_if(begin(), end(), [&](auto &&v) {
      auto &&[i, v_] = v;
      return i == key;
    });
  }

  void reshape(dr::index<I> shape) {
    bool all_inside = true;
    for (auto &&[index, v] : *this) {
      auto &&[i, j] = index;
      if (!(i < shape[0] && j < shape[1])) {
        all_inside = false;
        break;
      }
    }

    if (all_inside) {
      shape_ = shape;
      return;
    } else {
      coo_matrix<T, I> new_tuples(shape);
      for (auto &&[index, v] : *this) {
        auto &&[i, j] = index;
        if (i < shape[0] && j < shape[1]) {
          new_tuples.insert({index, v});
        }
      }
      shape_ = shape;
      assign_tuples(new_tuples.begin(), new_tuples.end());
    }
  }

  coo_matrix() = default;
  ~coo_matrix() = default;
  coo_matrix(const coo_matrix &) = default;
  coo_matrix(coo_matrix &&) = default;
  coo_matrix &operator=(const coo_matrix &) = default;
  coo_matrix &operator=(coo_matrix &&) = default;

  std::size_t nbytes() const noexcept {
    return tuples_.size() * sizeof(value_type);
  }

private:
  dr::index<I> shape_;
  backend_type tuples_;
};

} // namespace __detail

} // namespace dr::shp
