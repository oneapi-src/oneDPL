// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_CONCURRENT_QUEUE_H
#define _ONEDPL_CONCURRENT_QUEUE_H

#include <queue>
#include <mutex>
#include <deque>
#include <condition_variable>

namespace oneapi {
namespace dpl{
namespace experimental{
namespace util{
    template <typename T>
    class concurrent_queue
    {
    public:

        void pop(T& item)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            while (queue_.empty())
            {
                cond_.wait(mlock);
            }
            item = queue_.front();
            queue_.pop_front();
        }

        void push(T item)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            queue_.push_back(item);
            mlock.unlock();
            cond_.notify_one();
        }

        void pop_all(std::list<T>& item_list)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            item_list.insert(item_list.begin(),std::make_move_iterator(queue_.begin()), std::make_move_iterator(queue_.end()));
        }

        bool empty(){
            return queue_.empty();
        }

        concurrent_queue() = default;
        concurrent_queue(const concurrent_queue& q) = delete;
        concurrent_queue& operator=(const concurrent_queue& q) = delete;

    private:
        std::deque<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };
} // namespace util
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_CONCURRENT_QUEUE_H */
