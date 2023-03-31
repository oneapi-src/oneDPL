// -*- C++ -*-
//===---------------------------------------------------------------------===//
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

#ifndef _ONEDPL_CONCURRENT_QUEUE_H
#define _ONEDPL_CONCURRENT_QUEUE_H

#include <queue>
#include <mutex>
#include <list>
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
            queue_.pop();
        }

        void push(T item)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            queue_.push(item);
            mlock.unlock();
            cond_.notify_one();
        }

        void pop_all(std::list<T>& item_list)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            while (queue_.empty())
            {
                cond_.wait(mlock);
            }
            while(!queue_.empty()){
                auto item = queue_.front();
                queue_.pop();
                item_list.push_back(item);

            }
        }

        bool is_empty(){
            return queue_.empty();
        }

        concurrent_queue() = default;
        concurrent_queue(const concurrent_queue& q) : queue_(q.queue_) {};
        concurrent_queue& operator=(const concurrent_queue& q) {q=q.queue_; return *this;}

    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };
} // namespace util
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_CONCURRENT_QUEUE_H */
