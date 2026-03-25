#pragma once
#include <functional>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <algorithm>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 0) {
        if (numThreads == 0) {
            numThreads = std::max(1u, std::thread::hardware_concurrency() - 1);
        }
        m_workers.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            m_workers.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_shutdown = true;
        }
        m_condition.notify_all();
        for (auto& worker : m_workers) {
            if (worker.joinable())
                worker.join();
        }
    }

    template<typename F>
    auto enqueue(F&& task) -> std::future<std::invoke_result_t<F>> {
        using ReturnType = std::invoke_result_t<F>;
        auto taskPtr = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(task));
        std::future<ReturnType> result = taskPtr->get_future();

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_tasks.emplace([taskPtr]() { (*taskPtr)(); });
        }
        m_condition.notify_one();
        return result;
    }

    size_t numThreads() const { return m_workers.size(); }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this] { return m_shutdown || !m_tasks.empty(); });
                if (m_shutdown && m_tasks.empty())
                    return;
                if (!m_tasks.empty()) {
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
            }
            if (task)
                task();
        }
    }

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::atomic<bool> m_shutdown{false};
};
