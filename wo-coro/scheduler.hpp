#pragma once
#include <coroutine>
#include <list>
#include <queue>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace cudaWoCoro { // begin of namespace cudaWoCoro =========================

struct Task {
  std::function<void()> work;

  template <typename C>
  Task(C&& c): work{std::forward<C>(c)} {
  }
};


class Scheduler {

  public: 

    Scheduler(size_t num_threads = std::thread::hardware_concurrency());

    template <typename C>
    void emplace(C&& c);
    auto suspend(); 
    void schedule();
    void wait();


  private:

    std::vector<std::unique_ptr<Task>> _tasks;
    std::queue<Task*> _pending_tasks;

    std::vector<std::thread> _workers;
    std::mutex _mtx;
    std::condition_variable _cv;
    bool _stop{false};
    std::atomic<size_t> _finished{0};

    void _enqueue(Task* task);
    void _process(Task* task);
};

Scheduler::Scheduler(size_t num_threads) {
  _workers.reserve(num_threads);

  for(size_t t = 0; t < num_threads; ++t) {
    _workers.emplace_back([this]() {
        while(true) {
          Task* task{nullptr};
          {
            std::unique_lock<std::mutex> lock(_mtx);
            _cv.wait(lock, [this]{ return _stop || (!_pending_tasks.empty()); });
            if(_stop) {
              return;
            }

            task = _pending_tasks.front();
            _pending_tasks.pop();
          }
          if(task != nullptr) {
            _process(task);
          }
        }
      }
    );
  }
}

template <typename C>
void Scheduler::emplace(C&& c) {
  _tasks.emplace_back(std::make_unique<Task>(std::forward<C>(c)));
}

void Scheduler::schedule() {
  for(auto& task: _tasks) {
    _enqueue(task.get());
  }
}

void Scheduler::wait() {
  for(auto& w: _workers) {
    w.join();
  } 
}

void Scheduler::_enqueue(Task* task) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _pending_tasks.push(task);
  }
  _cv.notify_one();
}

void Scheduler::_process(Task* task) {
  task->work();

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    {
      std::unique_lock<std::mutex> lock(_mtx);
      _stop = true;
    }
    _cv.notify_all();
  }
}

} // end of namespace cudaWoCoro =========================
