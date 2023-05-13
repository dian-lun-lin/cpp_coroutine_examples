#pragma once
#include <coroutine>
#include <list>
#include <queue>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace cudaCoro { // begin of namespace cudaCoro =========================

struct Task {

  struct promise_type {
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    Task get_return_object() { return std::coroutine_handle<promise_type>::from_promise(*this); }
    void return_void() {}
    void unhandled_exception() {}
  };

  Task(std::coroutine_handle<promise_type> handle): handle{handle} {}

  auto get_handle() { return handle; }

  std::coroutine_handle<promise_type> handle;
};


class Scheduler {

  public: 

    Scheduler(size_t num_threads = std::thread::hardware_concurrency());

    void emplace(std::coroutine_handle<> task);
    auto suspend(); 
    void schedule();
    void wait();


  private:

    std::vector<std::coroutine_handle<>> _tasks;
    std::queue<std::coroutine_handle<>> _pending_tasks;

    std::vector<std::thread> _workers;
    std::mutex _mtx;
    std::condition_variable _cv;
    bool _stop{false};
    std::atomic<size_t> _finished{0};

    void _enqueue(std::coroutine_handle<> task);
    void _process(std::coroutine_handle<> task);
};

Scheduler::Scheduler(size_t num_threads) {
  _workers.reserve(num_threads);

  for(size_t t = 0; t < num_threads; ++t) {
    _workers.emplace_back([this]() {
        while(true) {
          std::coroutine_handle<> task;
          {
            std::unique_lock<std::mutex> lock(_mtx);
            _cv.wait(lock, [this]{ return _stop || (!_pending_tasks.empty()); });
            if(_stop) {
              return;
            }

            task = _pending_tasks.front();
            _pending_tasks.pop();
          }
          if(task) {
            _process(task);
          }
        }
      }
    );
  }
}

void Scheduler::emplace(std::coroutine_handle<> task) {
  _tasks.emplace_back(task);
}

void Scheduler::schedule() {
  for(auto task: _tasks) {
    _enqueue(task);
  }
}

auto Scheduler::suspend() {
  return std::suspend_always{};
}

void Scheduler::wait() {
  for(auto& w: _workers) {
    w.join();
  } 
}

void Scheduler::_enqueue(std::coroutine_handle<> task) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _pending_tasks.push(task);
  }
  _cv.notify_one();
}

void Scheduler::_process(std::coroutine_handle<> task) {
  task.resume();

  if(!task.done()) {
    _enqueue(task);
  }
  else {
    task.destroy();
    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      {
        std::unique_lock<std::mutex> lock(_mtx);
        _stop = true;
      }
      _cv.notify_all();
    }
  }
}

} // end of namespace cudaCoro =========================
