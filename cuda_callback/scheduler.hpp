#pragma once

#include <coroutine>
#include <list>
#include <queue>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace cudaCallback { // begin of namespace cudaCallback =========================

class Scheduler;

struct cudaCallbackData {
  Scheduler* sch;
  size_t task_id;
};

struct Task {

  struct promise_type {
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    Task get_return_object() { return std::coroutine_handle<promise_type>::from_promise(*this); }
    void return_void() {}
    void unhandled_exception() {}

    size_t id;
  };

  Task(std::coroutine_handle<promise_type> handle): handle{handle} {}

  auto get_handle() { return handle; }

  std::coroutine_handle<promise_type> handle;
};


class Scheduler {


  friend void CUDART_CB _cuda_callback(cudaStream_t st, cudaError_t stat, void* void_args);


  public: 

    Scheduler(size_t num_threads = std::thread::hardware_concurrency());

    void emplace(std::coroutine_handle<Task::promise_type> task);
    auto suspend(cudaStream_t stream); 
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

// cuda callback
void CUDART_CB _cuda_callback(cudaStream_t stream, cudaError_t stat, void* void_args) {

  // unpack
  auto* cbd = (cudaCallbackData*) void_args;
  Scheduler* sch = cbd->sch;
  size_t task_id = cbd->task_id;
  
  sch->_enqueue(sch->_tasks[task_id]);
}

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

void Scheduler::emplace(std::coroutine_handle<Task::promise_type> task) {
  task.promise().id = _tasks.size();
  _tasks.emplace_back(task);
}

void Scheduler::schedule() {
  for(auto task: _tasks) {
    _enqueue(task);
  }
}

auto Scheduler::suspend(cudaStream_t stream) {
  struct awaiter: std::suspend_always {
    Scheduler& sch;
    cudaStream_t stream;

    cudaCallbackData cbd;

    explicit awaiter(Scheduler& sch, cudaStream_t stream): sch{sch}, stream{stream} {}
    void await_suspend(std::coroutine_handle<Task::promise_type> coro_handle) {
      cbd.sch = &(sch);
      cbd.task_id = coro_handle.promise().id;
      cudaStreamAddCallback(stream, _cuda_callback, (void*)&cbd, 0);
    }
    
  };

  return awaiter{*this, stream};
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

  if(task.done()) {
    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      {
        std::unique_lock<std::mutex> lock(_mtx);
        _stop = true;
      }
      _cv.notify_all();
    }
  }
}

} // end of namespace cudaCallback =========================
