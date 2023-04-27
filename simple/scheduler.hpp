#include <coroutine>
#include <queue>

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

  std::queue<std::coroutine_handle<>> _tasks;

  public: 

    void emplace(std::coroutine_handle<> task) {
      _tasks.push(task);
    }

    void schedule() {
      while(!_tasks.empty()) {
        auto task = _tasks.front();
        _tasks.pop();
        task.resume();

        if(!task.done()) { 
          _tasks.push(task);
        }
      }
    }

    auto suspend() {
      return std::suspend_always{};
    }
};



