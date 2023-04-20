#include <coroutine>
#include <list>

struct Task {

  struct promise_type {
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    Task get_return_object() { return Task{}; }
    void unhandled_exception() {}
  };
};


class Scheduler {

  std::list<std::coroutine_handle<>> _tasks;


  public: 
    bool schedule() {
      auto task = _tasks.front();
      _tasks.pop_front();
      if(!task.done()) { task.resume(); }

      return !_tasks.empty();
    }

    auto suspend() {
      struct Awaiter: std::suspend_always {
        Scheduler& scheduler;
        Awaiter(Scheduler& sched): scheduler{sched} {}
        void await_suspend(std::coroutine_handle<> coro) {
          scheduler._tasks.push_back(coro);
        }
      };

      return Awaiter{*this};
    }
};



