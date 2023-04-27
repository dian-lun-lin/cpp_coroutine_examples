
struct Coro {

  struct promise_type {
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    Coro get_return_object() { return std::coroutine_handle<promise_type>::from_promise(*this); }
    void return_void() {}
    void unhandled_exception() {}
  };

  Coro(std::coroutine_handle<promise_type> handle): handle{handle} {}

  std::coroutine_handle<promise_type> handle;
};

co_yield <expression>
-> co_await promise.yield_value(<expression>);

co_return <expression>
-> promise.return_value(<expression>);
   goto end;


