
Coro gpu_work() {
  Coro::promise_type p();
  auto coro_obj = p.get_return_object();

  try {
    co_await p.initial_suspend();
    cudaStream_t stream;
    cudaStreamCreate(stream);
    gpu_matmul<<<8, 256, 0, stream>>>(matA, matB);
    while(cudaStreamQuery(stream) != cudaSuccess) {
      co_await std::suspend_always{};
    }
  } catch(...) {
    p.unhandled_exception();
  }
  co_await p.final_suspend();
}

