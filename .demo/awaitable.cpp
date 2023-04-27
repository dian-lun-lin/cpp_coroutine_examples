
Coro gpu_work() {
  cudaStream_t stream;
  cudaStreamCreate(stream);
  gpu_matmul<<<8, 256, 0, stream>>>(matA, matB);
  while(cudaStreamQuery(stream) != cudaSuccess) {
    co_await std::suspend_always{};
  }
}

// compiler transform
auto&& awaiter = std::suspend_always{};
if(awaiter.await_ready()) {
  awaiter.await_suspend(std::coroutine_handle<>..);
  //<suspend/resume>
}
awaiter.await_resume();



