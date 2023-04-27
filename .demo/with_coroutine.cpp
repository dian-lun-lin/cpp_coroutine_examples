
void cpu_work() {
  cpu_matmul(matA, matB, ...);
}

Coro gpu_work() {
  cudaStream_t stream;
  cudaStreamCreate(stream);
  gpu_matmul<<<8, 256, 0, stream>>>(matA, matB);
  while(cudaStreamQuery(stream) != cudaSuccess) {
    co_await std::suspend_always{};
  }
  cudaStreamDestory(stream);
}

// cpu_work and gpu_work are independent to each other
// assume we only have one CPU thread
int main() {
  auto coro = gpu_work();
  cpu_work();

  while(!coro.done()) { coro.resume(); }
}

