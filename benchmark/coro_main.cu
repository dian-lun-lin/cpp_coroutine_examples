#include <vector>
#include <algorithm>
#include <numeric>

#include <chrono>
#include <iostream>
#include "../cpu-gpu/scheduler.hpp"

// GPU loop kernel
__global__ void cuda_loop(
  size_t ms
) {
  long long loop_cycles = 1350000 * ms; // TODO: 1350MHZ is for 2080ti. change it to clock for arbitrary GPU 
  long long start = clock64();
  long long cycles_elapsed;
  do { 
    cycles_elapsed = clock64() - start; 
  } while (cycles_elapsed < loop_cycles);
}

// CPU loop task
void cpu_loop(int ms) {
  auto start = std::chrono::steady_clock::now();
  int a = 1;
  int b = a * 10 % 7;
  while(b != 0)
  {
    a = b * 10;
    b = a % 7;
    if(std::chrono::steady_clock::now() - start > std::chrono::milliseconds(ms)) 
      break;
  }
}


// ===================================================
//
// Definition of work 
//
// ===================================================

cudaCoro::Task work(
  cudaCoro::Scheduler& sch, dim3 dim_grid, dim3 dim_block, 
  int cpu_ms, int gpu_ms
) {
  cpu_loop(cpu_ms);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(gpu_ms);
  while(cudaStreamQuery(stream) != cudaSuccess) {
    co_await sch.suspend();
  }
  cudaStreamDestroy(stream);
}

// ===================================================

// main

// ===================================================

int main(int argc, char** argv) {

  if(argc != 5) {
    std::cerr << "usage: ./a.out num_threads num_tasks cpu_time(ms) gpu_time(ms) \n";
    std::exit(EXIT_FAILURE);
  }

  int num_threads = std::atoi(argv[1]);
  int num_tasks = std::atoi(argv[2]);
  int cpu_ms = std::atoi(argv[3]);
  int gpu_ms = std::atoi(argv[4]);
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(256, 1, 1);

  cudaCoro::Scheduler sch{(size_t)num_threads};
  for(size_t i = 0; i < num_tasks; ++i) {
    sch.emplace(work(sch, dim_grid, dim_block, cpu_ms, gpu_ms).get_handle());
  }
  auto beg_t = std::chrono::steady_clock::now();
  sch.schedule();
  sch.wait();
  auto end_t = std::chrono::steady_clock::now();

  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - beg_t).count();
  std::cout << "Coroutine-based scheduler execution time: " << dur << "ms\n";

}
