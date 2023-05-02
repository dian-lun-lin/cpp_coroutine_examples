#include <vector>
#include <algorithm>
#include <numeric>

#include <chrono>
#include <iostream>
#include "../cuda_callback/scheduler.hpp"
#include "../cuda_poll/scheduler.hpp"
#include "../cuda_wo_coro/scheduler.hpp"
#include "matmul.hpp"

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
// Defition of work 
//
// ===================================================

cudaPoll::Task work(
  cudaPoll::Scheduler& sch, dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE,  int cpu_ms, int gpu_ms
) {
  cpu_loop(cpu_ms);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(gpu_ms);
  while(cudaStreamQuery(stream) != cudaSuccess) {
    co_await sch.suspend();
  }
  cudaStreamDestroy(stream);
}

cudaCallback::Task work(
  cudaCallback::Scheduler& sch, dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE,  int cpu_ms, int gpu_ms
) {
  cpu_loop(cpu_ms);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(gpu_ms);
  co_await sch.suspend(stream);
  cudaStreamDestroy(stream);
}

void wo_coro_work(
  dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE, int cpu_ms, int gpu_ms
) {
  cpu_loop(cpu_ms);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(gpu_ms);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}


// ===================================================

// main

// ===================================================

int main(int argc, char** argv) {

  if(argc != 5) {
    std::cerr << "usage: ./a.out num_threads num_tasks cpu_time(ms) gpu_time(ms) \n";
  }

  int num_threads = std::atoi(argv[1]);
  int num_tasks = std::atoi(argv[2]);
  int cpu_ms = std::atoi(argv[3]);
  int gpu_ms = std::atoi(argv[3]);
  size_t BLOCK_SIZE = 32;
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  {
    cudaPoll::Scheduler sch{(size_t)num_threads};
    for(size_t i = 0; i < num_tasks; ++i) {
      sch.emplace(work(sch, dim_grid, dim_block, BLOCK_SIZE, cpu_ms, gpu_ms).get_handle());
    }
    auto beg_t = std::chrono::steady_clock::now();
    sch.schedule();
    sch.wait();
    auto end_t = std::chrono::steady_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - beg_t).count();
    std::cout << "Polling-based scheduler execution time: " << dur << "ms\n";
  }

  {
    cudaCallback::Scheduler sch{(size_t)num_threads};
    for(size_t i = 0; i < num_tasks; ++i) {
      sch.emplace(work(sch, dim_grid, dim_block, BLOCK_SIZE, cpu_ms, gpu_ms).get_handle());
    }
    auto beg_t = std::chrono::steady_clock::now();
    sch.schedule();
    sch.wait();
    auto end_t = std::chrono::steady_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - beg_t).count();
    std::cout << "Callback-based scheduler execution time: " << dur << "ms\n";
  }

  {
    // without coroutine
    cudaWoCoro::Scheduler sch{(size_t)num_threads};
    for(size_t i = 0; i < num_tasks; ++i) {
      sch.emplace(std::bind(wo_coro_work, dim_grid, dim_block, BLOCK_SIZE, cpu_ms, gpu_ms));
    }
    auto beg_t = std::chrono::steady_clock::now();
    sch.schedule();
    sch.wait();
    auto end_t = std::chrono::steady_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - beg_t).count();
    std::cout << "Without-coroutine scheduler execution time: " << dur << "ms\n";
  }

}
