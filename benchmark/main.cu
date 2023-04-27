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

// ===================================================

// Defition of Task

// ===================================================

cudaPoll::Task gpu_work(
  cudaPoll::Scheduler& sch, dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE, int ms
) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(ms);
  while(cudaStreamQuery(stream) != cudaSuccess) {
    co_await sch.suspend();
  }
  cudaStreamDestroy(stream);
}

cudaCallback::Task gpu_work(
  cudaCallback::Scheduler& sch, dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE, int ms
) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(ms);
  co_await sch.suspend(stream);
  cudaStreamDestroy(stream);
}

void gpu_work_wo_coro(
  dim3 dim_grid, dim3 dim_block, 
  size_t BLOCK_SIZE, int ms
) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuda_loop<<<dim_grid, dim_block, 0, stream>>>(ms);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}


// ===================================================

// main

// ===================================================

int main(int argc, char** argv) {

  if(argc != 4) {
    std::cerr << "usage: ./a.out num_threads num_tasks task_overhead(ms) \n";
  }

  int num_threads = std::atoi(argv[1]);
  int num_tasks = std::atoi(argv[2]);
  int ms = std::atoi(argv[3]);
  size_t BLOCK_SIZE = 32;
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  {
    cudaPoll::Scheduler sch{(size_t)num_threads};
    for(size_t i = 0; i < num_tasks; ++i) {
      sch.emplace(gpu_work(sch, dim_grid, dim_block, BLOCK_SIZE, ms).get_handle());
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
      sch.emplace(gpu_work(sch, dim_grid, dim_block, BLOCK_SIZE, ms).get_handle());
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
      sch.emplace(std::bind(gpu_work_wo_coro,dim_grid, dim_block, BLOCK_SIZE, ms));
    }
    auto beg_t = std::chrono::steady_clock::now();
    sch.schedule();
    sch.wait();
    auto end_t = std::chrono::steady_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - beg_t).count();
    std::cout << "Without-coroutine scheduler execution time: " << dur << "ms\n";
  }

}
