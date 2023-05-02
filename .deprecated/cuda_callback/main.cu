#include <iostream>
#include "scheduler.hpp"

using namespace cudaCallback;

template <typename T>
__global__
void gpu_count(T* count) {
  ++(*count);
}

Task TaskA(Scheduler& sch) {
  std::cout << "Start TaskA\n";
  int* counter;
  cudaStream_t stream;

  cudaMallocManaged(&counter, sizeof(int));
  cudaStreamCreate(&stream);
  gpu_count<<<8, 256, 0, stream>>>(counter);
 
  co_await sch.suspend(stream);

  std::cout << "TaskA is finished\n";
  cudaFreeAsync(counter, stream);
  cudaStreamDestroy(stream);
}

Task TaskB(Scheduler& sch) {

  std::cout << "Start TaskB\n";
  int* counter;
  cudaStream_t stream;

  cudaMallocManaged(&counter, sizeof(int));
  cudaStreamCreate(&stream);
  gpu_count<<<8, 256, 0, stream>>>(counter);
 
  co_await sch.suspend(stream);

  cudaFreeAsync(counter, stream);
  cudaStreamDestroy(stream);

  std::cout << "TaskB is finished\n";
}


int main() {

  Scheduler sch;

  sch.emplace(TaskA(sch).get_handle());
  sch.emplace(TaskB(sch).get_handle());

  std::cout << "Start scheduling...\n";

  sch.schedule();
  sch.wait();

}


