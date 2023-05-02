#include <iostream>
#include "scheduler.hpp"

using namespace cudaWoCoro;

template <typename T>
__global__
void gpu_count(T* count) {
  ++(*count);
}

void TaskA() {

  std::cout << "Start TaskA\n";
  int* counter;
  cudaStream_t stream;

  cudaMallocManaged(&counter, sizeof(int));
  cudaStreamCreate(&stream);
  gpu_count<<<8, 256, 0, stream>>>(counter);
 
  cudaStreamSynchronize(stream);

  std::cout << "TaskA is finished\n";
  cudaFreeAsync(counter, stream);
  cudaStreamDestroy(stream);
}

void TaskB() {

  std::cout << "Start TaskB\n";
  int* counter;
  cudaStream_t stream;

  cudaMallocManaged(&counter, sizeof(int));
  cudaStreamCreate(&stream);
  gpu_count<<<8, 256, 0, stream>>>(counter);
 
  cudaStreamSynchronize(stream);

  cudaFreeAsync(counter, stream);
  cudaStreamDestroy(stream);

  std::cout << "TaskB is finished\n";
}


int main() {

  Scheduler sch(1);

  sch.emplace(TaskA);
  sch.emplace(TaskB);

  std::cout << "Start scheduling...\n";

  sch.schedule();
  sch.wait();

}


