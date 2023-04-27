
void cpu_work() {
  cpu_matmul(matA, matB, ...);
}

void gpu_work() {
  cudaStream_t stream;
  cudaStreamCreate(stream);
  gpu_matmul<<<8, 256, 0, stream>>>(matA, matB, ...);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

// cpu_work and gpu_work are independent to each other
// assume we only have one CPU thread
int main() {
  cpu_work();
  gpu_work();

  // alternatively
  gpu_work();
  cpu_work();
}

