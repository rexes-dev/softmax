#include "softmax.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

struct CudaTimer {
  CudaTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
  }
  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }

  template <typename Func> double time(Func &&f) {
    cudaEventRecord(start);
    f();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    double us = static_cast<double>(ms) * 1000;
    return us;
  }

  cudaEvent_t start;
  cudaEvent_t end;
};

int getSMCount() {
  static int sm_cnt = [] {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.multiProcessorCount;
  }();

  return sm_cnt;
}

std::string to_string(SoftmaxType t) {
  switch (t) {
  case SoftmaxType::Naive:
    return "naive";
  case SoftmaxType::Safe:
    return "safe";
  case SoftmaxType::Online:
    return "online";
  }
  return "unknown";
}

void run_bench(int V, int batch_size, SoftmaxType softmax_type) {
  float *x;
  float *y;
  cudaMalloc(&x, V * batch_size * sizeof(float));
  cudaMalloc(&y, V * batch_size * sizeof(float));

  CudaTimer timer;
  constexpr int WarmUp = 3;
  for (int i = 0; i < WarmUp; ++i)
    softmax_dev(x, y, V, batch_size, softmax_type);
  cudaDeviceSynchronize();

  const int num_iter = 100;
  const auto us = timer.time([&] {
    for (int i = 0; i < num_iter; ++i)
      softmax_dev(x, y, V, batch_size, softmax_type);
  });
  const auto avg_us = us / num_iter;
  std::cout << std::setw(8) << V << ' ' << std::setw(7) << batch_size << ' '
            << std::setw(10) << avg_us << std::endl;
  cudaFree(x);
  cudaFree(y);
}

int main() {
  std::cout << "SM count: " << getSMCount() << std::endl;
  std::cout << '\n'
            << "Sweep: batch_size  (V = 128256)" << '\n'
            << std::setw(8) << "V" << ' ' << std::setw(7) << "batch" << ' '
            << std::setw(10) << "avg (us)\n";
  for (int batch_size : {1, 64, getSMCount(), 1024, 4096}) {
    const int V = 128256;
    run_bench(V, batch_size, SoftmaxType::Safe);
  }

  std::cout << '\n'
            << "Sweep: V  (batch_size = 4096)" << '\n'
            << std::setw(8) << "V" << ' ' << std::setw(7) << "batch" << ' '
            << std::setw(10) << "avg (us)\n";
  for (int V : {32000, 50257, 128256, 151936, 256000}) {
    const int batch_size = 4096;
    run_bench(V, batch_size, SoftmaxType::Safe);
  }

  return 0;
}
