#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void naive_softmax_host(const float *x, float *y, int V, int batch_size) {
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx, x += V, y += V) {
    float d = 0.0f;
    for (std::size_t i = 0; i < V; ++i)
      d += std::exp(x[i]); // load

    for (std::size_t i = 0; i < V; ++i)
      y[i] = std::exp(x[i]) / d; // load + store
  }
}

void safe_softmax_host(const float *x, float *y, int V, int batch_size) {
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx, x += V, y += V) {
    float m = -std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < V; ++i)
      m = std::max(m, x[i]); // load

    float d = 0.0f;
    for (std::size_t i = 0; i < V; ++i)
      d += std::exp(x[i] - m); // load

    for (std::size_t i = 0; i < V; ++i)
      y[i] = std::exp(x[i] - m) / d; // load and store
  }
}

void online_softmax_host(const float *x, float *y, int V, int batch_size) {
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx, x += V, y += V) {
    float m = -std::numeric_limits<float>::max();
    float d = 0.0f;
    for (std::size_t i = 0; i < V; ++i) {
      float next_m = std::max(m, x[i]);
      d = d * std::exp(m - next_m) + std::exp(x[i] - next_m);
      m = next_m;
    }
    for (std::size_t i = 0; i < V; ++i)
      y[i] = std::exp(x[i] - m) / d;
  }
}

void softmax_host(const float *x, float *y, int V, int batch_size,
                  SoftmaxType softmax_type) {
  switch (softmax_type) {
  case SoftmaxType::Naive: {
    naive_softmax_host(x, y, V, batch_size);
    break;
  }
  case SoftmaxType::Safe: {
    safe_softmax_host(x, y, V, batch_size);
    break;
  }
  case SoftmaxType::Online: {
    online_softmax_host(x, y, V, batch_size);
    break;
  }
  default:
    throw std::runtime_error("Unsupported SoftmaxType");
  }
}
