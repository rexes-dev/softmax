#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

void naive_softmax_host(const float *x, float *y, std::size_t V) {
  float d = 0.0f;
  for (std::size_t i = 0; i < V; ++i)
    d += std::exp(x[i]); // load

  for (std::size_t i = 0; i < V; ++i)
    y[i] = std::exp(x[i]) / d; // load + store
}

void safe_softmax_host(const float *x, float *y, std::size_t V) {
  float m = -std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < V; ++i)
    m = std::max(m, x[i]); // load

  float d = 0.0f;
  for (std::size_t i = 0; i < V; ++i)
    d += std::exp(x[i] - m); // load

  for (std::size_t i = 0; i < V; ++i)
    y[i] = std::exp(x[i] - m) / d; // load and store
}

void online_softmax_host(const float *x, float *y, std::size_t V) {
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
