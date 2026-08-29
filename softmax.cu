#include "softmax.h"

#include <cub/cub.cuh>
#include <cuda/std/limits>

constexpr int kBlockDim = 256;

__global__ void naive_softmax_kernel(const float *x, float *y, int V) {
  const int batch_idx = blockIdx.x;
  x += batch_idx * V;
  y += batch_idx * V;

  const int i = threadIdx.x;

  float d_partial = 0.0f;
  for (int j = i; j < V; j += kBlockDim)
    d_partial += __expf(x[j]);

  using BlockReduce = cub::BlockReduce<float, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const auto d_sum = BlockReduce(temp_storage).Sum(d_partial);

  __shared__ float d;
  if (i == 0)
    d = d_sum;
  __syncthreads();

  for (int j = i; j < V; j += kBlockDim)
    y[j] = __expf(x[j]) / d;
}

__global__ void safe_softmax_kernel(const float *x, float *y, int V) {
  const int batch_idx = blockIdx.x;
  x += batch_idx * V;
  y += batch_idx * V;

  const int i = threadIdx.x;

  float m_partial = -cuda::std::numeric_limits<float>::max();
  for (int j = i; j < V; j += kBlockDim)
    m_partial = max(m_partial, x[j]);

  using BlockReduce = cub::BlockReduce<float, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const auto m_max =
      BlockReduce(temp_storage).Reduce(m_partial, [](float a, float b) {
        return fmaxf(a, b);
      });

  __shared__ float m;
  if (i == 0)
    m = m_max;
  __syncthreads();

  float d_partial = 0.0f;
  for (int j = i; j < V; j += kBlockDim)
    d_partial += __expf(x[j] - m);

  const auto d_sum = BlockReduce(temp_storage).Sum(d_partial);

  __shared__ float d;
  if (i == 0)
    d = d_sum;
  __syncthreads();

  for (int j = i; j < V; j += kBlockDim)
    y[j] = __expf(x[j] - m) / d;
}

__global__ void online_softmax_kernel(const float *x, float *y, int V) {
  const int batch_idx = blockIdx.x;
  x += batch_idx * V;
  y += batch_idx * V;

  const int i = threadIdx.x;

  // associative && commutative op mentioned in the paper
  const auto bin_op = [](const float2 &md1, const float2 &md2) {
    const auto m1 = md1.x;
    const auto d1 = md1.y;
    const auto m2 = md2.x;
    const auto d2 = md2.y;
    const auto m = fmaxf(m1, m2);
    const auto d = d1 * __expf(m1 - m) + d2 * __expf(m2 - m);
    return make_float2(m, d);
  };

  // -inf - (-inf) = NaN, so I had to set it to max.
  auto md_partial = make_float2(-cuda::std::numeric_limits<float>::max(), 0);
  for (int j = i; j < V; j += kBlockDim)
    md_partial = bin_op(md_partial, make_float2(x[j], 1));

  using BlockReduce = cub::BlockReduce<float2, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const auto md_total = BlockReduce(temp_storage).Reduce(md_partial, bin_op);

  __shared__ float2 md;
  if (i == 0)
    md = md_total;
  __syncthreads();

  const auto m = md.x;
  const auto d = md.y;
  for (int j = i; j < V; j += kBlockDim)
    y[j] = __expf(x[j] - m) / d;
}

void softmax_dev(const float *x, float *y, int V, int batch_size,
                 SoftmaxType softmax_type) {
  switch (softmax_type) {
  case SoftmaxType::Naive: {
    naive_softmax_kernel<<<batch_size, kBlockDim>>>(x, y, V);
    break;
  }
  case SoftmaxType::Safe: {
    safe_softmax_kernel<<<batch_size, kBlockDim>>>(x, y, V);
    break;
  }
  case SoftmaxType::Online: {
    online_softmax_kernel<<<batch_size, kBlockDim>>>(x, y, V);
    break;
  }
  default:
    throw std::runtime_error("Unsupported SoftmaxType");
  }
}
