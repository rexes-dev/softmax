// WARNING: TESTS ARE CLAUDE GENERATED
#include "softmax.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;
constexpr float kThird = 1.0f / 3.0f;

// A valid softmax is a probability distribution: every element in [0, 1] and
// the whole thing sums to 1. The sum accumulates in double because at the
// larger V the device tests use, adding a few thousand floats sequentially
// drifts by more than kTol on its own -- that would be the checker failing
// rather than the softmax.
void ExpectIsDistribution(const std::vector<float> &y) {
  double sum = 0.0;
  for (const float v : y) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0, kTol);
}

// Reference softmax computed in double precision, so it is independent of the
// implementation under test.
std::vector<float> ReferenceSoftmax(const std::vector<float> &x) {
  double m = -HUGE_VAL;
  for (const float v : x)
    m = std::max(m, static_cast<double>(v));

  double d = 0.0;
  for (const float v : x)
    d += std::exp(static_cast<double>(v) - m);

  std::vector<float> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    y[i] = static_cast<float>(std::exp(static_cast<double>(x[i]) - m) / d);
  return y;
}

} // namespace

TEST(SoftmaxHost, UniformInputIsUniformOutput) {
  const std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> y(x.size());

  safe_softmax_host(x.data(), y.data(), x.size());

  for (const float v : y)
    EXPECT_NEAR(v, 0.25f, kTol);
  ExpectIsDistribution(y);
}

TEST(SoftmaxHost, MatchesReferenceOnModerateInput) {
  const std::vector<float> x = {1.0f, 2.0f, 3.0f};
  const std::vector<float> want = ReferenceSoftmax(x);
  std::vector<float> y(x.size());

  // All three variants agree with the reference when nothing overflows.
  naive_softmax_host(x.data(), y.data(), x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "naive, index " << i;

  safe_softmax_host(x.data(), y.data(), x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "safe, index " << i;
  ExpectIsDistribution(y);

  online_softmax_host(x.data(), y.data(), x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "online, index " << i;
  ExpectIsDistribution(y);
}

// Guards the online recurrence d = d * exp(m - m') + exp(x - m'), which has to
// rescale the running denominator every time the running max moves.
TEST(SoftmaxHost, OnlineMatchesSafe) {
  const std::vector<float> x = {-12.5f, 0.0f, 7.25f, -3.0f, 42.0f,
                                41.5f,  1.0f, -0.5f, 20.0f, -30.0f};
  std::vector<float> safe(x.size());
  std::vector<float> online(x.size());

  safe_softmax_host(x.data(), safe.data(), x.size());
  online_softmax_host(x.data(), online.data(), x.size());

  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(online[i], safe[i], kTol) << "index " << i;
  ExpectIsDistribution(online);
}

// The motivating case: exp(89.0f) ~= 4.5e38 exceeds FLT_MAX ~= 3.4e38, so the
// naive denominator becomes inf and every output is inf/inf = NaN.
TEST(SoftmaxStability, NaiveOverflowsWhereSafeDoesNot) {
  const std::vector<float> x = {89.0f, 89.0f, 89.0f};
  std::vector<float> naive(x.size());
  std::vector<float> safe(x.size());
  std::vector<float> online(x.size());

  naive_softmax_host(x.data(), naive.data(), x.size());
  safe_softmax_host(x.data(), safe.data(), x.size());
  online_softmax_host(x.data(), online.data(), x.size());

  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to overflow, got " << v;

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(safe[i], kThird, kTol) << "safe, index " << i;
    EXPECT_NEAR(online[i], kThird, kTol) << "online, index " << i;
  }
}

// The other end of the same problem: exp(-500.0f) underflows to 0, so the naive
// denominator is 0 and every output is 0/0 = NaN.
TEST(SoftmaxStability, NaiveUnderflowsOnLargeNegativeInput) {
  const std::vector<float> x = {-500.0f, -500.0f, -500.0f};
  std::vector<float> naive(x.size());
  std::vector<float> safe(x.size());
  std::vector<float> online(x.size());

  naive_softmax_host(x.data(), naive.data(), x.size());
  safe_softmax_host(x.data(), safe.data(), x.size());
  online_softmax_host(x.data(), online.data(), x.size());

  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to underflow, got " << v;

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(safe[i], kThird, kTol) << "safe, index " << i;
    EXPECT_NEAR(online[i], kThird, kTol) << "online, index " << i;
  }
}

namespace {

using SoftmaxDev = void (*)(const float *, float *, std::size_t);

// The kernels use __expf, a fast approximation, so the device tests compare on
// relative error rather than the tighter absolute tolerance the host tests use.
float DeviceTol(float want) { return 1e-4f * std::fabs(want) + 1e-9f; }

std::vector<float> Ramp(std::size_t V, float lo, float hi) {
  std::vector<float> x(V);
  for (std::size_t i = 0; i < V; ++i)
    x[i] = V == 1 ? lo
                  : lo + (hi - lo) * (static_cast<float>(i) /
                                      static_cast<float>(V - 1));
  return x;
}

// Every device test needs the same "is there a GPU" guard and the same
// malloc/copy/launch/copy/free dance. GTEST_SKIP() only skips the whole test
// when it runs in SetUp(), which is why this is a fixture rather than a free
// function. Naming the fixture SoftmaxDevice keeps the pre-existing test's full
// name, SoftmaxDevice.MatchesHost, unchanged.
class SoftmaxDevice : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
      GTEST_SKIP() << "no CUDA device available";
  }

  // Writes into an out-param because ASSERT_* only compiles inside a void
  // function. Call it through ASSERT_NO_FATAL_FAILURE: a failed ASSERT_ in here
  // returns from Run, not from the test, and without that wrapper the test
  // would carry on and compare an untouched output buffer.
  static void Run(SoftmaxDev softmax_dev, const std::vector<float> &x,
                  std::vector<float> &y) {
    const std::size_t bytes = x.size() * sizeof(float);
    y.assign(x.size(), 0.0f);

    float *dx = nullptr;
    float *dy = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dy, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);

    softmax_dev(dx, dy, x.size());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(cudaFree(dx), cudaSuccess);
    EXPECT_EQ(cudaFree(dy), cudaSuccess);
  }
};

} // namespace

TEST_F(SoftmaxDevice, MatchesHost) {
  // naive_softmax_dev is the unsafe variant, so keep the inputs in a range
  // where exp() cannot overflow -- otherwise both sides are NaN and the
  // comparison proves nothing.
  const std::vector<float> x = Ramp(1000, -10.0f, 10.0f);

  std::vector<float> want(x.size());
  naive_softmax_host(x.data(), want.data(), x.size());

  std::vector<float> got;
  ASSERT_NO_FATAL_FAILURE(Run(naive_softmax_dev, x, got));

  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
  ExpectIsDistribution(got);
}

// V < kBlockDim (256) leaves most threads with no element at all: their partial
// max stays -inf and their partial sum stays 0, and neither may poison the
// block reduction. 256 and 512 are exact multiples of the block size, the rest
// leave a ragged grid-stride tail, and 4096 makes every thread loop 16 times.
TEST_F(SoftmaxDevice, SafeMatchesReferenceAcrossSizes) {
  const std::vector<std::size_t> sizes = {1, 3, 255, 256, 257, 512, 1000, 4096};

  for (const std::size_t V : sizes) {
    SCOPED_TRACE("V = " + std::to_string(V));

    const std::vector<float> x = Ramp(V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmax(x);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(safe_softmax_dev, x, got));

    for (std::size_t i = 0; i < V; ++i)
      EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
    ExpectIsDistribution(got);
  }
}

// A monotone ramp puts the max at the last element, so a kernel that never
// really reduced -- one that kept x[V-1], or the max of a single thread's
// stride -- still passes the test above. Here the max sits at index 4, and
// 42.0f is far enough above the rest that picking the wrong one is visible.
TEST_F(SoftmaxDevice, SafeMatchesReferenceOnIrregularInput) {
  const std::vector<float> x = {-12.5f, 0.0f, 7.25f, -3.0f, 42.0f,
                                41.5f,  1.0f, -0.5f, 20.0f, -30.0f};
  const std::vector<float> want = ReferenceSoftmax(x);

  std::vector<float> got;
  ASSERT_NO_FATAL_FAILURE(Run(safe_softmax_dev, x, got));

  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
  ExpectIsDistribution(got);
}

// The device counterpart of SoftmaxStability.NaiveOverflowsWhereSafeDoesNot:
// __expf(89.0f) ~= 4.5e38 exceeds FLT_MAX, so the naive denominator saturates
// to inf and every output is inf/inf = NaN. This is the input that proves the
// max was actually subtracted -- every test above still passes if m is 0.
TEST_F(SoftmaxDevice, SafeIsStableWhereNaiveOverflows) {
  const std::vector<float> x = {89.0f, 89.0f, 89.0f};

  std::vector<float> naive;
  ASSERT_NO_FATAL_FAILURE(Run(naive_softmax_dev, x, naive));
  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to overflow, got " << v;

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(safe_softmax_dev, x, safe));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(safe[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(safe);
}

// The other end of the same problem: __expf(-500.0f) underflows to 0, so the
// naive denominator is 0 and every output is 0/0 = NaN. Shifting by the max
// makes the largest term exp(0) = 1, so the denominator can never reach 0.
TEST_F(SoftmaxDevice, SafeIsStableWhereNaiveUnderflows) {
  const std::vector<float> x = {-500.0f, -500.0f, -500.0f};

  std::vector<float> naive;
  ASSERT_NO_FATAL_FAILURE(Run(naive_softmax_dev, x, naive));
  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to underflow, got " << v;

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(safe_softmax_dev, x, safe));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(safe[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(safe);
}

// online_softmax_dev fuses the max and the denominator into a single reduction
// over (m, d) pairs, so it fails in ways the two-pass safe kernel cannot. These
// mirror the safe_softmax_dev cases so a divergence points at the fused
// recurrence rather than at softmax in general.
TEST_F(SoftmaxDevice, OnlineMatchesReferenceAcrossSizes) {
  const std::vector<std::size_t> sizes = {1, 3, 255, 256, 257, 512, 1000, 4096};

  for (const std::size_t V : sizes) {
    SCOPED_TRACE("V = " + std::to_string(V));

    const std::vector<float> x = Ramp(V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmax(x);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(online_softmax_dev, x, got));

    for (std::size_t i = 0; i < V; ++i)
      EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
    ExpectIsDistribution(got);
  }
}

TEST_F(SoftmaxDevice, OnlineMatchesReferenceOnIrregularInput) {
  const std::vector<float> x = {-12.5f, 0.0f, 7.25f, -3.0f, 42.0f,
                                41.5f,  1.0f, -0.5f, 20.0f, -30.0f};
  const std::vector<float> want = ReferenceSoftmax(x);

  std::vector<float> got;
  ASSERT_NO_FATAL_FAILURE(Run(online_softmax_dev, x, got));

  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
  ExpectIsDistribution(got);
}

// The device analogue of SoftmaxHost.OnlineMatchesSafe, and the test that
// actually targets the recurrence d = d * exp(m_old - m_new) + exp(x - m_new).
// Inside one thread's grid stride an ascending input moves the running max on
// every element, so the rescale factor is exercised V/kBlockDim times per
// thread; a descending input moves it once and then never again, leaving the
// other term of bin_op to absorb every remaining element. Both directions have
// to land on what the two-pass kernel already computes.
TEST_F(SoftmaxDevice, OnlineMatchesSafeOnBothRampDirections) {
  for (const bool ascending : {true, false}) {
    SCOPED_TRACE(ascending ? "ascending" : "descending");

    const std::vector<float> x =
        ascending ? Ramp(4096, -10.0f, 10.0f) : Ramp(4096, 10.0f, -10.0f);

    std::vector<float> safe;
    ASSERT_NO_FATAL_FAILURE(Run(safe_softmax_dev, x, safe));

    std::vector<float> online;
    ASSERT_NO_FATAL_FAILURE(Run(online_softmax_dev, x, online));

    for (std::size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(online[i], safe[i], DeviceTol(safe[i])) << "index " << i;
    ExpectIsDistribution(online);
  }
}

// Same overflow input as the safe kernel: the online form never materialises
// exp(x) either, since every term is already shifted by the running max.
TEST_F(SoftmaxDevice, OnlineIsStableWhereNaiveOverflows) {
  const std::vector<float> x = {89.0f, 89.0f, 89.0f};

  std::vector<float> online;
  ASSERT_NO_FATAL_FAILURE(Run(online_softmax_dev, x, online));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(online[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(online);
}

TEST_F(SoftmaxDevice, OnlineIsStableWhereNaiveUnderflows) {
  const std::vector<float> x = {-500.0f, -500.0f, -500.0f};

  std::vector<float> online;
  ASSERT_NO_FATAL_FAILURE(Run(online_softmax_dev, x, online));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(online[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(online);
}
