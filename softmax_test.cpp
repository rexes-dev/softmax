// WARNING: TESTS ARE CLAUDE GENERATED
#include "softmax.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;
constexpr float kThird = 1.0f / 3.0f;
// Attention masks encode "may not attend here" as a -inf logit; softmax must
// map those to exactly 0 without disturbing the unmasked positions.
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

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

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Safe);

  for (const float v : y)
    EXPECT_NEAR(v, 0.25f, kTol);
  ExpectIsDistribution(y);
}

TEST(SoftmaxHost, MatchesReferenceOnModerateInput) {
  const std::vector<float> x = {1.0f, 2.0f, 3.0f};
  const std::vector<float> want = ReferenceSoftmax(x);
  std::vector<float> y(x.size());

  // All three variants agree with the reference when nothing overflows.
  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Naive);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "naive, index " << i;

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Safe);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "safe, index " << i;
  ExpectIsDistribution(y);

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Online);
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

  softmax_host(x.data(), safe.data(), x.size(), 1, SoftmaxType::Safe);
  softmax_host(x.data(), online.data(), x.size(), 1, SoftmaxType::Online);

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

  softmax_host(x.data(), naive.data(), x.size(), 1, SoftmaxType::Naive);
  softmax_host(x.data(), safe.data(), x.size(), 1, SoftmaxType::Safe);
  softmax_host(x.data(), online.data(), x.size(), 1, SoftmaxType::Online);

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

  softmax_host(x.data(), naive.data(), x.size(), 1, SoftmaxType::Naive);
  softmax_host(x.data(), safe.data(), x.size(), 1, SoftmaxType::Safe);
  softmax_host(x.data(), online.data(), x.size(), 1, SoftmaxType::Online);

  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to underflow, got " << v;

  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(safe[i], kThird, kTol) << "safe, index " << i;
    EXPECT_NEAR(online[i], kThird, kTol) << "online, index " << i;
  }
}

// A causal mask leaves row 0 with a single unmasked logit, so the answer is
// exact: all the probability mass on element 0. This is the closed-form case
// where a NaN from mishandled -inf cannot hide behind a tolerance.
TEST(SoftmaxMasked, CausalRowZeroHost) {
  const std::vector<float> x = {0.0f, kNegInf, kNegInf, kNegInf};
  const std::vector<float> want = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Naive);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "naive, index " << i;

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Safe);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "safe, index " << i;
  ExpectIsDistribution(y);

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Online);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "online, index " << i;
  ExpectIsDistribution(y);
}

// A hole in the middle of the row: the max (index 3) sits after the mask, so
// the running max crosses an -inf on its way there.
TEST(SoftmaxMasked, InteriorMaskHost) {
  const std::vector<float> x = {0.0f, 1.0f, kNegInf, 3.0f};
  const std::vector<float> want = ReferenceSoftmax(x);
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Naive);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "naive, index " << i;

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Safe);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "safe, index " << i;
  ExpectIsDistribution(y);

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Online);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "online, index " << i;
  ExpectIsDistribution(y);
}

// A mask at the FRONT of the row is what actually broke the -inf sentinel in
// softmax_host(Online): on the first iteration m and x[0] were both -inf, so
// d = d * exp(-inf - (-inf)) = NaN before any real logit was seen. Padding
// masks produce exactly this shape. The finite -FLT_MAX sentinel keeps the
// subtraction at -inf - (-FLT_MAX) = -inf, whose exp is a harmless 0.
TEST(SoftmaxMasked, LeadingMaskHost) {
  const std::vector<float> x = {kNegInf, 0.5f, 2.0f};
  const std::vector<float> want = ReferenceSoftmax(x);
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Naive);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "naive, index " << i;

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Safe);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "safe, index " << i;
  ExpectIsDistribution(y);

  softmax_host(x.data(), y.data(), x.size(), 1, SoftmaxType::Online);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(y[i], want[i], kTol) << "online, index " << i;
  ExpectIsDistribution(y);
}

namespace {

// A batch is batch_size rows of V logits laid out contiguously, row r
// occupying x[r * V, (r + 1) * V). Every helper below assumes that layout,
// which is also what the host and device entry points take.

std::vector<float> Ramp(std::size_t V, float lo, float hi) {
  std::vector<float> x(V);
  for (std::size_t i = 0; i < V; ++i)
    x[i] = V == 1 ? lo
                  : lo + (hi - lo) * (static_cast<float>(i) /
                                      static_cast<float>(V - 1));
  return x;
}

// Row 0 of a causal attention mask: only element 0 unmasked.
std::vector<float> CausalRowZero(std::size_t V) {
  std::vector<float> x(V, kNegInf);
  x[0] = 0.0f;
  return x;
}

// Concatenates equal-length rows into one contiguous batch.
std::vector<float> Batch(const std::vector<std::vector<float>> &rows) {
  std::vector<float> x;
  for (const auto &row : rows)
    x.insert(x.end(), row.begin(), row.end());
  return x;
}

std::vector<float> Row(const std::vector<float> &x, int V, int r) {
  return std::vector<float>(x.begin() + static_cast<std::ptrdiff_t>(r) * V,
                            x.begin() + static_cast<std::ptrdiff_t>(r + 1) * V);
}

int Rows(const std::vector<float> &x, int V) {
  return static_cast<int>(x.size() / static_cast<std::size_t>(V));
}

// The reference applied row by row. Rows never see each other, which is
// exactly the property the batched tests are checking.
std::vector<float> ReferenceSoftmaxBatched(const std::vector<float> &x, int V) {
  std::vector<float> y;
  y.reserve(x.size());
  for (int r = 0; r < Rows(x, V); ++r) {
    const std::vector<float> row = ReferenceSoftmax(Row(x, V, r));
    y.insert(y.end(), row.begin(), row.end());
  }
  return y;
}

// batch_size rows that all differ from their neighbours. Row r is a ramp over
// [lo, hi] rotated so its max lands at a different index, then shifted by a
// row-dependent constant so its max value and its denominator differ too. An
// implementation that reads or writes the wrong row -- every block using row
// 0, an off-by-one in the row stride, a reduction that spans two rows --
// therefore lands on a wrong answer rather than a coincidentally right one.
// The shift cycles through 16 values so it never pushes the inputs into the
// range where the naive variant overflows.
std::vector<float> DistinctRows(int batch_size, int V, float lo, float hi) {
  std::vector<float> x;
  x.reserve(static_cast<std::size_t>(batch_size) * V);
  for (int r = 0; r < batch_size; ++r) {
    std::vector<float> row = Ramp(V, lo, hi);
    std::rotate(row.begin(), row.begin() + (r * 37) % V, row.end());
    for (float &v : row)
      v += 0.25f * static_cast<float>(r % 16);
    x.insert(x.end(), row.begin(), row.end());
  }
  return x;
}

float HostTol(float) { return kTol; }

// Element-wise comparison that names the row as well as the column, so a
// failure says which block (or which iteration of the host loop) went wrong.
template <class Tol>
void ExpectBatchNear(const std::vector<float> &got,
                     const std::vector<float> &want, int V, Tol tol) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol(want[i]))
        << "row " << i / V << ", index " << i % V;
}

void ExpectRowsAreDistributions(const std::vector<float> &y, int V) {
  for (int r = 0; r < Rows(y, V); ++r) {
    SCOPED_TRACE("row " + std::to_string(r));
    ExpectIsDistribution(Row(y, V, r));
  }
}

} // namespace

// Each row is its own softmax. A neighbour that overflows, underflows, or is
// masked must not leak into a row that is fine: the max, the denominator and
// the output of row r depend on row r alone. Row 0 in particular has to match
// the reference whatever comes after it. The naive variant turns rows 1 and 2
// into NaN by construction (see SoftmaxStability); those NaNs must stay
// confined to the rows that produced them.
TEST(SoftmaxHostBatch, RowsAreIndependent) {
  const int V = 3;
  const std::vector<float> x = Batch({
      {1.0f, 2.0f, 3.0f},
      {89.0f, 89.0f, 89.0f},
      {-500.0f, -500.0f, -500.0f},
      {0.0f, kNegInf, kNegInf},
      {4.0f, 5.0f, 6.0f},
  });
  const int batch_size = Rows(x, V);
  const std::vector<float> want = ReferenceSoftmaxBatched(x, V);
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Safe);
  ExpectBatchNear(y, want, V, HostTol);
  ExpectRowsAreDistributions(y, V);

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Online);
  ExpectBatchNear(y, want, V, HostTol);
  ExpectRowsAreDistributions(y, V);

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Naive);
  for (const int r : {0, 3, 4}) {
    const std::vector<float> got = Row(y, V, r);
    const std::vector<float> row_want = Row(want, V, r);
    for (int i = 0; i < V; ++i)
      EXPECT_NEAR(got[i], row_want[i], kTol) << "row " << r << ", index " << i;
  }
  for (const int r : {1, 2})
    for (const float v : Row(y, V, r))
      EXPECT_TRUE(std::isnan(v))
          << "expected naive softmax to overflow/underflow on row " << r
          << ", got " << v;
}

// 33 rows so the 16-way shift cycle in DistinctRows wraps around and rows 0,
// 16 and 32 share a shift but not a rotation. V = 10 is small enough that a
// failure message is readable.
TEST(SoftmaxHostBatch, MatchesReferenceOnDistinctRows) {
  const int V = 10;
  const int batch_size = 33;
  const std::vector<float> x = DistinctRows(batch_size, V, -10.0f, 10.0f);
  const std::vector<float> want = ReferenceSoftmaxBatched(x, V);
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Naive);
  {
    SCOPED_TRACE("naive");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Safe);
  {
    SCOPED_TRACE("safe");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Online);
  {
    SCOPED_TRACE("online");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }
}

// Masked rows of every shape the single-row tests cover, side by side in one
// batch: causal row 0, an interior hole, a leading (padding) mask, and an
// unmasked row after all of them so a sentinel that survived a masked row is
// caught by the row that follows it.
TEST(SoftmaxHostBatch, MaskedRowsInBatch) {
  const int V = 4;
  const std::vector<float> x = Batch({
      CausalRowZero(V),
      {0.0f, 1.0f, kNegInf, 3.0f},
      {kNegInf, 0.5f, 2.0f, 1.0f},
      {0.0f, 1.0f, 2.0f, 3.0f},
  });
  const int batch_size = Rows(x, V);
  const std::vector<float> want = ReferenceSoftmaxBatched(x, V);
  std::vector<float> y(x.size());

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Naive);
  {
    SCOPED_TRACE("naive");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Safe);
  {
    SCOPED_TRACE("safe");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }

  softmax_host(x.data(), y.data(), V, batch_size, SoftmaxType::Online);
  {
    SCOPED_TRACE("online");
    ExpectBatchNear(y, want, V, HostTol);
    ExpectRowsAreDistributions(y, V);
  }
}

namespace {

// The kernels use __expf, a fast approximation, so the device tests compare on
// relative error rather than the tighter absolute tolerance the host tests use.
float DeviceTol(float want) { return 1e-4f * std::fabs(want) + 1e-9f; }

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
  //
  // x holds x.size() / V rows of V logits each; every row gets its own block.
  static void Run(SoftmaxType type, const std::vector<float> &x, int V,
                  std::vector<float> &y) {
    ASSERT_GT(V, 0);
    ASSERT_EQ(x.size() % static_cast<std::size_t>(V), 0u)
        << "x.size() = " << x.size() << " is not a whole number of rows";
    const int batch_size = static_cast<int>(x.size() / V);
    const std::size_t bytes = x.size() * sizeof(float);
    y.assign(x.size(), 0.0f);

    float *dx = nullptr;
    float *dy = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dy, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    // Fill the output with NaN (all bits set) before the launch, so a row the
    // kernel never writes shows up as NaN in the comparison instead of as
    // whatever cudaMalloc happened to hand back -- which is often zero, and
    // zero is the right answer for a masked element.
    ASSERT_EQ(cudaMemset(dy, 0xFF, bytes), cudaSuccess);

    softmax_dev(dx, dy, V, batch_size, type);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(cudaFree(dx), cudaSuccess);
    EXPECT_EQ(cudaFree(dy), cudaSuccess);
  }

  // Single-row form: the whole of x is one row.
  static void Run(SoftmaxType type, const std::vector<float> &x,
                  std::vector<float> &y) {
    Run(type, x, static_cast<int>(x.size()), y);
  }
};

} // namespace

TEST_F(SoftmaxDevice, MatchesHost) {
  // SoftmaxType::Naive is the unsafe variant, so keep the inputs in a range
  // where exp() cannot overflow -- otherwise both sides are NaN and the
  // comparison proves nothing.
  const std::vector<float> x = Ramp(1000, -10.0f, 10.0f);

  std::vector<float> want(x.size());
  softmax_host(x.data(), want.data(), x.size(), 1, SoftmaxType::Naive);

  std::vector<float> got;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Naive, x, got));

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
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, got));

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
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, got));

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
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Naive, x, naive));
  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to overflow, got " << v;

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, safe));
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
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Naive, x, naive));
  for (const float v : naive)
    EXPECT_TRUE(std::isnan(v))
        << "expected naive softmax to underflow, got " << v;

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, safe));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(safe[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(safe);
}

// The online kernel fuses the max and the denominator into a single reduction
// over (m, d) pairs, so it fails in ways the two-pass safe kernel cannot. These
// mirror the safe-kernel cases so a divergence points at the fused
// recurrence rather than at softmax in general.
TEST_F(SoftmaxDevice, OnlineMatchesReferenceAcrossSizes) {
  const std::vector<std::size_t> sizes = {1, 3, 255, 256, 257, 512, 1000, 4096};

  for (const std::size_t V : sizes) {
    SCOPED_TRACE("V = " + std::to_string(V));

    const std::vector<float> x = Ramp(V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmax(x);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, got));

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
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, got));

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
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, safe));

    std::vector<float> online;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, online));

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
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, online));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(online[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(online);
}

TEST_F(SoftmaxDevice, OnlineIsStableWhereNaiveUnderflows) {
  const std::vector<float> x = {-500.0f, -500.0f, -500.0f};

  std::vector<float> online;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, online));
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(online[i], kThird, kTol) << "index " << i;
  ExpectIsDistribution(online);
}

// Masked (-inf) rows are the inputs the finite sentinel exists for, and each
// shape corners a different thread population:
//  - CausalRowZero(300): every thread except thread 0 reduces nothing but
//    -inf elements, so its partial is pure sentinel-and-mask arithmetic.
//  - {0, 1, -inf, 3}: a hole in the interior of an otherwise normal row.
//  - V = 7 with a mask: threads 7..255 contribute the bare sentinel -- the
//    case num_valid used to exclude from the reduction before 3dbdb65.
// With the old -inf sentinel the online merge computed d * __expf(-inf-(-inf))
// = NaN in all three; (-FLT_MAX, 0) is a true identity, so they must now match
// the double-precision reference exactly (masked outputs are exact zeros).
TEST_F(SoftmaxDevice, SafeHandlesMaskedInput) {
  const std::vector<std::vector<float>> cases = {
      CausalRowZero(300),
      {0.0f, 1.0f, kNegInf, 3.0f},
      {0.0f, 1.0f, kNegInf, 3.0f, kNegInf, 5.0f, 6.0f},
  };

  for (const auto &x : cases) {
    SCOPED_TRACE("V = " + std::to_string(x.size()));
    const std::vector<float> want = ReferenceSoftmax(x);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, got));

    for (std::size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
    ExpectIsDistribution(got);
  }
}

TEST_F(SoftmaxDevice, OnlineHandlesMaskedInput) {
  const std::vector<std::vector<float>> cases = {
      CausalRowZero(300),
      {0.0f, 1.0f, kNegInf, 3.0f},
      {0.0f, 1.0f, kNegInf, 3.0f, kNegInf, 5.0f, 6.0f},
  };

  for (const auto &x : cases) {
    SCOPED_TRACE("V = " + std::to_string(x.size()));
    const std::vector<float> want = ReferenceSoftmax(x);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, got));

    for (std::size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(got[i], want[i], DeviceTol(want[i])) << "index " << i;
    ExpectIsDistribution(got);
  }
}

// ---------------------------------------------------------------------------
// Batched device tests: one block per row.
// ---------------------------------------------------------------------------

namespace {

// (batch_size, V) shapes that corner different parts of the launch:
//  - (2, 3):       a second block exists at all, and most threads in each
//                  block have no element.
//  - (3, 1000):    several full grid-stride iterations per thread.
//  - (170, 257):   one block per SM on the 170-SM RTX 5090, ragged tail.
//  - (171, 256):   more blocks than SMs, so blocks share an SM and their
//                  __shared__ m / d instances have to stay per-block.
//  - (1024, 1000): a training-sized chunk scheduled over several waves.
const std::vector<std::pair<int, int>> kBatchShapes = {
    {2, 3}, {3, 1000}, {170, 257}, {171, 256}, {1024, 1000}};

std::string ShapeTrace(int batch_size, int V) {
  return "batch_size = " + std::to_string(batch_size) +
         ", V = " + std::to_string(V);
}

} // namespace

// The batched analogue of MatchesHost: the naive kernel against the naive host
// loop, on rows that differ so a block reading its neighbour's row is caught.
TEST_F(SoftmaxDevice, NaiveMatchesHostBatched) {
  const int V = 1000;
  const int batch_size = 64;
  const std::vector<float> x = DistinctRows(batch_size, V, -10.0f, 10.0f);

  std::vector<float> want(x.size());
  softmax_host(x.data(), want.data(), V, batch_size, SoftmaxType::Naive);

  std::vector<float> got;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Naive, x, V, got));

  ExpectBatchNear(got, want, V, DeviceTol);
  ExpectRowsAreDistributions(got, V);
}

TEST_F(SoftmaxDevice, SafeMatchesReferenceAcrossBatchShapes) {
  for (const auto &[batch_size, V] : kBatchShapes) {
    SCOPED_TRACE(ShapeTrace(batch_size, V));

    const std::vector<float> x = DistinctRows(batch_size, V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmaxBatched(x, V);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, V, got));

    ExpectBatchNear(got, want, V, DeviceTol);
    ExpectRowsAreDistributions(got, V);
  }
}

TEST_F(SoftmaxDevice, OnlineMatchesReferenceAcrossBatchShapes) {
  for (const auto &[batch_size, V] : kBatchShapes) {
    SCOPED_TRACE(ShapeTrace(batch_size, V));

    const std::vector<float> x = DistinctRows(batch_size, V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmaxBatched(x, V);

    std::vector<float> got;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, V, got));

    ExpectBatchNear(got, want, V, DeviceTol);
    ExpectRowsAreDistributions(got, V);
  }
}

// The device version of SoftmaxHostBatch.RowsAreIndependent, at a V that makes
// every thread loop and gives each row a different failure mode: a plain ramp,
// an overflowing row, an underflowing row, a causal row 0, an interior mask,
// a leading mask, and a final plain row that would inherit any state the
// masked rows leaked. For safe and online every row must match the reference;
// for naive rows 1 and 2 are NaN by construction and the rest must not be.
TEST_F(SoftmaxDevice, RowsAreIndependentBatched) {
  const int V = 300;
  std::vector<float> interior = Ramp(V, -10.0f, 10.0f);
  interior[V / 2] = kNegInf;
  std::vector<float> leading = Ramp(V, -10.0f, 10.0f);
  leading[0] = kNegInf;

  const std::vector<float> x = Batch({
      Ramp(V, -10.0f, 10.0f),
      std::vector<float>(V, 89.0f),
      std::vector<float>(V, -500.0f),
      CausalRowZero(V),
      interior,
      leading,
      Ramp(V, 10.0f, -10.0f),
  });
  const std::vector<float> want = ReferenceSoftmaxBatched(x, V);

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, V, safe));
  {
    SCOPED_TRACE("safe");
    ExpectBatchNear(safe, want, V, DeviceTol);
    ExpectRowsAreDistributions(safe, V);
  }

  std::vector<float> online;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, V, online));
  {
    SCOPED_TRACE("online");
    ExpectBatchNear(online, want, V, DeviceTol);
    ExpectRowsAreDistributions(online, V);
  }

  std::vector<float> naive;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Naive, x, V, naive));
  for (const int r : {0, 3, 4, 5, 6}) {
    SCOPED_TRACE("naive, row " + std::to_string(r));
    const std::vector<float> got = Row(naive, V, r);
    const std::vector<float> row_want = Row(want, V, r);
    for (int i = 0; i < V; ++i)
      EXPECT_NEAR(got[i], row_want[i], DeviceTol(row_want[i]))
          << "index " << i;
    ExpectIsDistribution(got);
  }
  for (const int r : {1, 2})
    for (const float v : Row(naive, V, r))
      EXPECT_TRUE(std::isnan(v))
          << "expected naive softmax to overflow/underflow on row " << r
          << ", got " << v;
}

// The batched form of OnlineMatchesSafeOnBothRampDirections: even rows ascend
// (the running max moves on every element of a thread's stride), odd rows
// descend (it moves once). With 170 blocks resident at once, a rescale that
// picked up a neighbouring block's m would show as a mismatch here.
TEST_F(SoftmaxDevice, OnlineMatchesSafeBatched) {
  const int V = 4096;
  const int batch_size = 170;
  std::vector<std::vector<float>> rows;
  rows.reserve(batch_size);
  for (int r = 0; r < batch_size; ++r)
    rows.push_back(r % 2 == 0 ? Ramp(V, -10.0f, 10.0f)
                              : Ramp(V, 10.0f, -10.0f));
  const std::vector<float> x = Batch(rows);

  std::vector<float> safe;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, V, safe));

  std::vector<float> online;
  ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, V, online));

  ExpectBatchNear(online, safe, V, DeviceTol);
  ExpectRowsAreDistributions(online, V);
}

// Real LM-head shapes: a decode batch of 64 over the Llama 2 vocabulary and a
// small batch over the Llama 3 vocabulary. V is far past any earlier test, so
// each thread loops hundreds of times and the block reduction sums partials
// that are themselves long serial sums.
TEST_F(SoftmaxDevice, VocabSizedRowsBatched) {
  const std::vector<std::pair<int, int>> shapes = {{64, 32000}, {8, 128256}};

  for (const auto &[batch_size, V] : shapes) {
    SCOPED_TRACE(ShapeTrace(batch_size, V));

    const std::vector<float> x = DistinctRows(batch_size, V, -10.0f, 10.0f);
    const std::vector<float> want = ReferenceSoftmaxBatched(x, V);

    std::vector<float> safe;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Safe, x, V, safe));
    {
      SCOPED_TRACE("safe");
      ExpectBatchNear(safe, want, V, DeviceTol);
      ExpectRowsAreDistributions(safe, V);
    }

    std::vector<float> online;
    ASSERT_NO_FATAL_FAILURE(Run(SoftmaxType::Online, x, V, online));
    {
      SCOPED_TRACE("online");
      ExpectBatchNear(online, want, V, DeviceTol);
      ExpectRowsAreDistributions(online, V);
    }
  }
}
