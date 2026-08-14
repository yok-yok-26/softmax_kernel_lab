#include "cpu_softmax.hpp"

#include <algorithm>
#include <cmath>

void cpu_softmax_reference(const std::vector<float>& input, std::vector<float>& output, int rows, int cols) {
  output.assign(static_cast<size_t>(rows) * cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* row_in = input.data() + static_cast<size_t>(r) * cols;
    float* row_out = output.data() + static_cast<size_t>(r) * cols;
    double max_v = static_cast<double>(row_in[0]);
    for (int c = 1; c < cols; ++c) {
      max_v = std::max(max_v, static_cast<double>(row_in[c]));
    }
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      double e = std::exp(static_cast<double>(row_in[c]) - max_v);
      row_out[c] = static_cast<float>(e);
      sum += e;
    }
    for (int c = 0; c < cols; ++c) {
      row_out[c] = static_cast<float>(static_cast<double>(row_out[c]) / sum);
    }
  }
}
