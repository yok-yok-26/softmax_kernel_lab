#pragma once

#include <vector>

void cpu_softmax_reference(const std::vector<float>& input, std::vector<float>& output, int rows, int cols);
