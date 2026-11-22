#pragma once
#include <vector>
#include <array>

// input: number of triangles, candidate edges
// output: selected edges indices
std::vector<int> max_weight_matching_lemon(
    int num_tris,
    const std::vector<std::array<int,3>>& cand   // {tA, tB, w_scaled}
);
