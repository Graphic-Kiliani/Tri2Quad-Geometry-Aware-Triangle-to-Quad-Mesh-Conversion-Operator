#include "matching_lemon.hpp"

#ifdef USE_LEMON
// 用 LEMON 的最大权匹配
#include <lemon/list_graph.h>
#include <lemon/matching.h>
#include <cassert>
#include <cmath>
#include <limits>

std::vector<int> max_weight_matching_lemon(
    int num_tris,
    const std::vector<std::array<int,3>>& cand)  // {tA, tB, w_scaled(整数)}
{
    using namespace lemon;
    ListGraph g;
    // build graph: each triangle is a node
    std::vector<ListGraph::Node> nodes(num_tris);
    for (int i=0;i<num_tris;++i) nodes[i] = g.addNode();

    // edge and weight
    ListGraph::EdgeMap<int> weight(g);
    std::vector<ListGraph::Edge> edges;
    edges.reserve(cand.size());
    for (auto &c : cand) {
        int a = c[0], b = c[1], w = c[2];
        auto e = g.addEdge(nodes[a], nodes[b]);
        weight[e] = w;
        edges.push_back(e);
    }

    MaxWeightedMatching<ListGraph, ListGraph::EdgeMap<int>> mwm(g, weight);
    mwm.run();

    std::vector<int> chosen;
    chosen.reserve(cand.size());
    for (int i=0;i<(int)edges.size(); ++i) {
        auto e = edges[i];
        if (mwm.matching(e)) {
            chosen.push_back(i);
        }
    }
    return chosen;
}

#else
#include <stdexcept>
std::vector<int> max_weight_matching_lemon(
    int,
    const std::vector<std::array<int,3>>&)
{
    throw std::runtime_error(
        "[tri2quad] LEMON backend not available. "
        "Please configure CMake with -DUSE_LEMON=ON (and internet) to use method=\"blossom\"."
    );
}
#endif
