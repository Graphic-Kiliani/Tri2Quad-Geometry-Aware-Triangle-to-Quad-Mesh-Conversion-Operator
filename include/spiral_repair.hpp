#pragma once
// Spiral loop 检测 + 方案B（交错链翻转，挑最优）修复。
// 检测器逐行移植自 spiral_detect.py::compute_open_ix（只保留绕圈边相关输出），
// 面/边编号顺序与 Python 版 _spiral_topo_from_quads_tris 完全一致。
#include "tri2quad.hpp"
#include <vector>
#include <unordered_map>

namespace tri2quad {

struct SpiralReport {
    int n_wind_edges = 0;          // edge_is_winding && edge_wind_pair_m >= k
    int n_chains = 0;              // 绕圈边连通分量数
    int max_pair_m = 0;            // max(edge_wind_pair_m)
    std::vector<Edge> wind_edges;  // 顶点对（升序）
};

// faces：先四边形后三角形（任意多边形亦可），顶点序即面环序
// stop_at_triangles=false（默认，三角形感知）：走带只在奇异点/边界处停，可沿三角形
//   邻边继续（与 Blender 的 Alt 选 edge loop 一致）。在 quad-dominant 网上，原规则会
//   把穿过三角形的螺旋切成短链而漏检。
// stop_at_triangles=true：原 sep_ix 行为（边一侧有非四边形面即停），仅用于对照。
SpiralReport detect_spirals(const std::vector<Vec3>& V,
                            const std::vector<std::vector<int>>& faces,
                            int k = 3, bool stop_at_triangles = false);

struct SpiralRepairParams {
    int  k = 3;           // 绕圈判定阈值 pair_m >= k
    int  topk = 40;       // 每轮评估的候选数
    int  lmax = 5;        // 交错链最多涉及的已配对边数
    int  max_steps = 1000;
    int  threads = 0;     // OpenMP 线程数，0 = 默认
    bool stop_at_triangles = false;  // 检测规则，见 detect_spirals
    bool verbose = false;
};

struct SpiralRepairStats {
    int wind_before = 0, wind_after = 0;
    int chains_after = 0, max_pair_m_after = 0;
    int steps = 0, evals = 0;
    double seconds = 0.0;
    std::vector<Edge> wind_edges_after;
};

// cand_info：内部边键 (min<<32|max) → {权重, 是否合法(前置过滤+后置验证均通过)}
// merged：当前匹配（合并边顶点对）。修复成功时就地改写 quads/tris/merged。
SpiralRepairStats repair_spirals(
    const std::vector<Vec3>& V,
    const std::vector<Tri>&  F,
    const std::unordered_map<long long, std::pair<double, bool>>& cand_info,
    std::vector<Edge>& merged,
    std::vector<Quad>& quads_out,
    std::vector<Tri>&  tris_out,
    const SpiralRepairParams& P);

} // namespace tri2quad
