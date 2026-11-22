#pragma once
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace tri2quad {

// 基本类型
using Vec3 = std::array<double, 3>;
using Tri  = std::array<int, 3>;
using Quad = std::array<int, 4>;
using Edge = std::array<int, 2>; // 顶点索引对（按升序存储）

struct Result {
    std::vector<Vec3>  new_vertices; // 本实现不改顶点位置，原样返回
    std::vector<Tri>   tris;         // 剩余三角形
    std::vector<Quad>  quads;        // 新生成四边形
    std::vector<Edge>  merge_edges;  // 被合并的内部边（按顶点索引）
    std::vector<double> weights;     // |E_int| 的权重（与 edges_in_order 对齐）
    // 统计
    int num_internal_edges = 0;
    int num_accepted_quads = 0;
    int num_rejected_quads = 0;
    int num_symmetry_protected_edges = 0; // 因落在对称面上而被保护的内部边数
    // Spiral 修复（方案B）统计
    bool   spiral_repair_ran = false;
    int    spiral_wind_before = 0;        // 修复前绕圈边数（pair_m >= k）
    int    spiral_wind_after = 0;         // 修复后绕圈边数
    int    spiral_chains_after = 0;
    int    spiral_max_pair_m_after = 0;
    int    spiral_repair_steps = 0;       // 接受的翻转次数
    int    spiral_repair_evals = 0;       // 检测器调用次数
    double spiral_repair_seconds = 0.0;
    std::vector<Edge> spiral_wind_edges;  // 修复后残留绕圈边
};

// —— 对称面自动检测结果 ——
struct SymmetryPlaneResult {
    bool   found = false;
    Vec3   normal{0.0, 0.0, 1.0};
    double d = 0.0;          // 平面方程 n·x + d = 0
    double coverage = 0.0;   // 最优候选的反射覆盖率 ∈ [0,1]
};

// 主入口
Result tri_to_quad(
    const std::vector<Vec3>& vertices,
    const std::vector<Tri>&  faces,
    double angle_threshold_deg = 150.0,         // 后置验证：保持原有
    const std::string& method = "greedy",
    bool verbose = false,

    // —— 新增：三类"前置过滤"参数（全有默认值，保持兼容） ——
    double pre_angle_min_deg = 30.0,            // 角度-下限预过滤（前置）
    double pre_angle_max_deg = 140.0,           // 角度-上限预过滤（前置）
    double feature_dihedral_thresh_deg = 50.0,  // 二面角阈值（前置），保护锐边
    bool   enable_concavity_prefilter = true,   // 凹四边形前置过滤
    double planarity_eps = 0.0,                 // 平面性阈值；<=0 表示禁用

    // —— Edge-Align 打分项（命名沿用论文 arXiv:2605.16813 中的 Q_align） ——
    bool   enable_edge_align = false,           // 是否启用 Q_align 打分
    double alpha_angle = 0.6,                   // Q_angle 权重
    double alpha_aspect = 0.4,                  // Q_aspect 权重
    double alpha_align = 0.0,                   // Q_align 权重（启用时建议 0.2）

    // —— 对称面保护边：落在任一平面上的内部边不进入候选池 ——
    const std::vector<double>& sym_planes = {}, // 每平面 4 个数: nx,ny,nz,d（n·x + d = 0）
    double sym_plane_tol = 0.0,                 // 端点到平面的绝对距离容差；<=0 禁用
    double sym_edge_beta = 0.0,                 // >0 时，单条边的有效容差再收紧为
                                                // min(sym_plane_tol, beta*边长)：只保护
                                                // 真正"贴"在面上的边，避免纤细结构
                                                // （被平面穿体而过的薄羽毛/薄片）的边被误保护

    // —— Spiral 修复（方案B）：匹配完成后检测 spiral，在绕圈边附近枚举交错链
    //    （翻转后四边形数不减、只引入合法候选），每轮取前 topk 个并行检测，
    //    接受绕圈边最少的一个，直到 0 或无改进 ——
    bool spiral_repair = false,
    int  spiral_repair_k = 3,           // 绕圈判定：pair_m >= k
    int  spiral_repair_topk = 40,       // 每轮评估候选数
    int  spiral_repair_lmax = 5,        // 交错链最多涉及的已配对边数
    int  spiral_repair_max_steps = 1000,
    int  spiral_repair_threads = 0      // OpenMP 线程数，0 = 默认
);

// 对称面自动检测（自包含：顶点 PCA + 反射覆盖率，无需 BVH/libigl）：
// 3 个候选平面 = 以 PCA 三主轴为法线、过质心的平面，再做 offset 细化；
// coverage = 顶点反射后 eps 内存在最近顶点的比例，eps = eps_ratio × bbox 对角线；
// coverage >= min_coverage 才认为 found。
// protect_tol > 0 时启用"中面退化检查"：若某候选平面 protect_tol 容差带内的顶点
// 占比 > max_inband_ratio（扁平/细杆物体自身的中面，会把大部分边都"保护"掉），
// 该候选轴被跳过。
SymmetryPlaneResult detect_symmetry_plane(
    const std::vector<Vec3>& vertices,
    const std::vector<Tri>&  faces,
    double eps_ratio = 0.01,
    double min_coverage = 0.9,
    bool verbose = false,
    double protect_tol = 0.0,
    double max_inband_ratio = 0.2
);

} // namespace tri2quad
