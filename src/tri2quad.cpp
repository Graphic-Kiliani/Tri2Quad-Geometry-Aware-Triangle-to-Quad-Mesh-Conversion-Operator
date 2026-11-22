#include "tri2quad.hpp"
#include "matching_lemon.hpp"
#include "spiral_repair.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <iostream>

static inline tri2quad::Edge edge_canon(int a, int b) {
    if (a > b) std::swap(a, b);
    return tri2quad::Edge{a, b};
}
static inline double dot3(const tri2quad::Vec3& a, const tri2quad::Vec3& b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static inline tri2quad::Vec3 sub3(const tri2quad::Vec3& a, const tri2quad::Vec3& b){
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
static inline tri2quad::Vec3 cross3(const tri2quad::Vec3& a, const tri2quad::Vec3& b){
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
static inline double norm3(const tri2quad::Vec3& a){
    return std::sqrt(dot3(a,a));
}
static inline double angle_between(const tri2quad::Vec3& u, const tri2quad::Vec3& v){
    double nu = norm3(u), nv = norm3(v);
    if (nu==0.0 || nv==0.0) return 0.0;
    double c = std::max(-1.0, std::min(1.0, dot3(u,v)/(nu*nv)));
    return std::acos(c); // radians
}

static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

static tri2quad::Vec3 normalize(const tri2quad::Vec3& v){
    double n = norm3(v);
    if (n==0) return {0,0,0};
    return {v[0]/n, v[1]/n, v[2]/n};
}

// 二面角（0..pi），不关心法向方向
static double dihedral_angle_between_tris(
    const std::vector<tri2quad::Vec3>& V,
    const tri2quad::Tri& A,
    const tri2quad::Tri& B)
{
    // 用前三个点定义两个三角的法向
    auto nA = normalize(cross3(sub3(V[A[1]],V[A[0]]), sub3(V[A[2]],V[A[0]])));
    auto nB = normalize(cross3(sub3(V[B[1]],V[B[0]]), sub3(V[B[2]],V[B[0]])));
    double c = std::max(-1.0, std::min(1.0, dot3(nA,nB)));
    return std::acos(c); // radians
}

// 近似平面性：四点对拟合平面的最大距离 / 平均边长
static double quad_planarity_ratio(const std::array<tri2quad::Vec3,4>& q){
    auto e01 = sub3(q[1],q[0]);
    auto e02 = sub3(q[2],q[0]);
    auto n = normalize(cross3(e01, e02));
    // 若法向为零，退化：视作非平面（返回大值）
    double mean_edge = 0.0;
    tri2quad::Vec3 edges[4] = { sub3(q[1],q[0]), sub3(q[2],q[1]), sub3(q[3],q[2]), sub3(q[0],q[3]) };
    for (int i=0;i<4;++i) mean_edge += norm3(edges[i]);
    mean_edge = mean_edge/4.0 + 1e-12;

    auto dist = [&](const tri2quad::Vec3& p)->double{
        auto v = sub3(p, q[0]);
        return std::abs(dot3(n,v));
    };
    double dmax = std::max(std::max(dist(q[0]), dist(q[1])), std::max(dist(q[2]), dist(q[3])));
    if (n[0]==0 && n[1]==0 && n[2]==0) return 1e9; // 强制视作不平面
    return dmax / mean_edge;
}

// 凹性检测（前置）：先做一次平面性近似，再在该平面方向上检查四个角的"同向性"
static bool is_concave_quad(const std::array<tri2quad::Vec3,4>& q){
    // 参考法向（用前三点）
    auto n = normalize(cross3(sub3(q[1],q[0]), sub3(q[2],q[0])));
    if (n[0]==0 && n[1]==0 && n[2]==0) return true; // 退化视作不合格

    int sign0 = 0;
    for (int i=0;i<4;++i){
        const auto& pi = q[i];
        const auto& pn = q[(i+1)%4];
        const auto& pp = q[(i+3)%4];
        auto c = cross3(sub3(pn,pi), sub3(pp,pi)); // (pi->pn) x (pi->pp)
        double s = dot3(c, n);
        int si = (s>1e-12) ? 1 : (s<-1e-12 ? -1 : 0);
        if (si==0) continue;
        if (sign0==0) sign0 = si;
        else if (si != sign0) return true; // 出现符号不一致：凹/自交
    }
    return false; // 全部同号或近似共线：视作凸
}

// 计算四边形四个内角（弧度），顶点按顺序 q[0..3]
static std::array<double,4> quad_angles(
    const std::array<tri2quad::Vec3,4>& q)
{
    std::array<double,4> A{};
    for (int i=0;i<4;++i){
        const auto& p  = q[i];
        const auto& p0 = q[(i+3)%4];
        const auto& p1 = q[(i+1)%4];
        auto v0 = sub3(p0,p);
        auto v1 = sub3(p1,p);
        A[i] = angle_between(v0, v1);
    }
    return A;
}

// quad quality score (0-1)
static double quad_quality(
    const std::array<tri2quad::Vec3,4>& q)
{
    // angle eval
    auto ang = quad_angles(q);
    double score_angle = 0.0;
    for (int i=0;i<4;++i){
        double deg = ang[i]*180.0/M_PI;
        score_angle += std::max(0.0, 90.0 - std::abs(deg - 90.0)); // ∈[0,90]
    }
    score_angle /= 360.0; // ∈[0,1]

    // aspect ratio eval
    auto e01 = norm3(sub3(q[1],q[0]));
    auto e12 = norm3(sub3(q[2],q[1]));
    auto e23 = norm3(sub3(q[3],q[2]));
    auto e30 = norm3(sub3(q[0],q[3]));
    auto d02 = norm3(sub3(q[2],q[0]));
    auto d13 = norm3(sub3(q[3],q[1]));

    auto min_e = std::min(std::min(e01,e12), std::min(e23,e30));
    auto max_e = std::max(std::max(e01,e12), std::max(e23,e30));
    double ratio_edge = (max_e>0)? (min_e/max_e) : 0.0; // ∈[0,1]

    auto dd = (std::min(d02,d13) / (std::max(d02,d13)+1e-12)); // ∈[0,1]
    double score_shape = 0.5*ratio_edge + 0.5*dd;

    // overall eval
    return 0.6*score_angle + 0.4*score_shape; // ∈[0,1]
}

// ============================================================
// Edge Flow Alignment: 预计算每个顶点的"主曲率方向"（离散 PCA 方法）
// 用途：估计 mesh 表面局部的自然 flow 方向
// ============================================================

// 计算每个顶点的主方向：用一环邻域的 edge 做 PCA，取最大特征值对应的方向
// 返回：per-vertex 方向向量（投影到切平面上的主方向）
static std::vector<tri2quad::Vec3> compute_vertex_principal_dirs(
    const std::vector<tri2quad::Vec3>& V,
    const std::vector<tri2quad::Tri>&  F)
{
    int nV = (int)V.size();
    // 1) 建顶点邻接表
    std::vector<std::vector<int>> adj(nV);
    for (auto& f : F){
        for (int i=0;i<3;++i){
            int a = f[i], b = f[(i+1)%3];
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }
    // 去重
    for (auto& nb : adj){
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    // 2) 估计顶点法向量（area-weighted face normal average）
    std::vector<tri2quad::Vec3> vnormals(nV, {0,0,0});
    for (auto& f : F){
        auto e1 = sub3(V[f[1]], V[f[0]]);
        auto e2 = sub3(V[f[2]], V[f[0]]);
        auto fn = cross3(e1, e2); // area-weighted
        for (int i=0;i<3;++i){
            vnormals[f[i]][0] += fn[0];
            vnormals[f[i]][1] += fn[1];
            vnormals[f[i]][2] += fn[2];
        }
    }
    for (auto& n : vnormals) n = normalize(n);

    // 3) 对每个顶点，收集一环 edge vectors，投影到切平面，做 2x2 PCA
    std::vector<tri2quad::Vec3> principal_dirs(nV, {1,0,0}); // fallback
    for (int vi=0; vi<nV; ++vi){
        if (adj[vi].empty()) continue;
        auto& N = vnormals[vi];
        if (N[0]==0 && N[1]==0 && N[2]==0) continue;

        // 在切平面上做协方差分析
        // 先构造切平面坐标系 (u, v)
        // 找一个不与 N 平行的向量作为参考
        tri2quad::Vec3 ref = {1,0,0};
        if (std::abs(dot3(N, ref)) > 0.9) ref = {0,1,0};
        auto u = normalize(cross3(N, ref));
        auto v = cross3(N, u); // 已经是单位向量

        // 对一环邻域所有 edge 做协方差
        double cov00=0, cov01=0, cov11=0;
        for (int nb : adj[vi]){
            auto e = sub3(V[nb], V[vi]);
            double len = norm3(e);
            if (len < 1e-12) continue;
            // 投影到切平面
            double pu = dot3(e, u);
            double pv = dot3(e, v);
            // 累加外积（用方向不带长度权重，或带长度权重）
            // 这里用归一化方向，避免被一条很长的边主导
            double plen = std::sqrt(pu*pu + pv*pv);
            if (plen < 1e-12) continue;
            pu /= plen; pv /= plen;
            cov00 += pu*pu;
            cov01 += pu*pv;
            cov11 += pv*pv;
        }

        // 2x2 对称矩阵的主特征向量
        // [[cov00, cov01],[cov01, cov11]]
        // 特征值: lambda = (tr ± sqrt(tr^2 - 4*det)) / 2
        double tr = cov00 + cov11;
        double det = cov00*cov11 - cov01*cov01;
        double disc = tr*tr - 4*det;
        if (disc < 0) disc = 0;
        double lambda1 = (tr + std::sqrt(disc)) / 2.0; // 较大特征值

        // 对应的特征向量
        double ex, ey;
        if (std::abs(cov01) > 1e-12) {
            ex = lambda1 - cov11;
            ey = cov01;
        } else {
            // 对角矩阵
            if (cov00 >= cov11) { ex = 1; ey = 0; }
            else { ex = 0; ey = 1; }
        }
        double elen = std::sqrt(ex*ex + ey*ey);
        if (elen < 1e-12) continue;
        ex /= elen; ey /= elen;

        // 转回 3D：principal_dir = ex * u + ey * v
        principal_dirs[vi] = {
            ex*u[0] + ey*v[0],
            ex*u[1] + ey*v[1],
            ex*u[2] + ey*v[2]
        };
    }
    return principal_dirs;
}

// Q_align: 评估合并边方向与局部 flow 方向（principal direction）的对齐度
// 合并边应该正交于 flow 方向（合并边消失后，保留的边沿 flow 方向）
// 返回值 ∈ [0, 1]，越高表示合并边越适合被合并
// 命名沿用论文 arXiv:2605.16813 中的 Q_align
static double compute_edge_align_score(
    const std::vector<tri2quad::Vec3>& V,
    const std::vector<tri2quad::Vec3>& principal_dirs,
    const tri2quad::Edge& edge)
{
    int v0 = edge[0], v1 = edge[1];
    auto edge_dir = normalize(sub3(V[v1], V[v0]));
    if (edge_dir[0]==0 && edge_dir[1]==0 && edge_dir[2]==0) return 0.5;

    // 取两端点主方向的平均作为局部 flow 方向
    auto d0 = principal_dirs[v0];
    auto d1 = principal_dirs[v1];
    tri2quad::Vec3 flow_dir = normalize({
        (d0[0]+d1[0])/2.0,
        (d0[1]+d1[1])/2.0,
        (d0[2]+d1[2])/2.0
    });
    if (flow_dir[0]==0 && flow_dir[1]==0 && flow_dir[2]==0) return 0.5;

    // |sin(angle)| = sqrt(1 - cos^2)
    // 合并边与 flow 方向正交时得分最高
    double cos_val = std::abs(dot3(edge_dir, flow_dir));
    cos_val = std::min(1.0, cos_val);
    double sin_val = std::sqrt(1.0 - cos_val*cos_val);
    return sin_val; // ∈ [0, 1]
}

// ============================================================
// 对称面自动检测（自包含：顶点 PCA + 反射覆盖率，无需 BVH/libigl）
// ============================================================

// 3x3 对称矩阵 Jacobi 特征分解（移植自 symmetry_detector，仅依赖 std::array + cmath）
static void jacobi_eigen_3x3(std::array<std::array<double,3>,3> a,
                             std::array<double,3>& values,
                             std::array<tri2quad::Vec3,3>& vectors)
{
    std::array<std::array<double,3>,3> v{};
    for (int i=0;i<3;++i) v[i][i] = 1.0;

    for (int iter=0; iter<50; ++iter){
        int p=0, q=1;
        double max_off = std::abs(a[0][1]);
        if (std::abs(a[0][2]) > max_off){ p=0; q=2; max_off=std::abs(a[0][2]); }
        if (std::abs(a[1][2]) > max_off){ p=1; q=2; max_off=std::abs(a[1][2]); }
        if (max_off < 1e-14) break;

        const double app=a[p][p], aqq=a[q][q], apq=a[p][q];
        const double theta = 0.5*std::atan2(2.0*apq, aqq-app);
        const double c = std::cos(theta), s = std::sin(theta);

        for (int k=0;k<3;++k){
            const double akp=a[k][p], akq=a[k][q];
            a[k][p] = c*akp - s*akq;
            a[k][q] = s*akp + c*akq;
        }
        for (int k=0;k<3;++k){
            const double apk=a[p][k], aqk=a[q][k];
            a[p][k] = c*apk - s*aqk;
            a[q][k] = s*apk + c*aqk;
        }
        for (int k=0;k<3;++k){
            const double vkp=v[k][p], vkq=v[k][q];
            v[k][p] = c*vkp - s*vkq;
            v[k][q] = s*vkp + c*vkq;
        }
    }

    values = { a[0][0], a[1][1], a[2][2] };
    vectors = {
        normalize({v[0][0], v[1][0], v[2][0]}),
        normalize({v[0][1], v[1][1], v[2][1]}),
        normalize({v[0][2], v[1][2], v[2][2]})
    };
}

// 均匀网格哈希：回答"查询点 eps 内是否存在顶点"（cell = eps 时 3x3x3 邻域即精确）
struct VertexEpsGrid {
    double cell;
    std::unordered_map<long long, std::vector<int>> buckets;
    const std::vector<tri2quad::Vec3>* V;

    static long long key_of(int ix, int iy, int iz){
        // 乘法空间哈希：允许碰撞（碰撞只影响性能，桶内仍按精确距离判定）
        return (long long)ix*73856093LL ^ (long long)iy*19349663LL ^ (long long)iz*83492791LL;
    }

    VertexEpsGrid(const std::vector<tri2quad::Vec3>& verts, double eps) : cell(eps), V(&verts) {
        for (int i=0;i<(int)verts.size();++i){
            int ix = (int)std::floor(verts[i][0]/cell);
            int iy = (int)std::floor(verts[i][1]/cell);
            int iz = (int)std::floor(verts[i][2]/cell);
            buckets[key_of(ix,iy,iz)].push_back(i);
        }
    }

    bool exists_within(const tri2quad::Vec3& p, double eps) const {
        int cx = (int)std::floor(p[0]/cell);
        int cy = (int)std::floor(p[1]/cell);
        int cz = (int)std::floor(p[2]/cell);
        const double eps2 = eps*eps;
        for (int dx=-1;dx<=1;++dx)
        for (int dy=-1;dy<=1;++dy)
        for (int dz=-1;dz<=1;++dz){
            auto it = buckets.find(key_of(cx+dx, cy+dy, cz+dz));
            if (it == buckets.end()) continue;
            for (int vi : it->second){
                const auto& q = (*V)[vi];
                double ddx = q[0]-p[0], ddy = q[1]-p[1], ddz = q[2]-p[2];
                if (ddx*ddx + ddy*ddy + ddz*ddz < eps2) return true;
            }
        }
        return false;
    }
};

// 顶点反射覆盖率：把每个顶点关于平面 (n, d) 反射，eps 内有最近顶点的比例
static double plane_reflection_coverage(
    const std::vector<tri2quad::Vec3>& V,
    const VertexEpsGrid& grid,
    const tri2quad::Vec3& n,
    double d,
    double eps)
{
    int hits = 0;
    for (const auto& p : V){
        double s = dot3(n, p) + d;
        tri2quad::Vec3 r = { p[0]-2.0*s*n[0], p[1]-2.0*s*n[1], p[2]-2.0*s*n[2] };
        if (grid.exists_within(r, eps)) ++hits;
    }
    return (double)hits / (double)V.size();
}

// 🔥 从两三角 + 共边，构造"有序四边形"顶点序列（保持法向量方向一致）
// 核心思路：通过检查共享边在每个三角形中的顺序，来确定正确的四边形绕向
// 输入：三角 triA=(a0,a1,a2), triB=(b0,b1,b2), shared edge=(s0,s1)
// 输出：四边形顶点坐标数组，保证绕向与原始三角形一致
static std::array<tri2quad::Vec3,4> build_ordered_quad_pts(
    const std::vector<tri2quad::Vec3>& V,
    const tri2quad::Tri& triA,
    const tri2quad::Tri& triB,
    const tri2quad::Edge& shared)
{
    int s0 = shared[0], s1 = shared[1];

    // 找到两个三角形中不在共享边上的顶点（opposite vertex）
    int a_opp = -1, b_opp = -1;
    for (int k = 0; k < 3; ++k) {
        if (triA[k] != s0 && triA[k] != s1) a_opp = triA[k];
        if (triB[k] != s0 && triB[k] != s1) b_opp = triB[k];
    }

    // 🔥 关键：找到共享边在triA中的方向
    // 在triA中，共享边可能是 s0→s1 或 s1→s0
    // 我们需要按照triA的顶点顺序来判断
    bool edge_forward_in_A = false;  // s0→s1 in triA?
    for (int k = 0; k < 3; ++k) {
        int v_curr = triA[k];
        int v_next = triA[(k + 1) % 3];
        if ((v_curr == s0 && v_next == s1)) {
            edge_forward_in_A = true;
            break;
        }
    }

    // 🔥 构建四边形：保持triA的绕向
    // triA的绕向是 ...→(共享边起点)→(共享边终点)→...
    // 四边形应该是：a_opp → (共享边起点) → b_opp → (共享边终点) → 回到a_opp
    std::array<tri2quad::Vec3,4> q;
    if (edge_forward_in_A) {
        // triA中是 s0→s1，所以四边形是 [a_opp, s0, b_opp, s1]
        q = {V[a_opp], V[s0], V[b_opp], V[s1]};
    } else {
        // triA中是 s1→s0，所以四边形是 [a_opp, s1, b_opp, s0]
        q = {V[a_opp], V[s1], V[b_opp], V[s0]};
    }

    return q;
}

namespace tri2quad {

SymmetryPlaneResult detect_symmetry_plane(
    const std::vector<Vec3>& V,
    const std::vector<Tri>&  F,
    double eps_ratio,
    double min_coverage,
    bool verbose,
    double protect_tol,
    double max_inband_ratio)
{
    (void)F; // 检测仅基于顶点，faces 保留在签名里以便将来扩展
    SymmetryPlaneResult res;
    const int nV = (int)V.size();
    if (nV < 4) return res;

    // bbox 对角线 & 质心
    Vec3 bmin = V[0], bmax = V[0];
    Vec3 c = {0,0,0};
    for (const auto& p : V){
        for (int k=0;k<3;++k){
            bmin[k] = std::min(bmin[k], p[k]);
            bmax[k] = std::max(bmax[k], p[k]);
            c[k] += p[k];
        }
    }
    double diag = norm3(sub3(bmax, bmin));
    if (diag <= 0.0) return res;
    for (int k=0;k<3;++k) c[k] /= nV;

    const double eps = std::max(eps_ratio * diag, 1e-12);

    // 顶点协方差（均匀权重）→ PCA 三主轴
    std::array<std::array<double,3>,3> cov{};
    for (const auto& p : V){
        double dx = p[0]-c[0], dy = p[1]-c[1], dz = p[2]-c[2];
        cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz;
        cov[1][0] += dy*dx; cov[1][1] += dy*dy; cov[1][2] += dy*dz;
        cov[2][0] += dz*dx; cov[2][1] += dz*dy; cov[2][2] += dz*dz;
    }
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) cov[i][j] /= nV;

    std::array<double,3> evals;
    std::array<Vec3,3> axes;
    jacobi_eigen_3x3(cov, evals, axes);

    VertexEpsGrid grid(V, eps);

    // 3 个候选平面：法线 = 各主轴，过质心；选覆盖率最高的轴
    int best_axis = -1;
    double best_cov = -1.0;
    for (int i=0;i<3;++i){
        double d0 = -dot3(axes[i], c);

        // 中面退化检查：容差带内顶点占比过高 ⇒ 该平面是物体自身的中面
        // （扁平/细杆物体），若以它做保护会吞掉大部分边，跳过该候选轴
        if (protect_tol > 0.0) {
            int inband = 0;
            for (const auto& p : V){
                if (std::abs(dot3(axes[i], p) + d0) < protect_tol) ++inband;
            }
            double inband_frac = (double)inband / (double)nV;
            if (inband_frac > max_inband_ratio){
                if (verbose){
                    std::cerr << "[sym detect] PCA axis " << i
                              << " SKIPPED (mid-surface plane: in-band vertex fraction="
                              << inband_frac << " > " << max_inband_ratio << ")\n";
                }
                continue;
            }
        }

        double cov_i = plane_reflection_coverage(V, grid, axes[i], d0, eps);
        if (verbose){
            std::cerr << "[sym detect] PCA axis " << i
                      << " n=(" << axes[i][0] << "," << axes[i][1] << "," << axes[i][2]
                      << ") coverage=" << cov_i << "\n";
        }
        if (cov_i > best_cov){ best_cov = cov_i; best_axis = i; }
    }
    if (best_axis < 0) return res;

    const Vec3& n = axes[best_axis];
    const double d_center = -dot3(n, c);

    // offset 细化：d = d_center + t，在 ±0.03·diag 内粗扫 + 两轮局部细化
    const double range = 0.03 * diag;
    double best_t = 0.0;
    {
        const int steps = 60;
        for (int i=0;i<=steps;++i){
            double t = -range + 2.0*range*i/steps;
            double cov_t = plane_reflection_coverage(V, grid, n, d_center + t, eps);
            if (cov_t > best_cov){ best_cov = cov_t; best_t = t; }
        }
        // 局部细化两轮（步长每次 /10）
        double step = 2.0*range/steps;
        for (int round=0; round<2; ++round){
            step /= 10.0;
            for (int i=-10;i<=10;++i){
                double t = best_t + i*step;
                double cov_t = plane_reflection_coverage(V, grid, n, d_center + t, eps);
                if (cov_t > best_cov){ best_cov = cov_t; best_t = t; }
            }
        }
    }

    res.coverage = best_cov;
    res.normal = n;
    res.d = d_center + best_t;
    res.found = (best_cov >= min_coverage);

    if (verbose){
        std::cerr << "[sym detect] best plane n=(" << n[0] << "," << n[1] << "," << n[2]
                  << ") d=" << res.d << " coverage=" << best_cov
                  << (res.found ? " ACCEPTED" : " REJECTED")
                  << " (min_coverage=" << min_coverage << ")\n";
    }
    return res;
}

Result tri_to_quad(
    const std::vector<Vec3>& V,
    const std::vector<Tri>&  F,
    double angle_threshold_deg,
    const std::string& method,
    bool verbose,
    // 新增参数（必须和 .hpp 顺序一致）
    double pre_angle_min_deg,
    double pre_angle_max_deg,
    double feature_dihedral_thresh_deg,
    bool   enable_concavity_prefilter,
    double planarity_eps,
    // Edge-Align 参数（命名沿用论文 arXiv:2605.16813 的 Q_align）
    bool   enable_edge_align,
    double alpha_angle,
    double alpha_aspect,
    double alpha_align,
    // 对称面保护边
    const std::vector<double>& sym_planes,
    double sym_plane_tol,
    double sym_edge_beta,
    // Spiral 修复（方案B：交错链翻转，挑最优）
    bool spiral_repair,
    int  spiral_repair_k,
    int  spiral_repair_topk,
    int  spiral_repair_lmax,
    int  spiral_repair_max_steps,
    int  spiral_repair_threads)
{
    Result R;
    R.new_vertices = V;

    // === [新增：把"度"参数转弧度，供前置过滤用] ===
    double pre_angle_min_rad = pre_angle_min_deg * M_PI / 180.0;
    double pre_angle_max_rad = pre_angle_max_deg * M_PI / 180.0;
    double feature_dihedral_thresh_rad = feature_dihedral_thresh_deg * M_PI / 180.0;

    // === [Edge-Align] 预计算主曲率方向 ===
    std::vector<Vec3> principal_dirs;
    if (enable_edge_align && alpha_align > 0.0) {
        principal_dirs = compute_vertex_principal_dirs(V, F);
        if (verbose) {
            std::cerr << "[tri2quad] Edge-align scoring ENABLED (alpha_align="
                      << alpha_align << ")\n";
        }
    }

    // 1) 建立 edge -> {tri idx...}
    struct TriRef { int t; int a,b; }; // t=三角索引; 边 a-b
    std::unordered_map<long long, std::vector<int>> edge2tris;
    auto key_of = [](int i, int j)->long long {
        if (i>j) std::swap(i,j);
        return ( (long long)i<<32 ) | (unsigned)j;
    };

    auto tri_edges = [&](const Tri& t)->std::array<Edge,3> {
        return std::array<Edge,3>{ edge_canon(t[0],t[1]),
                                   edge_canon(t[1],t[2]),
                                   edge_canon(t[2],t[0]) };
    };

    for (int ti=0; ti<(int)F.size(); ++ti){
        auto es = tri_edges(F[ti]);
        for (auto& e : es){
            edge2tris[key_of(e[0],e[1])].push_back(ti);
        }
    }

    // 2) 收集"内部边"候选（两侧各有一个三角）
    struct Candidate { Edge e; int tA; int tB; double w; };
    std::vector<Candidate> cands;
    // spiral 修复用：候选边 → {权重, 是否同时通过后置验证}（只有合法候选才可被翻转引入）
    std::unordered_map<long long, std::pair<double, bool>> repair_cand_info;
    const double post_thr_rad = angle_threshold_deg * M_PI / 180.0;
    std::vector<Edge> edges_in_order; // 与 R.weights 对齐
    for (auto& kv : edge2tris){
        auto& tris = kv.second;
        if (tris.size()!=2) continue; // 边界/非流形跳过
        int tA = tris[0], tB = tris[1];
        const auto& triA = F[tA];
        const auto& triB = F[tB];

        int s0 = (int)(kv.first>>32);
        int s1 = (int)(kv.first & 0xffffffff);
        Edge shared = edge_canon(s0,s1);

        // ====== [新增：前置过滤 ⓪ 对称面保护边] ======
        // 两端点都落在任一对称面容差内的边不进入候选池（永不合并）。
        // sym_edge_beta > 0 时，单条边的有效容差再收紧为 min(tol, beta*边长)：
        // 只保护真正"贴"在面上的边，纤细结构（被平面穿体而过的薄羽毛/薄片，
        // 端点距离≈边长量级）不会被误保护。
        if (sym_plane_tol > 0.0 && sym_planes.size() >= 4) {
            double thr = sym_plane_tol;
            if (sym_edge_beta > 0.0) {
                double edge_len = norm3(sub3(V[s0], V[s1]));
                thr = std::min(thr, sym_edge_beta * edge_len);
            }
            bool on_sym_plane = false;
            for (size_t pi = 0; pi + 3 < sym_planes.size(); pi += 4) {
                double nx = sym_planes[pi], ny = sym_planes[pi+1];
                double nz = sym_planes[pi+2], dd = sym_planes[pi+3];
                double d0 = std::abs(nx*V[s0][0] + ny*V[s0][1] + nz*V[s0][2] + dd);
                double d1 = std::abs(nx*V[s1][0] + ny*V[s1][1] + nz*V[s1][2] + dd);
                if (d0 < thr && d1 < thr) { on_sym_plane = true; break; }
            }
            if (on_sym_plane) {
                R.num_symmetry_protected_edges++;
                continue; // 保护：跳过该候选
            }
        }

        // 构造预合并四边形（与后续一致）
        auto qpts = build_ordered_quad_pts(V, triA, triB, shared);

        // ====== [新增：前置过滤 ① 角度预过滤] ======
        auto ang = quad_angles(qpts);
        double amax = std::max(std::max(ang[0],ang[1]), std::max(ang[2],ang[3]));
        double amin = std::min(std::min(ang[0],ang[1]), std::min(ang[2],ang[3]));
        // 注意：传入的 pre_angle_* 是"度"，这里比较"弧度"
        // 这些参数会在函数入参处转为弧度放到局部变量（见下文）
        if (amax > pre_angle_max_rad || amin < pre_angle_min_rad) {
            continue; // 前置剔除
        }

        // ====== [新增：前置过滤 ② 锐边/二面角预过滤] ======
        double dih = dihedral_angle_between_tris(V, triA, triB);
        if (dih > feature_dihedral_thresh_rad) {
            continue; // 二面角过大：保护特征边
        }

        // ====== [新增：前置过滤 ③ 凹四边形 + 平面性 预过滤] ======
        bool bad_concave = enable_concavity_prefilter && is_concave_quad(qpts);
        if (bad_concave) continue;

        if (planarity_eps > 0.0) {
            double r = quad_planarity_ratio(qpts);
            if (r > planarity_eps) continue; // 非平面性过大，提前剔除
        }

        // ---- 通过前置过滤，计算质量分 ----
        double q_geom = quad_quality(qpts); // 原始几何质量 (Q_angle + Q_aspect 内部)

        double w;
        if (enable_edge_align && alpha_align > 0.0) {
            // 分别计算各子项
            // Q_angle (从 quad_quality 中拆出来重新算)
            auto ang_scores = quad_angles(qpts);
            double score_angle = 0.0;
            for (int i=0;i<4;++i){
                double deg = ang_scores[i]*180.0/M_PI;
                score_angle += std::max(0.0, 90.0 - std::abs(deg - 90.0));
            }
            score_angle /= 360.0;

            // Q_aspect
            auto e01_l = norm3(sub3(qpts[1],qpts[0]));
            auto e12_l = norm3(sub3(qpts[2],qpts[1]));
            auto e23_l = norm3(sub3(qpts[3],qpts[2]));
            auto e30_l = norm3(sub3(qpts[0],qpts[3]));
            auto d02_l = norm3(sub3(qpts[2],qpts[0]));
            auto d13_l = norm3(sub3(qpts[3],qpts[1]));
            auto min_e_l = std::min(std::min(e01_l,e12_l), std::min(e23_l,e30_l));
            auto max_e_l = std::max(std::max(e01_l,e12_l), std::max(e23_l,e30_l));
            double ratio_edge_l = (max_e_l>0)? (min_e_l/max_e_l) : 0.0;
            auto dd_l = (std::min(d02_l,d13_l) / (std::max(d02_l,d13_l)+1e-12));
            double score_aspect = 0.5*ratio_edge_l + 0.5*dd_l;

            // Q_align (论文 arXiv:2605.16813)
            double score_align = compute_edge_align_score(V, principal_dirs, shared);

            // 加权组合（归一化使总权重为1；alpha_aspect 可置 0，此时 Q_aspect 退出）
            double total_alpha = alpha_angle + alpha_aspect + alpha_align;
            if (total_alpha <= 0.0) total_alpha = 1.0; // 兜底：避免除零
            w = (alpha_angle * score_angle + alpha_aspect * score_aspect + alpha_align * score_align) / total_alpha;
        } else {
            w = q_geom;
        }

        if (spiral_repair) {
            // 与第 4 步后置验证相同的判据：最大内角 > angle_threshold 或两三角法向相反 → 不合法
            double amax_post = std::max(std::max(ang[0],ang[1]), std::max(ang[2],ang[3]));
            auto nA_ = normalize(cross3(sub3(V[triA[1]], V[triA[0]]), sub3(V[triA[2]], V[triA[0]])));
            auto nB_ = normalize(cross3(sub3(V[triB[1]], V[triB[0]]), sub3(V[triB[2]], V[triB[0]])));
            bool post_ok = !(amax_post > post_thr_rad) && !(dot3(nA_, nB_) < 0.0);
            repair_cand_info[kv.first] = {w, post_ok};
        }

        cands.push_back({shared, tA, tB, w});
        edges_in_order.push_back(shared);
        R.weights.push_back(w);
    }
    R.num_internal_edges = (int)cands.size();

    // 3) 匹配：根据 method 选择 global(LEMON Blossom) 或 greedy
    std::vector<Candidate> chosen;
    chosen.reserve(cands.size());

    if (method == "global" || method == "blossom") {
        // method == "global" 走 LEMON 全局最大权匹配（基于 Edmonds Blossom 算法）；
        // 历史名 "blossom" 仍然兼容以避免破坏既有调用脚本。
        // 3.1 先把浮点权重归一化到整数（LEMON 接口用整数权）
        //     简单做法：w ∈ [0,1]，映射到 [0, 1e6]
        const int W_SCALE = 1000000;
        std::vector<std::array<int,3>> cand_for_lemon; // {tA, tB, w_scaled}
        cand_for_lemon.reserve(cands.size());
        for (auto &c : cands) {
            int ws = (int)std::llround(std::max(0.0, std::min(1.0, c.w)) * W_SCALE);
            cand_for_lemon.push_back({c.tA, c.tB, ws});
        }

        // 3.2 跑最大权匹配
        auto chosen_idx = max_weight_matching_lemon((int)F.size(), cand_for_lemon);
        chosen.reserve(chosen_idx.size());
        for (int idx : chosen_idx) {
            chosen.push_back(cands[idx]);
        }
    }
    else {
        // 3.x greedy 兜底（原实现）
        std::sort(cands.begin(), cands.end(),
                [](const Candidate& a, const Candidate& b){ return a.w > b.w; });
        std::vector<char> tri_used(F.size(), 0);
        for (auto& c : cands){
            if (!tri_used[c.tA] && !tri_used[c.tB]){
                chosen.push_back(c);
                tri_used[c.tA] = tri_used[c.tB] = 1;
            }
        }
    }

    // 4) 应用合并 + 几何验证（> angle_threshold 则回退）
    double thr = angle_threshold_deg * M_PI / 180.0;
    std::unordered_set<long long> merged_tris; merged_tris.reserve(chosen.size()*2);
    int flipped_count = 0;  // 统计翻转次数

    for (auto& c : chosen){
        // 四边形点序（使用build_ordered_quad_pts获取初始顺序）
        auto qpts = build_ordered_quad_pts(V, F[c.tA], F[c.tB], c.e);
        auto ang = quad_angles(qpts);
        double max_ang = std::max(std::max(ang[0],ang[1]), std::max(ang[2],ang[3]));
        if (max_ang > thr){
            R.num_rejected_quads++;
            continue; // 回退：保持为两个三角
        }

        // Step 1: 检查两个三角形的法向量是否一致
        const auto& triA = F[c.tA];
        const auto& triB = F[c.tB];

        // 计算两个三角形的法向量
        auto normalA = normalize(cross3(sub3(V[triA[1]], V[triA[0]]), sub3(V[triA[2]], V[triA[0]])));
        auto normalB = normalize(cross3(sub3(V[triB[1]], V[triB[0]]), sub3(V[triB[2]], V[triB[0]])));

        // 如果两个三角形法向量方向不一致（dot < 0），不应该合并！
        double dot_normals = dot3(normalA, normalB);
        if (dot_normals < 0.0) {
            R.num_rejected_quads++;
            continue; // 跳过：法向量方向相反，不合并
        }

        // 使用triA的法向量作为参考
        auto ref_normal = normalA;

        // Step 2: 生成四边形索引顺序（与build_ordered_quad_pts一致）
        int s0 = c.e[0], s1 = c.e[1];

        auto opp_of = [&](const Tri& t)->int{
            for (int k=0;k<3;++k){ if (t[k]!=s0 && t[k]!=s1) return t[k]; }
            return -1;
        };
        int a_opp = opp_of(triA);
        int b_opp = opp_of(triB);

        // 确定共享边在triA中的方向
        bool edge_forward_in_A = false;
        for (int k = 0; k < 3; ++k) {
            int v_curr = triA[k];
            int v_next = triA[(k + 1) % 3];
            if (v_curr == s0 && v_next == s1) {
                edge_forward_in_A = true;
                break;
            }
        }

        // 构建初始四边形索引
        Quad q_idx;
        if (edge_forward_in_A) {
            q_idx = { a_opp, s0, b_opp, s1 };
        } else {
            q_idx = { a_opp, s1, b_opp, s0 };
        }

        // Step 3: 计算四边形的法向量
        // 方法：将四边形 [v0,v1,v2,v3] 沿对角线拆成两个三角形，取法向量平均
        // 三角形1: [v0, v1, v2]
        // 三角形2: [v0, v2, v3]
        const auto& v0 = V[q_idx[0]];
        const auto& v1 = V[q_idx[1]];
        const auto& v2 = V[q_idx[2]];
        const auto& v3 = V[q_idx[3]];

        // 三角形1的法向量: (v1-v0) × (v2-v0)
        auto n1 = cross3(sub3(v1, v0), sub3(v2, v0));

        // 三角形2的法向量: (v2-v0) × (v3-v0)
        auto n2 = cross3(sub3(v2, v0), sub3(v3, v0));

        // 取平均并归一化
        Vec3 quad_normal = {
            (n1[0] + n2[0]) / 2.0,
            (n1[1] + n2[1]) / 2.0,
            (n1[2] + n2[2]) / 2.0
        };
        quad_normal = normalize(quad_normal);

        // Step 4: 检查法向量方向，如果相反则翻转四边形顶点顺序
        double dot_product = dot3(quad_normal, ref_normal);
        if (dot_product < 0) {
            // 法向量方向相反，翻转顶点顺序：[v0, v1, v2, v3] → [v3, v2, v1, v0]
            std::reverse(q_idx.begin(), q_idx.end());
            flipped_count++;
        }

        R.quads.push_back(q_idx);
        R.merge_edges.push_back(c.e);
        merged_tris.insert( (long long)c.tA );
        merged_tris.insert( (long long)c.tB );
        R.num_accepted_quads++;
    }

    if (verbose && flipped_count > 0) {
        std::cerr << "[tri2quad] Flipped " << flipped_count << "/" << R.num_accepted_quads
                  << " quads to maintain consistent normal direction\n";
    }

    // 5) 剩余未并的三角形直接保留
    R.tris.reserve(F.size());
    for (int ti=0; ti<(int)F.size(); ++ti){
        if (merged_tris.count(ti)==0){
            R.tris.push_back(F[ti]);
        }
    }

    // 6) Spiral 修复（方案B）：检测 → 交错链翻转（挑最优）→ 迭代，四边形数不减、只用合法候选
    if (spiral_repair) {
        SpiralRepairParams P;
        P.k = spiral_repair_k; P.topk = spiral_repair_topk; P.lmax = spiral_repair_lmax;
        P.max_steps = spiral_repair_max_steps; P.threads = spiral_repair_threads; P.verbose = verbose;
        auto S = repair_spirals(V, F, repair_cand_info, R.merge_edges, R.quads, R.tris, P);
        R.spiral_repair_ran = true;
        R.spiral_wind_before = S.wind_before;
        R.spiral_wind_after = S.wind_after;
        R.spiral_chains_after = S.chains_after;
        R.spiral_max_pair_m_after = S.max_pair_m_after;
        R.spiral_repair_steps = S.steps;
        R.spiral_repair_evals = S.evals;
        R.spiral_repair_seconds = S.seconds;
        R.spiral_wind_edges = S.wind_edges_after;
        R.num_accepted_quads = (int)R.quads.size();
    }

    if (verbose){
        std::cerr << "[tri2quad] internal_edges=" << R.num_internal_edges
                  << " accepted_quads=" << R.num_accepted_quads
                  << " rejected=" << R.num_rejected_quads
                  << " tris_left=" << R.tris.size();
        if (R.num_symmetry_protected_edges > 0){
            std::cerr << " symmetry_protected=" << R.num_symmetry_protected_edges;
        }
        if (R.spiral_repair_ran){
            std::cerr << " spiral_wind=" << R.spiral_wind_before << "->" << R.spiral_wind_after
                      << " repair_steps=" << R.spiral_repair_steps;
        }
        std::cerr << "\n";
    }
    return R;
}

}
