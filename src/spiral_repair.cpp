// Spiral loop 检测（移植自 spiral_detect.py::compute_open_ix）+ 方案B 修复。
#include "spiral_repair.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tri2quad {

static inline long long ekey(int a, int b) {
    if (a > b) std::swap(a, b);
    return ((long long)a << 32) | (unsigned)b;
}

// ============================================================
// 检测器
// ============================================================
namespace {

double median_of(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    if (n % 2) return xs[n / 2];
    return (xs[n / 2 - 1] + xs[n / 2]) / 2.0;
}

const double DOMINANT_GAP_RATIO = 3.0;

// _twopass_gap_index；返回 -1 表示 None
int twopass_gap_index(const std::vector<double>& gaps) {
    int n = (int)gaps.size();
    if (n < 3) return -1;
    int k = 0;
    for (int i = 1; i < n; ++i) if (gaps[i] > gaps[k]) k = i;  // 首个最大
    double gmax = gaps[k];
    std::vector<double> others;
    for (int i = 0; i < n; ++i) if (i != k) others.push_back(gaps[i]);
    if (others.empty()) return -1;
    double mx_o = *std::max_element(others.begin(), others.end());
    double med_o = median_of(others);
    if (gmax <= DOMINANT_GAP_RATIO * mx_o) return -1;
    if (med_o > 0.0 && gmax <= DOMINANT_GAP_RATIO * med_o) return -1;
    if (k + 1 < 2 || n - k < 2) return -1;
    return k;
}

inline double dist3(const Vec3& a, const Vec3& b) {
    double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// _i_sub_gaps：inter 为升序向量
void i_sub_gaps(const std::vector<int>& order, const std::vector<int>& inter,
                const std::vector<Vec3>& V, std::vector<double>& ge, std::vector<int>& gt) {
    std::vector<int> subpos;  // sub 元素在 order 中的位置（递增）
    for (int i = 0; i < (int)order.size(); ++i)
        if (std::binary_search(inter.begin(), inter.end(), order[i])) subpos.push_back(i);
    ge.clear(); gt.clear();
    for (int s = 0; s + 1 < (int)subpos.size(); ++s) {
        int i = subpos[s], j = subpos[s + 1];
        double eu = 0.0;
        for (int k = i; k < j; ++k) eu += dist3(V[order[k]], V[order[k + 1]]);
        ge.push_back(eu); gt.push_back(j - i);
    }
}

struct WLabel { int eu_lab; bool drop_u, drop_v; };

WLabel winding_label(const std::vector<int>& ou, const std::vector<int>& ov,
                     const std::vector<int>& inter, const std::vector<Vec3>& V) {
    std::vector<double> ge_u, ge_v; std::vector<int> gt_u, gt_v;
    i_sub_gaps(ou, inter, V, ge_u, gt_u);
    i_sub_gaps(ov, inter, V, ge_v, gt_v);
    // len(sub)<2 等价于 gaps 为空
    if (ge_u.empty() || ge_v.empty()) return {0, false, false};
    double eu_u = 0.0, eu_v = 0.0;
    for (double x : ge_u) eu_u += x;
    for (double x : ge_v) eu_v += x;
    bool du = false, dv = false;
    int ku = twopass_gap_index(ge_u), kv = twopass_gap_index(ge_v);
    double med_u = median_of(ge_u), med_v = median_of(ge_v);
    if (ku >= 0) {
        std::vector<double> rest;
        for (int i = 0; i < (int)ge_u.size(); ++i) if (i != ku) rest.push_back(ge_u[i]);
        if (median_of(rest) < med_v) { eu_u -= ge_u[ku]; du = true; }
    }
    if (kv >= 0) {
        std::vector<double> rest;
        for (int i = 0; i < (int)ge_v.size(); ++i) if (i != kv) rest.push_back(ge_v[i]);
        if (median_of(rest) < med_u) { eu_v -= ge_v[kv]; dv = true; }
    }
    int lab = (eu_u > eu_v) ? 1 : ((eu_v > eu_u) ? -1 : 0);
    return {lab, du, dv};
}

} // namespace

SpiralReport detect_spirals(const std::vector<Vec3>& V,
                            const std::vector<std::vector<int>>& faces, int k,
                            bool stop_at_triangles) {
    SpiralReport rep;
    const int n_f = (int)faces.size();
    // ---- 边编号（与 _spiral_topo_from_quads_tris 相同：按面序、面内环序首次出现编号）----
    std::unordered_map<long long, int> eid;
    eid.reserve(faces.size() * 3);
    std::vector<std::array<int, 2>> E;
    std::vector<int> fe_off(n_f + 1, 0), fe;   // face -> edges
    std::vector<int> nsides(n_f);
    std::vector<int> ef_cnt;                    // edge -> #faces
    fe.reserve(faces.size() * 4);
    for (int fi = 0; fi < n_f; ++fi) {
        const auto& f = faces[fi];
        int m = (int)f.size();
        nsides[fi] = m;
        for (int kk = 0; kk < m; ++kk) {
            int a = f[kk], b = f[(kk + 1) % m];
            long long key = ekey(a, b);
            auto it = eid.find(key);
            int e;
            if (it == eid.end()) {
                e = (int)E.size(); eid.emplace(key, e);
                E.push_back({std::min(a, b), std::max(a, b)});
                ef_cnt.push_back(0);
            } else e = it->second;
            ef_cnt[e]++;
            fe.push_back(e);
        }
        fe_off[fi + 1] = (int)fe.size();
    }
    const int n_e = (int)E.size();
    if (n_e == 0) return rep;
    // edge -> faces (CSR，按面序)
    std::vector<int> ef_off(n_e + 1, 0);
    for (int e = 0; e < n_e; ++e) ef_off[e + 1] = ef_off[e] + ef_cnt[e];
    std::vector<int> ef(ef_off[n_e]);
    {
        std::vector<int> fill(ef_off.begin(), ef_off.end() - 1);
        for (int fi = 0; fi < n_f; ++fi)
            for (int j = fe_off[fi]; j < fe_off[fi + 1]; ++j) ef[fill[fe[j]]++] = fi;
    }
    int n_v = 0;
    for (auto& e : E) n_v = std::max(n_v, e[1] + 1);
    // deg：顶点所在面数（_face_vertex_cycle 对简单多边形即面顶点集）
    std::vector<int> deg(n_v, 0);
    for (int fi = 0; fi < n_f; ++fi) {
        std::vector<int> vs(faces[fi].begin(), faces[fi].end());
        std::sort(vs.begin(), vs.end()); vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
        for (int v : vs) if (v >= 0 && v < n_v) deg[v]++;
    }
    // adj：顶点 -> 关联边（升序）
    std::vector<int> av_off(n_v + 1, 0);
    for (auto& e : E) { av_off[e[0] + 1]++; av_off[e[1] + 1]++; }
    for (int v = 0; v < n_v; ++v) av_off[v + 1] += av_off[v];
    std::vector<int> av(av_off[n_v]);
    {
        std::vector<int> fill(av_off.begin(), av_off.end() - 1);
        for (int e = 0; e < n_e; ++e) { av[fill[E[e][0]]++] = e; av[fill[E[e][1]]++] = e; }
    }
    std::vector<char> bd_v(n_v, 0), tri_edge(n_e, 0);
    for (int e = 0; e < n_e; ++e) {
        if (ef_cnt[e] == 1) { bd_v[E[e][0]] = 1; bd_v[E[e][1]] = 1; }
        if (stop_at_triangles)
            for (int j = ef_off[e]; j < ef_off[e + 1]; ++j)
                if (nsides[ef[j]] != 4) { tri_edge[e] = 1; break; }
    }
    auto other_end = [&](int e, int v) { return E[e][0] == v ? E[e][1] : E[e][0]; };
    auto share_face = [&](int e1, int e2) {
        for (int i = ef_off[e1]; i < ef_off[e1 + 1]; ++i)
            for (int j = ef_off[e2]; j < ef_off[e2 + 1]; ++j)
                if (ef[i] == ef[j]) return true;
        return false;
    };
    auto opposite_edge = [&](int v, int e) -> int {
        int b = av_off[v], n = av_off[v + 1] - b;
        if (n != 4) return -1;
        int opp = -1, cnt = 0;
        for (int i = 0; i < 4; ++i) {
            int e2 = av[b + i];
            if (e2 == e) continue;
            if (!share_face(e, e2)) { opp = e2; ++cnt; }
        }
        return cnt == 1 ? opp : -1;
    };
    // two_families：返回 e0, e1（各族代表边），失败返回 false
    auto two_families = [&](int v, int& e0, int& e1) -> bool {
        int b = av_off[v];
        if (av_off[v + 1] - b != 4) return false;
        e0 = av[b];
        int o0 = opposite_edge(v, e0);
        if (o0 < 0) return false;
        int rest[4], nr = 0;
        for (int i = 0; i < 4; ++i) { int x = av[b + i]; if (x != e0 && x != o0) rest[nr++] = x; }
        if (nr != 2) return false;
        e1 = rest[0];
        int o1 = opposite_edge(v, e1);
        if (o1 < 0 || o1 != rest[1]) return false;
        return true;
    };

    // ---- 从奇异点游走涂色 + 并查集 ----
    std::vector<int> parent(n_e);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<char> painted(n_e, 0);
    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<int> path;
    std::unordered_set<int> seen;
    for (int sv = 0; sv < n_v; ++sv) {
        if (bd_v[sv] || deg[sv] == 4) continue;
        for (int j = av_off[sv]; j < av_off[sv + 1]; ++j) {
            int e = av[j];
            path.clear(); path.push_back(e);
            seen.clear(); seen.insert(e);
            int v = other_end(e, sv);
            while (true) {
                if (bd_v[v] || deg[v] != 4) break;
                if (tri_edge[e]) break;
                int opp = opposite_edge(v, e);
                if (opp < 0) break;
                if (seen.count(opp)) break;
                path.push_back(opp); seen.insert(opp);
                e = opp; v = other_end(e, v);
            }
            for (int x : path) painted[x] = 1;
            for (size_t i = 0; i + 1 < path.size(); ++i) {
                int ra = find(path[i]), rb = find(path[i + 1]);
                if (ra != rb) parent[rb] = ra;
            }
        }
    }
    // 链：root -> 顶点集合（升序）、边（升序）
    std::unordered_map<int, int> cidx;  // root -> chord idx
    std::vector<std::vector<int>> cverts, cedges;
    for (int e = 0; e < n_e; ++e) {
        if (!painted[e]) continue;
        int r = find(e);
        auto it = cidx.find(r);
        int c;
        if (it == cidx.end()) { c = (int)cverts.size(); cidx.emplace(r, c); cverts.emplace_back(); cedges.emplace_back(); }
        else c = it->second;
        cverts[c].push_back(E[e][0]); cverts[c].push_back(E[e][1]);
        cedges[c].push_back(e);
    }
    for (auto& vs : cverts) { std::sort(vs.begin(), vs.end()); vs.erase(std::unique(vs.begin(), vs.end()), vs.end()); }

    // ---- 正交链对与 m ----
    std::unordered_map<long long, int> pidx;
    std::vector<std::pair<int, int>> pairs;  // (chord u, chord v)，按 root 升序同 Python key
    std::vector<int> pair_m;
    for (int v = 0; v < n_v; ++v) {
        if (bd_v[v] || deg[v] != 4) continue;
        int e0, e1;
        if (!two_families(v, e0, e1)) continue;
        if (!(painted[e0] && painted[e1])) continue;
        int ru = find(e0), rv = find(e1);
        if (ru == rv) continue;
        if (ru > rv) std::swap(ru, rv);
        long long key = ((long long)ru << 32) | (unsigned)rv;
        if (pidx.count(key)) continue;
        int cu = cidx[ru], cv = cidx[rv];
        std::vector<int> inter;
        std::set_intersection(cverts[cu].begin(), cverts[cu].end(), cverts[cv].begin(), cverts[cv].end(),
                              std::back_inserter(inter));
        pidx.emplace(key, (int)pairs.size());
        pairs.push_back({cu, cv});
        pair_m.push_back((int)inter.size());
    }

    // ---- 路序（_ordered_path_vertices，惰性）----
    std::vector<int> path_state(cverts.size(), 0);  // 0 未算, 1 路, 2 非路
    std::vector<std::vector<int>> porder(cverts.size());
    auto get_order = [&](int c) -> const std::vector<int>* {
        if (path_state[c] == 0) {
            const auto& eids = cedges[c];
            std::unordered_map<int, int> slot;  // 顶点 -> 局部下标（插入序）
            std::vector<int> vorder;
            std::vector<std::vector<std::pair<int, int>>> nb;
            auto get = [&](int v) {
                auto it = slot.find(v);
                if (it != slot.end()) return it->second;
                int s = (int)vorder.size(); slot.emplace(v, s); vorder.push_back(v); nb.emplace_back();
                return s;
            };
            for (int e : eids) {
                int a = E[e][0], b = E[e][1];
                int sa = get(a); nb[sa].push_back({b, e});
                int sb = get(b); nb[sb].push_back({a, e});
            }
            bool ok = true;
            std::vector<int> ends;
            for (size_t i = 0; i < vorder.size(); ++i) {
                if (nb[i].size() > 2) { ok = false; break; }
                if (nb[i].size() == 1) ends.push_back((int)i);
            }
            if (ok && ends.size() != 2) ok = false;
            if (ok) {
                std::vector<int> order{vorder[ends[0]]};
                std::unordered_set<int> se;
                int curr = vorder[ends[0]];
                while (true) {
                    int nxt = -1;
                    for (auto& pr : nb[slot[curr]]) {
                        if (se.count(pr.second)) continue;
                        nxt = pr.first; se.insert(pr.second); break;
                    }
                    if (nxt < 0) break;
                    order.push_back(nxt); curr = nxt;
                }
                if (order.size() != vorder.size()) ok = false;
                else porder[c] = std::move(order);
            }
            path_state[c] = ok ? 1 : 2;
        }
        return path_state[c] == 1 ? &porder[c] : nullptr;
    };

    // ---- 绕圈归属 ----
    std::vector<int> lab(pairs.size(), 0);
    std::vector<char> probe(cverts.size(), 0);
    for (size_t pi = 0; pi < pairs.size(); ++pi) {
        if (pair_m[pi] < 2) continue;
        int cu = pairs[pi].first, cv = pairs[pi].second;
        const auto* ou = get_order(cu);
        const auto* ov = get_order(cv);
        if (!ou || !ov || ou->size() < 2 || ov->size() < 2) continue;
        std::vector<int> inter;
        std::set_intersection(cverts[cu].begin(), cverts[cu].end(), cverts[cv].begin(), cverts[cv].end(),
                              std::back_inserter(inter));
        WLabel w = winding_label(*ou, *ov, inter, V);
        if (w.drop_u) probe[cu] = 1;
        if (w.drop_v) probe[cv] = 1;
        lab[pi] = w.eu_lab;
    }
    std::vector<char> is_w(n_e, 0);
    std::vector<int> wpm(n_e, 0);
    for (size_t pi = 0; pi < pairs.size(); ++pi) {
        if (lab[pi] == 0) continue;
        int wr = lab[pi] > 0 ? pairs[pi].first : pairs[pi].second;
        if (probe[wr]) { lab[pi] = 0; continue; }
    }
    for (size_t pi = 0; pi < pairs.size(); ++pi) {
        if (lab[pi] == 0) continue;
        int wr = lab[pi] > 0 ? pairs[pi].first : pairs[pi].second;
        int mv = pair_m[pi];
        for (int e : cedges[wr]) {
            is_w[e] = 1;
            if (mv > wpm[e]) wpm[e] = mv;
        }
    }
    // ---- 汇总 ----
    std::unordered_map<int, std::vector<int>> g;
    for (int e = 0; e < n_e; ++e) {
        rep.max_pair_m = std::max(rep.max_pair_m, wpm[e]);
        if (is_w[e] && wpm[e] >= k) {
            rep.wind_edges.push_back({E[e][0], E[e][1]});
            g[E[e][0]].push_back(E[e][1]); g[E[e][1]].push_back(E[e][0]);
        }
    }
    rep.n_wind_edges = (int)rep.wind_edges.size();
    std::unordered_set<int> vis;
    for (auto& kv : g) {
        if (vis.count(kv.first)) continue;
        rep.n_chains++;
        std::vector<int> st{kv.first}; vis.insert(kv.first);
        while (!st.empty()) {
            int x = st.back(); st.pop_back();
            for (int y : g[x]) if (!vis.count(y)) { vis.insert(y); st.push_back(y); }
        }
    }
    return rep;
}

// ============================================================
// 方案B 修复
// ============================================================
namespace {

struct Flip { std::vector<int> rem, add; };

struct RGraph {
    const std::vector<Vec3>* V;
    const std::vector<Tri>* F;
    std::vector<int> tA, tB;            // 每条内部边两侧三角形（tA 为面序较小者）
    std::vector<Edge> ek;               // 边顶点对（升序），编号=首次出现序
    std::vector<double> w;
    std::vector<char> valid;
    std::vector<std::vector<int>> tri_valid_edges;  // 三角形 -> 合法内部边
    std::vector<std::vector<int>> vt;   // 顶点 -> 三角形
    std::vector<Quad> Q;                // 每条边合成的四边形（已做朝向修正）

    int other(int e, int t) const { return tA[e] == t ? tB[e] : tA[e]; }
};

void build_graph(RGraph& G, const std::vector<Vec3>& V, const std::vector<Tri>& F,
                 const std::unordered_map<long long, std::pair<double, bool>>& info,
                 std::unordered_map<long long, int>& key2e) {
    G.V = &V; G.F = &F;
    std::unordered_map<long long, std::vector<int>> e2t;
    std::vector<long long> order;
    e2t.reserve(F.size() * 3);
    for (int t = 0; t < (int)F.size(); ++t) {
        for (int k = 0; k < 3; ++k) {
            long long key = ekey(F[t][k], F[t][(k + 1) % 3]);
            auto it = e2t.find(key);
            if (it == e2t.end()) { e2t.emplace(key, std::vector<int>{t}); order.push_back(key); }
            else it->second.push_back(t);
        }
    }
    int nV = (int)V.size();
    G.vt.assign(nV, {});
    for (int t = 0; t < (int)F.size(); ++t) for (int k = 0; k < 3; ++k) G.vt[F[t][k]].push_back(t);
    G.tri_valid_edges.assign(F.size(), {});
    for (long long key : order) {
        const auto& ts = e2t[key];
        if (ts.size() != 2) continue;
        int e = (int)G.tA.size();
        key2e.emplace(key, e);
        int s0 = (int)(key >> 32), s1 = (int)(key & 0xffffffff);
        G.tA.push_back(ts[0]); G.tB.push_back(ts[1]);
        G.ek.push_back({s0, s1});
        auto it = info.find(key);
        bool ok = it != info.end() && it->second.second;
        G.w.push_back(it != info.end() ? it->second.first : 0.0);
        G.valid.push_back(ok ? 1 : 0);
        if (ok) { G.tri_valid_edges[ts[0]].push_back(e); G.tri_valid_edges[ts[1]].push_back(e); }
        // 四边形点序：与 build_ordered_quad_pts / lib.Graph.quad_of 一致，再按 triA 法向定朝向
        const Tri& A = F[ts[0]]; const Tri& B = F[ts[1]];
        int ao = -1, bo = -1;
        for (int k = 0; k < 3; ++k) {
            if (A[k] != s0 && A[k] != s1) ao = A[k];
            if (B[k] != s0 && B[k] != s1) bo = B[k];
        }
        bool fwd = false;
        for (int k = 0; k < 3; ++k) if (A[k] == s0 && A[(k + 1) % 3] == s1) fwd = true;
        Quad q = fwd ? Quad{ao, s0, bo, s1} : Quad{ao, s1, bo, s0};
        auto sub = [&](int a, int b) { return Vec3{V[a][0] - V[b][0], V[a][1] - V[b][1], V[a][2] - V[b][2]}; };
        auto cr = [](const Vec3& a, const Vec3& b) { return Vec3{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; };
        Vec3 nA = cr(sub(A[1], A[0]), sub(A[2], A[0]));
        Vec3 n1 = cr(sub(q[1], q[0]), sub(q[2], q[0]));
        Vec3 n2 = cr(sub(q[2], q[0]), sub(q[3], q[0]));
        double d = (n1[0]+n2[0])*nA[0] + (n1[1]+n2[1])*nA[1] + (n1[2]+n2[2])*nA[2];
        if (d < 0) std::reverse(q.begin(), q.end());
        G.Q.push_back(q);
    }
}

std::vector<std::vector<int>> assemble(const RGraph& G, const std::vector<char>& M) {
    std::vector<std::vector<int>> faces;
    std::vector<char> used(G.F->size(), 0);
    for (int e = 0; e < (int)M.size(); ++e) {
        if (!M[e]) continue;
        faces.push_back({G.Q[e][0], G.Q[e][1], G.Q[e][2], G.Q[e][3]});
        used[G.tA[e]] = used[G.tB[e]] = 1;
    }
    for (int t = 0; t < (int)G.F->size(); ++t)
        if (!used[t]) faces.push_back({(*G.F)[t][0], (*G.F)[t][1], (*G.F)[t][2]});
    return faces;
}

// 与 d4_multi.enum_flips 相同语义
void enum_flips(const RGraph& G, const std::vector<int>& mate, const std::vector<int>& seeds,
                int LMAX, std::set<std::pair<std::vector<int>, std::vector<int>>>& out) {
    struct St { int cur; std::vector<int> rem, add, used; };
    auto in = [](const std::vector<int>& u, int x) { return std::find(u.begin(), u.end(), x) != u.end(); };
    auto emit = [&](std::vector<int> r, std::vector<int> a) {
        std::sort(r.begin(), r.end()); std::sort(a.begin(), a.end());
        out.emplace(std::move(r), std::move(a));
    };
    std::vector<St> stack;
    for (int t0 : seeds) {
        int e0 = mate[t0];
        stack.clear();
        if (e0 >= 0) {
            int t1 = G.other(e0, t0);
            stack.push_back({t1, {e0}, {}, {t0, t1}});
            while (!stack.empty()) {
                St s = std::move(stack.back()); stack.pop_back();
                for (int e : G.tri_valid_edges[s.cur]) {
                    if (e == mate[s.cur]) continue;
                    int nx = G.other(e, s.cur);
                    if (nx == t0 && (int)s.rem.size() >= 2) {
                        auto a = s.add; a.push_back(e); emit(s.rem, a); continue;
                    }
                    if (in(s.used, nx)) continue;
                    if (mate[nx] < 0) { auto a = s.add; a.push_back(e); emit(s.rem, a); continue; }
                    if ((int)s.rem.size() >= LMAX) continue;
                    int em = mate[nx], nn = G.other(em, nx);
                    if (in(s.used, nn)) continue;
                    St ns{nn, s.rem, s.add, s.used};
                    ns.rem.push_back(em); ns.add.push_back(e); ns.used.push_back(nx); ns.used.push_back(nn);
                    stack.push_back(std::move(ns));
                }
            }
        } else {
            stack.push_back({t0, {}, {}, {t0}});
            while (!stack.empty()) {
                St s = std::move(stack.back()); stack.pop_back();
                for (int e : G.tri_valid_edges[s.cur]) {
                    if (e == mate[s.cur]) continue;
                    int nx = G.other(e, s.cur);
                    if (in(s.used, nx)) continue;
                    if (mate[nx] < 0) { auto a = s.add; a.push_back(e); emit(s.rem, a); continue; }
                    if ((int)s.rem.size() >= LMAX) continue;
                    int em = mate[nx], nn = G.other(em, nx);
                    if (in(s.used, nn)) continue;
                    { auto r = s.rem; r.push_back(em); auto a = s.add; a.push_back(e); emit(r, a); }
                    St ns{nn, s.rem, s.add, s.used};
                    ns.rem.push_back(em); ns.add.push_back(e); ns.used.push_back(nx); ns.used.push_back(nn);
                    stack.push_back(std::move(ns));
                }
            }
        }
    }
}

} // namespace

SpiralRepairStats repair_spirals(
    const std::vector<Vec3>& V, const std::vector<Tri>& F,
    const std::unordered_map<long long, std::pair<double, bool>>& cand_info,
    std::vector<Edge>& merged, std::vector<Quad>& quads_out, std::vector<Tri>& tris_out,
    const SpiralRepairParams& P)
{
    auto t0 = std::chrono::steady_clock::now();
    SpiralRepairStats S;
    RGraph G;
    std::unordered_map<long long, int> key2e;
    build_graph(G, V, F, cand_info, key2e);
    const int nE = (int)G.ek.size();
    std::vector<char> M(nE, 0);
    for (auto& m : merged) {
        auto it = key2e.find(ekey(m[0], m[1]));
        if (it != key2e.end()) M[it->second] = 1;
    }
    SpiralReport cur = detect_spirals(V, assemble(G, M), P.k, P.stop_at_triangles);
    S.evals = 1;
    S.wind_before = cur.n_wind_edges;
    if (P.verbose)
        std::cerr << "[spiral repair] wind_before=" << cur.n_wind_edges << " chains=" << cur.n_chains << "\n";

    while (cur.n_wind_edges > 0 && S.steps < P.max_steps) {
        std::vector<int> mate(F.size(), -1);
        for (int e = 0; e < nE; ++e) if (M[e]) { mate[G.tA[e]] = e; mate[G.tB[e]] = e; }
        std::vector<char> is_seed(F.size(), 0);
        std::vector<int> seeds;
        for (auto& we : cur.wind_edges)
            for (int v : {we[0], we[1]})
                if (v < (int)G.vt.size())
                    for (int t : G.vt[v]) if (!is_seed[t]) { is_seed[t] = 1; seeds.push_back(t); }
        std::set<std::pair<std::vector<int>, std::vector<int>>> flips;
        enum_flips(G, mate, seeds, P.lmax, flips);
        struct C { int dq; double dW; const std::pair<std::vector<int>, std::vector<int>>* f; std::vector<long long> tie; };
        std::vector<C> cands;
        for (auto& f : flips) {
            int dq = (int)f.second.size() - (int)f.first.size();
            if (dq < 0) continue;
            double dW = 0.0;
            for (int e : f.second) dW += G.w[e];
            for (int e : f.first) dW -= G.w[e];
            std::vector<long long> tie;
            for (int e : f.first) tie.push_back(ekey(G.ek[e][0], G.ek[e][1]));
            tie.push_back(-1);
            for (int e : f.second) tie.push_back(ekey(G.ek[e][0], G.ek[e][1]));
            cands.push_back({dq, dW, &f, std::move(tie)});
        }
        if (cands.empty()) break;
        std::sort(cands.begin(), cands.end(), [](const C& a, const C& b) {
            if (a.dq != b.dq) return a.dq > b.dq;
            if (a.dW != b.dW) return a.dW > b.dW;
            return a.tie < b.tie;
        });
        if ((int)cands.size() > P.topk) cands.resize(P.topk);
        const int nc = (int)cands.size();
        std::vector<SpiralReport> outs(nc);
#ifdef _OPENMP
        int nt = P.threads > 0 ? P.threads : omp_get_max_threads();
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
#endif
        for (int i = 0; i < nc; ++i) {
            std::vector<char> Mi = M;
            for (int e : cands[i].f->first) Mi[e] = 0;
            for (int e : cands[i].f->second) Mi[e] = 1;
            outs[i] = detect_spirals(V, assemble(G, Mi), P.k, P.stop_at_triangles);
        }
        S.evals += nc;
        int best = 0;
        for (int i = 1; i < nc; ++i) {
            if (outs[i].n_wind_edges < outs[best].n_wind_edges ||
                (outs[i].n_wind_edges == outs[best].n_wind_edges && cands[i].dW > cands[best].dW))
                best = i;
        }
        if (outs[best].n_wind_edges >= cur.n_wind_edges) break;
        for (int e : cands[best].f->first) M[e] = 0;
        for (int e : cands[best].f->second) M[e] = 1;
        cur = std::move(outs[best]);
        S.steps++;
        if (P.verbose)
            std::cerr << "[spiral repair] step " << S.steps << " dW=" << cands[best].dW
                      << " -> wind=" << cur.n_wind_edges << "\n";
    }
    S.wind_after = cur.n_wind_edges;
    S.chains_after = cur.n_chains;
    S.max_pair_m_after = cur.max_pair_m;
    S.wind_edges_after = cur.wind_edges;
    if (S.steps > 0) {
        quads_out.clear(); tris_out.clear(); merged.clear();
        std::vector<char> used(F.size(), 0);
        for (int e = 0; e < nE; ++e) {
            if (!M[e]) continue;
            quads_out.push_back(G.Q[e]);
            merged.push_back(G.ek[e]);
            used[G.tA[e]] = used[G.tB[e]] = 1;
        }
        for (int t = 0; t < (int)F.size(); ++t) if (!used[t]) tris_out.push_back(F[t]);
    }
    S.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

} // namespace tri2quad
