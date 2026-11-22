#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tri2quad.hpp"
#include "spiral_repair.hpp"

namespace py = pybind11;
using namespace tri2quad;

static std::vector<Vec3> as_vec3(py::array_t<double, py::array::c_style | py::array::forcecast> arr){
    if (arr.ndim()!=2 || arr.shape(1)!=3) throw std::runtime_error("vertices must be (N,3) float/double");
    std::vector<Vec3> out(arr.shape(0));
    auto r = arr.unchecked<2>();
    for (ssize_t i=0;i<r.shape(0);++i){
        out[i] = { r(i,0), r(i,1), r(i,2) };
    }
    return out;
}
static std::vector<Tri> as_tri(py::array_t<int, py::array::c_style | py::array::forcecast> arr){
    if (arr.ndim()!=2 || arr.shape(1)!=3) throw std::runtime_error("faces must be (M,3) int");
    std::vector<Tri> out(arr.shape(0));
    auto r = arr.unchecked<2>();
    for (ssize_t i=0;i<r.shape(0);++i){
        out[i] = { r(i,0), r(i,1), r(i,2) };
    }
    return out;
}
static py::array_t<double> vec3_to_np(const std::vector<Vec3>& V){
    py::array_t<double> arr({(ssize_t)V.size(), (ssize_t)3});
    auto w = arr.mutable_unchecked<2>();
    for (ssize_t i=0;i<(ssize_t)V.size();++i){
        w(i,0)=V[i][0]; w(i,1)=V[i][1]; w(i,2)=V[i][2];
    }
    return arr;
}
template<int K>
static py::array_t<int> faces_to_np(const std::vector<std::array<int,K>>& F){
    py::array_t<int> arr({(ssize_t)F.size(), (ssize_t)K});
    auto w = arr.mutable_unchecked<2>();
    for (ssize_t i=0;i<(ssize_t)F.size();++i)
        for (int j=0;j<K;++j) w(i,j)=F[i][j];
    return arr;
}

PYBIND11_MODULE(quad_converter, m) {
    m.doc() = "Triangle-to-Quad conversion (greedy / global with prefilters + edge-align scoring)";

    m.def("tri_to_quad",
        [](py::array vertices,
           py::array faces,
           double angle_threshold_deg,
           std::string method,
           bool verbose,
           double pre_angle_min_deg,
           double pre_angle_max_deg,
           double feature_dihedral_thresh_deg,
           bool   enable_concavity_prefilter,
           double planarity_eps,
           bool   enable_edge_align,
           double alpha_angle,
           double alpha_aspect,
           double alpha_align,
           std::vector<double> sym_planes,
           double sym_plane_tol,
           double sym_edge_beta,
           bool spiral_repair,
           int  spiral_repair_k,
           int  spiral_repair_topk,
           int  spiral_repair_lmax,
           int  spiral_repair_max_steps,
           int  spiral_repair_threads)
        {
            auto V = as_vec3(vertices.cast<py::array_t<double>>());
            auto F = as_tri (faces.cast<py::array_t<int>>());
            tri2quad::Result R;
            {
            py::gil_scoped_release nogil;  // 纯 C++ 计算，释放 GIL 便于外部多线程调用
            R = tri2quad::tri_to_quad(
                V, F,
                angle_threshold_deg,
                method,
                verbose,
                pre_angle_min_deg,
                pre_angle_max_deg,
                feature_dihedral_thresh_deg,
                enable_concavity_prefilter,
                planarity_eps,
                enable_edge_align,
                alpha_angle,
                alpha_aspect,
                alpha_align,
                sym_planes,
                sym_plane_tol,
                sym_edge_beta,
                spiral_repair,
                spiral_repair_k,
                spiral_repair_topk,
                spiral_repair_lmax,
                spiral_repair_max_steps,
                spiral_repair_threads
            );
            }
            py::dict out;
            out["new_vertices"] = vec3_to_np(R.new_vertices);
            out["tris"]         = faces_to_np<3>(R.tris);
            out["quads"]        = faces_to_np<4>(R.quads);
            out["merge_edges"]  = faces_to_np<2>(R.merge_edges);
            py::array_t<double> W({(ssize_t)R.weights.size()});
            auto w = W.mutable_unchecked<1>();
            for (ssize_t i=0;i<(ssize_t)R.weights.size();++i) w(i)=R.weights[i];
            out["weights"] = W;

            py::dict stat;
            stat["num_internal_edges"]  = R.num_internal_edges;
            stat["num_accepted_quads"]  = R.num_accepted_quads;
            stat["num_rejected_quads"]  = R.num_rejected_quads;
            stat["num_symmetry_protected_edges"] = R.num_symmetry_protected_edges;
            stat["tris_left"]           = (int)R.tris.size();
            out["merge_stats"] = stat;
            if (R.spiral_repair_ran) {
                py::dict sp;
                sp["wind_before"]    = R.spiral_wind_before;
                sp["wind_after"]     = R.spiral_wind_after;
                sp["chains_after"]   = R.spiral_chains_after;
                sp["max_pair_m_after"] = R.spiral_max_pair_m_after;
                sp["steps"]          = R.spiral_repair_steps;
                sp["evals"]          = R.spiral_repair_evals;
                sp["seconds"]        = R.spiral_repair_seconds;
                sp["wind_edges"]     = faces_to_np<2>(R.spiral_wind_edges);
                out["spiral_repair"] = sp;
            }
            return out;
        },
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("angle_threshold_deg") = 150.0,
        py::arg("method") = "greedy",
        py::arg("verbose") = false,
        // —— 前置过滤参数 ——
        py::arg("pre_angle_min_deg") = 30.0,
        py::arg("pre_angle_max_deg") = 140.0,
        py::arg("feature_dihedral_thresh_deg") = 50.0,
        py::arg("enable_concavity_prefilter") = true,
        py::arg("planarity_eps") = 0.0,
        // —— Edge-Align 参数（论文 arXiv:2605.16813 Q_align） ——
        py::arg("enable_edge_align") = false,
        py::arg("alpha_angle") = 0.6,
        py::arg("alpha_aspect") = 0.4,
        py::arg("alpha_align") = 0.0,
        // —— 对称面保护边 ——
        py::arg("sym_planes") = std::vector<double>{},
        py::arg("sym_plane_tol") = 0.0,
        py::arg("sym_edge_beta") = 0.0,
        // —— Spiral 修复（方案B：交错链翻转，挑最优；四边形数不减、只用合法候选） ——
        py::arg("spiral_repair") = false,
        py::arg("spiral_repair_k") = 3,
        py::arg("spiral_repair_topk") = 40,
        py::arg("spiral_repair_lmax") = 5,
        py::arg("spiral_repair_max_steps") = 1000,
        py::arg("spiral_repair_threads") = 0
    );

    m.def("detect_spirals",
        [](py::array vertices, py::array quads, py::array tris, int k, bool stop_at_triangles)
        {
            auto V = as_vec3(vertices.cast<py::array_t<double>>());
            std::vector<std::vector<int>> faces;
            auto addf = [&](py::array a, int K){
                auto arr = a.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
                if (arr.size() == 0) return;
                if (arr.ndim()!=2 || arr.shape(1)!=K) throw std::runtime_error("bad face array shape");
                auto r = arr.unchecked<2>();
                for (ssize_t i=0;i<r.shape(0);++i){
                    std::vector<int> f(K);
                    for (int j=0;j<K;++j) f[j]=r(i,j);
                    faces.push_back(std::move(f));
                }
            };
            addf(quads, 4); addf(tris, 3);   // 与 Python 版相同：先四边形后三角形
            tri2quad::SpiralReport rep;
            { py::gil_scoped_release nogil; rep = tri2quad::detect_spirals(V, faces, k, stop_at_triangles); }
            py::dict out;
            out["n_wind_edges"] = rep.n_wind_edges;
            out["n_chains"]     = rep.n_chains;
            out["max_pair_m"]   = rep.max_pair_m;
            out["wind_edges"]   = faces_to_np<2>(rep.wind_edges);
            return out;
        },
        py::arg("vertices"), py::arg("quads"), py::arg("tris"), py::arg("k") = 3,
        py::arg("stop_at_triangles") = false
    );

    m.def("detect_symmetry_plane",
        [](py::array vertices,
           py::array faces,
           double eps_ratio,
           double min_coverage,
           bool verbose,
           double protect_tol,
           double max_inband_ratio)
        {
            auto V = as_vec3(vertices.cast<py::array_t<double>>());
            auto F = as_tri (faces.cast<py::array_t<int>>());
            auto R = tri2quad::detect_symmetry_plane(V, F, eps_ratio, min_coverage,
                                                     verbose, protect_tol, max_inband_ratio);
            py::dict out;
            out["found"] = R.found;
            py::array_t<double> n({(ssize_t)3});
            auto w = n.mutable_unchecked<1>();
            for (int i=0;i<3;++i) w(i) = R.normal[i];
            out["normal"] = n;
            out["d"] = R.d;
            out["coverage"] = R.coverage;
            return out;
        },
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("eps_ratio") = 0.01,
        py::arg("min_coverage") = 0.9,
        py::arg("verbose") = false,
        py::arg("protect_tol") = 0.0,
        py::arg("max_inband_ratio") = 0.2
    );
}
