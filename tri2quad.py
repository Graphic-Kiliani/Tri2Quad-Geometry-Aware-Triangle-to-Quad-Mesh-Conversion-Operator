"""Command-line tools for triangle-to-quad mesh conversion."""

import argparse
import os
import sys

import numpy as np
import trimesh

import quad_converter


def load_obj_vertices_faces(obj_path):
    verts = []
    faces = []

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                verts.append([x, y, z])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    raise ValueError(
                        f"[Error] Only triangle faces are supported, detected {len(parts)} edges: {parts}"
                    )
                idx = []
                for p in parts:
                    if "/" in p:
                        p = p.split("/")[0]
                    vi = int(p)
                    if vi < 0:
                        vi = len(verts) + 1 + vi
                    idx.append(vi - 1)
                faces.append(idx)

    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("[Error] Vertex array dimension is incorrect, should be (N,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("[Error] Face array dimension is incorrect, should be (M,3)")
    return V, F


def _load_glb_vertices_faces(glb_path):
    m = trimesh.load(glb_path, force='mesh', process=False)
    if isinstance(m, trimesh.Trimesh):
        mesh = m
    else:
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("[Error] GLB found no triangle mesh")
        mesh = trimesh.util.concatenate(geoms)

    V = mesh.vertices.astype(np.float64, copy=False)
    F = mesh.faces.astype(np.int32, copy=False)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("[Error] GLB is not a triangle mesh")
    return V, F


# ============================================================================
# FBX loading via pyassimp (loop-level: pyassimp expands to 3 verts / triangle,
# preserving per-corner UVs). We then optionally do a UV-aware dedup so that
# each UV island becomes a topologically-independent connected component.
# ============================================================================

def _load_fbx_loop_level(fbx_path):
    """
    Read a .fbx via pyassimp. pyassimp's default fbx loader materialises each
    triangle corner as its own vertex, so we get:
        V_split: (3M, 3) float64 — per-corner positions
        F_raw:   (M, 3)  int32   — trivially [[0,1,2],[3,4,5],...]
        UV_split:(3M, 2) float64 — per-corner UV coords (channel 0)

    This "loop-level" representation is exactly what we need for the UV-aware
    dedup pass in `_uv_aware_dedup`. If a mesh has no UV channel we fill zeros
    so the caller can still proceed (dedup would then merge everything by
    position alone, i.e. behave like a plain vertex weld).
    """
    try:
        import pyassimp
    except ImportError as e:
        raise RuntimeError(
            "[Error] pyassimp is required to read .fbx files. Install with:\n"
            "  pip install pyassimp\n(assimp shared lib must also be present)"
        ) from e

    V_list, F_list, UV_list, off = [], [], [], 0
    with pyassimp.load(fbx_path) as scene:
        if not scene.meshes:
            raise ValueError(f"[Error] No meshes found in {fbx_path}")
        for m in scene.meshes:
            v = np.asarray(m.vertices, dtype=np.float64)
            f = np.asarray(m.faces, dtype=np.int32)
            if f.ndim != 2 or f.shape[1] != 3:
                raise ValueError(
                    f"[Error] fbx sub-mesh has non-triangle faces (F shape={f.shape}); "
                    "pre-triangulate the mesh before feeding to tri2quad."
                )
            if m.texturecoords is not None and len(m.texturecoords) > 0:
                uv = np.asarray(m.texturecoords[0], dtype=np.float64)[:, :2]
            else:
                uv = np.zeros((len(v), 2), dtype=np.float64)
            V_list.append(v)
            F_list.append(f + off)
            UV_list.append(uv)
            off += len(v)
    V_split = np.vstack(V_list)
    F_raw   = np.vstack(F_list)
    UV_split = np.vstack(UV_list)
    return V_split, F_raw, UV_split


def _uv_aware_dedup(V_split, UV_split, F_raw, tol_pos=1e-6, tol_uv=1e-6):
    """
    Merge vertices whose (position, uv) tuple matches within tolerance.

    Effect on UV seams: two corners at the same 3D location but different UVs
    have DIFFERENT keys → they stay as two vertices with different indices →
    the seam edge between them is topologically split (matches the "seam is
    boundary" semantics we want).

    Effect inside a UV island: two corners at the same 3D location and same
    UV get merged into one index → island interior edges become internal.

    Args
    ----
    V_split  : (N, 3) per-corner positions (from pyassimp)
    UV_split : (N, 2) per-corner UVs
    F_raw    : (M, 3) faces indexing V_split
    tol_pos  : rounding tolerance for position keys (metres, world units)
    tol_uv   : rounding tolerance for uv keys (in [0,1]-normalised UV space)

    Returns
    -------
    V_uv     : (K, 3)   deduped positions
    F_uv     : (M, 3)   faces remapped to V_uv indices
    UV_uv    : (K, 2)   deduped UVs (aligned with V_uv)
    remap    : (N,)     per-corner mapping V_split_index -> V_uv_index
    """
    dp = max(0, int(round(-np.log10(tol_pos)))) if tol_pos > 0 else 12
    du = max(0, int(round(-np.log10(tol_uv))))  if tol_uv  > 0 else 12
    pos_r = np.round(V_split,  dp)
    uv_r  = np.round(UV_split, du)
    seen = {}
    remap = np.empty(len(V_split), dtype=np.int32)
    new_V, new_UV = [], []
    for i in range(len(V_split)):
        k = (pos_r[i, 0], pos_r[i, 1], pos_r[i, 2],
             uv_r [i, 0], uv_r [i, 1])
        idx = seen.get(k)
        if idx is None:
            idx = len(new_V)
            seen[k] = idx
            new_V.append(V_split[i])
            new_UV.append(UV_split[i])
        remap[i] = idx
    V_uv  = np.asarray(new_V,  dtype=np.float64)
    UV_uv = np.asarray(new_UV, dtype=np.float64)
    F_uv  = remap[F_raw].astype(np.int32)
    return V_uv, F_uv, UV_uv, remap


def _face_islands_by_index(F_uv, n_vertices):
    """
    Group faces into vertex-adjacency connected components. On a UV-aware
    dedup output, each component IS one UV island (any face in island A shares
    a vertex-index with some other face in A; no face in A shares indices with
    faces in island B, because that would require crossing a seam, but seam
    endpoints have different UVs and thus different indices after dedup).
    """
    from collections import defaultdict, deque
    v2f = defaultdict(list)
    for fi, t in enumerate(F_uv):
        for v in t:
            v2f[int(v)].append(fi)
    seen = np.zeros(len(F_uv), dtype=bool)
    islands = []
    for start in range(len(F_uv)):
        if seen[start]:
            continue
        q = deque([start]); seen[start] = True; comp = []
        while q:
            fi = q.popleft(); comp.append(fi)
            for v in F_uv[fi]:
                for nb in v2f[int(v)]:
                    if not seen[nb]:
                        seen[nb] = True
                        q.append(nb)
        islands.append(comp)
    return islands


def load_with_uv_islands(path, tol_pos=1e-6, tol_uv=1e-6):
    """
    UV-aware loader used when --split_by_uv is set.

    Currently supports .fbx (via pyassimp loop-level read + dedup). Other
    formats fall back to `load_vertices_faces_auto` and treat the whole mesh
    as a single "island" — this keeps the code path uniform.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fbx":
        V_split, F_raw, UV_split = _load_fbx_loop_level(path)
        V_uv, F_uv, UV_uv, _ = _uv_aware_dedup(
            V_split, UV_split, F_raw, tol_pos=tol_pos, tol_uv=tol_uv
        )
        islands = _face_islands_by_index(F_uv, len(V_uv))
        return V_uv, F_uv, UV_uv, islands, {
            "loaded_via": "pyassimp+uv_dedup",
            "loop_level_verts": int(len(V_split)),
            "deduped_verts":     int(len(V_uv)),
            "n_faces":           int(len(F_uv)),
            "n_islands":         int(len(islands)),
        }
    # Non-fbx formats: reuse existing loaders, no UV, single "island"
    V, F = load_vertices_faces_auto(path)
    UV = np.zeros((len(V), 2), dtype=np.float64)
    islands = [list(range(len(F)))]
    return V, F, UV, islands, {
        "loaded_via": f"legacy({ext})",
        "loop_level_verts": int(len(V)),
        "deduped_verts":     int(len(V)),
        "n_faces":           int(len(F)),
        "n_islands":         1,
    }


def load_vertices_faces_auto(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return load_obj_vertices_faces(path)
    if ext in (".glb", ".gltf"):
        return _load_glb_vertices_faces(path)
    if ext == ".fbx":
        # NB: pyassimp explodes each triangle into 3 independent corners, so
        # the returned mesh is "index-unshared". Downstream tri_to_quad will
        # see 0 internal edges and produce 0 quads. If you actually want
        # merged quads on an fbx input, use --split_by_uv (recommended) or
        # pre-weld with an external tool.
        V_split, F_raw, _UV = _load_fbx_loop_level(path)
        return V_split, F_raw

    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
        if head == b'glTF':
            return _load_glb_vertices_faces(path)
    except Exception:
        pass

    raise ValueError(f"[Error] Format not supported for {ext}, Please use .obj / .glb / .gltf / .fbx")


def save_obj_mixed(output_path, V, quads, tris, UV=None):
    """
    Write a mixed quad+tri OBJ. If UV (aligned to V) is given, emit `vt` lines
    and `f v/vt v/vt ...` face syntax so the seam-preserving indices survive.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# quad-dominant mesh generated by quad_converter\n")
        for v in V:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        if UV is not None:
            for uv in UV:
                f.write(f"vt {uv[0]:.9f} {uv[1]:.9f}\n")
            for q in quads:
                a, b, c, d = int(q[0]) + 1, int(q[1]) + 1, int(q[2]) + 1, int(q[3]) + 1
                f.write(f"f {a}/{a} {b}/{b} {c}/{c} {d}/{d}\n")
            for t in tris:
                a, b, c = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        else:
            for q in quads:
                a, b, c, d = int(q[0]) + 1, int(q[1]) + 1, int(q[2]) + 1, int(q[3]) + 1
                f.write(f"f {a} {b} {c} {d}\n")
            for t in tris:
                a, b, c = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
                f.write(f"f {a} {b} {c}\n")


def _run_tri_to_quad_per_island(V_uv, F_uv, islands, quad_kwargs,
                                 min_island_size=2, verbose=False):
    """
    Run tri_to_quad on each UV island independently, then splice the outputs
    back into the global V_uv index space.

    Islands with fewer than `min_island_size` faces skip the C++ call and
    emit their triangles unchanged (a 1-face island cannot form a quad
    anyway; we save the pybind11 marshalling cost).
    """
    all_quads_global = []
    all_tris_global  = []
    per_island_stats = {"n_islands": len(islands),
                        "islands_run":     0,
                        "islands_skipped": 0,
                        "quads_total":     0,
                        "tris_total":      0,
                        "internal_edges":  0,
                        "accepted_quads":  0,
                        "rejected_quads":  0,
                        "symmetry_protected": 0,
                        "spiral_repair_islands": 0,   # 跑了修复的岛数
                        "spiral_wind_before": 0,      # 各岛修复前绕圈边之和
                        "spiral_wind_after": 0,       # 各岛修复后绕圈边之和
                        "spiral_repair_steps": 0,
                        "spiral_repair_evals": 0,
                        "spiral_repair_seconds": 0.0}
    for island_idx, face_ids in enumerate(islands):
        F_i_global = F_uv[face_ids]

        if len(face_ids) < min_island_size:
            for t in F_i_global:
                all_tris_global.append([int(t[0]), int(t[1]), int(t[2])])
            per_island_stats["islands_skipped"] += 1
            continue

        used_global = np.unique(F_i_global)
        g2l = {int(g): li for li, g in enumerate(used_global)}
        F_i_local = np.array(
            [[g2l[int(v)] for v in t] for t in F_i_global], dtype=np.int32
        )
        V_i = np.ascontiguousarray(V_uv[used_global], dtype=np.float64)
        F_i_local = np.ascontiguousarray(F_i_local, dtype=np.int32)

        try:
            res = quad_converter.tri_to_quad(V_i, F_i_local, **quad_kwargs)
        except Exception as e:
            print(f"  [Warn] island {island_idx} (F={len(face_ids)}): "
                  f"tri_to_quad failed ({e}); keeping as tris")
            for t in F_i_global:
                all_tris_global.append([int(t[0]), int(t[1]), int(t[2])])
            per_island_stats["islands_skipped"] += 1
            continue

        for q in res["quads"]:
            all_quads_global.append([int(used_global[q[k]]) for k in range(4)])
        for t in res["tris"]:
            all_tris_global.append([int(used_global[t[k]]) for k in range(3)])
        stats = res.get("merge_stats", {})
        per_island_stats["islands_run"]     += 1
        per_island_stats["internal_edges"]  += int(stats.get("num_internal_edges", 0))
        per_island_stats["accepted_quads"]  += int(stats.get("num_accepted_quads", 0))
        per_island_stats["rejected_quads"]  += int(stats.get("num_rejected_quads", 0))
        per_island_stats["symmetry_protected"] += int(stats.get("num_symmetry_protected_edges", 0))
        if "spiral_repair" in res:
            sr = res["spiral_repair"]
            per_island_stats["spiral_repair_islands"] += 1
            per_island_stats["spiral_wind_before"]    += int(sr["wind_before"])
            per_island_stats["spiral_wind_after"]     += int(sr["wind_after"])
            per_island_stats["spiral_repair_steps"]   += int(sr["steps"])
            per_island_stats["spiral_repair_evals"]   += int(sr["evals"])
            per_island_stats["spiral_repair_seconds"] += float(sr["seconds"])

    quads_arr = (np.asarray(all_quads_global, dtype=np.int32)
                 if all_quads_global else np.zeros((0, 4), dtype=np.int32))
    tris_arr  = (np.asarray(all_tris_global,  dtype=np.int32)
                 if all_tris_global  else np.zeros((0, 3), dtype=np.int32))
    per_island_stats["quads_total"] = int(len(quads_arr))
    per_island_stats["tris_total"]  = int(len(tris_arr))
    return quads_arr, tris_arr, per_island_stats


# ============================================================================
# Public high-level API
# ============================================================================

def _detect_sym_planes(V, F, enable, eps_ratio=0.01, min_coverage=0.9,
                       tol_ratio=0.005, max_inband_ratio=0.2, strict_ratio=1e-3,
                       verbose=False):
    """
    Run the embedded symmetry-plane detection on the WHOLE mesh (once).

    Returns (sym_planes, sym_plane_tol):
        sym_planes    : flat list [nx, ny, nz, d] (empty when disabled / not found)
        sym_plane_tol : absolute distance tolerance — starts from
                        tol_ratio * bbox diagonal as a CAP, then auto-tightened
                        into the largest gap in the sorted vertex-to-plane
                        distances (only when a decisive >=30x gap exists), so
                        off-plane neighbour vertices are never swallowed.

    The detected plane is expressed in world coordinates, so it can be passed
    straight into every per-island tri_to_quad call of the UV pipeline.

    Candidate planes whose protect-tolerance band swallows more than
    `max_inband_ratio` of all vertices are rejected inside the C++ detector as
    degenerate mid-surface planes (flat / rod-like objects).
    """
    if not enable:
        return [], 0.0
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = tol_ratio * diag
    det = quad_converter.detect_symmetry_plane(
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int32),
        eps_ratio=eps_ratio,
        min_coverage=min_coverage,
        verbose=verbose,
        protect_tol=tol,
        max_inband_ratio=max_inband_ratio,
    )
    if not det["found"]:
        print(f"[Warn] symmetry protect enabled but no plane passed the acceptance "
              f"threshold (best coverage={det['coverage']:.4f} < {min_coverage} "
              f"or degenerate mid-surface); continuing WITHOUT symmetry protection")
        return [], 0.0
    n = det["normal"]
    d = float(det["d"])
    # Auto-tighten the protection tolerance: sort vertex distances to the plane
    # and look for a clear gap between the on-plane vertex cluster and off-plane
    # neighbours. If found, place tol inside the gap instead of blindly using
    # tol_ratio*diag — a coarse cap would otherwise swallow neighbour rings
    # (edges with only one, or even no, endpoint truly on the plane would get
    # "protected" and never merge).
    dist = np.abs(V @ n + d)
    near = np.sort(dist[dist < tol])
    tightened = False
    if len(near) >= 2:
        gaps = near[1:] / np.maximum(near[:-1], 1e-12)
        i = int(np.argmax(gaps))
        # Only a DECISIVE gap (>=30x) proves a crisp on-plane vertex cluster
        # separated from off-plane neighbours (e.g. 656x in our fbx tests).
        if gaps[i] >= 30.0:
            tol = float(np.sqrt(near[i] * near[i + 1]))  # geometric mid of the gap
            tightened = True
    if not tightened:
        # No crisp on-plane cluster: fall back to a STRICT tolerance
        # (numerical-level) instead of the coarse cap. Meshes without exact
        # on-plane vertices (continuous distance distribution) then protect
        # essentially nothing — matching the "both endpoints exactly on the
        # plane" semantics. The strict floor still covers the small spread
        # caused by the detected plane's slight tilt / float precision.
        tol = min(tol, strict_ratio * diag)
    print(f"[Info] symmetry plane detected: n=({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}) "
          f"d={d:.6f} coverage={det['coverage']:.4f} "
          f"(tol={tol:.3e}, cap={tol_ratio * diag:.3e}, on-plane verts={int((dist < tol).sum())})")
    return [float(n[0]), float(n[1]), float(n[2]), d], tol


# ============================================================================
# Spiral loop 检测（C++ 实现，src/spiral_repair.cpp::detect_spirals）
#
# 从内部非 4 价顶点沿"顶点处对边"（不共享面的入边/出边）游走涂色开链，两条
# 开链的公共顶点数即相交次数 m_uv；m_uv >= k 且 I-弧长较长的一条判为绕圈链
# （spiral）。算法移植自 sep_ix 的 compute_open_ix，唯一改动：走带对三角形感知（只在奇异点/边界停，可沿三角形
# 邻边继续，与 Blender Alt 选 edge loop 一致）；原规则会在 quad-dominant 网上把穿过
# 三角形的螺旋切断而漏检。原规则可用 detect_spirals(..., stop_at_triangles=True)。
# 修复见 tri_to_quad(..., spiral_repair=True)（方案B：交错链翻转）。
# ============================================================================

def detect_spiral_report(V, quads, tris, k=3):
    """
    在 quad+tri 网格上检测 spiral loop。

    返回 dict:
        n_wind_edges : 绕圈边数（pair_m >= k）
        n_chains     : 绕圈边构成的连通分量数（≈ spiral 条数）
        max_pair_m   : 最大相交次数（≈ 最大绕数）
        wind_edges   : set[(a,b)] 顶点对（升序），索引空间与 quads/tris/V 一致
    """
    quads = np.ascontiguousarray(np.asarray(quads).reshape(-1, 4), dtype=np.int32)
    tris = np.ascontiguousarray(np.asarray(tris).reshape(-1, 3), dtype=np.int32)
    if len(quads) == 0:
        return {"n_wind_edges": 0, "n_chains": 0, "max_pair_m": 0, "wind_edges": set()}
    rec = quad_converter.detect_spirals(np.asarray(V, dtype=np.float64), quads, tris, int(k))
    wind = set((int(a), int(b)) for a, b in np.asarray(rec["wind_edges"]).reshape(-1, 2))
    return {"n_wind_edges": int(rec["n_wind_edges"]), "n_chains": int(rec["n_chains"]),
            "max_pair_m": int(rec["max_pair_m"]), "wind_edges": wind}


def _export_spiral_obj(out_path, V, wind_edges):
    """把绕圈边写成 v+l 的 OBJ（顶点与输出网格同序），无绕圈边则不写。返回路径或 None。"""
    if not wind_edges:
        return None
    sp_path = os.path.splitext(out_path)[0] + "_spiral.obj"
    with open(sp_path, "w", encoding="utf-8") as f:
        f.write("# spiral loop edges (winding chains, pair_m >= k)\n")
        for v in np.asarray(V, dtype=np.float64):
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for a, b in sorted(wind_edges):
            f.write(f"l {a + 1} {b + 1}\n")
    return sp_path



def convert_fbx_to_quad(
    local_file_path,
    output_path=None,
    method="global",
    angle_threshold_deg=150.0,
    pre_angle_min_deg=30.0,
    pre_angle_max_deg=140.0,
    feature_dihedral_thresh_deg=45.0,
    enable_concavity_prefilter=True,
    planarity_eps=0.0,
    enable_edge_align=False,
    alpha_angle=0.6,
    alpha_aspect=0.4,
    alpha_align=0.0,
    dedup_tol_pos=1e-6,
    dedup_tol_uv=1e-6,
    min_island_size=2,
    enable_symmetry_protect=False,
    sym_detect_eps_ratio=0.01,
    sym_min_coverage=0.9,
    sym_plane_tol_ratio=0.005,
    sym_max_inband_ratio=0.2,
    sym_edge_beta=0.2,
    sym_strict_tol_ratio=1e-3,
    detect_spirals=False,
    repair_spirals=False,
    spiral_k=3,
    spiral_repair_threads=0,
    verbose=False,
):
    """
    Convert a triangle-mesh .fbx to a quad-dominant .obj via the UV-island
    pipeline (loop-level load → UV-aware dedup → per-island tri_to_quad →
    splice back into a global (V, UV) index space, write OBJ with `vt` lines).

    Parameters
    ----------
    local_file_path : str
        Path to the input .fbx file. Must exist. Other formats are rejected
        (use tri2quad.py CLI for .obj / .glb / .gltf).
    output_path : str or None, optional
        Where to write the output OBJ. If None, defaults to
        `<basename>_quad.obj` next to the input file.

    method, angle_threshold_deg, pre_angle_min_deg, pre_angle_max_deg,
    feature_dihedral_thresh_deg, enable_concavity_prefilter, planarity_eps,
    enable_edge_align, alpha_angle, alpha_aspect, alpha_align
        Passed straight through to `quad_converter.tri_to_quad` per island.
        Defaults match the "Example 1" configuration in tri2quad.py's
        docstring (0.6·Q_angle + 0.4·Q_aspect scoring, global matching).

    dedup_tol_pos, dedup_tol_uv : float
        Rounding tolerances used by the (pos, uv) dedup pass.
    min_island_size : int
        UV islands smaller than this (in # triangles) skip the C++ call and
        are emitted as unmerged tris.
    enable_symmetry_protect : bool
        When True, run the embedded symmetry-plane detection ONCE on the
        whole mesh and protect edges whose BOTH endpoints lie on the detected
        plane (within sym_plane_tol) from being merged, in every per-island
        tri_to_quad call. When no plane passes sym_min_coverage, the run
        continues without protection (a warning is printed).
    sym_detect_eps_ratio, sym_min_coverage, sym_plane_tol_ratio : float
        Detection tolerance / acceptance coverage / edge-protection tolerance,
        all relative to the bbox diagonal.
    sym_max_inband_ratio : float
        Mid-surface degeneracy guard: candidate planes whose protect-tolerance
        band holds more than this fraction of ALL vertices are skipped
        (flat / rod-like meshes). Default 0.2; set >1 to disable.
    sym_edge_beta : float
        Per-edge tightening: an edge is protected only if both endpoints are
        within min(sym_plane_tol, beta * edge_length) of the plane — skips
        thin structures pierced by the plane. Default 0.2; set 0 to disable.
    detect_spirals : bool
        Run the C++ spiral detector on the final spliced mesh and report it in
        features["spiral"] (QC only, mesh unchanged).
    repair_spirals : bool
        Scheme-B spiral repair inside every per-island tri_to_quad call
        (alternating-chain flips; quad count never decreases, only legal
        candidates are introduced). Default False.
    spiral_k : int
        Winding threshold pair_m >= k for detection and repair. Default 3.
    spiral_repair_threads : int
        OpenMP threads for the repair (0 = all cores). When the caller already
        processes many FBX files in parallel, use 1–4 to avoid oversubscription.
    verbose : bool
        Forwarded to the C++ core.

    Returns
    -------
    output_obj_path : str
        Absolute path of the written OBJ (same as `output_path` when given).
    features : dict
        A serialisable summary of the run:

            {
              "input_fbx"        : absolute input path,
              "output_obj"       : absolute output path,
              "elapsed_seconds"  : float, wall-clock end-to-end,
              "loop_verts"       : int, pyassimp loop-level vertex count,
              "dedup_verts"      : int, vertex count after (pos, uv) dedup,
              "n_faces"          : int, triangle count in the input,
              "n_islands"        : int, number of UV islands (connected
                                   components after dedup),
              "n_quads"          : int, quads in the output,
              "n_tris"           : int, triangles left in the output,
              "quad_face_ratio"  : float, n_quads / (n_quads + n_tris),
              "tri_merge_ratio"  : float, 2*n_quads / n_faces (fraction of
                                   input triangles that got merged into a
                                   quad),
              "islands_run"      : int, islands that actually reached the
                                   C++ core (>= min_island_size),
              "islands_skipped"  : int, islands emitted as tris without a
                                   C++ call,
              "internal_edges"   : int, sum over islands of candidate
                                   internal edges seen by tri_to_quad,
              "accepted_quads"   : int, sum over islands of C++-side
                                   accepted merges,
              "rejected_quads"   : int, sum over islands of C++-side
                                   posterior-gate rejections,
              "config"           : dict, echo of the tuning parameters
                                   above (for reproducibility),
            }

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError.

    Example
    -------
    >>> from tri2quad import convert_fbx_to_quad
    >>> obj_path, feats = convert_fbx_to_quad(
    ...     "some/input.fbx",
    ...     output_path="some/out.obj",
    ...     enable_edge_align=True, alpha_angle=0.8,
    ...     alpha_aspect=0.0, alpha_align=0.2,
    ... )
    >>> print(feats["n_quads"], feats["n_tris"], feats["n_islands"])
    """
    import time
    t_start = time.time()

    in_path = os.path.abspath(local_file_path)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input fbx not found: {in_path}")
    ext = os.path.splitext(in_path)[1].lower()
    if ext != ".fbx":
        raise ValueError(
            f"convert_fbx_to_quad expects a .fbx file, got '{ext}'. "
            "For .obj / .glb / .gltf use the tri2quad.py CLI or "
            "quad_converter.tri_to_quad directly."
        )

    if output_path is None:
        out_path = os.path.splitext(in_path)[0] + "_quad.obj"
    else:
        out_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    V_uv, F_uv, UV_uv, islands, load_info = load_with_uv_islands(
        in_path,
        tol_pos=dedup_tol_pos,
        tol_uv=dedup_tol_uv,
    )

    # Embedded symmetry detection (once, on the whole mesh) — the plane is in
    # world coordinates and applies to every per-island tri_to_quad call.
    sym_planes, sym_plane_tol = _detect_sym_planes(
        V_uv, F_uv,
        enable=enable_symmetry_protect,
        eps_ratio=sym_detect_eps_ratio,
        min_coverage=sym_min_coverage,
        tol_ratio=sym_plane_tol_ratio,
        max_inband_ratio=sym_max_inband_ratio,
        strict_ratio=sym_strict_tol_ratio,
        verbose=verbose,
    )

    quad_kwargs = dict(
        angle_threshold_deg=angle_threshold_deg,
        method=method,
        verbose=verbose,
        pre_angle_min_deg=pre_angle_min_deg,
        pre_angle_max_deg=pre_angle_max_deg,
        feature_dihedral_thresh_deg=feature_dihedral_thresh_deg,
        enable_concavity_prefilter=enable_concavity_prefilter,
        planarity_eps=planarity_eps,
        enable_edge_align=enable_edge_align,
        alpha_angle=alpha_angle,
        alpha_aspect=alpha_aspect,
        alpha_align=alpha_align,
        sym_planes=sym_planes,
        sym_plane_tol=sym_plane_tol,
        sym_edge_beta=sym_edge_beta,
    )
    if repair_spirals:
        quad_kwargs["spiral_repair"] = True
        quad_kwargs["spiral_repair_k"] = spiral_k
        quad_kwargs["spiral_repair_threads"] = spiral_repair_threads

    quads, tris, isl_stats = _run_tri_to_quad_per_island(
        V_uv, F_uv, islands,
        quad_kwargs=quad_kwargs,
        min_island_size=min_island_size,
        verbose=verbose,
    )
    save_obj_mixed(out_path, V_uv, quads, tris, UV=UV_uv)
    if detect_spirals:
        sp_fin = detect_spiral_report(V_uv, quads, tris, k=spiral_k)
    elif repair_spirals:   # 未单独检测时，用各岛修复后的残留之和（链不跨岛，二者相等）
        sp_fin = {"n_wind_edges": isl_stats["spiral_wind_after"], "n_chains": -1, "max_pair_m": -1}
    else:
        sp_fin = {"n_wind_edges": 0, "n_chains": 0, "max_pair_m": 0}

    n_faces = int(load_info["n_faces"])
    n_quads = int(quads.shape[0])
    n_tris  = int(tris.shape[0])
    total_out = n_quads + n_tris
    features = {
        "input_fbx"       : in_path,
        "output_obj"      : out_path,
        "elapsed_seconds" : round(time.time() - t_start, 4),
        "loop_verts"      : int(load_info["loop_level_verts"]),
        "dedup_verts"     : int(load_info["deduped_verts"]),
        "n_faces"         : n_faces,
        "n_islands"       : int(load_info["n_islands"]),
        "n_quads"         : n_quads,
        "n_tris"          : n_tris,
        "quad_face_ratio" : (n_quads / total_out) if total_out else 0.0,
        "tri_merge_ratio" : (2 * n_quads / n_faces) if n_faces else 0.0,
        "islands_run"     : int(isl_stats["islands_run"]),
        "islands_skipped" : int(isl_stats["islands_skipped"]),
        "internal_edges"  : int(isl_stats["internal_edges"]),
        "accepted_quads"  : int(isl_stats["accepted_quads"]),
        "rejected_quads"  : int(isl_stats["rejected_quads"]),
        "spiral"          : {
            "enabled"          : bool(detect_spirals),
            "repair"           : bool(repair_spirals),
            "wind_edges_before_repair": int(isl_stats["spiral_wind_before"]),
            "repair_steps"     : int(isl_stats["spiral_repair_steps"]),
            "repair_evals"     : int(isl_stats["spiral_repair_evals"]),
            "repair_seconds"   : round(float(isl_stats["spiral_repair_seconds"]), 4),
            "wind_edges_final" : int(sp_fin["n_wind_edges"]),
            "chains_final"     : int(sp_fin["n_chains"]),
            "max_pair_m"       : int(sp_fin["max_pair_m"]),
        },
        "config"          : {
            "method"                       : method,
            "angle_threshold_deg"          : angle_threshold_deg,
            "pre_angle_min_deg"            : pre_angle_min_deg,
            "pre_angle_max_deg"            : pre_angle_max_deg,
            "feature_dihedral_thresh_deg"  : feature_dihedral_thresh_deg,
            "enable_concavity_prefilter"   : enable_concavity_prefilter,
            "planarity_eps"                : planarity_eps,
            "enable_edge_align"            : enable_edge_align,
            "alpha_angle"                  : alpha_angle,
            "alpha_aspect"                 : alpha_aspect,
            "alpha_align"                  : alpha_align,
            "dedup_tol_pos"                : dedup_tol_pos,
            "dedup_tol_uv"                 : dedup_tol_uv,
            "min_island_size"              : min_island_size,
            "enable_symmetry_protect"      : enable_symmetry_protect,
            "sym_detect_eps_ratio"         : sym_detect_eps_ratio,
            "sym_min_coverage"             : sym_min_coverage,
            "sym_plane_tol_ratio"          : sym_plane_tol_ratio,
            "sym_max_inband_ratio"         : sym_max_inband_ratio,
            "sym_edge_beta"                : sym_edge_beta,
            "sym_strict_tol_ratio"         : sym_strict_tol_ratio,
            "sym_planes"                   : sym_planes,
            "detect_spirals"               : detect_spirals,
            "repair_spirals"               : repair_spirals,
            "spiral_k"                     : spiral_k,
            "spiral_repair_threads"        : spiral_repair_threads,
        },
    }
    return out_path, features


def main():
    parser = argparse.ArgumentParser(
        description="Convert a triangle mesh to a quad-dominant OBJ."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input triangle mesh (.obj, .glb, .gltf or .fbx)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output OBJ path (default: same name as input with '_quad.obj' suffix)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="global",
        choices=["greedy", "global", "blossom"],
        help="Matching method: 'greedy' (fast, local-optimum) or 'global' (LEMON max-weight matching = Blossom algorithm, global-optimum). Default: global.",
    )
    parser.add_argument(
        "--angle_threshold",
        type=float,
        default=150.0,
        help="POSTERIOR maximum quad internal angle (degrees). After a quad is built, if its largest angle exceeds this, the merge is rejected and the two triangles are kept. Default: 150.",
    )
    parser.add_argument(
        "--pre_angle_min",
        type=float,
        default=30.0,
        help="PRE-FILTER lower bound for quad internal angles (degrees). Candidate quads with any angle below this are discarded before matching. Default: 30.",
    )
    parser.add_argument(
        "--pre_angle_max",
        type=float,
        default=140.0,
        help="PRE-FILTER upper bound for quad internal angles (degrees). Candidate quads with any angle above this are discarded before matching. Default: 140.",
    )
    parser.add_argument(
        "--feature_dihedral",
        type=float,
        default=45.0,
        help="Feature-edge protection threshold (degrees). If the dihedral angle between the two triangles sharing the candidate edge exceeds this, the edge is treated as a sharp feature and protected (not merged). Default: 45.",
    )
    parser.add_argument(
        "--disable_concavity_prefilter", "--disable_concavity_check",
        action="store_true",
        help="Disable the concavity pre-filter. By default candidate quads detected as concave (or self-intersecting) are discarded; pass this flag to keep them.",
    )
    parser.add_argument(
        "--planarity_eps",
        type=float,
        default=0.0,
        help="Planarity epsilon: max(point-to-plane distance) / mean(edge length). Quads with a ratio above this are discarded. 0 (default) disables the planarity pre-filter.",
    )
    # —— Edge-align scoring (Q_align, paper arXiv:2605.16813) ——
    parser.add_argument(
        "--enable_edge_align",
        action="store_true",
        help="Enable the Q_align scoring term. When set, the quality function becomes (alpha_angle*Q_angle + alpha_aspect*Q_aspect + alpha_align*Q_align)/(sum). When NOT set, the C++ core falls back to the default 0.6*Q_angle + 0.4*Q_aspect.",
    )
    parser.add_argument(
        "--alpha_angle",
        type=float,
        default=0.6,
        help="Weight of Q_angle in the (renormalised) quality function. Only used when --enable_edge_align is set. Default: 0.6.",
    )
    parser.add_argument(
        "--alpha_aspect",
        type=float,
        default=0.4,
        help="Weight of Q_aspect (edge & diagonal balance) in the (renormalised) quality function. Pass 0 to drop Q_aspect entirely. Default: 0.4.",
    )
    parser.add_argument(
        "--alpha_align",
        type=float,
        default=0.0,
        help="Weight of Q_align (edge alignment with local principal direction). Default: 0.0. Recommended: 0.2 when paired with alpha_angle=0.8 and alpha_aspect=0.0 to match the paper's 0.8/0.2 setting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stage statistics from the C++ core.",
    )
    # —— UV-aware pipeline (needed for .fbx and any 'separate_by_uv' mesh) ——
    parser.add_argument(
        "--split_by_uv",
        action="store_true",
        help="Enable the UV-island pipeline. Loads the mesh at the LOOP level "
             "(one vertex per triangle-corner), deduplicates by (position, uv) "
             "so that UV seams stay topologically split, groups faces into "
             "connected components (one component per UV island), and calls "
             "tri_to_quad ONCE PER ISLAND. Seam edges therefore never enter "
             "the candidate pool and never get merged. Default: off (legacy "
             "single-call pipeline). Strongly recommended for .fbx inputs.",
    )
    parser.add_argument(
        "--dedup_tol_pos",
        type=float,
        default=1e-6,
        help="Rounding tolerance on 3D position when doing UV-aware dedup "
             "(--split_by_uv only). Default: 1e-6 (world units).",
    )
    parser.add_argument(
        "--dedup_tol_uv",
        type=float,
        default=1e-6,
        help="Rounding tolerance on UV coords when doing UV-aware dedup "
             "(--split_by_uv only). Default: 1e-6 in [0,1] UV space.",
    )
    parser.add_argument(
        "--min_island_size",
        type=int,
        default=2,
        help="UV islands smaller than this (in # triangles) skip the C++ call "
             "and their triangles are emitted unchanged. Default: 2 (a 1-face "
             "island cannot form a quad).",
    )
    # —— Symmetry-plane edge protection (embedded detection, one step) ——
    parser.add_argument(
        "--enable_symmetry_protect",
        action="store_true",
        help="Enable symmetry-plane edge protection. The symmetry plane is "
             "detected automatically (PCA candidate planes + reflection "
             "coverage, offset-refined) ONCE on the whole input mesh, then any "
             "internal edge whose BOTH endpoints lie on the plane (within "
             "--sym_plane_tol_ratio * bbox_diag) never enters the merge "
             "candidate pool. Works with both the legacy and --split_by_uv "
             "pipelines in a single run. Default: off.",
    )
    parser.add_argument(
        "--sym_detect_eps_ratio",
        type=float,
        default=0.01,
        help="Detection tolerance: a reflected vertex counts as matched if a "
             "mesh vertex lies within eps = ratio * bbox diagonal. Default: 0.01.",
    )
    parser.add_argument(
        "--sym_min_coverage",
        type=float,
        default=0.9,
        help="Acceptance threshold: fraction of vertices whose reflection must "
             "find a match for the plane to be accepted. Default: 0.9.",
    )
    parser.add_argument(
        "--sym_plane_tol_ratio",
        type=float,
        default=0.005,
        help="Edge-protection tolerance: both endpoints within "
             "ratio * bbox diagonal of the plane => edge protected. Default: 0.005.",
    )
    parser.add_argument(
        "--sym_max_inband_ratio",
        type=float,
        default=0.2,
        help="Mid-surface degeneracy guard: a candidate plane whose "
             "protect-tolerance band contains more than this fraction of ALL "
             "vertices is the object's own mid-surface (flat / rod-like "
             "meshes) and is skipped. Default: 0.2. Set >1 to disable.",
    )
    parser.add_argument(
        "--sym_edge_beta",
        type=float,
        default=0.2,
        help="Per-edge tightening: an edge is protected only if BOTH endpoints "
             "are within min(sym_plane_tol, beta * edge_length) of the plane. "
             "This protects only edges truly HUGGING the plane and skips thin "
             "structures pierced by it (thin feathers/sheets, endpoint "
             "distance ~ edge length). Default: 0.2. Set 0 to disable.",
    )
    parser.add_argument(
        "--sym_strict_tol_ratio",
        type=float,
        default=1e-3,
        help="Strict fallback tolerance (ratio * bbox diagonal) used when the "
             "mesh has NO crisp on-plane vertex cluster (no decisive gap in "
             "the vertex-to-plane distances). Enforces 'both endpoints "
             "(numerically) exactly on the plane' semantics: meshes without "
             "exact on-plane vertices protect essentially nothing. "
             "Default: 1e-3.",
    )
    parser.add_argument(
        "--detect_spirals",
        action="store_true",
        help="After conversion, detect spiral loops (open opposite-edge strips "
             "crossed by the same strip >= spiral_k times) on the output mesh "
             "(C++ detector), print stats, and write a "
             "'<output>_spiral.obj' (v + l lines) when any are found. "
             "Default: off.",
    )
    parser.add_argument(
        "--repair_spirals",
        action="store_true",
        help="Scheme-B spiral repair inside the C++ core: after matching, detect "
             "spirals and flip short alternating chains near winding edges "
             "(quad count never decreases, only default-filter-legal quads are "
             "introduced); each round evaluates the top-K flips in parallel and "
             "keeps the best. Default: off.",
    )
    parser.add_argument(
        "--spiral_repair_threads",
        type=int,
        default=0,
        help="OpenMP threads for --repair_spirals (0 = all cores).",
    )
    parser.add_argument(
        "--spiral_k",
        type=int,
        default=3,
        help="Intersection-count threshold: an open chain crossed by the same "
             "chain at least this many times counts as a spiral (paper "
             "convention: >= 3 for robustness). Default: 3.",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.isfile(in_path):
        print(f"[Error] Input files not found: {in_path}")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_quad.obj"
    else:
        out_path = os.path.abspath(args.output)

    # Kwargs shared by both pipelines
    quad_kwargs = dict(
        angle_threshold_deg=args.angle_threshold,
        method=args.method,
        verbose=args.verbose,
        pre_angle_min_deg=args.pre_angle_min,
        pre_angle_max_deg=args.pre_angle_max,
        feature_dihedral_thresh_deg=args.feature_dihedral,
        enable_concavity_prefilter=(not args.disable_concavity_prefilter),
        planarity_eps=args.planarity_eps,
        enable_edge_align=args.enable_edge_align,
        alpha_angle=args.alpha_angle,
        alpha_aspect=args.alpha_aspect,
        alpha_align=args.alpha_align,
        sym_edge_beta=args.sym_edge_beta,
    )
    if args.repair_spirals:
        quad_kwargs["spiral_repair"] = True
        quad_kwargs["spiral_repair_k"] = args.spiral_k
        quad_kwargs["spiral_repair_threads"] = args.spiral_repair_threads

    if args.split_by_uv:
        # UV-island pipeline: load loop-level, dedup by (pos, uv), split into
        # islands, run tri_to_quad per-island, splice results.
        print(f"[Info] Reading (UV-island pipeline): {in_path}")
        V_uv, F_uv, UV_uv, islands, load_info = load_with_uv_islands(
            in_path,
            tol_pos=args.dedup_tol_pos,
            tol_uv=args.dedup_tol_uv,
        )
        print(f"[Info] loaded_via={load_info['loaded_via']}  "
              f"loop_verts={load_info['loop_level_verts']} → deduped_verts="
              f"{load_info['deduped_verts']}  faces={load_info['n_faces']}  "
              f"islands={load_info['n_islands']}")

        # Symmetry detection runs ONCE on the whole mesh; the plane is in world
        # coordinates and is forwarded to every per-island tri_to_quad call.
        sym_planes, sym_plane_tol = _detect_sym_planes(
            V_uv, F_uv,
            enable=args.enable_symmetry_protect,
            eps_ratio=args.sym_detect_eps_ratio,
            min_coverage=args.sym_min_coverage,
            tol_ratio=args.sym_plane_tol_ratio,
            max_inband_ratio=args.sym_max_inband_ratio,
            strict_ratio=args.sym_strict_tol_ratio,
            verbose=args.verbose,
        )
        quad_kwargs["sym_planes"] = sym_planes
        quad_kwargs["sym_plane_tol"] = sym_plane_tol

        print(f"[Info] Running tri_to_quad per island (method={args.method}) ...")

        quads, tris, isl_stats = _run_tri_to_quad_per_island(
            V_uv, F_uv, islands,
            quad_kwargs=quad_kwargs,
            min_island_size=args.min_island_size,
            verbose=args.verbose,
        )
        if args.verbose:
            print("[Stats]", isl_stats)
        if args.repair_spirals:
            print(f"[SpiralRepair] wind_edges(k>={args.spiral_k}): "
                  f"{isl_stats['spiral_wind_before']} -> {isl_stats['spiral_wind_after']} "
                  f"(islands={isl_stats['spiral_repair_islands']}, steps={isl_stats['spiral_repair_steps']}, "
                  f"evals={isl_stats['spiral_repair_evals']}, {isl_stats['spiral_repair_seconds']:.2f}s)")

        # Preserve UV in the output OBJ (V_uv and UV_uv are aligned).
        save_obj_mixed(out_path, V_uv, quads, tris, UV=UV_uv)
        print(f"[OK] Saved to {out_path}")
        print(f"[Summary] quads={quads.shape[0]}, tris={tris.shape[0]}, "
              f"vertices={V_uv.shape[0]} (UV-preserved)")
        if args.detect_spirals:
            # 全局复检 + 导出（--repair_spirals 时修复已在各岛 tri_to_quad 内完成）
            fin = detect_spiral_report(V_uv, quads, tris, k=args.spiral_k)
            print(f"[Spiral] global check: wind_edges(k>={args.spiral_k})="
                  f"{fin['n_wind_edges']}, chains={fin['n_chains']}, "
                  f"max_pair_m={fin['max_pair_m']}")
            sp = _export_spiral_obj(out_path, V_uv, fin["wind_edges"])
            if sp:
                print(f"[Spiral] winding edges written to {sp}")
    else:
        # Legacy pipeline (obj / glb / gltf; also fbx but pyassimp will hand us
        # loop-level 3M vertices with 0 internal edges → 0 quads. That's
        # expected: use --split_by_uv for fbx.).
        print(f"[Info] Reading: {in_path}")
        V, F = load_vertices_faces_auto(in_path)
        print(f"[Info] Vertices={V.shape[0]}, Tri_faces={F.shape[0]}")

        # Symmetry detection runs ONCE on the whole mesh, then the plane is
        # forwarded to the single tri_to_quad call.
        sym_planes, sym_plane_tol = _detect_sym_planes(
            V, F,
            enable=args.enable_symmetry_protect,
            eps_ratio=args.sym_detect_eps_ratio,
            min_coverage=args.sym_min_coverage,
            tol_ratio=args.sym_plane_tol_ratio,
            max_inband_ratio=args.sym_max_inband_ratio,
            strict_ratio=args.sym_strict_tol_ratio,
            verbose=args.verbose,
        )
        quad_kwargs["sym_planes"] = sym_planes
        quad_kwargs["sym_plane_tol"] = sym_plane_tol

        print(f"[Info] Using quad_converter.tri_to_quad (method={args.method}) ...")
        res = quad_converter.tri_to_quad(V, F, **quad_kwargs)

        if "spiral_repair" in res:
            r = res["spiral_repair"]
            print(f"[SpiralRepair] wind_edges(k>={args.spiral_k}): {r['wind_before']} -> {r['wind_after']} "
                  f"(chains={r['chains_after']}, steps={r['steps']}, evals={r['evals']}, {r['seconds']:.2f}s)")
        V_new = res["new_vertices"]  # (N,3)
        tris  = res["tris"]          # (R,3)
        quads = res["quads"]         # (K,4)
        stats = res["merge_stats"]

        if args.verbose:
            print("[Stats]", dict(stats))

        save_obj_mixed(out_path, V_new, quads, tris)
        print(f"[OK] Saved to {out_path}")
        print(f"[Summary] quads={quads.shape[0]}, tris={tris.shape[0]}, "
              f"vertices={V_new.shape[0]}")
        if args.detect_spirals:
            fin = detect_spiral_report(V_new, quads, tris, k=args.spiral_k)
            print(f"[Spiral] wind_edges(k>={args.spiral_k})={fin['n_wind_edges']}, "
                  f"chains={fin['n_chains']}, max_pair_m={fin['max_pair_m']}")
            sp = _export_spiral_obj(out_path, V_new, fin["wind_edges"])
            if sp:
                print(f"[Spiral] winding edges written to {sp}")


if __name__ == "__main__":
    main()
