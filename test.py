import numpy as np
import quad_converter as qc

vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])
triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
result = qc.tri_to_quad(vertices, triangles, method="global", spiral_repair=True)
assert result["quads"].shape == (1, 4)
assert len(result["tris"]) == 0
assert result["spiral_repair"]["wind_after"] == 0

normal = 2 ** -0.5
protected = qc.tri_to_quad(
    vertices, triangles, method="global",
    sym_planes=[normal, -normal, 0.0, 0.0], sym_plane_tol=1e-6,
    spiral_repair=True,
)
assert len(protected["quads"]) == 0
assert len(protected["tris"]) == 2
assert protected["merge_stats"]["num_symmetry_protected_edges"] == 1
print("Conversion, symmetry protection and spiral repair passed.")
