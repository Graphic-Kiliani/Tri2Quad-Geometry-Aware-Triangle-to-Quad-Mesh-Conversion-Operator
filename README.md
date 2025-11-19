### 1. Introduction
I developed a high-performance triangle-to-quad-dominant conversion operator and it reaches SOTA effect with rich geometric filter control and globally post-merge consistent normals. Come and try it !!!


### 2. Installation
```bash
pip install quad_converter-1.0.0-cp311-cp311-linux_x86_64.whl
```

### 3. Test

####  Instructions
```bash
# Test Python API
python -c "import quad_converter; print(quad_converter.__version__)"

# Test Command Line Tools, we provide high control of geometric attributes, check it here!
quad-convert --help

# Test conversion
# Basic Usage
quad-convert --input test.obj --output test_quad.obj --verbose

# Usage with parameter
quad-convert \
  --input model.obj \
  --output model_quad.obj \
  --angle_threshold 150 \
  --method blossom \
  --verbose

# test.py for tri-obj input and ouput quad-output
python test.py
```



####  Python API

```python
import quad_converter
import numpy as np

# Prepare Mesh Data
vertices = np.array([...], dtype=np.float64)  # (N, 3)
triangles = np.array([...], dtype=np.int32)   # (M, 3)

# Convert to quad dominant mesh
result = quad_converter.convert_tri_to_quad(
    vertices=vertices,
    triangles=triangles,
    angle_threshold_deg=150.0,
    matching_method="blossom"
)

# Results
new_vertices = result["new_vertices"]
quad_faces = result["quads"]
tri_faces = result["tris"]
stats = result["merge_stats"]

print(f"Generate {len(quad_faces)} quads")
print(f"Left {len(tri_faces)} tris")
```
