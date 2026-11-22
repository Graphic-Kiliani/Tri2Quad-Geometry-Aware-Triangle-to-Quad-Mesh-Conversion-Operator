from .quad_converter import detect_spirals, detect_symmetry_plane, tri_to_quad

__version__ = "1.1.0"
convert_tri_to_quad = tri_to_quad

__all__ = ["tri_to_quad", "convert_tri_to_quad", "detect_symmetry_plane", "detect_spirals"]


def get_version():
    return __version__
