import pycrcc
import lanelet_bounds
import numpy as np


def get_rectangle(width, v1, v2):
    """builds a rectangle object which has the line v1,v2 as middle line"""
    v = v2 - v1
    r_x = np.linalg.norm(v) / 2
    r_y = width / 2
    orientation = np.arctan2(v[1], v[0])
    center = v1 + (v) / 2
    return pycrcc.RectOBB(r_x, r_y, orientation, *center)


def build_rectangles(lanelet_network, rectangles, params):
    """Build rectangles that enclose the lanelet network along its outer boundaries
    """
    width = params.get('width', 0.2)

    def add_rectangle(v1, v2):
        rectangles.add_shape(get_rectangle(width, v1, v2))

    for left_vertices, right_vertices in lanelet_bounds.lateral_bounds(lanelet_network):
        previous = left_vertices[0]
        for v in left_vertices[1:]:
            add_rectangle(previous, v)
            previous = v
        previous = right_vertices[0]
        for v in right_vertices[1:]:
            add_rectangle(previous, v)
            previous = v
    for v1, v2 in lanelet_bounds.longitudinal_bounds(lanelet_network.lanelets):
        add_rectangle(v1, v2)


def build_shell(lanelet_network, rectangles, triangles, params):
    """Build a shell that encloses the lanelet network along its outer boundaries
        Unlike build_rectangles, the shell rectangles lie completely outside the lanelets
    """
    width = params.get('width', 0.2)
    margin = params.get('margin', 0.02)
    max_depth = params.get('max_depth', 5)

    def add_rectangle(v1, v2, local_width=width, depth=0):
        """adds rectangle only if it doesn't collide with a lanelet triangle
            if it collides with a triangle, do a quadtree process"""
        r = get_rectangle(local_width, v1, v2)
        if r.collide(triangles):
            if not depth >= max_depth:
                depth += 1
                rotation_matrix = np.array([[0, -1], [1, 0]])
                line = v1 - v2
                tangent = np.dot(rotation_matrix, line)
                if np.linalg.norm(tangent) != 0:
                    tangent = local_width / 4 * tangent / np.linalg.norm(tangent)
                middle = ((v1 - v2) / 2) + v2

                def recursive(v, w):
                    add_rectangle(v, w, local_width / 2, depth)

                recursive(v1 + tangent, middle + tangent)
                recursive(middle + tangent, v2 + tangent)
                recursive(v1 - tangent, middle - tangent)
                recursive(middle - tangent, v2 - tangent)
        else:
            rectangles.add_shape(r)

    offset = width / 2 + margin
    for bound in lanelet_bounds.offset_bounds(lanelet_network, offset):
        previous = bound[0]
        for v in bound[1:]:
            add_rectangle(previous, v)
            previous = v
