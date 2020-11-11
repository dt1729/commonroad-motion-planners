from commonroad_cc.visualization.draw_dispatch import draw_object
import pycrcc
import triangle_builder, rectangle_builder, lanelet_bounds, scenario_bounds
import plot_scenario

import timeit
import numpy as np


def construct(scenario, build_order, draw_order=[], plot_order=['plot'], boundary_margin=20):
    """Construct road bounding boxes, using
        :param scenario: CommonRoad scenario (use CommonRoadFileReader to read in a scenario xml file)
        :param build_order: list of strings that describe which shapes to construct,
            see build(orders) for the possible options
        :param draw_order: list of strings that describe which shapes to draw,
            see draw(orders) for the possible options
        :param plot_order: list of strings that describe what should happen to the plot of the scenario
        :param boundary_margin: the axis aligned rectangle, which surrounds all lanelets of the scenario,
            is enlarged at each corner by this value

    """

    def build(orders):
        if orders == []:
            return

        # helper functions to handle the two different types of road triangles
        def copy_into_road(shapegroup):
            """copy the elements of a shapegroup into the road shapegroup"""
            for object in shapegroup.unpack():
                road.add_shape(object)

        def build_simple_triangles():
            triangle_builder.build_simple_triangles(lanelet_network.lanelets, simple_triangles)
            copy_into_road(simple_triangles)

        def build_section_triangles():
            for lane_polyline in lanelet_bounds.lane_hull(lanelet_network):
                lane_polyline = lane_polyline.tolist()
                triangle_builder.triangulate(lanelet_bounds.polyline_edges(lane_polyline), [], None, section_triangles,
                                             {})
            copy_into_road(section_triangles)

        def help_build_quadtree(params):
            # detect opportunity to use speedup automatically, if not explicitly given
            if 'speedup' not in params:
                params['speedup'] = (rectangles.unpack() != [])
            build_quadtree(0, *boundaries, params)

        def build_lane_sections(width=0.2):
            for vertices in lanelet_bounds.lane_hull(lanelet_network):
                for p1, p2 in lanelet_bounds.polyline_edges(vertices):
                    rectangles.add_shape(rectangle_builder.get_rectangle(width, p1, p2))

        def help_triangulate(params):
            # hull: flag that indicates whether a hull is build around the road to bound the triangulation area
            # -True: hull is built from a minkowski sum
            # -False: the corners of the scenario are used instead
            hull = params.get('hull', True)
            offset = params.get('hull_offset', 2)
            bounds = [bound for bound in lanelet_bounds.pairwise_bounds(lanelet_network)]

            if hull:
                points = [point for lane_section in lanelet_bounds.lane_hull(lanelet_network) for point in lane_section]
                hull_points = simple_minkowski(points, offset, road)
                triangle_builder.triangulate(bounds, hull_points, road, triangulation, params)
            else:
                corner_vertices = [(corners[0], corners[2]), (corners[1], corners[2]),
                                   (corners[1], corners[3]), (corners[0], corners[3])]
                corner_edges = lanelet_bounds.polyline_edges(corner_vertices)
                triangle_builder.triangulate(bounds + corner_edges, [], road, triangulation, params)

        def build_critical_area(params):
            width = params.get('width', 2)
            triangle_builder.build_offset_section_triangles(lanelet_network, critical_area, width)

        params = {}
        options = {
            "quads": lambda: help_build_quadtree(params),
            "simple_triangles": lambda: build_simple_triangles(),
            "section_triangles": lambda: build_section_triangles(),
            "rectangles": lambda: rectangle_builder.build_rectangles(lanelet_network, rectangles, params),
            "lane_sections": lambda: build_lane_sections(**params),
            "shell": lambda: rectangle_builder.build_shell(lanelet_network, shell, road, params),
            "triangulation": lambda: help_triangulate(params),
            "critical_area": lambda: build_critical_area(params),
            "runtime": lambda: measure_runtime(params)
        }
        for order in orders:
            params = {}
            if not type(order) == str:
                # the first element of order determines which option is taken, the following is a dictionary with
                # parameters passed to the functions
                # example: order = ('quads', {'max_depth': 10})
                # result: build_quadtree(..., {'max_depth': 10})
                params = order[1]
                order = order[0]
            if order in options:
                options.get(order)()

    # Can be used in runtime evaluation (call with lambda function) to reset the quads and rectangles
    def reset_shapegroups():
        def init():
            return pycrcc.ShapeGroup()

        global quads, rectangles
        quads = init()
        rectangles = init()

    def measure_runtime(params):
        stmt = params.get('build', 'pass')
        setup = params.get('setup', 'pass')
        n = params.get('number', 100)

        f = lambda: build(stmt)
        times = timeit.repeat(f, setup, repeat=n, number=1, globals=globals())
        runtime.append(min(times))

    def draw(orders):
        if orders == []:
            return
        options = {
            "quads": quads,
            "simple_triangles": simple_triangles,
            "section_triangles": section_triangles,
            "rectangles": rectangles,
            "antiquads": antiquads,
            "shell": shell,
            "triangulation": triangulation,
            "critical_area": critical_area,
        }
        default_colors = {'rectobb': {'facecolor': '#b2df8a'},
                          'rectaabb': {'facecolor': '#a6cee3'},
                          'triangle': {'facecolor': '#ff9999'},
                          'circle': {'facecolor': '#1f78b4', },
                          'point': {'color': '#e31a1c'},
                          'boundingvolume': {'facecolor': 'none',
                                             'draw_recursively': True,
                                             'draw_contained_collision_object': True}}

        special_colors = {'simple_triangles': {'triangle': {'facecolor': '#fb9a99'}},
                          'section_triangles': {'triangle': {'facecolor': '#fdbf6f'}},
                          'triangulation': {'triangle': {'facecolor': '#ff7f00'}},
                          'shell': {'rectobb': {'facecolor': '#33a02c'}},
                          'critical_area': {'triangle': {'facecolor': '#cab2d6', 'zorder': 10}},
                          }

        def add_edge_and_zorder(object):
            # all collision objects should be drawn with black borders on the same plane if not explicitly changed
            # order of the draw calls determines the visibility of objects on the same plane
            if 'edgecolor' not in object:
                object['edgecolor'] = '#000000'
            if 'zorder' not in object:
                object['zorder'] = 20

        for entry in default_colors:
            add_edge_and_zorder(default_colors[entry])
        for entry in special_colors:
            subdictionary = special_colors[entry]
            for subentry in subdictionary:
                add_edge_and_zorder(subdictionary[subentry])

        for order in orders:
            if order in options:
                draw_object(options.get(order), draw_params={'no_parent': {'collision': special_colors.get(order, default_colors)}})

    def plot(orders):
        if orders == []:
            return
        draw_object(scenario)
        params = {}
        options = {
            'plot': lambda: plot_scenario.show(corners, params),
            'save': lambda: plot_scenario.save_scenario(corners, params),
        }

        for order in orders:
            params = {}
            if not type(order) == str:
                params = order[1]
                order = order[0]
            if order in options:
                options.get(order)()

    def handle_return():
        dict = {
            "quads": quads,
            "simple_triangles": simple_triangles,
            "section_triangles": section_triangles,
            "rectangles": rectangles,
            "antiquads": antiquads,
            "shell": shell,
            "triangulation": triangulation,
            "critical_area": critical_area,
            "corners": corners,
            "runtime": runtime,
        }

        return dict

    def build_quadtree(depth, rx, ry, x, y, params):
        max_depth = params.get('max_depth', 10)
        build_antiquads = params.get('build_antiquads', False)
        speedup = params.get('speedup', False)

        quad = pycrcc.RectAABB(rx, ry, x, y)
        if quad.collide(road):
            if depth >= max_depth or (speedup and not quad.collide(rectangles)):
                if (build_antiquads):
                    antiquads.add_shape(quad)
            else:
                depth += 1

                def recursive(x, y):
                    build_quadtree(depth, rx, ry, x, y, params)

                # expand
                rx /= 2
                ry /= 2
                recursive(x + rx, y + ry)
                recursive(x - rx, y + ry)
                recursive(x + rx, y - ry)
                recursive(x - rx, y - ry)
        else:
            quads.add_shape(quad)

    def simple_minkowski(points, radius, collision_mesh):
        """
        Basic algorithm to enlarge a polygon by a constant amount.
        Can be interpreted as a minkowski sum between the polygon defined by the points and a circle with the radius.
        Computes the normal for each point and offsets them by half the radius.
        :param points: List of points of the polygon.
        :param radius: Distance by which the polygon is enlarged. Each point is moved by half the radius to the outside.
        :param collision_mesh: Collision objects that define the inside area of the polygon.
        :return: point list with offset points
        """
        normals = lanelet_bounds.polyline_normals(points)
        offsets = [normal * radius / 2 for normal in normals]
        # The normals could point inside or outside the polygon, offset in both directions
        sum_points = [[p - offset, p + offset] for offset, p in zip(offsets, points)]

        # uncomment for visualization
        # for p, offsets in zip(points, sum_points):
        #    draw_object(rectangle_builder.get_rectangle(0.1, p, offsets[0]))
        #    draw_object(rectangle_builder.get_rectangle(0.05, p, offsets[1]))

        # Flatten
        sum_points = [point for tuple in sum_points for point in tuple]

        return [p for p in sum_points if not pycrcc.Point(*p).collide(collision_mesh)]

    lanelet_network = scenario.lanelet_network
    lanelets = lanelet_network.lanelets

    def init():
        return pycrcc.ShapeGroup()

    quads = init()
    road = init()
    simple_triangles = init()
    section_triangles = init()
    rectangles = init()
    antiquads = init()
    shell = init()
    triangulation = init()
    critical_area = init()

    # min and max coordinates of the scenario
    corners = scenario_bounds.calc_corners(lanelets, boundary_margin)
    # coordinates of an axis aligned box that surrounds the entire scenario
    boundaries = scenario_bounds.calc_boundary_box(corners)
    runtime = []

    build(build_order)
    draw(draw_order)
    plot(plot_order)

    return handle_return()
