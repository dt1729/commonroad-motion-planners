import os

from commonroad.common.file_reader import CommonRoadFileReader
import pycrcc

from construction import construct
import triangle_builder
import measurement
import lanelet_bounds

import Polygon, Polygon.Utils

import numpy as np
from commonroad_cc.visualization.draw_dispatch import draw_object
import matplotlib.pyplot as plt

def eval_restrictor(scenario, construct_build=['section_triangles', ('triangulation', {'hull': False})],
             construct_kwargs={'draw_order': [], 'plot_order': []}, count=10_000, repeat = 10):
    t = construct(scenario, construct_build, **construct_kwargs)
    boundary_rep = t.get('triangulation')
    corners = t.get('corners')

    lanelet_network = scenario.lanelet_network
    lanelets = lanelet_network.lanelets


    def inside_road(vehicle, rep, fill_holes):
        candidates = [polygon for polygon,collision_mesh in rep if collision_mesh.collide(vehicle)]

        if not candidates:
            return False

        polygon = candidates[0]
        for p in candidates[1:]:
            polygon = polygon | p

        if fill_holes:
            polygon = Polygon.Utils.fillHoles(polygon)

        def rectobb_vertices(rect):
            rx = np.array(rect.local_x_axis()) * np.array(rect.r_x())
            ry = np.array(rect.local_y_axis()) * np.array(rect.r_y())
            center = np.array(rect.center())
            return [rx + ry, rx - ry, -rx - ry, -rx + ry] + center

        vehicle_polygon = Polygon.Polygon(rectobb_vertices(vehicle))
        return polygon.covers(vehicle_polygon)


    def lanelet_rep_setup(lanelets):
        lanelet_rep = []
        for lanelet in lanelets:
            triangles = pycrcc.ShapeGroup()
            triangle_builder.build_simple_triangles([lanelet], triangles)
            polyline = np.concatenate((lanelet.left_vertices, np.flipud(lanelet.right_vertices)))
            polygon = Polygon.Polygon(polyline)
            lanelet_rep.append((polygon, triangles))
        return lanelet_rep

    draw_object(scenario)
    def lane_rep_setup(lanelet_network):
        lane_rep = []
        for lane_polyline in lanelet_bounds.lane_hull(lanelet_network):
            triangles = pycrcc.ShapeGroup()
            triangle_builder.triangulate(lanelet_bounds.polyline_edges(lane_polyline.tolist()), [], None, triangles, {})
            polygon = Polygon.Polygon(lane_polyline)
            lane_rep.append((polygon, triangles))
        return lane_rep

    def inside_polygon(vehicle, polygon):
        def rectobb_vertices(rect):
            rx = np.array(rect.local_x_axis()) * np.array(rect.r_x())
            ry = np.array(rect.local_y_axis()) * np.array(rect.r_y())
            center = np.array(rect.center())
            return [rx + ry, rx - ry, -rx - ry, -rx + ry] + center

        vehicle_polygon = Polygon.Polygon(rectobb_vertices(vehicle))
        return polygon.covers(vehicle_polygon)

    def whole_polygon_setup(polylines):
        polygon = Polygon.Polygon()
        for polyline in polylines:
            polygon = polygon | Polygon.Polygon(polyline)

        return polygon

    lanelet_rep = lanelet_rep_setup(lanelets)
    lane_rep = lane_rep_setup(lanelet_network)
    whole_lane_polygon = whole_polygon_setup([lane for lane in lanelet_bounds.lane_hull(lanelet_network)])
    whole_lanelet_polygon = whole_polygon_setup(
        [np.concatenate((lanelet.left_vertices, np.flipud(lanelet.right_vertices))) for lanelet in lanelets])

    #write polygon representations to file
    from Polygon import IO
    IO.writeSVG('lanelet_rep.svg', [polygon for polygon, triangle in lanelet_rep], stroke_width= [0.05])
    IO.writeSVG('lane_rep.svg', [polygon for polygon, triangle in lane_rep], stroke_width= [0.05])
    IO.writeSVG('whole_lane_polygon.svg', whole_lane_polygon, stroke_width= [0.05])
    IO.writeSVG('whole_lanelet_polygon.svg', whole_lanelet_polygon, stroke_width= [0.05])

    #Get random seed that is used in the functions
    import time
    current_milli_time = lambda: int(round(time.time() * 1000))
    seed = current_milli_time()

    # Compare different vehicle restricting functions:
    # Create a lambda function and send it to the measurement.restrictor function
    # There the results and the time of individual random samples is recorded
    # Additionally, the total time for each measurement is recorded here
    runs = [
        ('lanelet_polygons', lambda vehicle, data: not inside_road(vehicle, data, True), lanelet_rep),
        ('lane_polygons', lambda vehicle, data: not inside_road(vehicle, data, False), lane_rep),
        ('whole_lane_polygon', lambda vehicle, data: not inside_polygon(vehicle, data), whole_lane_polygon),
        ('whole_lanelet_polygon', lambda vehicle, data: not inside_polygon(vehicle, data), whole_lanelet_polygon),
        ('boundary', lambda vehicle, boundary: vehicle.collide(boundary), boundary_rep),
    ]
    result = []
    for name, function, data in runs:
        total_time = current_milli_time()
        run_result = measurement.restrictor(function,data, corners, {'count': count, 'seed': seed, 'repeat': repeat})
        total_time = current_milli_time() - total_time
        result.append((name, run_result, total_time))

    def avg(l):
        if len(l) == 0:
            return float('nan')
        return round(sum(l) / len(l),5)

    print('vehicle_restrictor evaluation')
    print('count', count)
    print('#name, total_time, avg_time, avg_time_false, avg_time_true')
    for name, output, total_time in result:
        avg_time = avg([time for binary, time, vehicle in output])
        avg_time_false = avg([time for binary, time, vehicle in output if not binary])
        avg_time_true = avg([time for binary, time, vehicle in output if binary])
        print(name, total_time, avg_time, avg_time_false, avg_time_true, sep=',')

    draw_object(scenario)

    # Print the cases where not all outputs are identical and draw the vehicles where the difference happened
    vehicles = [vehicle for decision, time, vehicle in result[0][1]]
    decisions = []
    for output in [output for name, output, total_time in result]:
        decisions.append([decision for decision, time, vehicle in output])

    def is_uniform(x):
        x0 = x[0]
        for xi in x[1:]:
            if xi != x0:
                return False
        return True
    differences = [(i,x) for i,x in enumerate(zip(*decisions)) if not is_uniform(x)]
    if differences:
        print('#Differences:', len(differences), 'of', count)
        for i,x in differences:
            print(i, *x, '')
            draw_object(vehicles[i])

    #for polygon, collision_mesh in lane_rep:
    #    draw_object(collision_mesh)
    draw_object(boundary_rep)

    corners = t.get('corners')
    plt.axis([*corners])
    plt.gca().set_aspect('equal')
    plt.show()


