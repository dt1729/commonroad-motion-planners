from commonroad.common.file_reader import CommonRoadFileReader
import os

from construction import construct
from vehicle_restrictor import eval_restrictor
import measurement
import collections


def open_scenario(scenario_file):
    """Opens a scenario from file"""
    return CommonRoadFileReader(os.getcwd() + scenario_file).open_scenario()

def open_measurable(scenario_file):
    """Opens a scenario from an .xml file in /boundary/scenarios/measurable/"""
    return open_scenario('/boundary/scenarios/measurable/' + scenario_file + '.xml')

def construct_from_file(scenario_file, *vargs, **kwargs):
    """Opens the scenario_file and then calls construct"""
    scenario = open_scenario(scenario_file)
    return construct(scenario, *vargs, **kwargs)

def quad_test():
    """Builds a quadtree for an example scenario"""
    build = ['simple_triangles', 'quads']
    draw = ['quads']
    scenario = open_measurable('GER_Ffb_2')
    construct(scenario, build, draw)

def rectangle_test():
    build = ['simple_triangles', 'section_triangles', 'rectangles']
    draw = ['rectangles']
    scenario = open_measurable('GER_Ffb_2')
    construct(scenario, build, draw)


def shell_test():
    build = ['simple_triangles', 'section_triangles', 'shell']
    draw = ['shell']
    scenario = open_measurable('GER_Ffb_2')
    construct(scenario, build, draw)


def tri_test():
    build = ['section_triangles', 'triangulation']
    draw = ['triangulation']
    scenario = open_measurable('GER_Ffb_2')
    construct(scenario, build, draw)


def quadtree_eval_depth(folder='/boundary/scenarios/measurable/', scenario='GER_Muc_1a', maximum=13):
    results = []
    for i in range(0, maximum):
        build = ['simple_triangles', 'rectangles', ('quads', {'max_depth': i})]
        t = construct_from_file(folder + scenario + '.xml', build)
        boundingboxes = t['quads']
        road = t['simple_triangles']
        corners = t['corners']
        test_result = measurement.monte_carlo(boundingboxes, road, corners, 200_000)

        test_result = prepend_result('max_depth', i, test_result)
        print(test_result)
        results.append(test_result)

    filename = 'quadtree_depth_' + scenario + '.csv'
    write_evaluation(filename, results)


def shell_eval_depth(folder='/boundary/scenarios/measurable/', scenario='GER_Muc_1a', maximum=6):
    results = []
    for i in range(0, maximum):
        build = ['simple_triangles', ('shelltriangulation', {'width': 1, 'max_depth': i})]
        t = construct_from_file(folder + scenario + '.xml', build)
        boundingboxes = t['shelltriangulation']
        road = t['simple_triangles']
        corners = t['corners']
        test_result = measurement.monte_carlo(boundingboxes, road, corners, 200_000)

        test_result = prepend_result('max_depth', i, test_result)
        print(test_result)
        results.append(test_result)

    filename = 'shell_depth_' + scenario + '.csv'
    write_evaluation(filename, results)


def quadtree_eval_depth_ca(folder='/boundary/scenarios/measurable/', scenario='GER_Muc_1a', maximum=13, ca_width=2):
    results = []
    for i in range(0, maximum):
        build = ['simple_triangles', 'rectangles', ('quads', {'max_depth': i}), ('critical_area', {'width': ca_width})]
        t = construct_from_file(folder + scenario + '.xml', build)
        boundingboxes = t['quads']
        road = t['simple_triangles']
        critical_area = t['critical_area']
        corners = t['corners']
        test_result = measurement.monte_carlo_critical_area(boundingboxes, road, critical_area, corners, 100_000)

        test_result = prepend_result('max_depth', i, test_result)
        print(test_result)
        results.append(test_result)

    filename = 'quadtree_depth_ca_' + scenario + '.csv'
    write_evaluation(filename, results)


def shell_eval_depth_ca(folder='/boundary/scenarios/measurable/', scenario='GER_Muc_1a', maximum=6, ca_width=2):
    results = []
    for i in range(0, maximum):
        build = ['simple_triangles', ('shell', {'width': ca_width * 2, 'max_depth': i}),
                 ('critical_area', {'width': ca_width})]
        t = construct_from_file(folder + scenario + '.xml', build)
        boundingboxes = t['shell']
        road = t['simple_triangles']
        critical_area = t['critical_area']
        corners = t['corners']
        test_result = measurement.monte_carlo_critical_area(boundingboxes, road, critical_area, corners, 100_000)

        test_result = prepend_result('max_depth', i, test_result)
        print(test_result)
        results.append(test_result)

    filename = 'shell_depth_ca_' + scenario + '.csv'
    write_evaluation(filename, results)


def eval_precision_all(folder='/boundary/scenarios/measurable/', quad_max=13, shell_max=6, ca_width=2, scenarios=
['B471_Garching_Lanesplit', 'B471_Lustheim_Roundabout', 'GER_Ffb_2', 'GER_Muc_1a',
 'Intersection_Leopold_Hohenzollern_v3', 'NGSIM_US_101']):
    for scenario in scenarios:
        print(scenario)
        quadtree_eval_depth(folder, scenario, quad_max)
        quadtree_eval_depth_ca(folder, scenario, quad_max, ca_width)
        shell_eval_depth(folder, scenario, shell_max)
        shell_eval_depth_ca(folder, scenario, shell_max, ca_width)


def eval_number(folder, scenario, iterator, boundingbox_name, maximum, iteratorname):
    """iterator: lambda function that provides a list of inputs for the construct function, with an input step
        boundingboxes: name of shapegroup that provides the bounding boxes
    """

    results = []
    for i in range(0, maximum):
        t = construct_from_file(folder + scenario + '.xml', *iterator(i))
        boundingboxes = t[boundingbox_name]

        number = len(boundingboxes.unpack())
        number_result = collections.namedtuple('number_result', iteratorname + ' number')
        test_result = number_result(i, number)
        results.append(test_result)
        print(test_result)

    filename = boundingbox_name + '_number_' + iteratorname + '_ ' + scenario + '.csv'
    write_evaluation(filename, results)


def eval_number_all(folder='/boundary/scenarios/measurable/', quad_max=14, shell_max=7, scenarios=
['B471_Garching_Lanesplit', 'B471_Lustheim_Roundabout', 'GER_Ffb_2', 'GER_Muc_1a',
 'Intersection_Leopold_Hohenzollern_v3', 'NGSIM_US_101']):
    quadtree_orders = lambda i: [['simple_triangles', ('quads', {'max_depth': i})]]
    shell_orders = lambda i: [['simple_triangles', ('shell', {'max_depth': i})]]
    triangle_orders = lambda i: [['triangulation']]
    for scenario in scenarios:
        print(scenario)
        eval_number(folder, scenario, quadtree_orders, 'quads', quad_max, 'depth')
        eval_number(folder, scenario, shell_orders, 'shell', shell_max, 'depth')
        eval_number(folder, scenario, triangle_orders, 'triangulation', 1, 'x')


def eval_runtime(folder, scenario, iterator, boundingbox_name, maximum, iteratorname):
    results = []
    for i in range(0, maximum):
        runtime = construct_from_file(folder + scenario + '.xml', *iterator(i))['runtime'][0]

        runtime_result = collections.namedtuple('runtime_result', iteratorname + ' runtime')
        test_result = runtime_result(i, runtime)
        results.append(test_result)
        print(test_result)

    filename = boundingbox_name + '_runtime_' + iteratorname + '_ ' + scenario + '.csv'
    write_evaluation(filename, results)


def eval_runtime_all(folder='/boundary/scenarios/measurable/', quad_max=13, shell_max=7, number_tests=20, scenarios=
['B471_Garching_Lanesplit', 'B471_Lustheim_Roundabout', 'GER_Ffb_2', 'GER_Muc_1a',
 'Intersection_Leopold_Hohenzollern_v3', 'NGSIM_US_101']):
    quadtree = lambda i: [
        ['simple_triangles', ('runtime', {'number': number_tests, 'build': [('quads', {'max_depth': i})]})]]
    shell = lambda i: [
        ['simple_triangles', ('runtime', {'number': number_tests, 'build': [('shell', {'max_depth': i})]})]]
    triangle = lambda i: [['simple_triangles', ('runtime', {'number': number_tests, 'build': ['triangulation']})]]
    speedup_without_rectangles = lambda i: [['simple_triangles', 'rectangles', ('runtime', {'number': number_tests,
                                                                                            'build': [('quads',
                                                                                                       {'speedup': True,
                                                                                                        'max_depth': i})]})]]
    rectangle_setup = 'lambda : reset_shapegroups()'
    speedup_with_rectangles = lambda i: [['simple_triangles', ('runtime', {'number': number_tests,
                                                                           'build': ['rectangles', (
                                                                               'quads',
                                                                               {'speedup': True, 'max_depth': i})],
                                                                           'setup': rectangle_setup})]]

    for scenario in scenarios:
        print(scenario)
        eval_runtime(folder, scenario, quadtree, 'quads', quad_max, 'depth')
        eval_runtime(folder, scenario, speedup_with_rectangles, 'speedup_with', quad_max, 'depth')
        eval_runtime(folder, scenario, speedup_without_rectangles, 'speedup_without', quad_max, 'depth')
        eval_runtime(folder, scenario, shell, 'shell', shell_max, 'depth')
        eval_runtime(folder, scenario, triangle, 'triangulation', 1, 'x')


def prepend_result(field, value, result):
    newresult = collections.namedtuple('result', (field,) + result._fields)
    return newresult(value, *list(result))


def write_evaluation(filename, evaluation):
    with open(filename, 'w') as file:
        file.write(','.join(evaluation[0]._fields) + '\n')
        for result in evaluation:
            file.write(','.join([str(x) for x in list(result)]) + '\n')

def repair_scenario(scenario):
    #TODO: add params
    from repair import repair
    scenario.lanelet_network = repair(scenario.lanelet_network)
    return scenario

def repair_test(max_distance):
    from commonroad.scenario import lanelet
    import numpy as np
    v = [(0,1), (1,1.5), (2,1), (3.5,0.5), (4.5, 1)]
    w = [(0.,0.), (1.,0.), (2.,0.), (3.,0.), (4.,0.), (5.,0.), (6.,0.)]
    x = [(-3.,0.1),(0.2,0.12)]

    v = np.array([np.array(x) for x in v])
    w = np.array([np.array(x) for x in w])
    x = np.array([np.array(x) for x in x])

    left = v + np.array([0, 2])
    right = v
    a = lanelet.Lanelet(left_vertices = left, right_vertices = right, center_vertices= (left + right)/2, lanelet_id = 0)
    left = np.flipud(w + np.array([0,-2]))
    right = np.flipud(w)
    b = lanelet.Lanelet(left_vertices = left, right_vertices= right, center_vertices=(left + right)/ 2, lanelet_id = 1)
    left = x + np.array([0, 1.7])
    right = x
    c = lanelet.Lanelet(left_vertices=left, right_vertices=right, center_vertices=(left + right) / 2, lanelet_id=2)
    a.adj_right, a.adj_right_same_direction = (1, False)
    b.adj_right, b.adj_right_same_direction = (0, False)
    c.successor = [1]
    a.predecessor = [2]

    from commonroad.scenario.lanelet import LaneletNetwork
    lanelet_network = LaneletNetwork.create_from_lanelet_list([a, b, c])
    from repair import repair
    lanelet_network = repair(lanelet_network, max_distance=max_distance)

    from commonroad_cc.visualization.draw_dispatch import draw_object
    draw_params = {'lanelet': {'left_bound_color': '#111111',
                               'right_bound_color': '#555555',
                               'center_bound_color': '#dddddd',
                               'draw_left_bound': True,
                               'draw_right_bound': True,
                               'draw_center_bound': True,
                               'draw_border_vertices': True,
                               'draw_start_and_direction': False,
                               'show_label': True}}
    draw_params = {'no_parent': {'scenario': draw_params}}
    draw_object(lanelet_network, draw_params=draw_params)
    import matplotlib.pyplot as plt
    plt.gca().set_aspect('equal')
    plt.show()