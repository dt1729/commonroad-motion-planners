import pycrcc
import random
import collections
import math
import timeit
from commonroad_cc.visualization.draw_dispatch import draw_object


def restrictor(callable, data, corners, params):
    vehicle_width = params.get('vehicle_width', 1.9)
    vehicle_length = params.get('vehicle_length', 5)
    count = params.get('count', 100_000)
    draw_vehicle = params.get('draw_vehicle', False)

    #Time measurement parameters
    repeat = params.get('repeat', 100)
    setup = params.get('setup', 'pass')
    critical_area = params.get('critical_area')

    const_seed = params.get('seed')
    random.seed(const_seed)

    output = []
    for i in range(0, count):
        x = random.uniform(*corners[:2])
        y = random.uniform(*corners[2:])

        point = pycrcc.Point(x, y)
        if critical_area and not point.collide(critical_area):
            continue

        angle = random.uniform(-math.pi, math.pi)
        vehicle = pycrcc.RectOBB(vehicle_width/2, vehicle_length/2, angle, x, y)

        is_offroad = callable(vehicle, data)
        times = timeit.repeat(lambda: callable(vehicle, data), setup, repeat=repeat, number=1)
        output.append((is_offroad, min(times), vehicle))

        if draw_vehicle:
            draw_object(vehicle)
    return output


def monte_carlo(boundingboxes, road, corners, count=100_000):
    """
    Perform Monte Carlo measurement
    :param count: number of random points
    :param shapegroups: container that has fields: boundingboxes, road
    :param corners: scenario boundaries, within which the random points are distributed uniformly
    :param depth: depth of the iteration, only used as a field for the namedtuple
    :return namedtuple that contains fields: count, bbhit, roadhit, miss
    """

    random.seed()
    bbhit = roadhit = miss = 0

    for i in range(0, count):
        x = random.uniform(*corners[:2])
        y = random.uniform(*corners[2:])
        point = pycrcc.Point(x, y)
        if point.collide(road):
            roadhit += 1
        elif point.collide(boundingboxes):
            bbhit += 1
        else:
            miss += 1

    result = collections.namedtuple('result', 'count bbhit roadhit miss precision error')
    return result(count, bbhit, roadhit, miss, precision(count, roadhit, miss), error(count, miss))


def monte_carlo_critical_area(boundingboxes, road, critical_area, corners, count=100_000):
    """Do Monte Carlo in scenario but only count points that are within the critical_area"""
    random.seed()
    critical_area_hit = bbhit = roadhit = miss = 0

    while critical_area_hit < count:
        x = random.uniform(*corners[0:2])
        y = random.uniform(*corners[2:4])
        point = pycrcc.Point(x, y)
        if point.collide(critical_area):
            critical_area_hit += 1
            if point.collide(road):
                roadhit += 1
            elif point.collide(boundingboxes):
                bbhit += 1
            else:
                miss += 1

    result = collections.namedtuple('result', 'count bbhit roadhit miss precision error')
    return result(count, bbhit, roadhit, miss, precision(count, roadhit, miss), error(count, miss))


def precision(count, roadhit, miss):
    """
    Calculate the precision
    """
    return (count - roadhit - miss) / (count - roadhit)


def error(count, miss):
    """
    Calculate the error
    """
    return miss / count
