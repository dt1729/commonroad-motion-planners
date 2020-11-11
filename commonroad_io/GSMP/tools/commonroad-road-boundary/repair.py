import numpy as np
from fvks.geometry.polyline import polyline_nearest_point, resample_polyline, line_segment_nearest_point


def align_corners(lanelets, max_distance):
    """
    Ensures that all adjacent and successor lanelets share two corners. Two points are aligned by
    :param max_distance: If two points that should align have a distance bigger than this value,
        they are not matched. Instead a console output is made, informing that the adjacency is likely incorrect
    :return: None
    """
    class Point:
        def __init__(self, position, weight):
            self.position = position
            self.weight = weight

    class StickyPoint:
        """Wrapper for Point, allows to apply the result of the add operation to all involved points"""
        def __init__(self, position, weight):
            self.point = Point(position, weight)  # Attribute refers either to a Point or a StickyPoint

        def leaf(self):
            if type(self.point) is StickyPoint:
                return self.point.leaf()
            else:
                return self

        def add(self, other):
            """Computes the middle point between self and other StickyPoint and assigns this new position to both of them"""
            a = self.leaf()
            b = other.leaf()
            v = a.point
            w = b.point
            new_weight = v.weight + w.weight
            new_position = (v.position * v.weight + w.position * w.weight) / new_weight

            #create new leaf that represents the middle point of the two points
            sum_point = StickyPoint(new_position, new_weight)
            a.point = sum_point
            b.point = sum_point

        def close(self, other):
            return np.linalg.norm(self.p - other.p) <= max_distance

        @property
        def p(self):
            """Shortcut for getting the point coordinates"""
            return self.leaf().point.position

    class StickyLanelet:
        def __init__(self, lanelet):
            self.lanelet = lanelet
            self.corners = [StickyPoint(lanelet.left_vertices[0], 1), StickyPoint(lanelet.right_vertices[0], 1),
                            StickyPoint(lanelet.left_vertices[-1], 1), StickyPoint(lanelet.right_vertices[-1], 1)]

        def update(self):
            """Ensures that the lanelet vertices are set according to the corners of the Sticky Lanelet."""
            self.set_left(self.lanelet.left_vertices)
            self.set_right(self.lanelet.right_vertices)

        def set_left(self, vertices):
            vertices[0] = self.corners[0].p
            vertices[-1] = self.corners[2].p
            self.lanelet.left_vertices = vertices

        def set_right(self, vertices):
            vertices[0] = self.corners[1].p
            vertices[-1] = self.corners[3].p
            self.lanelet.right_vertices = vertices

    def add_corners(a1, b1, a2, b2, a, b):
        corner_a1 = a.corners[a1]
        corner_b1 = b.corners[b1]
        corner_a2 = a.corners[a2]
        corner_b2 = b.corners[b2]

        if corner_a1.close(corner_b1) and corner_a2.close(corner_b2):
            corner_a1.add(corner_b1)
            corner_a2.add(corner_b2)
        else:
            print("Lanelet Repair: Possibly wrong adjacency between Lanelets", a.lanelet.lanelet_id, "and", b.lanelet.lanelet_id)

    network = {lanelet.lanelet_id: StickyLanelet(lanelet) for lanelet in lanelets}
    line_cache = []
    for key in network:
        a = network.get(key)
        if a.lanelet.adj_left != None:
            b = network.get(a.lanelet.adj_left)

            # The adjacencies/successor relationships are symmetric
            # Avoid adding the same corners twice:
            # if adjacent lanelets have same direction, let the rightmost lanelet handle the repair
            # else, let the first lanelet to be called handle the repair, and cache the lanelet pair
            a.update()
            b.update()
            if a.lanelet.adj_left_same_direction:
                add_corners(0, 1, 2, 3, a, b)

            else:
                # compute line only once, otherwise the second line will be different to the first
                if key < a.lanelet.adj_left:
                    # pair is ordered ascending
                    pair = (key, a.lanelet.adj_left)
                else:
                    pair = (a.lanelet.adj_left, key)

                if pair not in line_cache:
                    line_cache.append(pair)
                    add_corners(0, 2, 2, 0, a, b)

        if a.lanelet.adj_right != None:
            b = network.get(a.lanelet.adj_right)
            a.update()
            b.update()
            if not a.lanelet.adj_right_same_direction:
                if key < a.lanelet.adj_right:
                    # pair is ordered ascending
                    pair = (key, a.lanelet.adj_right)
                else:
                    pair = (a.lanelet.adj_right, key)

                if pair not in line_cache:
                    line_cache.append(pair)
                    add_corners(1, 3, 3, 1,a,b)

        for x in a.lanelet.predecessor:
            b = network.get(x)
            a.update()
            b.update()
            add_corners(0, 2, 1, 3, a, b)

    for key in network:
        network.get(key).update()

def douglas_peucker_multiline(lines, epsilon):
    """Douglas Peucker for multiple, similar polylines(lists of points) with same length
    Returns coloumns TODO:explain what that means
    """

    # find point with maximum distance
    d_max = 0
    index = 0
    results = []
    for i in range(1, len(lines[0])):
        d = 0
        for polyline in lines:
            d += line_segment_nearest_point(polyline[0], polyline[-1], polyline[i])[1]

        if d > d_max:
            index = i
            d_max = d

    # simplify polyline if maximum distance is bigger than epsilon
    if d_max >= epsilon:
        polylines_1 = [polyline[0:index] for polyline in lines]
        polylines_2 = [polyline[index:-1] for polyline in lines]
        results_1 = douglas_peucker_multiline(polylines_1, epsilon)
        results_2 = douglas_peucker_multiline(polylines_2, epsilon)
        # store results
        results.extend(results_1)
        results.extend(results_2)
    else:
        results_1 = [polyline[0] for polyline in lines]
        results_2 = [polyline[-1] for polyline in lines]
        results.append(results_1)
        results.append(results_2)
    return results



def project_adjacent_borders(lanelet_network, dp_epsilon, resample_step):
    """ """
    lanelets = lanelet_network.lanelets
    lanelet_id_list = list(map(lambda l: l.lanelet_id, lanelets))

    def project(a, b):
        """returns projection of the points of polyline a onto polyline b"""
        projection = []
        def non_duplicate_polyline(polyline):
            return np.array([polyline[0]] + [q for p, q in zip(polyline, polyline[1:]) if not np.equal(p, q).all()])

        non_duplicate_b = non_duplicate_polyline(b)
        for p in a:
            nearest_point, distance, line_distance, index = polyline_nearest_point(non_duplicate_b, p)
            projection.append(nearest_point)
        #Keep first and last points
        projection[0] = b[0]
        projection[-1] = b[-1]

        return projection


    def project_lanelet(l, direction):
        """performs projection for lanelet"""
        if direction:
            l.left_vertices = project(l.right_vertices, l.left_vertices)
            return l.left_vertices
        else:
            l.right_vertices = project(l.left_vertices, l.right_vertices)
            return np.flipud(l.right_vertices)



    while lanelet_id_list != []:
        # search for the leftmost and rightmost lanelets adjacent to the current one
        id = lanelet_id_list.pop()
        current = lanelet_network.find_lanelet_by_id(id)
        direction = True

        # ____________________ go to the rightmost lanelet _______________________
        while (direction and current.adj_right != None) or (not direction and current.adj_left != None):
            if (direction):
                next_id = current.adj_right
                direction = (direction == current.adj_right_same_direction)
            else:
                next_id = current.adj_left
                direction = (direction == current.adj_left_same_direction)
            current = lanelet_network.find_lanelet_by_id(next_id)

        # Rightmost lanelets
        rightmost = current
        rightmost_direction = direction
        lines = []
        if direction:
            last_vertex = current.right_vertices[-1]
            current.right_vertices = resample_polyline(current.right_vertices, resample_step)
            current.right_vertices[-1] = last_vertex #ensure that last point remains the same
            lines.append(current.right_vertices)
        else:
            last_vertex = current.left_vertices[-1]
            current.left_vertices = resample_polyline(current.left_vertices, resample_step)
            current.left_vertices[-1] = last_vertex  # ensure that last point remains the same
            lines.append(np.flipud(current.left_vertices))

        lines.append(project_lanelet(current, direction))



        # ___________________ expand left and collect all polylines____________________________
        # expand left while there is a lanelet on the left, looking from the direction of the first lanelet
        while (direction and current.adj_left != None) or (not direction and current.adj_right != None):
            if direction:
                next_id = current.adj_left
                next_direction = (direction == current.adj_left_same_direction)  # flip of direction
            else:
                next_id = current.adj_right
                next_direction = (direction == current.adj_right_same_direction)

            next = lanelet_network.find_lanelet_by_id(next_id)
            if next_id in lanelet_id_list:
                lanelet_id_list.remove(next_id)


            #assign the shared line between the current and next lanelet
            if direction:
                shared_line = current.left_vertices
            else:
                shared_line = np.flipud(current.right_vertices)
            if next_direction:
                next.right_vertices = shared_line
            else:
                next.left_vertices = np.flipud(shared_line)

            direction = next_direction
            current = next

            lines.append(project_lanelet(current, direction))

        # refit lines so that all polylines are duplicate free but still have the same amount of points
        # necessary points contains information for each point whether to keep it
        # meanings: True: keep point, False: remove point
        assert (all(len(a) == len(b) for a, b in zip(lines, lines[1:])))
        necessary_points = [True for point in lines[0]]

        for polyline in lines:
            numerated_non_duplicate = [0] + [i + 1 for i, (p, q) in enumerate(zip(polyline, polyline[1:]))
                                             if not np.equal(p, q).all()]

            numerated_non_duplicate[-1] = len(necessary_points) - 1
            for i in range(0,len(necessary_points) - 1):
                if i not in numerated_non_duplicate:
                    necessary_points[i] = False


        for i, polyline in enumerate(lines):
            lines[i] = [point for j,point in enumerate(polyline) if necessary_points[j]]

        assert (all(len(a) == len(b) for a, b in zip(lines, lines[1:])))
        #Reduce polylines
        for i, polyline in enumerate(lines):
            lines[i] = [p for p in polyline] #convert to list if polyline is an np array

        coloumns = douglas_peucker_multiline(lines, dp_epsilon*len(lines)) #TODO:explain coloumns to lines
        new_lines = [[] for polyline in lines]
        for c in coloumns:
            for i, point in enumerate(c):
                new_lines[i].append(point)

        #TODO: investigate why it is necessary to set the end point, douglas peucker should keep the last point
        for i, new_p in enumerate(new_lines):
            new_p[-1] = lines[i][-1]

        lines = new_lines
        assert (all(all(p[0] == new_p[0]) for p, new_p in zip(lines,new_lines)))
        assert (all(all(p[-1] == new_p[-1]) for p, new_p in zip(lines, new_lines)))
        lines = [np.array(polyline) for polyline in lines]
        # ___________________ expand left and assign polylines____________________________
        def assign_poylines(lanelet, direction):
            if direction:
                lanelet.right_vertices = lines.pop(0)
                lanelet.left_vertices = lines[0]
            else:
                lanelet.left_vertices = np.flipud(lines.pop(0))
                lanelet.right_vertices = np.flipud(lines[0])

        current = rightmost
        direction = rightmost_direction


        assign_poylines(current, direction)
        while (direction and current.adj_left != None) or (not direction and current.adj_right != None):
            if direction:
                next_id = current.adj_left
                direction = (direction == current.adj_left_same_direction)  # flip of direction
            else:
                next_id = current.adj_right
                direction = (direction == current.adj_right_same_direction)

            current = lanelet_network.find_lanelet_by_id(next_id)
            assign_poylines(current, direction)

def repair(lanelet_network, dp_epsilon=0.1, max_distance=0.5, resample_step=0.5):
    """
    Changes the lanelet borders so that:
     - each pair of adjacent lanelets shares an identical line between them
     - each pair of successing lanelets shares a start/end line
    :param lanelet_network: Lanelet Network that is copied and repaired
    :param dp_epsilon: Douglas Peucker epsilon value for line reduction
    :param max_distance: Maximum distance that is tolerated between aligning corners
    :param resample_step: Distance between two points on the resampled polylines
    :return: Lanelet_Network with repaired lanelet borders.
        Especially, adjacent lanelets share the same np array between them.
        Therefore, modification to a lanelets left_vertices or right_vertices might impact adjacent lanelets
    """
    import copy
    repaired_network = copy.deepcopy(lanelet_network)
    lanelets = repaired_network.lanelets

    align_corners(lanelets, max_distance)
    project_adjacent_borders(repaired_network, dp_epsilon, resample_step)

    for l in lanelets:
        left = l.left_vertices
        right = l.right_vertices
        assert type(left) == type(np.array([])) == type(right)
        assert len(left) == len(right), \
            "\nLanelet:" + l.lanelet_id + "doesn't have same length borders after repair"
        assert len(left) > 1
        l.center_vertices = (left + right) / 2





    #Draw lanelets and their border vertices
    draw_params = {'lanelet': {'left_bound_color': '#111111',
                               'right_bound_color': '#555555',
                               'center_bound_color': '#dddddd',
                               'draw_left_bound': True,
                               'draw_right_bound': True,
                               'draw_center_bound': True,
                               'draw_border_vertices': True,
                               'draw_start_and_direction': False,
                               'show_label': False}}
    #draw_params = {'no_parent': {'scenario': draw_params}}
    #from fvks.visualization.draw_dispatch import draw_object
    #import matplotlib.pyplot as plt
    #draw_object(repaired_network, draw_params=draw_params)
    #plt.gca().set_aspect('equal')
    #plt.show()

    return repaired_network
