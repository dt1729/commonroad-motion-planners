__author__ = "Anna-Katharina Rettinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["CoPlan"]
__version__ = "0.1"
__maintainer__ = "Anna-Katharina Rettinger"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Beta"

import copy
from typing import *

import construction
import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import Polygon, ShapeGroup, Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.scenario import Scenario
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad.planning.planning_problem import PlanningProblem
from automata.HelperFunctions import add_initial_state_to_automata
from automata.helper_tree_search import MotionPrimitiveStatus, initial_visualization, \
    update_visualization
from automata.MotionPrimitive import MotionPrimitive
from automata.MotionAutomata import MotionAutomata
from config import Default
from abc import abstractmethod, ABC
from automata.queue import FIFOQueue, LIFOQueue, PriorityQueue
from automata.node import Node, PrioNode


class MotionPlanner(ABC):
    def __init__(self, scenario: Scenario, planningProblem: PlanningProblem, automaton: MotionAutomata, config=Default):
        # store input parameters
        self.scenario: Scenario = scenario
        self.planningProblem: PlanningProblem = planningProblem
        automaton, initial_motion_primitive = add_initial_state_to_automata(automaton, planningProblem)
        self.automaton: MotionAutomata = automaton
        self.egoShape: Rectangle = automaton.egoShape

        self.initial_state: State = self.planningProblem.initial_state
        self.initial_mp = initial_motion_primitive

        # visualization params
        self.config = config

        # remove unneeded attributes of initial state
        if hasattr(self.initial_state, 'yaw_rate'):
            del self.initial_state.yaw_rate

        if hasattr(self.initial_state, 'slip_angle'):
            del self.initial_state.slip_angle

        # set specifications from given goal state
        if hasattr(self.planningProblem.goal.state_list[0], 'time_step'):
            self.desired_time = self.planningProblem.goal.state_list[0].time_step
        else:
            self.desired_time = Interval(0, np.inf)

        # construct commonroad boundaries and collision checker object
        build = ['section_triangles', 'triangulation']
        boundary = construction.construct(self.scenario, build, [], [])
        road_boundary_shape_list = list()
        initial_state = None
        for r in boundary['triangulation'].unpack():
            initial_state = State(position=np.array([0, 0]), orientation=0.0, time_step=0)
            p = Polygon(np.array(r.vertices()))
            road_boundary_shape_list.append(p)
        road_bound = StaticObstacle(obstacle_id=scenario.generate_object_id(), obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                    obstacle_shape=ShapeGroup(road_boundary_shape_list), initial_state=initial_state)
        self.collisionChecker = create_collision_checker(self.scenario)
        self.collisionChecker.add_collision_object(create_collision_object(road_bound))
    path_fig: Union[str, None]

    def reached_goal(self, path: List[State]) -> bool:
        """
        Goal-test every state of the path and returns true if one of the state satisfies all conditions for the goal
        region: position, orientation, velocity, time.

        :param path: the path to be goal-tested

        """
        for i in range(len(path)):
            if self.planningProblem.goal.is_reached(path[i]):
                return True
        return False

    def remove_states_behind_goal(self, path: List[List[State]]) -> List[List[State]]:
        """
        Removes all states that are behind the state which satisfies goal state conditions and returns the pruned path.

        :param path: the path to be pruned
        """

        for i in range(len(path[-1])):
            if self.planningProblem.goal.is_reached(path[-1][i]):
                for j in range(i + 1, len(path[-1])):
                    path[-1].pop()
                return path
        return path

    def check_collision_free(self, path: List[State]) -> bool:
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise.

        :param path: The path you want to check
        """
        trajectory = Trajectory(path[0].time_step, path)

        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self.egoShape)

        # create a collision object using the trajectory prediction of the ego vehicle
        co = create_collision_object(traj_pred)

        # check collision for motion primitive
        if self.collisionChecker.collide(co):
            return False
        return True

    @staticmethod
    def translate_primitive_to_current_state(primitive: MotionPrimitive, path_current: List[State]) -> List[State]:
        """
        Uses the trajectory defined in the given primitive, translates it towards the last state of current path and
        returns the list of new path.
        In the newly appended part (created through translation of the primitive) of the path, the position,
        orientation and time step are changed, but the velocity is not changed.
        Attention: The input primitive itself will not be changed after this operation.

        :param primitive: the primitive to be translated
        :param path_current: the path whose last state is the goal state for the translation
        """
        return primitive.appendTrajectoryToState(path_current[-1])

    @staticmethod
    def append_path(path_current: List[State], newPath: List[State]) -> List[State]:
        """
        Appends a new path to the current path and returns the whole path.

        :param path_current: current path which is to be extended
        :param newPath: new path which is going to be added to the current path
        """
        path = path_current[:]
        path.extend(newPath)
        return path

    def plot_solution(self, solution_path, node_status, all_node_status):
        node_status = update_visualization(mp=solution_path[-1], status=MotionPrimitiveStatus.SOLUTION,
                                           node_status=node_status, path_fig=self.path_fig, config=self.config,
                                           count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))
        for prim in solution_path:
            node_status = update_visualization(mp=prim, status=MotionPrimitiveStatus.SOLUTION,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))
        return all_node_status

    def plot_collision_primitives(self, current_node: Union[Node, PrioNode], path_translated, node_status,
                                  all_node_status):
        node_status = update_visualization(mp=[current_node.path[-1][-1]] + path_translated,
                                           status=MotionPrimitiveStatus.INVALID, node_status=node_status,
                                           path_fig=self.path_fig, config=self.config, count=len(all_node_status))
        if self.config.PLOT_COLLISION_STEPS:
            all_node_status.append(copy.copy(node_status))

        return all_node_status, node_status

    @abstractmethod
    def search_alg(self):
        """
        In inherited classes, the search is implemented.
        :return:
        """
        pass


class IterativeNoCostSearchAlgorithm(MotionPlanner, ABC):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(IterativeNoCostSearchAlgorithm, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                             automaton=automaton, config=config)

    frontier: Union[FIFOQueue, LIFOQueue]
    path_fig: Union[str, None]

    def search_alg(self):
        """
        Implementation of BFS/DFS (tree search) using a FIFO/LIFO queue
        the frontier is defined in inherited classes
        """
        # for visualization in jupyter notebook
        all_node_status = []
        node_status: Dict[int, Tuple] = {}

        # First node
        initial_node = Node(path=[[self.initial_state]], primitives=[self.initial_mp], tree_depth=0)
        initial_visualization(self.scenario, self.initial_state, self.egoShape, self.planningProblem, self.config,
                              self.path_fig)

        # check if we already reached the goal state
        if self.reached_goal(initial_node.path[-1]):
            return self.remove_states_behind_goal(initial_node.path), initial_node.primitives, all_node_status

        # add current node to the frontier
        self.frontier.insert(initial_node)

        node_status = update_visualization(mp=initial_node.path[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                           node_status=node_status, path_fig=self.path_fig, config=self.config,
                                           count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))

        while not self.frontier.empty():
            # Pop the shallowest node
            current_node: Node = self.frontier.pop()

            node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

            # Check all possible successor primitives(i.e., actions) for current node
            for succ_primitive in current_node.get_successors():

                # translate/rotate motion primitive to current position
                current_primitive_list = copy.copy(current_node.primitives)
                path_translated = self.translate_primitive_to_current_state(succ_primitive, current_node.path[-1])

                # check for collision, if is not collision free it is skipped
                if not self.check_collision_free(path_translated):
                    all_node_status, node_status = self.plot_collision_primitives(current_node=current_node,
                                                                                  path_translated=path_translated,
                                                                                  node_status=node_status,
                                                                                  all_node_status=all_node_status)
                    continue

                current_primitive_list.append(succ_primitive)

                # Goal test
                if self.reached_goal(path_translated):
                    path_new = current_node.path + [[current_node.path[-1][-1]] + path_translated]
                    solution_path = self.remove_states_behind_goal(path_new)
                    all_node_status = self.plot_solution(solution_path=solution_path, node_status=node_status,
                                                         all_node_status=all_node_status)
                    # return solution
                    return self.remove_states_behind_goal(path_new), current_primitive_list, all_node_status

                # Inserting the child to the frontier:
                path_new = current_node.path + [[current_node.path[-1][-1]] + path_translated]
                child = Node(path=path_new, primitives=current_primitive_list, tree_depth=current_node.tree_depth + 1)
                node_status = update_visualization(mp=path_new[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                                   node_status=node_status, path_fig=self.path_fig, config=self.config,
                                                   count=len(all_node_status))
                all_node_status.append(copy.copy(node_status))
                self.frontier.insert(child)

                if path_translated[-1].time_step > self.desired_time.end:
                    # prevent algorithm from running infinitely long for DFS (search failed)
                    print('Algorithm is in infinite loop and will not find a solution')
                    return None, None, all_node_status

            node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.EXPLORED,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

        return None, None, all_node_status


class BreadthFirstSearch(IterativeNoCostSearchAlgorithm):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(BreadthFirstSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                 automaton=automaton, config=config)
        self.frontier = FIFOQueue()
        if config.SAVE_FIG:
            self.path_fig = './figures/bfs/'
        else:
            self.path_fig = None


class DepthFirstSearch(IterativeNoCostSearchAlgorithm):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(DepthFirstSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                               automaton=automaton, config=config)
        self.frontier = LIFOQueue()
        if config.SAVE_FIG:
            self.path_fig = './figures/dfs/'
        else:
            self.path_fig = None


class DepthLimitedSearch(MotionPlanner):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(DepthLimitedSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                 automaton=automaton, config=config)

        if config.SAVE_FIG:
            self.path_fig = './figures/dls/'
        else:
            self.path_fig = None

    def search_alg(self, limit=7):
        """
        Depth-Limited Search implementation
        """
        # for visualization in jupyter notebook
        all_node_status = []
        node_status: Dict[int, Tuple] = {}

        # First node
        initial_node = Node(path=[[self.initial_state]], primitives=[self.initial_mp], tree_depth=0)
        initial_visualization(self.scenario, self.initial_state, self.egoShape, self.planningProblem, self.config,
                              self.path_fig)

        node_status = update_visualization(mp=initial_node.path[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                           node_status=node_status, path_fig=self.path_fig, config=self.config,
                                           count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))

        result = self.recursive_dls(all_node_status, node_status, initial_node, limit)
        if result is None:
            return None, None, all_node_status
        path = result[0]
        list_primitives = result[1]
        all_node_status = result[2]

        return path, list_primitives, all_node_status

    def recursive_dls(self, all_node_status: List[Dict[int, Tuple]], node_status: Dict[int, Tuple], current_node: Node,
                      limit: int):
        """
        Recursive implementation of Depth-Limited Search.
        @param all_node_status: List which stores the changes in the node _status for plotting
        @param node_status: Dict with status of each node
        @param current_node: consists of path, list of primitives and the current tree_depth
        @param limit: current Limit
        @return:
        """
        node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                           node_status=node_status, path_fig=self.path_fig, config=self.config,
                                           count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))

        # Goal test
        if self.reached_goal(current_node.path[-1]):
            solution_path = self.remove_states_behind_goal(current_node.path)
            all_node_status = self.plot_solution(solution_path=solution_path, node_status=node_status,
                                                 all_node_status=all_node_status)
            # return solution
            return solution_path, current_node.primitives, all_node_status

        elif limit == 0:
            node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.EXPLORED,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))
            return 'cutoff'

        else:
            cutoff_occurred = False

        for succ_primitive in reversed(current_node.get_successors()):
            # translate/rotate motion primitive to current position
            current_primitive_list = copy.copy(current_node.primitives)
            path_translated = self.translate_primitive_to_current_state(succ_primitive, current_node.path[-1])

            # check for collision, if is not collision free it is skipped
            if not self.check_collision_free(path_translated):
                all_node_status, node_status = self.plot_collision_primitives(current_node=current_node,
                                                                              path_translated=path_translated,
                                                                              node_status=node_status,
                                                                              all_node_status=all_node_status)
                continue

            # Continue search with child node
            current_primitive_list.append(succ_primitive)
            path_new = current_node.path + [[current_node.path[-1][-1]] + path_translated]
            child = Node(path=path_new, primitives=current_primitive_list, tree_depth=current_node.tree_depth + 1)

            node_status = update_visualization(mp=path_new[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

            result = self.recursive_dls(all_node_status=all_node_status, node_status=node_status, current_node=child,
                                        limit=limit - 1)

            if result == 'cutoff':
                cutoff_occurred = True

            elif result is not None:
                return result

        if cutoff_occurred:
            node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.EXPLORED,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

        return 'cutoff' if cutoff_occurred else None


class BestFirstSearchAlgorithm(MotionPlanner, ABC):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(BestFirstSearchAlgorithm, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                       automaton=automaton, config=config)
        self.frontier = PriorityQueue()

        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            self.desired_position = self.calc_goal_interval(self.planningProblem.goal.state_list[0].position.vertices)
        else:
            self.desired_position = None

    path_fig: Union[str, None]

    @abstractmethod
    def evaluation_function(self, current_node: PrioNode):
        """
        Function in inherited classes, evaluates f(n).
        @param current_node:
        @return: cost
        """
        return

    @staticmethod
    def calc_goal_interval(vertices):
        """
        Calculate the maximum Intervals of the goal position given as vertices.
        @param: vertices: vertices which describe the goal position.
        """
        min_x = np.inf
        max_x = -np.inf

        min_y = np.inf
        max_y = -np.inf
        for vertex in vertices:
            if vertex[0] < min_x:
                min_x = vertex[0]
            if vertex[0] > max_x:
                max_x = vertex[0]
            if vertex[1] < min_y:
                min_y = vertex[1]
            if vertex[1] > max_y:
                max_y = vertex[1]
        return [Interval(start=min_x, end=max_x), Interval(start=min_y, end=max_y)]

    def calc_euclidean_distance(self, current_node: PrioNode) -> float:
        """
        Calculates the euclidean distance to the desired goal position.
        @param current_node:
        @return:
        """
        if self.desired_position[0].contains(current_node.path[-1][-1].position[0]):
            delta_x = 0.0
        else:
            delta_x = min([abs(self.desired_position[0].start - current_node.path[-1][-1].position[0]),
                           abs(self.desired_position[0].end - current_node.path[-1][-1].position[0])])
        if self.desired_position[1].contains(current_node.path[-1][-1].position[1]):
            delta_y = 0
        else:
            delta_y = min([abs(self.desired_position[1].start - current_node.path[-1][-1].position[1]),
                           abs(self.desired_position[1].end - current_node.path[-1][-1].position[1])])

        return np.sqrt(delta_x**2 + delta_y**2)

    def heuristic_function(self, current_node: PrioNode) -> float:
        """
        This heuristic function estimates the time to reach the goal
        @param current_node:
        @return:
        """
        if self.reached_goal(current_node.path[-1]):
            return 0.0
        if self.desired_position is None:
            return self.desired_time.start - current_node.path[-1][-1].time_step
        return self.calc_euclidean_distance(current_node=current_node)/current_node.path[-1][-1].velocity

    def search_alg(self):
        """
        Implementation of BestFirstSearch (tree search) using a Priority queue
        The evaluation function is implemented in the inherited classes
        """
        # for visualization in jupyter notebook
        all_node_status = []
        node_status: Dict[int, Tuple] = {}

        # First node
        initial_node = PrioNode(path=[[self.initial_state]], primitives=[self.initial_mp], tree_depth=0, current_cost=0)
        initial_visualization(self.scenario, self.initial_state, self.egoShape, self.planningProblem, self.config,
                              self.path_fig)

        # add current node (i.e., current path and primitives) to the frontier
        f = self.evaluation_function(initial_node)
        self.frontier.insert(item=initial_node, priority=f)

        node_status = update_visualization(mp=initial_node.path[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                           node_status=node_status, path_fig=self.path_fig, config=self.config,
                                           count=len(all_node_status))
        all_node_status.append(copy.copy(node_status))

        while not self.frontier.empty():
            # Pop the shallowest node
            current_node: PrioNode = self.frontier.pop()

            node_status = update_visualization(mp=current_node.path[-1],
                                               status=MotionPrimitiveStatus.CURRENTLY_EXPLORED, node_status=node_status,
                                               path_fig=self.path_fig, config=self.config, count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

            # Goal test
            if self.reached_goal(current_node.path[-1]):
                solution_path = self.remove_states_behind_goal(current_node.path)
                all_node_status = self.plot_solution(solution_path=solution_path, node_status=node_status,
                                                     all_node_status=all_node_status)
                # return solution
                return solution_path, current_node.primitives, all_node_status

            # Check all possible successor primitives(i.e., actions) for current node
            for succ_primitive in current_node.get_successors():

                # translate/rotate motion primitive to current position
                current_primitive_list = copy.copy(current_node.primitives)
                path_translated = self.translate_primitive_to_current_state(succ_primitive, current_node.path[-1])
                # check for collision, if is not collision free it is skipped
                if not self.check_collision_free(path_translated):
                    all_node_status, node_status = self.plot_collision_primitives(current_node=current_node,
                                                                                  path_translated=path_translated,
                                                                                  node_status=node_status,
                                                                                  all_node_status=all_node_status)
                    continue

                current_primitive_list.append(succ_primitive)

                path_new = current_node.path + [[current_node.path[-1][-1]] + path_translated]
                child_node = PrioNode(path=path_new, primitives=current_primitive_list,
                                      tree_depth=current_node.tree_depth + 1, current_cost=current_node.current_cost)
                f = self.evaluation_function(current_node=child_node)

                # Inserting the child to the frontier:
                node_status = update_visualization(mp=child_node.path[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                                   node_status=node_status, path_fig=self.path_fig, config=self.config,
                                                   count=len(all_node_status))
                all_node_status.append(copy.copy(node_status))
                self.frontier.insert(item=child_node, priority=f)

            node_status = update_visualization(mp=current_node.path[-1], status=MotionPrimitiveStatus.EXPLORED,
                                               node_status=node_status, path_fig=self.path_fig, config=self.config,
                                               count=len(all_node_status))
            all_node_status.append(copy.copy(node_status))

        return None, None, all_node_status


class UniformCostSearch(BestFirstSearchAlgorithm):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(UniformCostSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                automaton=automaton, config=config)

        if config.SAVE_FIG:
            self.path_fig = './figures/ucs/'
        else:
            self.path_fig = None

    def evaluation_function(self, current_node: PrioNode) -> float:
        """
        Evaluation function of UCS is f(n) = g(n)
        """

        # calculate g(n)
        if self.reached_goal(current_node.path[-1]):
            current_node.path = self.remove_states_behind_goal(current_node.path)
        current_node.current_cost += (len(current_node.path[-1])-1) * self.scenario.dt

        return current_node.current_cost


class GreedyBestFirstSearch(BestFirstSearchAlgorithm):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(GreedyBestFirstSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                                    automaton=automaton, config=config)

        if config.SAVE_FIG:
            self.path_fig = './figures/gbfs/'
        else:
            self.path_fig = None

    def evaluation_function(self, current_node: PrioNode) -> float:
        """
        Evaluation function of GBFS is f(n) = h(n)
        """

        current_node.current_cost = self.heuristic_function(current_node=current_node)
        return current_node.current_cost


class AStarSearch(BestFirstSearchAlgorithm):
    def __init__(self, scenario, planningProblem, automaton, config=Default):
        super(AStarSearch, self).__init__(scenario=scenario, planningProblem=planningProblem,
                                          automaton=automaton, config=config)

        if config.SAVE_FIG:
            self.path_fig = './figures/astar/'
        else:
            self.path_fig = None

    def evaluation_function(self, current_node: PrioNode) -> float:
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """

        if self.reached_goal(current_node.path[-1]):
            current_node.path = self.remove_states_behind_goal(current_node.path)
        # calculate g(n)
        current_node.current_cost += (len(current_node.path[-1])-1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return current_node.current_cost + self.heuristic_function(current_node=current_node)
