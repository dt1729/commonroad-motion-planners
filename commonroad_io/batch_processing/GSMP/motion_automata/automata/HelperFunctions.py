import sys
import os

path_commonroad_search = "../../"
sys.path.append(os.path.join(path_commonroad_search, "GSMP/tools/"))
sys.path.append(os.path.join(path_commonroad_search, "GSMP/tools/commonroad-collision-checker"))
sys.path.append(os.path.join(path_commonroad_search, "GSMP/tools/commonroad-road-boundary"))
sys.path.append(os.path.join(path_commonroad_search, "GSMP/motion_automata/vehicle_model"))

import time
from multiprocessing import Manager, Process
import enum
import ipywidgets as widgets

import numpy as np
from math import atan2

import matplotlib.pyplot as plt
from IPython import display
from ipywidgets import widgets
#from IPython.display import display

# import CommonRoad-io modules
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.scenario.scenario import Scenario
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import StaticObstacle, ObstacleRole, ObstacleType, DynamicObstacle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_cc.visualization.draw_dispatch import draw_object as draw_it
from commonroad.common.solution_writer import CommonRoadSolutionWriter, VehicleType, CostFunction

# import Motion Automata modules
from automata.MotionAutomata import MotionAutomata
from automata.MotionPrimitive import MotionPrimitive
from automata.States import FinalState
from typing import List, Dict, Tuple


# ================================== scenario ================================== #
def load_scenario(scenario_path):
    # open and read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()

    return scenario, planning_problem_set


# ================================== automata ================================== #
def generate_automata(veh_type_id, mp_file=None, search_tutorial=False):
    """
    1. We first create a Motion Automata object, then load pre-computed motion primitives into it. 
    Note that every motion primitive is a short trajectory with varied initial states that lead to different final states.

    2. We create a connectivity list for every pritimive in the Motion Automata, 
    which contains all connectable primitives from the focused primitive. 
    If the velocity and steering angel of the start state of primitive B match 
    those of the final state of primitive A, we say B is connectable from A.

    :param veh_type_id: id for the vehicle model in the motion primitives
    :param mp_file: optional parameter for specifying the path to a motion primitives xml file
    :param search_tutorial: boolean value to specify if the search is used for a tutorial ->
    motion primitives are sorted by their y-coordinate in the final state
    """

    # the primitives vary for different vehicle models 1, 2 and 3.
    assert veh_type_id in (1, 2, 3), "Input vehicle type id is not valid! Must be either 1, 2 or 3."

    # step 1
    automata = MotionAutomata(veh_type_id)

    print("Reading motion primitives...")
    try:
        prefix = '../../GSMP/motion_automata/motion_primitives/'
        if mp_file is not None:
            automata.readFromXML(prefix + mp_file)
        else:
            if veh_type_id == 1:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-0.91_0.91_SAstep_0.23_T_0.5_Model_FORD_ESCORT.xml')
            elif veh_type_id == 2:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-1.066_1.066_SAstep_0.27_T_0.5_Model_BMW320i.xml')
            elif veh_type_id == 3:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-1.023_1.023_SAstep_0.26_T_0.5_Model_VW_VANAGON.xml')

    except Exception:
        prefix = '../../../GSMP/motion_automata/motion_primitives/'

        if mp_file is not None:
            automata.readFromXML(prefix + mp_file)
        else:
            if veh_type_id == 1:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-0.91_0.91_SAstep_0.23_T_0.5_Model_FORD_ESCORT.xml')
            elif veh_type_id == 2:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-1.066_1.066_SAstep_0.27_T_0.5_Model_BMW320i.xml')
            elif veh_type_id == 3:
                automata.readFromXML(
                    prefix + 'V_0.0_20.0_Vstep_1.0_SA_-1.023_1.023_SAstep_0.26_T_0.5_Model_VW_VANAGON.xml')

    # additional step for search examples -> sorting motion primitives by max. y-coordinate
    if search_tutorial:
        automata.sort_primitives()

    # step 2
    automata.createConnectivityLists()

    # assign vehicle type id to all primitives
    automata.setVehicleTypeIdPrimitives()

    print("Automata created.")
    print('Number of loaded primitives: ' + str(len(automata.Primitives)))

    return automata


def add_initial_state_to_automata(automata, planning_problem):
    # set initial steering angle to zero
    planning_problem.initial_state.steering_angle = 0.0

    # the initial velocity of the planning problem may be any value, we need to obtain the closest velocity to it
    # from StartStates of the primitives in order to get the feasible successors of the planning problem
    velocity_closest = automata.getClosestStartVelocity(planning_problem.initial_state.velocity)
    planning_problem.initial_state.velocity = velocity_closest

    initial_state = planning_problem.initial_state

    # turn initial state into a motion primitive to check for connectivity to subsequent motion primitives
    final_state_primitive = FinalState(x=initial_state.position[0],
                                       y=initial_state.position[1],
                                       steering_angle=initial_state.steering_angle,
                                       velocity=initial_state.velocity,
                                       orientation=initial_state.orientation,
                                       time_step=initial_state.time_step)

    initial_motion_primitive = MotionPrimitive(startState=None, finalState=final_state_primitive, timeStepSize=0,
                                               trajectory=None)

    # create connectivity list for this imaginary motion primitive
    automata.createConnectivityListPrimitive(initial_motion_primitive)

    return automata, initial_motion_primitive


# ================================== search for solution ================================== #
"""
1. Searching for a possible solution could be time consuming, and also we would like to visualize the intermediate states during the search.
Thus, we create a separate Process to handle the search, and visualize the intermediate states via Main Process. 
More information on python Prcess could be found at [1](https://docs.python.org/3/library/multiprocessing.html) and
[2](http://www.blog.pythonlibrary.org/2016/08/02/python-201-a-multiprocessing-tutorial/). 

2. We need multiprocessing.Manager.Value() to help us share variables between different Processes.

3. To execute the search, we need to define an individual wrapper function and set it as the target of the new Process.
"""


def execute_search(motion_planner, initial_motion_primitive, path_shared, dict_status_shared, max_tree_depth):
    """
    definition of the wrapper function for search
    calls search algorithm and update the status dictionary for visualization in search algorithm
    note that dict_status_shared is being shared between difference Processes
    """
    result = motion_planner.search_alg(initial_motion_primitive.Successors, max_tree_depth, dict_status_shared)

    # result is in form of (final path, used_primitives)
    if result is None:
        # no path found
        return
    else:
        display.clear_output(wait=True)
        path_shared.value = result[0]

    if len(result[1]) > 0:
        print("Found primitives")

    # for primitive in result[1]:
    #     print('\t', primitive)


def start_search(scenario, planning_problem, automata, motion_planner, initial_motion_primitive,
                 flag_plot_intermediate_results=True, flag_plot_planning_problem=True):
    # create Manager object to manage shared variables between different Processes
    process_manager = Manager()

    # create shared variables between Processes
    path_shared = process_manager.Value('path_shared', None)
    dict_status_shared = process_manager.Value('dict_status_shared', None)

    # initialize status with a dictionary.
    # IMPORTANT NOTE: if the shared variable contains complex object types (list, dict, ...), to change its value,
    # we need to do it via another object (here dict_status) and pass it to the shared variable at the end
    dict_status = {'cost_current': 0, 'path_current': []}
    dict_status_shared.value = dict_status

    # maximum number of concatenated primitives
    max_tree_depth = 100

    # create and start the Process for search
    process_search = Process(target=execute_search,
                             args=(motion_planner,
                                   initial_motion_primitive,
                                   path_shared,
                                   dict_status_shared,
                                   max_tree_depth))
    process_search.start()

    if flag_plot_intermediate_results:
        # create a figure, plot the scenario and planning problem
        fig = plt.figure(figsize=(9, 9))
        plt.clf()
        draw_object(scenario)
        if flag_plot_planning_problem:
            draw_object(planning_problem)
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.show()

    refresh_rate_plot = 5.0

    while True:
        time.sleep(1 / refresh_rate_plot)
        # break the loop if process_search is not alive anymore
        if not process_search.is_alive():
            break

        # print current status
        string_status = "Cost so far:", round(dict_status_shared.value['cost_current'], 2)
        string_status += "Time step:", round(dict_status_shared.value['path_current'][-1].time_step, 2)
        string_status += "Orientation:", round(dict_status_shared.value['path_current'][-1].orientation, 2)
        string_status += "Velocity:", round(dict_status_shared.value['path_current'][-1].velocity, 2)
        print(string_status, end='\r')

        # if there is a path to be visualized
        if len(dict_status_shared.value['path_current']) > 0:
            if flag_plot_intermediate_results:
                plt.clf()
                draw_object(scenario)
                if flag_plot_planning_problem:
                    draw_object(planning_problem)
                plt.gca().set_aspect('equal')
                plt.gca().set_axis_off()
                plt.margins(0, 0)
                plt.show()

            # create a Trajectory from the current states of the path
            # Trajectory is essentially a list of states
            trajectory = Trajectory(initial_time_step=0, state_list=dict_status_shared.value['path_current'])

            # create an occupancy Prediction via the generated trajectory and shape of ego vehicle
            # the Prediction includes a Trajectory and Shape of the object
            prediction = TrajectoryPrediction(trajectory=trajectory, shape=automata.egoShape)

            # create a collision object from prediction
            # here it is used for visualization purpose
            collision_object = create_collision_object(prediction)

            # draw this collision object
            draw_it(collision_object, draw_params={'collision': {'facecolor': 'magenta'}})

        # visualize current trajectory
        if flag_plot_intermediate_results is True:
            fig.canvas.draw()

        # wait till process_search terminates
    process_search.join()
    time.sleep(0.5)
    print("Search finished.")

    if path_shared.value is not None and len(path_shared.value) > 1:
        print("Solution successfully found :)")
    else:
        print("Finding solution failed :(")

    return path_shared.value, dict_status_shared.value



