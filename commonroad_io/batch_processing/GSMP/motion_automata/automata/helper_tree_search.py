__author__ = "Anna-Katharina Rettinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["CoPlan"]
__version__ = "0.1"
__maintainer__ = "Anna-Katharina Rettinger"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Beta"

import enum
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython import display
from ipywidgets import widgets
from IPython.display import display

# import CommonRoad-io modules
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import Scenario
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle

# import Motion Automaton modules
from automata.MotionPrimitive import MotionPrimitive
from typing import List, Dict, Tuple


@enum.unique
class MotionPrimitiveStatus(enum.Enum):
    IN_FRONTIER = 0
    INVALID = 1
    CURRENTLY_EXPLORED = 2
    EXPLORED = 3
    SOLUTION = 4


def plot_legend(plotting_config):
    if hasattr(plotting_config, 'LABELS'):
        node_status, labels = plotting_config.LABELS
    else:
        node_status = [status.value for status in MotionPrimitiveStatus]
        labels = ['Frontier', 'Collision', 'Currently Exploring', 'Explored', 'Final Solution']

    custom_lines = []
    for value in node_status:

        custom_lines.append(Line2D([0], [0], color=plotting_config.PLOTTING_PARAMS[value][0],
                                   linestyle=plotting_config.PLOTTING_PARAMS[value][1],
                                   linewidth=plotting_config.PLOTTING_PARAMS[value][2]))
    legend = plt.legend(handles=custom_lines, labels=labels, loc='lower left',
                        bbox_to_anchor=(0.02, 0.51), prop={'size': 18})
    legend.set_zorder(30)
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.shadow"] = True


def plot_search_scenario(scenario, initial_state, ego_shape, planning_problem, config):
    plt.figure(figsize=(22.5, 4.5))
    plt.axis('equal')
    plt.xlim(55, 97)
    plt.ylim(-2.5, 5.5)
    draw_params = {'scenario': {'lanelet': {'facecolor': '#F8F8F8'}}}
    draw_object(scenario, draw_params=draw_params)
    ego_vehicle = DynamicObstacle(obstacle_id=scenario.generate_object_id(), obstacle_type=ObstacleType.CAR,
                                  obstacle_shape=ego_shape,
                                  initial_state=initial_state)
    draw_object(ego_vehicle)
    draw_object(planning_problem)
    if config.PLOT_LEGEND:
        plot_legend(plotting_config=config)


def initial_visualization(scenario, initial_state, ego_shape, planning_problem, config, path_fig):
    if not config.JUPYTER_NOTEBOOK:
        plot_search_scenario(scenario, initial_state, ego_shape, planning_problem, config)
        if path_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.axis('off')
            try:
                plt.savefig(path_fig + 'initial_scenario.' + config.OUTPUT_FORMAT, format=config.OUTPUT_FORMAT,
                            bbox_inches='tight')
            except:
                print('Saving was not successful')
        else:
            plt.show(block=False)


def plot_state(state: State, color='red'):
    plt.plot(state.position[0], state.position[1], color=color, marker='o', markersize=6)


def plot_motion_primitive(mp: MotionPrimitive, color='red'):
    """
    Plots an object of class MotionPrimitive with marker at initial state and end state
    @param mp: object of class Motion Primitive
    @param color: color for plotting Motion Primitive
    @return:
    """
    plot_state(state=mp.trajectory.state_list[0])
    plot_state(state=mp.trajectory.state_list[-1])
    x = []
    y = []
    for state in mp.trajectory.state_list:
        x.append(state.position[0])
        y.append(state.position[1])
    plt.plot(x, y, color=color, marker="")


def plot_primitive_path(mp: List[State], status: MotionPrimitiveStatus, plotting_params):
    plt.plot(mp[-1].position[0], mp[-1].position[1], color=plotting_params[status.value][0], marker='o', markersize=8,
             zorder=27)
    x = []
    y = []
    for state in mp:
        x.append(state.position[0])
        y.append(state.position[1])
    plt.plot(x, y, color=plotting_params[status.value][0], marker="", linestyle=plotting_params[status.value][1],
             linewidth=plotting_params[status.value][2], zorder=25)


def update_visualization(mp: List[State], status: MotionPrimitiveStatus, node_status: Dict[int, Tuple],
                         path_fig, config, count):
    assert isinstance(status, MotionPrimitiveStatus), "Status in not of type MotionPrimitiveStatus."

    node_status.update({hash(mp[-1]): (mp, status)})
    # only plot if run with python script
    if not config.JUPYTER_NOTEBOOK:
        plot_primitive_path(mp=mp, status=status, plotting_params=config.PLOTTING_PARAMS)
        if path_fig:
            plt.axis('off')
            plt.rcParams['svg.fonttype'] = 'none'
            try:
                plt.savefig(path_fig + 'solution_step_' + str(count) + '.' + config.OUTPUT_FORMAT,
                            format=config.OUTPUT_FORMAT, bbox_inches='tight')
            except:
                print('Saving was not successful')
        else:
            plt.pause(0.4)
    return node_status


def show_scenario(scenario_data: Tuple[Scenario, State, Rectangle, PlanningProblem], node_status: Dict[int, Tuple],
                  config):
    plot_search_scenario(scenario=scenario_data[0], initial_state=scenario_data[1], ego_shape=scenario_data[2],
                         planning_problem=scenario_data[3], config=config)
    for node in node_status.values():
        plot_primitive_path(node[0], node[1], config.PLOTTING_PARAMS)

    plt.show()


def display_steps(scenario_data, config, algorithm, **args):
    def slider_callback(iteration):
        # don't show graph for the first time running the cell calling this function
        try:
            show_scenario(scenario_data, node_status=all_node_status[iteration], config=config)
        except:
            pass

    def visualize_callback(Visualize):
        if Visualize is True:
            button.value = False
            global all_node_status

            if 'limit' in args:
                path, primitives, all_node_status = algorithm(limit=args['limit'])
            else:
                path, primitives, all_node_status = algorithm()

            slider.max = len(all_node_status) - 1

            for i in range(slider.max + 1):
                slider.value = i
                # time.sleep(.5)

    slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
    slider_visual = widgets.interactive(slider_callback, iteration=slider)
    display(slider_visual)

    button = widgets.ToggleButton(value=False)
    button_visual = widgets.interactive(visualize_callback, Visualize=button)
    display(button_visual)