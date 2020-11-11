from automata.HelperFunctions import *
import automata.tree_search as tree_search
from commonroad.visualization.draw_dispatch_cr import draw_object
import matplotlib.pyplot as plt

from config import StudentScript

veh_type_id = 2
mp_path = 'motion_primitives_search_tutorial/' + 'V_9.0_9.0_Vstep_0_SA_-0.2_0.2_SAstep_0.4_T_0.5_Model_BMW320i.xml'
scenario_path = '../../scenarios/tutorial/ZAM_Tutorial_Urban-3_2.xml'
config = StudentScript

# load scenario
scenario, planning_problem_set = load_scenario(scenario_path)

# plot scenario
plt.figure(figsize=(8, 8))
draw_object(scenario)
draw_object(planning_problem_set)
plt.gca().set_aspect('equal')
plt.margins(0, 0)
plt.show()

# create maneuver automaton and planning problem
automaton = generate_automata(veh_type_id, mp_file=mp_path, search_tutorial=True)

planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# BFS
# construct bfs motion planner and set up the initial state for planning problem
bfs_planner = tree_search.BreadthFirstSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                             config=config)
# start search
bfs_planner.search_alg()
plt.clf()

# DFS
dfs_planner = tree_search.DepthFirstSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                           config=config)
dfs_planner.search_alg()
plt.clf()
# DLS
dls = tree_search.DepthLimitedSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                     config=config)

dls.search_alg(limit=7)
plt.clf()
# Uniform Cost Search
ucs = tree_search.UniformCostSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                    config=config)
ucs.search_alg()
plt.clf()
# Greedy Best First Search
gbfs = tree_search.GreedyBestFirstSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                         config=config)
gbfs.search_alg()
plt.clf()
# A* Search

astar = tree_search.AStarSearch(scenario=scenario, planningProblem=planning_problem, automaton=automaton,
                                config=config)
astar.search_alg()
plt.clf()
print('Done')
