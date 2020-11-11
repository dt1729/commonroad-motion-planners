import os
import sys
sys.path.append("../vehicle_model/")

import yaml
import xml.etree.ElementTree as et
from xml.dom import minidom

import copy
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sin, cos

# vehicle model
from vehicleDynamics_KS import vehicleDynamics_KS
from parameters_vehicle1 import parameters_vehicle1
from parameters_vehicle2 import parameters_vehicle2
from parameters_vehicle3 import parameters_vehicle3

# solution checker
from solution_checker import TrajectoryValidator, KSTrajectory, KinematicSingleTrackModel

# commonroad objects
from commonroad.scenario.trajectory import State, Trajectory

# solution writer
from commonroad.common.solution_writer import CommonRoadSolutionWriter, VehicleModel, VehicleType, CostFunction


class Parameter:
	'''
	A class to hold all parameters
	'''
	def __init__(self, configuration):
		# outputs
		self.directory_output = configuration['outputs']['output_directory']

		# vehicle related
		self.veh_type_id = configuration['vehicles']['veh_type_id']
		if self.veh_type_id == 1:
			self.veh_type = VehicleType.FORD_ESCORT
			self.veh_param = parameters_vehicle1()
			self.veh_name = "FORD_ESCORT"
		elif self.veh_type_id == 2:
			self.veh_type = VehicleType.BMW_320i
			self.veh_param = parameters_vehicle2()
			self.veh_name = "BMW320i"
		elif self.veh_type_id == 3:
			self.veh_type = VehicleType.VW_VANAGON
			self.veh_param = parameters_vehicle3()
			self.veh_name = "VW_VANAGON"

		# time
		self.T = configuration['primitives']['T']
		self.dt_simulation = configuration['primitives']['dt_simulation']
		# times 100 for two significant digits accuracy, turns into centi-seconds
		self.time_stamps = ((np.arange(0, self.T, self.dt_simulation) + self.dt_simulation) * 100).astype(int)
		
		# velocity
		self.velocity_sample_min = configuration['primitives']['velocity_sample_min']
		self.velocity_sample_max = configuration['primitives']['velocity_sample_max']
		self.num_sample_velocity = configuration['primitives']['num_sample_velocity']
		
		# steering angle
		self.steering_angle_sample_min = configuration['primitives']['steering_angle_sample_min']
		self.steering_angle_sample_max = configuration['primitives']['steering_angle_sample_max']
		self.num_sample_steering_angle = configuration['primitives']['num_sample_steering_angle']

		if int(self.steering_angle_sample_max) == 0:
			self.steering_angle_sample_max = self.veh_param.steering.max

		# sample trajectories
		self.num_segment_trajectory   = configuration['sample_trajectories']['num_segment_trajectory']
		self.num_simulations 	      = configuration['sample_trajectories']['num_simulations']


def load_configuration(path_file_config):
	'''
	load input configuration file
	'''
	with open(path_file_config, 'r') as stream:
		try:
			configuration = yaml.load(stream)
			return configuration

		except yaml.YAMLError as exc:
			print(exc)


def generate_list_of_samples(parameter: Parameter):
	'''
	generate list of samples for velocity and steering angle based on the input parameters
	'''
	list_samples_v = np.linspace(parameter.velocity_sample_min, 
                             	 parameter.velocity_sample_max, 
                             	 parameter.num_sample_velocity)

	list_samples_steering_angle = np.linspace(parameter.steering_angle_sample_min, 
                                         	  parameter.steering_angle_sample_max,
                                         	  parameter.num_sample_steering_angle)

	return list_samples_v, list_samples_steering_angle


def check_acceleration_steering_rate_constraint(v_start, v_end, d_start, d_end, parameter):
	'''
	compute the required inputs and check whether they are of permissible values
	'''
	# compute required inputs
	a_input = (v_end - v_start) / parameter.T
	steering_rate_input = (d_end - d_start) / parameter.T
        
    # check if the constraints are respected
	if abs(a_input) > abs(parameter.veh_param.longitudinal.a_max) or \
	   abs(steering_rate_input) > abs(parameter.veh_param.steering.v_max):
		return False
	else:
		return True

def simulate_list_of_states(v_start, v_end, d_start, d_end, parameter):
	'''
	forward simulation of states with friction constraint taken into account
	elements of a state: x[m], y[m], steering angle[rad], velocity[m/s], orientation[rad], time step[decisecond]
	'''

	# compute required inputs
	a_input = (v_end - v_start) / parameter.T
	steering_rate_input = (d_end - d_start) / parameter.T

	# list to store the states
	list_states = []
	
	# trajectory always starts at position (0, 0) m with orientation of 0 rad
	x_input = np.array([0.0, 0.0, d_start, v_start, 0.0])

	# we assume constant input through the whole duration of T
	u_input = np.array([steering_rate_input, a_input])

	# time stamp of first state is 0 
	list_states.append(np.append(x_input, 0))
	
	is_friction_constraint_satisfied = True
	
	# forward simulation of states
	# ref: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf, page 4
	for time_stamp in parameter.time_stamps:
        # simulate state transition
		x_dot = np.array(vehicleDynamics_KS(x_input, u_input, parameter.veh_param))
        
        # check friction circle constraint
		if (a_input ** 2 + (x_input[3] * x_dot[4]) ** 2) ** 0.5 > parameter.veh_param.longitudinal.a_max:
			is_friction_constraint_satisfied = False
			break
        
        # generate new state
		x_output = x_input + x_dot * parameter.dt_simulation
        
        # downsample the states with step size of 10 centiseconds (equivalent to 0.1 seconds)
		if time_stamp % 10 == 0:
            # add state to list, note that time stamps of states are in unit of deciseconds
			list_states.append(np.append(x_output, time_stamp / 10))
         
        # prepare for next iteration
		x_input = x_output

    # return None if the friction constraint is not satisfied
	if is_friction_constraint_satisfied:
		return list_states
	else:
		return None


def create_trajectory_from_list_of_states(list_states):
    '''
    a helper function to create CommonRoad Trajectory from a list of States
    '''
    # list to hold states for final trajectory
    list_states_new = list()
    
    # iterate through trajectory states
    for state in list_states:
        # feed in required slots
        kwarg = {'position': np.array([state[0], state[1]]),
                 'velocity': state[3],
                 'steering_angle': state[2], 
                 'orientation': state[4],
                 'time_step': state[5].astype(int)}

        # append state
        list_states_new.append(State(**kwarg))         
        
    # create new trajectory for evaluation
    trajectory_new = Trajectory(initial_time_step=0, state_list=list_states_new)
    
    return trajectory_new

def check_validity(trajectory_input, parameter):
    '''
    check whether the input trajectory satisfies the kinematic single track model
	note: to be replaced by the new feasiblity checker
    '''
    csw = CommonRoadSolutionWriter(output_dir=os.getcwd(),              
                                   scenario_id=0,                 
                                   step_size=0.1,                          
                                   vehicle_type=parameter.veh_type,
                                   vehicle_model=VehicleModel.KS,
                                   cost_function=CostFunction.JB1)

    # use solution writer to generate target xml file
    csw.add_solution_trajectory(trajectory=trajectory_input, planning_problem_id=100)
    xmlTree = csw.root_node

    # generate states to be checked
    [node] = xmlTree.findall('ksTrajectory')
    veh_trajectory = KSTrajectory.from_xml(node)
    veh_model = KinematicSingleTrackModel(parameter.veh_type_id, veh_trajectory, None)

    # validate
    result = TrajectoryValidator.is_trajectory_valid(veh_trajectory, veh_model, 0.1)
    
    return result

def create_mirrored_primitives(list_traj, parameter):
	# make sure to make a deep copy
	list_traj_mirrored = copy.deepcopy(list_traj)

	count_accepted = 0
	for traj in list_traj:
		list_states_mirrored = []
	    
		for state in traj.state_list:
			# add mirrored state into list
			list_states_mirrored.append([state.position[0],
	                                    -state.position[1],
	                                    -state.steering_angle,
	                                     state.velocity,
	                                    -state.orientation,
	                                     state.time_step])
	    
		trajectory_new = create_trajectory_from_list_of_states(list_states_mirrored)
	    
		# double check the validity before adding to the list
		if check_validity(trajectory_new, parameter):
			list_traj_mirrored.append(trajectory_new)
			count_accepted += 1

	return list_traj_mirrored

def compute_number_successors_average(list_traj):
	'''
	compute the average number of successors (branching factor)
	'''
	list_count_succesors = []

	for traj_main in list_traj:
		count_successors = 0
		# 2bc = to be connected
		for traj_2bc in list_traj:
		
			state_final_traj_main = traj_main.state_list[-1]
			state_initial_traj_2bc = traj_2bc.state_list[0]
	
			if abs(state_final_traj_main.velocity - state_initial_traj_2bc.velocity) < 0.02 and \
			   abs(state_final_traj_main.steering_angle - state_initial_traj_2bc.steering_angle) < 0.02:
				count_successors += 1
	
		list_count_succesors.append(count_successors)

	return np.mean(list_count_succesors)


def generate_sample_trajectories(list_traj_input, parameter):
	'''
	generate sample trajectories and check if they pass the solution checker
	'''

	random.seed()

	for count_run in range(parameter.num_simulations):
		count_segment_traj = 0
		num_traj = len(list_traj_input)
		# get a random starting trajectory id
		idx_traj = random.randrange(num_traj)
		list_traj = []

		while count_segment_traj< parameter.num_segment_trajectory:
			# retrieve trajectory
			traj_main = copy.deepcopy(list_traj_input[idx_traj])
			list_traj.append(traj_main)
			list_successors_traj_main = []
			count_segment_traj += 1

			# obtain its successors
			for j in range(num_traj):
				# 2bc = to be connected
				traj_2bc = list_traj_input[j]

				state_final_traj_main = traj_main.state_list[-1]
				state_initial_traj_2bc = traj_2bc.state_list[0]

				if abs(state_final_traj_main.velocity - state_initial_traj_2bc.velocity) < 0.02 and \
				   abs(state_final_traj_main.steering_angle - state_initial_traj_2bc.steering_angle) < 0.02:

					list_successors_traj_main.append(j)

			# start over if a trajectory does not have a valid successor
			num_successors_traj_main = len(list_successors_traj_main)
			if num_successors_traj_main == 0:
				count_segment_traj = 0
				idx_traj = random.randrange(num_traj)
				list_traj = []
			else:
				# retrieve a random successor id
				idx_successor = random.randrange(num_successors_traj_main)
				idx_traj = list_successors_traj_main[idx_successor]

		fig = plt.figure()
		# plot first trajectory
		list_states_final = copy.deepcopy(list_traj[0].state_list)
		list_x = [state.position[0] for state in list_states_final]
		list_y = [state.position[1] for state in list_states_final]
		plt.scatter(list_x, list_y)

		# plot remaining trajectories
		for i in range(1, len(list_traj)):
			traj_pre = list_traj[i - 1]
			traj_cur = list_traj[i]   

			# retrieve states
			state_final_traj_pre   = traj_pre.state_list[-1]
			state_initial_traj_cur = traj_cur.state_list[0]

			# rotate + translate with regard to the last state of preivous trajectory
			traj_cur.translate_rotate(np.zeros(2), state_final_traj_pre.orientation)
			traj_cur.translate_rotate(state_final_traj_pre.position, 0)

			# retrieve new states
			state_final_traj_pre   = traj_pre.state_list[-1]
			state_initial_traj_cur = traj_cur.state_list[0]

			list_x = [state.position[0] for state in traj_cur.state_list]
			list_y = [state.position[1] for state in traj_cur.state_list]
			plt.scatter(list_x, list_y)

			# discard the first state of second trajectory onward to prevent duplication
			traj_cur.state_list.pop(0)
			list_states_final += traj_cur.state_list

		list_x = [state.position[0] for state in list_states_final]
		list_y = [state.position[1] for state in list_states_final]
		plt.xlim([min(list_x) - 2, max(list_x) + 2])
		plt.ylim([min(list_y) - 2, max(list_y) + 2])
		plt.axis('equal')
		# plt.show()

		# save as cr node to be validated via the solution checker

		list_states_for_traj = []
		count = 0
		for state in list_states_final:
			list_states_for_traj.append([state.position[0],
										 state.position[1],
										 state.steering_angle,
										 state.velocity,
										 state.orientation,
										 np.int64(count)])
			count += 1

		trajectory_new = create_trajectory_from_list_of_states(list_states_for_traj)
		is_valid = check_validity(trajectory_new, parameter)
		if not is_valid:
			print("This is not good :(")

def create_xml_node_from_trajectory(list_traj):
	# create Trajectories tag
	node_trajectories = et.Element('Trajectories')

	for trajectory in list_traj:

	    # create a tag for individual trajectory
	    node_trajectory = et.SubElement(node_trajectories, 'Trajectory')

	    # add time duration tag
	    node_duration = et.SubElement(node_trajectory, 'Duration')
	    node_duration.set('unit','deci-second')
	    node_duration.text = "10"

	    list_states = trajectory.state_list

	    # add start state
	    node_start = et.SubElement(node_trajectory, 'Start')
	    node_x = et.SubElement(node_start, 'x')
	    node_y = et.SubElement(node_start, 'y')
	    node_sa = et.SubElement(node_start, 'steering_angle')
	    node_v = et.SubElement(node_start, 'velocity')
	    node_o = et.SubElement(node_start, 'orientation')
	    node_t = et.SubElement(node_start, 'time_step')

	    state_start = list_states[0]

	    node_x.text = str(state_start.position[0])
	    node_y.text = str(state_start.position[1])
	    node_sa.text = str(state_start.steering_angle)
	    node_v.text = str(state_start.velocity)
	    node_o.text = str(state_start.orientation)
	    node_t.text = str(state_start.time_step)

	    # add final state
	    node_final = et.SubElement(node_trajectory, 'Final')
	    node_x = et.SubElement(node_final, 'x')
	    node_y = et.SubElement(node_final, 'y')
	    node_sa = et.SubElement(node_final, 'steering_angle')
	    node_v = et.SubElement(node_final, 'velocity')
	    node_o = et.SubElement(node_final, 'orientation')
	    node_t = et.SubElement(node_final, 'time_step')

	    state_final = list_states[-1]

	    node_x.text = str(state_final.position[0])
	    node_y.text = str(state_final.position[1])
	    node_sa.text = str(state_final.steering_angle)
	    node_v.text = str(state_final.velocity)
	    node_o.text = str(state_final.orientation)
	    node_t.text = str(state_final.time_step)

	    # add states in between
	    list_states_in_between = list_states[1:-1]

	    node_path = et.SubElement(node_trajectory, 'Path')

	    for state in list_states_in_between:
	        node_state = et.SubElement(node_path, 'State')

	        node_x = et.SubElement(node_state, 'x')
	        node_y = et.SubElement(node_state, 'y')
	        node_sa = et.SubElement(node_state, 'steering_angle')
	        node_v = et.SubElement(node_state, 'velocity')
	        node_o = et.SubElement(node_state, 'orientation')
	        node_t = et.SubElement(node_state, 'time_step')

	        node_x.text = str(state.position[0])
	        node_y.text = str(state.position[1])
	        node_sa.text = str(state.steering_angle)
	        node_v.text = str(state.velocity)
	        node_o.text = str(state.orientation)
	        node_t.text = str(state.time_step)

	return node_trajectories

def save_motion_primitives(list_traj, parameter):
	node_xml = create_xml_node_from_trajectory(list_traj)

	prefix = parameter.directory_output

	step_v = round((parameter.velocity_sample_max - parameter.velocity_sample_min) /
		           (parameter.num_sample_velocity - 1), 2)

	step_sa = round(parameter.steering_angle_sample_max * 2 / 
		           (parameter.num_sample_steering_angle - 1), 2)

	file_name = "V_{}_{}_Vstep_{}_SA_{}_{}_SAstep_{}_T_{}_Model_{}.xml".format(parameter.velocity_sample_min, 
																			   parameter.velocity_sample_max,
	                                                                           step_v,
	                                                                           -parameter.steering_angle_sample_max, 
	                                                                           parameter.steering_angle_sample_max,
	                                                                           step_sa,
	                                                                           round(parameter.T, 1), 
	                                                                           parameter.veh_name)

	xml_prettified = minidom.parseString(et.tostring(node_xml)).toprettyxml(indent="   ")

	with open(prefix + file_name, "w") as f:
	    f.write(xml_prettified)
	    print("file saved: {}".format(file_name))


def generate_motion_primitives(parameter):
	'''
	main function to generate motion primitives.
	'''
	# create lists of possible samples for velocity and steering angle
	list_samples_v, list_samples_steering_angle = generate_list_of_samples(parameter)

	# for some statistics
	# calculate number of posible states
	num_possible_states = len(list_samples_v) * len(list_samples_steering_angle)
	# calculate number of possible start state to end state combination
	num_possible_start_end_combinations = num_possible_states ** 2
	print("Total possible combinations of states: ", num_possible_start_end_combinations)

	count_processed = 0
	count_validated =0
	count_accepted = 0

	# for saving results
	list_traj_accepted = []

	# v = velocity, d = steering_angle
	# iterate through possible instances of the Cartesian product of list_samples_v and list_samples_steering_angle
	# create all possible combinations of (v_start, d_start) and (v_end, d_end)

	print("Examining combinations...")
	for (v_start, d_start) in itertools.product(list_samples_v, list_samples_steering_angle):
		for (v_end, d_end) in itertools.product(list_samples_v, list_samples_steering_angle):
			count_processed += 1

			# Print progress
			if count_processed % 5000 == 0 and count_processed:
				print(f"Progress: {count_processed} combinations checked.")
				
			# check constraints
			if not check_acceleration_steering_rate_constraint(v_start, v_end, d_start, d_end, parameter):
				continue
			
			# forward simulation
			list_states_simulated = simulate_list_of_states(v_start, v_end, d_start, d_end, parameter)
			
			if list_states_simulated is None:
				continue
			
			# create trajectory from the list of simulated states
			trajectory_new = create_trajectory_from_list_of_states(list_states_simulated)
			
			# check whether the generated trajectory pass the validity check
			result = check_validity(trajectory_new, parameter)
			count_validated += 1
			
			if result: 
				count_accepted += 1
				list_traj_accepted.append(trajectory_new)

	print(f"Progress: {count_processed} combinations checked.")
	print("============================================\n")

	if count_validated != 0:
		percentage_accept = round(count_accepted / count_validated, 2) * 100
	else:
		percentage_accept = 0
		
	print(f"Feasible combinations: {count_accepted}")

	return list_traj_accepted