import os
import numpy as np
import xml.etree.ElementTree as et
from typing import List
from copy import deepcopy
from automata.States import StartState, FinalState
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.trajectory import State

from commonroad.common.solution_writer import CommonRoadSolutionWriter, VehicleType, CostFunction
from commonroad.common.solution_writer import VehicleModel as VM
from solution_checker import *

class MotionPrimitive:
    def __init__(self, startState: StartState, finalState: FinalState, timeStepSize, trajectory: Trajectory):
        """
        Initialisation of a motion primitive.

        :param startState: start state of the primitive
        :param finalState: final state of the primitive
        :param trajectory: trajectory of the primitive
        """
        self.startState = startState
        self.finalState = finalState
        self.timeStepSize = timeStepSize
        self.trajectory = trajectory
        self.vehicle_type = VehicleType.BMW_320i
        self.vehicle_model = VM.KS
        # a list to store connectable successive primitives from current primitive
        self.Successors = []
        self.veh_type_id = 1 # default

    def __str__(self):
        return "Primitive: \n\t {}\t=>\n\t{}".format(str(self.startState), str(self.finalState))

    def print_info(self):
        kwarg = {'position'      : np.array([self.startState.x, self.startState.y]), 
                 'velocity'      : self.startState.velocity,
                 'steering_angle': self.startState.steering_angle, 
                 'orientation'   : self.startState.orientation, 
                 'time_step'     : self.startState.time_step}
        state_start = State(**kwarg)

        # add final state into the trajectory
        kwarg = {'position'      : np.array([self.finalState.x, self.finalState.y]), 
                 'velocity'      : self.finalState.velocity,
                 'steering_angle': self.finalState.steering_angle, 
                 'orientation'   : self.finalState.orientation, 
                 'time_step'     : self.finalState.time_step}
        state_final = State(**kwarg)

        print(state_start)

        for state in self.trajectory.state_list:
            print(state)

        print(state_final)

    def mirror(self) -> None:
        """
        Mirrors the current primitive so that it describes a primitive in the opposite direction.
        """
        self.finalState.y = - self.finalState.y
        self.finalState.orientation = - self.finalState.orientation
        for i in range(len(self.trajectory.state_list)):
            state = self.trajectory.state_list[i]
            self.trajectory.state_list[i].position[1] = - self.trajectory.state_list[i].position[1]
            state.orientation = - state.orientation
            self.trajectory.state_list[i] = state

    def checkConnectivityToNext(self, other: 'MotionPrimitive') -> bool: 
        """
        Any primitive whose first state's velocity is in the velocity range of the last state of the current primitive is considered as connected.
        Returns true if the two primitives are connected.

        :param other: the next motion primitive to be checked
        """
        # if (self.finalState.velocity - self.finalState.vRange >= other.startState.velocity - other.startState.vRange and
        #     self.finalState.velocity + self.finalState.vRange <= other.startState.velocity + other.startState.vRange):
        #     return True

        if abs(self.finalState.velocity       - other.startState.velocity) < 0.01 and \
           abs(self.finalState.steering_angle - other.startState.steering_angle) < 0.01 :
            return True
            
        return False

    def appendTrajectoryToState(self, state: State) -> List[State]: 
        """
        Appends the primitive to the given state and returns the new list of states including the intermediate steps.

        :param state: the state to which the trajectory will be appended
        """

        # deep copy to prevent manipulation of original data
        trajectory = deepcopy(self.trajectory)

        # rotate the trajectory by the orientation of the given state
        trajectory.translate_rotate(np.zeros(2), state.orientation)
        # translate the trajectory by the position of the given state
        trajectory.translate_rotate(state.position, 0)

        # trajectory includes all states of the primitive
        # as its first state is the same as the last state of the primitive to be connected
        # we pop out the first state
        trajectory.state_list.pop(0)

        # we modify the time steps of the trajectory
        time_step_state = int(state.time_step)
        for i in range(len(trajectory.state_list)):
            trajectory.state_list[i].time_step = time_step_state + i + 1

        # check connectability between the first state of the trajectory and the state
        state_start = deepcopy(state)
        state_end = deepcopy(trajectory.state_list[0])

        state_start.time_step = 0
        state_end.time_step = 1

        trajectory_to_be_cheked = Trajectory(0, [state_start, state_end])

        if self.check_validity(trajectory_to_be_cheked):
            return trajectory.state_list
        else:
            return None

    def check_validity(self, trajectory_to_be_cheked):
        """
        Checks feasibility of the trajectory
        """

        if self.veh_type_id == 1:
            veh_type = VehicleType.FORD_ESCORT
        elif self.veh_type_id == 2:
            veh_type = VehicleType.BMW_320i
        elif self.veh_type_id == 3:
            veh_type = VehicleType.VW_VANAGON

        csw = CommonRoadSolutionWriter(output_dir=os.getcwd(),              
                                       scenario_id=0,                 
                                       step_size=0.1,                          
                                       vehicle_type=veh_type,
                                       vehicle_model=self.vehicle_model,
                                       cost_function=CostFunction.JB1)

        # use solution writer to generate target xml tree
        csw.add_solution_trajectory(trajectory=trajectory_to_be_cheked, planning_problem_id=100)
        xmlTree = csw.root_node
        # generate states to be checked
        [node] = xmlTree.findall('ksTrajectory')

        veh_trajectory = KSTrajectory.from_xml(node)
        veh_model = KinematicSingleTrackModel(self.veh_type_id, veh_trajectory, None)

        # validate
        result = TrajectoryValidator.is_trajectory_valid(veh_trajectory, veh_model, 0.1)
        
        return result

    def costs(self) -> float:
        """
        Returns euclidean distance between the last state of current primitive and [0,0].
        """
        pos1 = [0,0]
        pos2 = [self.finalState.x, self.finalState.y]
        return np.sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]))
