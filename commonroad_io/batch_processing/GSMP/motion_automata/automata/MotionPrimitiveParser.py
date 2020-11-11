import numpy as np
from commonroad.scenario.trajectory import Trajectory, State
from automata.MotionPrimitive import MotionPrimitive
from automata.States import StartState, FinalState


class MotionPrimitiveParser:
    """
    Parse and create trajectories from given XML node
    """
    @classmethod
    def createFromNode(cls, xmlNode) -> MotionPrimitive:
        """
        Create a motion primitive from given XML node.

        :param cls:
        :param xmlNode: cls and xmlNode are often given together as an element of the list returned by xmlTree.findall(...)

        """
        # read from xml node and create start state
        startNode = xmlNode.find('Start')
        startX = float(startNode.find('x').text)
        startY = float(startNode.find('y').text)
        startSteeringAngle = float(startNode.find('steering_angle').text)
        startVelocity = float(startNode.find('velocity').text)
        startOrientation = float(startNode.find('orientation').text)
        startTimeStep = int(startNode.find('time_step').text)

        startState = StartState(startX, startY, startSteeringAngle, startVelocity, startOrientation, startTimeStep)

        # read from xml node and create final state
        finalNode = xmlNode.find('Final')
        finalX = float(finalNode.find('x').text)
        finalY = float(finalNode.find('y').text)
        finalSteeringAngle = float(finalNode.find('steering_angle').text)
        finalVelocity = float(finalNode.find('velocity').text)
        finalOrientation = float(finalNode.find('orientation').text)
        finalTimeStep = int(finalNode.find('time_step').text)

        finalState = FinalState(finalX, finalY, finalSteeringAngle, finalVelocity, finalOrientation, finalTimeStep)

        # create trajectory from path node and start/final states
        pathNode = xmlNode.find('Path')
        duration = xmlNode.find('Duration')
        timeStepSize = float(duration.text) / (len(pathNode) + 1)

        trajectory = MotionPrimitiveParser.createTrajectoryFromPathStates(pathNode, startState, finalState, timeStepSize)

        # create and return motion primitive
        return MotionPrimitive(startState, finalState, timeStepSize, trajectory)

    @classmethod
    def createTrajectoryFromPathStates(cls, xmlNode, startState: StartState, finalState: FinalState, timeStepSize: float):
        """
        Creates trajectory state list from the path values described in the xml file.

        :param cls:
        :param xmlNode: cls and xmlNode are often given together as the return value of xmlNode.find(...)
        :param startState: start State of the trajectory
        :param finalState: final State of the trajectory
        :param timeStepSize: time step size
        """
        vertices = []
        steering_angles = []
        velocities = []
        orientations = []
        time_steps = []

        # insert start state
        vertices.append([startState.x, startState.y])
        steering_angles.append(startState.steering_angle)
        velocities.append(startState.velocity)
        orientations.append(startState.orientation)  
        time_steps.append(int(startState.time_step))

        # insert trajectory states
        skipCounter = 0
        if xmlNode is not None:
            list_states = xmlNode.findall('State')
            for state in list_states:

                # # todo: down-sampling of trajectory list_states
                # if skipCounter < (1 / timeStepSize) * (20 / np.max([startState.velocity, finalState.velocity])):
                #     # (len(states) / (timeStepSize * 10)):# * 20 / ((startState.velocity + finalState.velocity) / 2):
                #     skipCounter = skipCounter + 1
                #     continue
                # skipCounter = 0

                x = float(state.find('x').text)
                y = float(state.find('y').text)
                steering_angle = float(state.find('steering_angle').text)
                velocity = float(state.find('velocity').text)
                orientation = float(state.find('orientation').text)
                time_step = int(state.find('time_step').text)

                vertices.append([x, y])
                steering_angles.append(steering_angle)
                velocities.append(velocity)
                orientations.append(orientation)
                time_steps.append(time_step)

        # insert final state
        vertices.append([finalState.x, finalState.y])
        steering_angles.append(finalState.steering_angle)
        velocities.append(finalState.velocity)
        orientations.append(finalState.orientation)  
        time_steps.append(int(finalState.time_step))

        trajectory = MotionPrimitiveParser.create_from_vertices(np.array(vertices),
                                                                np.array(steering_angles),
                                                                np.array(velocities),
                                                                np.array(orientations),
                                                                np.array(time_steps))

        return trajectory

    @classmethod
    def create_from_vertices(cls, vertices: np.ndarray, steering_angles: np.ndarray, velocities: np.ndarray,
                             orientations: np.ndarray, time_steps: np.ndarray) -> Trajectory:
        """
        Creates a trajectory from a set of given positions, orientations, velocities and a starting time step.

        :param vertices: a set of positions for every state of the trajectory
        :param t0: starting time step of the trajectory
        :param orientation: a set of orientations for every state of the trajectory
        :param velocity: a set of velocities for every state of the trajectory
        """
       
        assert len(vertices) == len(steering_angles) == len(velocities) == len(orientations) == len(time_steps), "The sizes of state elements should be equal!"

        list_states = []

        for i in range(len(vertices)):
            kwarg = {'position': np.array([vertices[i][0], vertices[i][1]]),
                    'velocity': velocities[i],
                    'steering_angle': steering_angles[i],
                    'orientation': orientations[i],
                    'time_step': time_steps[i]}

            # append state
            list_states.append(State(**kwarg)) 


        # create trajectory
        trajectory = Trajectory(initial_time_step=int(time_steps[0]), state_list=list_states)

        return trajectory

