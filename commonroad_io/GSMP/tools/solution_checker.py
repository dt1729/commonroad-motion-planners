import sys
sys.path.append("../../GSMP/motion_automata/vehicle_model")
from abc import ABC, abstractmethod, abstractproperty
from copy import copy
from typing import List
import math
import numpy as np
from commonroad.scenario.trajectory import State, Trajectory
from parameters_vehicle1 import parameters_vehicle1
from parameters_vehicle2 import parameters_vehicle2
from parameters_vehicle3 import parameters_vehicle3
from scipy.integrate import odeint
from scipy.optimize import minimize, Bounds
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_object
from vehicleDynamics_KS import vehicleDynamics_KS
from vehicleDynamics_MB import vehicleDynamics_MB
from vehicleDynamics_ST import vehicleDynamics_ST


class VelocityConverter:

    @classmethod
    def convert_to_directed(cls, x_velocity, y_velocity):
        velocity = math.sqrt(x_velocity ** 2 + y_velocity ** 2)

        if x_velocity == 0:
            if y_velocity > 0:
                orientation = 1 / 2 * math.pi
            elif y_velocity < 0:
                orientation = 3 / 2 * math.pi
            else:
                orientation = 0.0  # default case if there is no movement

        else:
            orientation = math.atan(y_velocity / x_velocity)
            if x_velocity < 0:
                if y_velocity >= 0:
                    orientation += math.pi
                else:
                    orientation -= math.pi

        return velocity, orientation

    @classmethod
    def convert_from_directed(cls, velocity, orientation):
        velocity_x = math.cos(orientation) * velocity
        velocity_y = math.sin(orientation) * velocity
        return velocity_x, velocity_y


class Input:
    __slots__ = [
        'acceleration',
        'steering_angle_speed',
        'time_step',
    ]

    def __init__(self, **kwargs):
        """ Elements of input are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def values(self):
        return np.array([self.steering_angle_speed, self.acceleration]) # The place is switched in the dynemaics

    @classmethod
    def from_xml(cls, state_xml):
        return Input(
            acceleration=float(state_xml.find("acceleration").text),
            steering_angle_speed=float(state_xml.find("steeringAngleSpeed").text),
            time_step=int(state_xml.find("time").text),
        )

    def __eq__(self, other):
        return self.time_step == other.time_step

    def __lt__(self, other):
        return self.time_step < other.time_step


class PMInput(Input):
    __slots__ = [
        'acceleration_x',
        'acceleration_y',
        'time_step',
    ]

    @property
    def values(self):
        return np.array([self.acceleration_x, self.acceleration_y])

    @classmethod
    def from_xml(cls, state_xml):
        return PMInput(
            acceleration_x=float(state_xml.find("xAcceleration").text),
            acceleration_y=float(state_xml.find("yAcceleration").text),
            time_step=int(state_xml.find("time").text),
        )


class InputVectorException(Exception):
    pass


class InputVector:
    INPUT_VECTOR_DICT = {
        'inputVector': lambda input_vector_xml: InputVector.from_xml(input_vector_xml),
        'pmInputVector': lambda input_vector_xml: PMInputVector.from_xml(input_vector_xml),
    }

    type = 'inputVector'
    input_type = Input

    def __init__(self, initial_time_step: int, input_list: List[Input]):
        self.initial_time_step = initial_time_step
        self.input_list = input_list

    @property
    def input_list(self):
        return self._input_list

    @input_list.setter
    def input_list(self, input_list):
        if not all(type(input) == self.input_type for input in input_list):
            raise InputVectorException('Input list contains invalid input type for %s' % type(self))

        self._input_list = input_list

    @classmethod
    def from_xml(cls, input_vector_xml):
        input_vector_type = input_vector_xml.tag
        if input_vector_type not in cls.INPUT_VECTOR_DICT:
            raise InputVectorException('Input vector type %s not defined.' % input_vector_type)

        if input_vector_type == 'pmInputVector':
            return PMInputVector.from_xml(input_vector_xml)

        input_list = sorted([Input.from_xml(input_xml) for input_xml in input_vector_xml])
        initial_time_step = input_list[0].time_step
        return InputVector(initial_time_step, input_list)


class PMInputVector(InputVector):
    type = 'pmInputVector'
    input_type = PMInput

    @classmethod
    def from_xml(cls, input_vector_xml):
        input_vector_type = input_vector_xml.tag
        if not input_vector_type == 'pmInputVector':
            raise InputVectorException('Invalid input vector type %s for %s.' % (input_vector_type, type(cls)))

        input_list = sorted([PMInput.from_xml(input_xml) for input_xml in input_vector_xml])
        initial_time_step = input_list[0].time_step
        return PMInputVector(initial_time_step, input_list)


class VehicleState(State, ABC):
    """
    Interface for adding additional functionality to the CommonRoad state class.
    """
    @property
    def values(self):
        vals = [self.position[0], self.position[1]]
        return np.array(vals + [getattr(self, attr) for attr in self.__slots__[1:-1]])

    @classmethod
    @abstractmethod
    def from_xml(cls, state_xml):
        pass

    @classmethod
    @abstractmethod
    def from_values(cls, values, time_step):
        pass

    @classmethod
    @abstractmethod
    def convert_init_state(cls, init_state, parameters):
        # init state fields: ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        pass

    def forward_simulation(self, input: List[float], dt, vehicle_model):
        values = odeint(vehicle_model.dynamics,
                        self.values,
                        [0.0, dt],
                        args=(input, vehicle_model.parameters),
                        tfirst=True)[1]
        return self.from_values(values, self.time_step + 1)

    def __eq__(self, other):
        return self.time_step == other.time_step

    def __lt__(self, other):
        return self.time_step < other.time_step


class PMState(VehicleState):
    """
        The possible state elements are defined as slots, which comprise the necessary state elements to describe the
        states of PM vehicle model.
    """
    __slots__ = [
        'position',
        #'velocity_x',  # This fields are converted to velocity and orientation
        #'velocity_y',  # This fields are converted to velocity and orientation
        'velocity',
        'orientation',
        'time_step',
    ]

    @property
    def values(self):
        velocity_x, velocity_y = VelocityConverter.convert_from_directed(self.velocity, self.orientation)
        return np.array([
            self.position[0],
            self.position[1],
            velocity_x,
            velocity_y,
        ])

    @classmethod
    def from_xml(cls, state_xml):
        velocity_x = float(state_xml.find("xVelocity").text)
        velocity_y = float(state_xml.find("yVelocity").text)
        velocity, orientation = VelocityConverter.convert_to_directed(velocity_x, velocity_y)
        return PMState(
            position=np.array([float(state_xml.find("x").text), float(state_xml.find("y").text)]),
            # velocity_x=velocity_x,
            # velocity_y=velocity_y,
            velocity=velocity,
            orientation=orientation,
            time_step=int(state_xml.find("time").text),
        )

    @classmethod
    def from_values(cls, values, time_step):
        velocity_x = values[2]
        velocity_y = values[3]
        velocity, orientation = VelocityConverter.convert_to_directed(velocity_x, velocity_y)
        return PMState(
            position=np.array([values[0], values[1]]),
            # velocity_x=velocity_x,
            # velocity_y=velocity_y,
            velocity=velocity,
            orientation=orientation,
            time_step=time_step,
        )

    @classmethod
    def convert_init_state(cls, init_state, parameters):
        # init state fields: ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        # velocity_x, velocity_y = VelocityConverter.convert_from_directed(init_state.velocity, init_state.orientation)
        return PMState(
            position=init_state.position,
            # velocity_x=velocity_x,
            # velocity_y=velocity_y,
            velocity=init_state.velocity,
            orientation=init_state.orientation,
            time_step=init_state.time_step,
        )


class KSState(VehicleState):
    """
        The possible state elements are defined as slots, which comprise the necessary state elements to describe the
        states of KS vehicle model.
    """
    __slots__ = [
        'position',
        'steering_angle',
        'velocity',
        'orientation',
        'time_step',
    ]

    @classmethod
    def from_xml(cls, state_xml):
        return KSState(
            position=np.array([float(state_xml.find("x").text), float(state_xml.find("y").text)]),
            steering_angle=float(state_xml.find("steeringAngle").text),
            orientation=float(state_xml.find("orientation").text),
            velocity=float(state_xml.find("velocity").text),
            time_step=int(state_xml.find("time").text),
        )

    @classmethod
    def from_values(cls, values, time_step):
        return KSState(
            position=np.array([values[0], values[1]]),
            steering_angle=values[2],
            velocity=values[3],
            orientation=values[4],
            time_step=time_step,
        )

    @classmethod
    def convert_init_state(cls, init_state, parameters):
        # init state fields: ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        return KSState(
            position=init_state.position,
            steering_angle=0.0,
            orientation=init_state.orientation,
            velocity=init_state.velocity,
            time_step=init_state.time_step,
        )


class STState(VehicleState):
    """
        The possible state elements are defined as slots, which comprise the necessary state elements to describe the
        states of ST vehicle model.
    """
    __slots__ = [
        'position',
        'steering_angle',
        'velocity',
        'orientation',
        'yaw_rate',
        'slip_angle',
        'time_step',
    ]

    @classmethod
    def from_xml(cls, state_xml):
        return STState(
            position=np.array([float(state_xml.find("x").text), float(state_xml.find("y").text)]),
            steering_angle=float(state_xml.find("steeringAngle").text),
            velocity=float(state_xml.find("velocity").text),
            orientation=float(state_xml.find("orientation").text),
            yaw_rate=float(state_xml.find("yawRate").text),
            slip_angle=float(state_xml.find("slipAngle").text),
            time_step=int(state_xml.find("time").text),
        )

    @classmethod
    def from_values(cls, values, time_step):
        return STState(
            position=np.array([values[0], values[1]]),
            steering_angle=values[2],
            velocity=values[3],
            orientation=values[4],
            yaw_rate=values[5],
            slip_angle=values[6],
            time_step=time_step,
        )

    @classmethod
    def convert_init_state(cls, init_state, parameters):
        # init state fields: ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        return STState(
            position=init_state.position,
            steering_angle=0.0,
            velocity=init_state.velocity,
            orientation=init_state.orientation,
            yaw_rate=init_state.yaw_rate,
            slip_angle=init_state.slip_angle,
            time_step=init_state.time_step
        )


class MBState(VehicleState):
    """
    The possible state elements are defined as slots, which comprise the necessary state elements to describe the
    states of MB vehicle model.
    """
    __slots__ = [
        'position',
        'steering_angle',
        'velocity',
        'orientation',
        'yaw_rate',
        'roll_angle',
        'roll_rate',
        'pitch_angle',
        'pitch_rate',
        'velocity_y',
        'position_z',
        'velocity_z',
        'roll_angle_front',
        'roll_rate_front',
        'velocity_y_front',
        'position_z_front',
        'velocity_z_front',
        'roll_angle_rear',
        'roll_rate_rear',
        'velocity_y_rear',
        'position_z_rear',
        'velocity_z_rear',
        'left_front_wheel_angular_speed',
        'right_front_wheel_angular_speed',
        'left_rear_wheel_angular_speed',
        'right_rear_wheel_angular_speed',
        'delta_y_f',
        'delta_y_r',
        'time_step',
    ]

    @classmethod
    def from_xml(cls, state_xml):
        return MBState(
            position=np.array([float(state_xml.find("x").text), float(state_xml.find("y").text)]),
            steering_angle=float(state_xml.find("steeringAngle").text),
            velocity=float(state_xml.find("velocity").text),
            orientation=float(state_xml.find("orientation").text),
            yaw_rate=float(state_xml.find("yawRate").text),
            roll_angle=float(state_xml.find("rollAngle").text),
            roll_rate=float(state_xml.find("rollRate").text),
            pitch_angle=float(state_xml.find("pitchAngle").text),
            pitch_rate=float(state_xml.find("pitchRate").text),
            velocity_y=float(state_xml.find("yVelocity").text),
            position_z=float(state_xml.find("zPosition").text),
            velocity_z=float(state_xml.find("zVelocity").text),
            roll_angle_front=float(state_xml.find("rollAngleFront").text),
            roll_rate_front=float(state_xml.find("rollRateFront").text),
            velocity_y_front=float(state_xml.find("yVelocityFront").text),
            position_z_front=float(state_xml.find("zPositionFront").text),
            velocity_z_front=float(state_xml.find("zVelocityFront").text),
            roll_angle_rear=float(state_xml.find("rollAngleRear").text),
            roll_rate_rear=float(state_xml.find("rollRateRear").text),
            velocity_y_rear=float(state_xml.find("yVelocityRear").text),
            position_z_rear=float(state_xml.find("zPositionRear").text),
            velocity_z_rear=float(state_xml.find("zVelocityRear").text),
            left_front_wheel_angular_speed=float(state_xml.find("leftFrontWheelAngularSpeed").text),
            right_front_wheel_angular_speed=float(state_xml.find("rightFrontWheelAngularSpeed").text),
            left_rear_wheel_angular_speed=float(state_xml.find("leftRearWheelAngularSpeed").text),
            right_rear_wheel_angular_speed=float(state_xml.find("rightRearWheelAngularSpeed").text),
            delta_y_f=float(state_xml.find("deltaYf").text),
            delta_y_r=float(state_xml.find("deltaYr").text),
            time_step=int(state_xml.find("time").text),
        )

    @classmethod
    def from_values(cls, values, time_step):
        return MBState(
            position=np.array([values[0], values[1]]),
            steering_angle=values[2],
            velocity=values[3],
            orientation=values[4],
            yaw_rate=values[5],
            roll_angle=values[6],
            roll_rate=values[7],
            pitch_angle=values[8],
            pitch_rate=values[9],
            velocity_y=values[10],
            position_z=values[11],
            velocity_z=values[12],
            roll_angle_front=values[13],
            roll_rate_front=values[14],
            velocity_y_front=values[15],
            position_z_front=values[16],
            velocity_z_front=values[17],
            roll_angle_rear=values[18],
            roll_rate_rear=values[19],
            velocity_y_rear=values[20],
            position_z_rear=values[21],
            velocity_z_rear=values[22],
            left_front_wheel_angular_speed=values[23],
            right_front_wheel_angular_speed=values[24],
            left_rear_wheel_angular_speed=values[25],
            right_rear_wheel_angular_speed=values[26],
            delta_y_f=values[27],
            delta_y_r=values[28],
            time_step=time_step,
        )

    @classmethod
    def convert_init_state(cls, init_state, parameters):
        # ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        if init_state.velocity == 0:
            init_state.velocity = 10 ** -15

        velocity_x, velocity_y = VelocityConverter.convert_from_directed(init_state.velocity, init_state.orientation)

        # create equivalent bicycle parameters
        g = 9.81  # [m/s^2]

        # auxiliary initial states
        F0_z_f = parameters.m_s * parameters.b * g / (parameters.a + parameters.b) + parameters.m_uf * g
        F0_z_r = parameters.m_s * parameters.a * g / (parameters.a + parameters.b) + parameters.m_ur * g

        return MBState(
                position=init_state.position,
                steering_angle=0.0,
                velocity=velocity_x,
                orientation=init_state.orientation,
                yaw_rate=init_state.yaw_rate,
                roll_angle=0.0,
                roll_rate=0.0,
                pitch_angle=0.0,
                pitch_rate=0.0,
                velocity_y=velocity_y,
                position_z=0.0,
                velocity_z=0.0,
                roll_angle_front=0.0,
                roll_rate_front=0.0,
                velocity_y_front=velocity_y + parameters.a * init_state.yaw_rate,
                position_z_front=F0_z_f / parameters.K_zt * 2,
                velocity_z_front=0.0,
                roll_angle_rear=0.0,
                roll_rate_rear=0.0,
                velocity_y_rear=velocity_y + parameters.b * init_state.yaw_rate,
                position_z_rear=F0_z_r / parameters.K_zt * 2,
                velocity_z_rear=0.0,
                left_front_wheel_angular_speed=velocity_x / parameters.R_w,
                right_front_wheel_angular_speed=velocity_x / parameters.R_w,
                left_rear_wheel_angular_speed=velocity_x / parameters.R_w,
                right_rear_wheel_angular_speed=velocity_x / parameters.R_w,
                delta_y_f=0.0,
                delta_y_r=0.0,
                time_step=init_state.time_step,
            )


class TrajectoryException(Exception):
    pass


class VehicleTrajectory(Trajectory, ABC):

    TRAJECTORY_DICT = {
        'pmTrajectory': lambda trajectory_xml: PMTrajectory.from_xml(trajectory_xml),
        'ksTrajectory': lambda trajectory_xml: KSTrajectory.from_xml(trajectory_xml),
        'stTrajectory': lambda trajectory_xml: STTrajectory.from_xml(trajectory_xml),
        'mbTrajectory': lambda trajectory_xml: MBTrajectory.from_xml(trajectory_xml),
    }

    def __init__(self, initial_time_step: int, state_list: List[State]):
        if not all(isinstance(state, self.state_type) for state in state_list):
            raise TrajectoryException('State list contains invalid state type for %s' % type(self))

        super(VehicleTrajectory, self).__init__(initial_time_step, state_list)

    @abstractproperty
    def type(self):
        pass

    @abstractproperty
    def state_type(self):
        pass

    @classmethod
    def from_xml(cls, trajectory_xml):
        trajectory_type = trajectory_xml.tag

        if trajectory_type not in cls.TRAJECTORY_DICT:
            raise TrajectoryException('Trajectory type %s not defined.' % trajectory_type)

        return cls.TRAJECTORY_DICT[trajectory_type](trajectory_xml)

    @classmethod
    def from_input_vector(cls, init_state, input_vector, dt, vehicle_model):
        x0 = vehicle_model.trajectory_type.state_type.convert_init_state(init_state, vehicle_model.parameters)
        states = [x0]
        x_current = copy(x0)
        for u in input_vector.input_list:
            x_next = x_current.forward_simulation(u.values, dt, vehicle_model)
            states.append(x_next)
            x_current = x_next
        return vehicle_model.trajectory_type(x0.time_step, states)


class PMTrajectory(VehicleTrajectory):
    type = 'pmTrajectory'
    state_type = PMState

    @classmethod
    def from_xml(cls, trajectory_xml):
        trajectory_type = trajectory_xml.tag
        if trajectory_type != 'pmTrajectory':
            raise TrajectoryException('Invalid trajectory type %s' % trajectory_type)

        state_list = sorted([PMState.from_xml(state_xml) for state_xml in trajectory_xml])
        initial_time_step = state_list[0].time_step
        return PMTrajectory(initial_time_step, state_list)


class KSTrajectory(VehicleTrajectory):
    type = 'ksTrajectory'
    state_type = KSState

    @classmethod
    def from_xml(cls, trajectory_xml):
        trajectory_type = trajectory_xml.tag
        if trajectory_type != 'ksTrajectory':
            raise TrajectoryException('Invalid trajectory type %s' % trajectory_type)

        state_list = sorted([KSState.from_xml(state_xml) for state_xml in trajectory_xml])
        initial_time_step = state_list[0].time_step
        return KSTrajectory(initial_time_step, state_list)


class STTrajectory(VehicleTrajectory):
    type = 'stTrajectory'
    state_type = STState

    @classmethod
    def from_xml(cls, trajectory_xml):
        trajectory_type = trajectory_xml.tag
        if trajectory_type != 'stTrajectory':
            raise TrajectoryException('Invalid trajectory type %s' % trajectory_type)

        state_list = sorted([STState.from_xml(state_xml) for state_xml in trajectory_xml])
        initial_time_step = state_list[0].time_step
        return STTrajectory(initial_time_step, state_list)


class MBTrajectory(VehicleTrajectory):
    type = 'mbTrajectory'
    state_type = MBState

    @classmethod
    def from_xml(cls, trajectory_xml):
        trajectory_type = trajectory_xml.tag
        if trajectory_type != 'mbTrajectory':
            raise TrajectoryException('Invalid trajectory type %s' % trajectory_type)

        state_list = sorted([MBState.from_xml(state_xml) for state_xml in trajectory_xml])
        initial_time_step = state_list[0].time_step
        return MBTrajectory(initial_time_step, state_list)


class TrajectoryValidator:

    VALIDATION_ERROR_MARGIN = 2e-2   # Error must be below VALIDATION_ERROR_MARGIN

    @classmethod
    def calculate_difference(cls, u: List[float], x0: VehicleState, x1: VehicleState, dt: float, vehicle_model):
        """
        Simulates the next state for the given x0 state, and then takes its positional difference
        with the actual next state x1.

        :param u: Input
        :param x0: Current state
        :param x1: Next State
        :param dt: Delta time
        :param vehicle_model: Vehicle Model
        :return np.array: 2 length array of positional difference
        """

        # Remove array indexing for calculating the difference for all of the state fields
        return x1.values[:2] - x0.forward_simulation(u, dt, vehicle_model).values[:2]

    @classmethod
    def calculate_cost(cls, u: List[float], x0: VehicleState, x1: VehicleState, dt: float, vehicle_model):
        """
        Calculates the squared sum cost between simulated state and the actual next state x1.

        :param u: Input
        :param x0: Current state
        :param x1: Next State
        :param dt: Delta time
        :param vehicle_model: Vehicle Model
        :return float: Cost
        """
        difference = cls.calculate_difference(u, x0, x1, dt, vehicle_model)
        return np.sqrt(np.sum(np.square(difference*1000)))  # squared sum

    @classmethod
    def is_state_transition_feasible(cls, x0: VehicleState, x1: VehicleState, dt: float, vehicle_model):
        """
        Checks if the state transition is feasible between the given two states.

        :param x0: Current state
        :param x1: Next State
        :param dt: Delta time
        :param vehicle_model: Vehicle Model
        :return: Boolean
        """
        # print(hasattr(vehicle_model, 'width'))
        # print(hasattr(vehicle_model, 'length'))
        # print(hasattr(vehicle_model, 'shape'))
        # Minimize difference between simulated state and next state by varying input u
        u0 = np.array([0, 0])

        # print('u0_init:', u0)
        u0 = minimize(cls.calculate_cost, u0, args=(x0, x1, dt, vehicle_model),
                      options={'disp': False, 'ftol': 1e-6, 'maxiter': 100},
                      method='SLSQP', bounds=vehicle_model.input_bounds).x
        # print('u0:', u0)
        difference = cls.calculate_difference(u0, x0, x1, dt, vehicle_model)
        # print('difference:', difference)

        return all((all(np.abs(difference) < cls.VALIDATION_ERROR_MARGIN), vehicle_model.is_valid_input(u0)))

    @classmethod
    def is_trajectory_valid(cls, trajectory, vehicle_model, dt):
        """
        Checks if the given trajectory is valid for the given vehicle model and delta time.

        :param trajectory: Vehicle Trajectory
        :param vehicle_model: Vehicle Model
        :param dt: Delta time
        :return: Boolean
        """
        for x0, x1 in zip(trajectory.state_list[:-1], trajectory.state_list[1:]):
            if not cls.is_state_transition_feasible(x0, x1, dt, vehicle_model):
                # print(x1.time_step)
                return False
        return True


class VehicleModelException(Exception):
    pass


class VehicleModel(ABC):

    VEHICLE_MODEL_DICT = {
        'PM': lambda parameter, trajectory, input_vector: PointMassModel(parameter, trajectory, input_vector),
        'KS': lambda parameter, trajectory, input_vector: KinematicSingleTrackModel(parameter, trajectory, input_vector),
        'ST': lambda parameter, trajectory, input_vector: SingleTrackModel(parameter, trajectory, input_vector),
        'MB': lambda parameter, trajectory, input_vector: MultiBodyModel(parameter, trajectory, input_vector),
    }

    PARAMETERS_DICT = {
        1: parameters_vehicle1(),
        2: parameters_vehicle2(),
        3: parameters_vehicle3(),
    }

    def __init__(self, parameter_id: int, trajectory: VehicleTrajectory = None, input_vector: InputVector = None):
        if parameter_id not in self.PARAMETERS_DICT:
            raise VehicleModelException("Vehicle model is not defined! Vehicle Model ID:", parameter_id)

        self.id = self.type + str(parameter_id)
        self.parameters = self.PARAMETERS_DICT[parameter_id]
        self.width = self.parameters.w
        self.length = self.parameters.l
        self.shape = Rectangle(length=self.length, width=self.width)

        self.trajectory = trajectory
        self.input_vector = input_vector
        self.input_bounds = Bounds([self.parameters.steering.v_min, -self.parameters.longitudinal.a_max],
                                   [self.parameters.steering.v_max, self.parameters.longitudinal.a_max])

    @abstractproperty
    def type(self):
        pass

    @abstractproperty
    def trajectory_type(self):
        pass

    @abstractproperty
    def input_vector_type(self):
        pass

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory):
        if trajectory is not None and not isinstance(trajectory, self.trajectory_type):
            raise VehicleModelException("Invalid trajectory type %s for vehicle model type %s"
                                        % (type(trajectory), type(self)))
        self._trajectory = trajectory

    @property
    def input_vector(self):
        return self._input_vector

    @input_vector.setter
    def input_vector(self, input_vector):
        if input_vector is not None and not isinstance(input_vector, self.input_vector_type):
            raise VehicleModelException("Invalid input vector type %s for vehicle model type %s"
                                        % (type(input_vector), type(self)))
        self._input_vector = input_vector

    # @property
    # def input_bounds(self):
    #     return Bounds([self.parameters.steering.v_min, -self.parameters.longitudinal.a_max],
    #                   [self.parameters.steering.v_max, self.parameters.longitudinal.a_max])

    @classmethod
    def from_vehicle_id(cls, vehicle_id: str, trajectory: VehicleTrajectory = None, input_vector: InputVector = None):
        if not len(vehicle_id) == 3:
            raise VehicleModelException("Invalid vehicle model! Vehicle Model ID:", vehicle_id)

        vehicle_type = vehicle_id[:2]
        parameter_id = int(vehicle_id[2])

        if vehicle_type not in cls.VEHICLE_MODEL_DICT:
            raise VehicleModelException("Vehicle model is not defined! Vehicle Model ID:", vehicle_id)

        return cls.VEHICLE_MODEL_DICT[vehicle_type](parameter_id, trajectory, input_vector)

    @classmethod
    @abstractmethod
    def dynamics(cls, t, x, u, p):
        pass

    def create_trajectory_from_input_vector(self, init_state, dt, input_vector=None):
        vector = self.input_vector if input_vector is None else input_vector
        if vector is None:
            VehicleModelException('Input vector of the vehicle model is None.')

        return VehicleTrajectory.from_input_vector(init_state, dt, vector, self)

    def is_valid_input(self, u: np.ndarray):
        return all((np.abs(u[0]) <= self.parameters.longitudinal.a_max,
                    self.parameters.steering.v_min <= u[0] <= self.parameters.steering.v_max))

    def create_collision_object(self):
        if self.trajectory is None:
            raise VehicleModelException('Trajectory is not defined, could not create collision object!')

        return create_collision_object(TrajectoryPrediction(trajectory=self.trajectory, shape=self.shape))

    def has_valid_trajectory(self, dt):
        if self.trajectory is None:
            raise VehicleModelException('Trajectory is not defined, could not perform trajectory validation!')

        return TrajectoryValidator.is_trajectory_valid(self.trajectory, self, dt)


class PointMassModel(VehicleModel):

    type = 'PM'
    trajectory_type = PMTrajectory
    input_vector_type = PMInputVector
    supported_partial_costs = ['A', 'J', 'V', 'LC', 'L', 'T', 'TO', 'O', 'D', 'TG', 'Y']

    @property
    def input_bounds(self):
        return Bounds([-self.parameters.longitudinal.a_max, -self.parameters.longitudinal.a_max],
                      [self.parameters.longitudinal.a_max, self.parameters.longitudinal.a_max])

    # Note: Apply input constraints to the point mass model? Others apply it in dynamics TODO
    # "With the exception of the point-mass model, all vehicle models
    # respect steering and acceleration constraints." - Vehicle Model doc. p.2
    #
    # def apply_input_constraints(self, state_values, input_values):
    #     dir_a, dir_ao = self._convert_to_directed(input_values[0], input_values[1])
    #     dir_v, dir_vo = self._convert_to_directed(state_values[2], state_values[3])
    #     dir_a_constrained = accelerationConstraints(dir_v, dir_a, self.parameters.longitudinal)
    #     velocity_x, velocity_y = self._convert_from_directed(dir_a_constrained, dir_ao)
    #     return velocity_x, velocity_y

    @classmethod
    def dynamics(cls, t, x, u, p):
        return [
            x[2],  # velocity x
            x[3],  # velocity y
            u[0],  # input acceleration x
            u[1],  # input acceleration y
        ]

    def is_valid_input(self, u: np.ndarray):
        return all((np.abs(u[0]) <= self.parameters.longitudinal.a_max,
                    np.abs(u[1]) <= self.parameters.longitudinal.a_max))


class KinematicSingleTrackModel(VehicleModel):
    type = 'KS'
    trajectory_type = KSTrajectory
    input_vector_type = InputVector

    @classmethod
    def dynamics(cls, t, x, u, p):
        return vehicleDynamics_KS(x, u, p)


class SingleTrackModel(VehicleModel):
    type = 'ST'
    trajectory_type = STTrajectory
    input_vector_type = InputVector

    @classmethod
    def dynamics(cls, t, x, u, p):
        return vehicleDynamics_ST(x, u, p)


class MultiBodyModel(VehicleModel):
    type = 'MB'
    trajectory_type = MBTrajectory
    input_vector_type = InputVector

    @classmethod
    def dynamics(cls, t, x, u, p):
        return vehicleDynamics_MB(x, u, p)
