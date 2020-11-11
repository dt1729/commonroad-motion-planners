class StartState:
    """
    class to store start state of a motion primitive
    """
    def __init__(self, x, y, steering_angle, velocity, orientation, time_step = 0):
        """
        Initialisation of class StartState.

        :param x: the position in x axis
        :param y: the position in y axis
        :param steering_angle: the steering angle of vehicle
        :param velocity: the velocity of the state
        :param orientation: the orientation of the final state
        """
        self.x = x
        self.y = y
        self.steering_angle = steering_angle
        self.velocity = velocity
        self.orientation = orientation
        self.time_step = time_step

    def __str__(self) -> str:
        """
        Returns the velocity information of StartState.
        """
        return "pos: ({}, {})m, vel: {} m/s, ori: {} rad, steer: {} rad".format(round(self.x, 2),
                                                                                round(self.y, 2),
                                                                                round(self.velocity, 2),
                                                                                round(self.orientation, 2),
                                                                                round(self.steering_angle, 2))

class FinalState:
    """
    class to store final state of a motion primitive
    """
    def __init__(self, x, y, steering_angle, velocity, orientation, time_step = 0):
        """
        Initialisation of class StartState.

        :param x: the position in x axis
        :param y: the position in y axis
        :param steering_angle: the steering angle of vehicle
        :param velocity: the velocity of the state
        :param orientation: the orientation of the final state
        """
        self.x = x
        self.y = y
        self.steering_angle = steering_angle
        self.velocity = velocity
        self.orientation = orientation
        self.time_step = time_step

    def __str__(self) -> str:
        """
        Returns the velocity, orientation and position information of the final state.
        """
        return "pos: ({}, {})m, vel: {} m/s, ori: {} rad, steer: {} rad".format(round(self.x, 2),
                                                                                round(self.y, 2),
                                                                                round(self.velocity, 2),
                                                                                round(self.orientation, 2),
                                                                                round(self.steering_angle, 2))