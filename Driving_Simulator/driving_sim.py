import numpy as np
import random

class RobotCar:
    def __init__(self, length=20.0):

        self.length = length
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.drift = 0.0

    def set_robot(self, x_i, y_i, theta_i):
        """
        Sets the initial position of the robot
        :param x_i:
        :param y_i:
        :param theta_i:
        :return:
        """

        self.x = x_i
        self.y = y_i
        self.orientation = theta_i % (2.0 * np.pi)

    def set_noise(self, steer_noise, dist_noise, robot_drift):
        """
        Set noise and drift parameters for the robot
        :param steering_noise:
        :param distance_noise:
        :param robot_drift:
        :return:
        """
        self.steering_noise = steer_noise
        self.distance_noise = dist_noise
        self.drift = robot_drift

    def move(self, theta, dist, tolerance=0.001, max_angle=np.pi/4.0):
        """
        :param pos: total distance driven
        :param theta: front wheel steering angle
        :param max_angle: limit of the front wheel steering angle
        :return:
        """

        # Trim steering
        if theta > max_angle:
            theta = max_angle
        elif theta < -max_angle:
            theta = -max_angle
        if dist < 0.0:
            dist = 0.0

        # Apply Noise and Drift
        steering = random.gauss(theta, self.steering_noise) + self.drift
        distance = random.gauss(dist, self.distance_noise)

        # Execute Motion
        turn = np.tan(steering) * distance / self.length  # Derived from Ackerman's formula

        if abs(turn) < tolerance:
            # Approximation as straight-line motion
            self.x += distance*np.cos(self.orientation)
            self.y += distance*np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # Approximation of bicycle motion
            radius = distance/turn
            cx = self.x - (np.sin(self.orientation)*radius)
            cy = self.y + (np.cos(self.orientation)*radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation)*radius)
            self.y = cy - (np.cos(self.orientation)*radius)

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)
