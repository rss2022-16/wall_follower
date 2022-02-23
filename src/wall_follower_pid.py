#!/usr/bin/env python2

# This method only makes decisions in the moment, doesn't care about observation history
# or future predictions of trajectory

# How to deal with pure pursuit edge cases?

import numpy as np

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_tools import *

class WallFollower:
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT
    SCAN_TOPIC = rospy.get_param("wall_follower/scan_topic")
    DRIVE_TOPIC = rospy.get_param("wall_follower/drive_topic")
    SIDE = rospy.get_param("wall_follower/side")
    VELOCITY = rospy.get_param("wall_follower/velocity")
    DESIRED_DISTANCE = rospy.get_param("wall_follower/desired_distance")

    ## Detector params
    LINE_DETECTOR_MIN = 0
    LINE_DETECTOR_MAX = np.pi/2
    LINE_DETECTOR_DIST_X = 3.25 # Meters
    LINE_DETECTOR_DIST_Y = 2.75
    WALL_TOPIC = "/wall"
    L = .15 # Length of vehicle

    ## PID params
    KP =  2 * (L / VELOCITY)
    KD = .33
    KI = 0

    KP_Theta = 2 * (L / VELOCITY)

    DT = 0.05

    def __init__(self):

        # Pub and sub
        self.drive_pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        self.line_pub = rospy.Publisher(self.WALL_TOPIC, Marker, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_cb)

        # PID Stuff
        self.dist = 0
        self.dist_integral = 0
        self.last_dist = None
        self.theta = 0

        self.lost = False

        rospy.Timer(rospy.Duration(self.DT), self.cb)
        rospy.on_shutdown(self.hook)

    def scan_cb(self, scan):
        """
        Receives LaserScan data and uses linear regression to generate a line to follow.
        Selects a sub-region of laser scan data depending on value of self.SIDE
        """
        # Convert to cartesian
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, num=len(scan.ranges))
        cartesian = np.stack( (np.multiply(ranges,np.cos(angles)), \
            np.multiply(ranges,np.sin(angles))), axis=1)

        # Get rid of : |x| > max_dist, |y| > max_dist, y > zero for right or y < zero for left
        bad_indices = [i for i in range(len(cartesian)) if abs(cartesian[i,0]) >= self.LINE_DETECTOR_DIST_X \
            or abs(cartesian[i,1]) >= self.LINE_DETECTOR_DIST_Y or self.SIDE*cartesian[i,1] <= 0]
        cartesian = np.delete(cartesian, bad_indices, axis=0)
        
        # Try looking ahead, but if we don't see anything, look behind
        try:
            less_than_zero_indices = [i for i in range(len(cartesian)) if cartesian[i,0] <= 1]
            cartesian_final = np.delete(cartesian, less_than_zero_indices, axis=0)
            [m, b] = np.polyfit(cartesian_final[:,0], cartesian_final[:,1], 1)
        except:
            try:
                greater_than_zero_indices = [i for i in range(len(cartesian)) if cartesian[i,0] >= 1]
                cartesian_final = np.delete(cartesian, greater_than_zero_indices, axis=0)
                [m, b] = np.polyfit(cartesian_final[:,0], cartesian_final[:,1], 1)
            except:
                self.lost = True
                return

        self.dist = -1 * (b + m - self.SIDE * self.DESIRED_DISTANCE)
        self.theta = -np.arctan(m)

        self.lost = False
        back_x = min(cartesian[:,0])
        x = np.linspace(back_x, float(self.LINE_DETECTOR_DIST_X), num=20)
        y = m * x + b - self.SIDE * self.DESIRED_DISTANCE
        VisualizationTools.plot_line(x, y, self.line_pub, frame="/laser")

    def cb(self, event):
        """
        Main callback running purepursuit. 
        Calculates turning angle at each timestep, using the most recently parsed scan data.
        """
        ## Control law
        if self.lost:
            eta = self.SIDE * .15
        else:
            eta = self.pid()

        ## Publish msg
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = eta
        msg.drive.speed = self.VELOCITY
        self.drive_pub.publish(msg)

    def pid(self):
        self.dist_integral += self.DT * self.dist
        if self.last_dist is not None:
            dist_derivative = (self.dist - self.last_dist) / self.DT
        else:
            dist_derivative = self.dist / self.DT
        self.last_dist = self.dist
        
        P = self.KP * self.dist
        I = self.KI * self.dist_integral
        D = self.KD * dist_derivative
        P_Theta = self.KP_Theta * self.theta

        eta = -1 * (P + I + D + P_Theta)

        return eta

    def hook(self):
        """
        Safe shutdown behavior. Send 0 commands to vehicle
        """
        # Publish msg
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = 0
        msg.drive.speed = 0
        self.drive_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.spin()