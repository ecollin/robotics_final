#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
import numpy as np
import math
from random import randint, random, uniform
from robotics_final.msg import BallCommand, BallResult, BallInitState

class Play_commands:

    def __init__(self):
        # once everything is setup initialized will be set to true
        self.initialized = False        
        rospy.init_node('play_commands')
        
        # Setup publishers and subscribers
        # subscribe to the lidar scan from the robot
        self.ball_command_pub = rospy.Publisher("robotics_final/BallCommand", BallCommand, queue_size=10)
        rospy.Subscriber("/robotics_final/ball_state", BallInitState, self.ball_state_received)
        rospy.Subscriber("/robotics_final/ball_result", BallResult, self.ball_result_received)

        self.initialized = True


    def ball_state_received(self, data: BallInitState):
        x = data.x
        y = data.y
        angle = data.angle
        print("ball starts at %2f , %2f @ %2f degrees" % (x, y, angle*180/math.pi))

    def ball_result_received(self, data: BallResult):
        self.reward = data.reward
        print("reward = %d" % self.reward)
        
    def run(self):
        rate = rospy.Rate(1)
        connections = self.ball_command_pub.get_num_connections()
        while connections < 1:
            rate.sleep()
            connections = self.ball_command_pub.get_num_connections()

        for i in range(10):
            self.reward = 0
            self.ball_command_pub.publish("send")
            while (self.reward == 0):
                rospy.sleep(1)
        
        rospy.spin()

        
if __name__=="__main__":
    node = Play_commands()
    node.run()








