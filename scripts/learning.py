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




class Learn:
    def __init__(self):
        self.initialized = False
        rospy.init_node('learning_algorithm')
        # subscribe to Ball_state, ball_result
        rospy.Subscriber("/robotics_final/ball_state", BallInitState, self.ball_state_received)
        rospy.Subscriber("/robotics_final/ball_result", BallResult, self.ball_result_received)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.current_state = None
        self.state_num = 0
        self.current_reward = None
        self.reward_num = 0
        self.iter_num = 1
        self.initialized = True

    def ball_state_received(self, data):
        print("Ball's initial state received")
        self.current_state = data
        self.state_num += 1

    def ball_result_received(self, data):
        print("Reward Received")
        self.current_reward = data
        self.reward_num += 1

    def algorithm(self):
        if self.reward_num == self.state_num == self.iter_num:
            ## we have a correct state, reward pair
            robot_state = self.get_state('turtlebot3_waffle_pi','world')
            print("Robot state:")
            print(robot_state)
            self.iter_num += 1

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = Learn()
    while(1):
        node.algorithm()
    node.run()
