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
import random
from robotics_final.msg import BallCommand, BallResult, BallInitState, RobotAction

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion


import constants as C

class Learn:

    # Variables representing the actions
    MOVE_LEFT = 0
    STAY_PUT = 1
    MOVE_RIGHT = 2


    # field goes from (-7, 0) to (-1.8, 7)
    FIELD_XLEFT = -7.0
    FIELD_DX = 7.0-1.8
    FIELD_DY = 7.0
    RESOLUTION = 1 # side length of square (m)
    BOXES_X = int(np.ceil(FIELD_DX/RESOLUTION))
    BOXES_Y = int(np.ceil(FIELD_DY/RESOLUTION))
    NUM_STATES = (BOXES_X*2-1)*(BOXES_Y*2-1)
    print("Resolution =",RESOLUTION,"boxes/meter")
    print("X boxes:", BOXES_X)
    print("Y boxes:", BOXES_Y)
    print("Number of states:", NUM_STATES)

    def __init__(self):
        rospy.init_node('learning_algorithm')
        # subscribe to Ball_state, ball_result
        rospy.Subscriber("/robotics_final/ball_state", BallInitState, self.ball_state_received)
        rospy.Subscriber("/robotics_final/ball_result", BallResult, self.ball_result_received)
        self.action_pub = rospy.Publisher("robotics_final/robot_action", RobotAction, queue_size=10)

        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.Q = np.zeros((Learn.NUM_STATES, 3), dtype=int)
        self.count = 0
        self.state_num = 0
        self.reward_num = 0
        self.reward = None

    def get_state_num(self):
        # mapping the robot/ball orientation to a number
        robot_state = self.get_state('turtlebot3_waffle_pi','world')
        ball_state = self.get_state('soccer_ball','world')
        # each object is in a "box" that is RESOLUTION meters wide.
        robot_xbox = np.ceil((robot_state.pose.position.x-Learn.FIELD_XLEFT)/Learn.RESOLUTION)
        robot_ybox = np.ceil(robot_state.pose.position.y/Learn.RESOLUTION)
        ball_xbox = np.ceil((ball_state.pose.position.x-Learn.FIELD_XLEFT)/Learn.RESOLUTION)
        ball_ybox = np.ceil(ball_state.pose.position.y/Learn.RESOLUTION)
        # the state is the combination of dx and dy.
        dx = int(ball_xbox - robot_xbox)
        dy = int(ball_ybox - robot_ybox)
        #adjusting so I no longer have negative values
        dx += Learn.BOXES_X-1
        dy += Learn.BOXES_Y-1
        #converting to unique number between 0 and NSTATES-1:
        return (2*Learn.BOXES_Y-1)*dy+dx

    def ball_state_received(self, data):
        print("Ball's initial state received")
        self.current_state = data
        self.state_num += 1

    def ball_result_received(self, data):
        print(f"Reward {self.reward_num} Received, val {data.reward}")
        self.reward = data.reward
        self.reward_num += 1


    def apply_action(self, action):
        if action == Learn.MOVE_LEFT:
            print("Sending action to move left")
            self.action_pub.publish(C.ACTION_MOVE_LEFT)
        elif action == Learn.MOVE_RIGHT:
            print("Sending action to move right")
            self.action_pub.publish(C.ACTION_MOVE_RIGHT)
        else:
            print("Staying put")


    def algorithm(self):
        threshold = 50
        alpha = 1
        gamma = 0.5
        while self.count < threshold:
            print('------\nIterations without update:', self.count, '/', threshold)
            # select a possible action (any of them)
            s = self.get_state_num()
            print("Initial state:", s)
            a = random.choice(np.arange(3))
            self.apply_action(a)
            while self.reward == None:
                print("Sleeping to wait for reward")
                rospy.sleep(1)
            reward = self.reward
            self.reward = None
            next_state = self.get_state_num()
            mx = np.amax(self.Q[next_state])
            update = self.Q[s][a] + alpha*(reward+gamma*mx-self.Q[s][a])
            if self.Q[s][a] != update:
                print("Update Q matrix")
                self.Q[s][a] = update
                print(self.Q)
                self.count = 0
            else:
                self.count += 1

        robot_state = self.get_state('turtlebot3_waffle_pi','world')


    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = Learn()
    node.algorithm()
    node.run()
