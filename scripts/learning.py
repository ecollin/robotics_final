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
from robotics_final.msg import BallCommand, BallResult, BallInitState


import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

np.set_printoptions(threshold=np.inf)


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
    RESOLUTION = 0.4 # side length of square (m)
    BOXES_X = int(np.ceil(FIELD_DX/RESOLUTION))
    BOXES_Y = int(np.ceil(FIELD_DY/RESOLUTION))
    NUM_STATES = (BOXES_X*2-1)*(BOXES_Y*2-1)
    print("Resolution =",RESOLUTION,"boxes/meter")
    print("X boxes:", BOXES_X)
    print("Y boxes:", BOXES_Y)
    print("Number of states:", NUM_STATES)

    def __init__(self):
        rospy.init_node('learning_algorithm')
        # Setup listeners
        rospy.Subscriber("/robotics_final/ball_result", BallResult, self.ball_result_received)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.Q = np.zeros((Learn.NUM_STATES, 3), dtype=int)
        self.count = 0
        self.state_num = 0
        self.reward_num = 0
        self.reward = None
        self.move_dist = 0.5

    def get_state_num(self):
        """
        This function uses self.get_state to find the locations of the robot
        and ball and returns a number in [0, NUM_STATES) representing that state
        """
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
        # adjusting to remove negative values for states
        dx += Learn.BOXES_X-1
        dy += Learn.BOXES_Y-1
        # converting to unique number between 0 and NSTATES-1:
        return (2*Learn.BOXES_X-1)*dy+dx


    def ball_result_received(self, data):
        """
        Handler for reward received event. Stores the reward on the class
        """
        #print(f"Reward {self.reward_num} Received, val {data.reward}")
        self.reward = data.reward
        self.reward_num += 1

    def set_robot(self, x, y):
        """
        Given (x, y) coordinates for the gazebo world, moves the turtlebot
        to that location using self.set
        Reference:
        https://www.programcreek.com/python/?code=marooncn%2Fnavbot%2Fnavbot-master%2Frl_nav%2Fscripts%2Fenv.py
        """
        state = ModelState()
        state.model_name = 'turtlebot3_waffle_pi'
        state.reference_frame = 'world'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")



    def apply_action(self, action):
        """
        Given an action in (self.MOVE_LEFT, self.STAY_PUT, self.MOVE_RIGHT],
        performs that action by moving the turtlebot accordingly.
        """
        robot_state = self.get_state('turtlebot3_waffle_pi','world')
        robot_x = robot_state.pose.position.x
        robot_y = robot_state.pose.position.y
        # Set the distance moved in an action such that it is at least as large as the
        # minimum distance that would let a robot in the middle of the goal go to either side
        #self.move_dist = max(((C.GOAL_TOP + C.GOAL_BOTTOM) / 2) / C.NUM_POS_SENDS, 0.5)
        if action == Learn.MOVE_LEFT:
            print("Move left")
            self.set_robot(robot_x, robot_y+self.move_dist)
        elif action == Learn.MOVE_RIGHT:
            print("Move right")
            self.set_robot(robot_x, robot_y-self.move_dist)
        else:
            print("Stay put")

    def algorithm(self):
        """
        Peform the QLearning algorithm until convergence of self.Q
        """
        convergence_threshold = 50
        reward_num_threshold = 300
        alpha = 1
        gamma = 0.5
        while (self.reward_num < reward_num_threshold) and (self.count<convergence_threshold):
            print('------')
            print('Iteration', self.reward_num, '/', reward_num_threshold)
            print('Iterations w/out Q-update:', self.count, '/', convergence_threshold)
            # select a possible action (any of them; all are valid)
            s = self.get_state_num()
            print("Initial state:", s)
            a = random.choice(np.arange(3))
            self.apply_action(a)
            while self.reward == None:
                #print("Sleeping to wait for reward")
                rospy.sleep(0.5)
            reward = self.reward
            print("REWARD =", reward)
            self.reward = None
            if reward == 0:
                next_state = self.get_state_num()
                mx = np.amax(self.Q[next_state])
            else:
                ## There is no next state if nonzero reward seen
                mx = 0
            update = self.Q[s][a] + alpha*(reward+gamma*mx-self.Q[s][a])
            if self.Q[s][a] != update:
                print("Update Q matrix")
                self.Q[s][a] = update
                self.count = 0
            else:
                self.count += 1

        print("Finished calculating Q-Matrix\n\n\n\n\n\n\n")


    def execute_best_actions(self):
        """
        Call after self.Q has converged to, on an infinite loop,
        get the current state and perform the action in that state with
        highest q-value.
        """
        while True:
            print("In execute_best_actions")
            s = self.get_state_num()
            qvals = self.Q[s]
            # Get action with largest qval
            best_action = np.argmax(qvals)
            # We don't actually update with rewards,
            # but use them to know when to perform next action
            # We want to travel 0.5 m in action's direction.
            self.apply_action(best_action)
            while self.reward == None:
                rospy.sleep(0.5)
            print("Reward =", self.reward)
            self.reward = None


if __name__ == "__main__":
    node = Learn()
    node.algorithm()
    node.execute_best_actions()
