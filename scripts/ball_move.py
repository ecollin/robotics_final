#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import numpy as np
import math
from random import randint, random, uniform

import constants as C

from robotics_final.msg import BallCommand, BallResult, BallInitState

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""
    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])
    return yaw


class BallMove:

    def __init__(self):
        # once everything is setup initialized will be set to true
        self.initialized = False        
        # initialize this particle filter node
        rospy.init_node('ball_move')

        self.base_frame = "base_footprint"
        self.map_topic = "map"
        self.odom_frame = "odom"
        self.scan_topic = "scan"
        
        # Setup publishers and subscribers
        # subscribe to the lidar scan from the robot
        #rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)
        rospy.Subscriber("/robotics_final/BallCommand", BallCommand, self.command_received)
        self.ball_state_pub = rospy.Publisher("robotics_final/ball_state", BallInitState, queue_size=10)
        self.ball_res_pub = rospy.Publisher("robotics_final/ball_result", BallResult, queue_size=10)

        #setup the soccer field parameters
        self.south_goal_line = 0.415
        self.north_goal_line = 6.36
        self.mid_field_y = 3.4
        self.mid_field_x = -1.8
        self.mid_goal_x = -6.23
        self.mid_goal_y = 3.45
        self.ball_velocity = 2
        self.last_ball_x = float('inf')
        
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.initialized = True


    def command_received(self, data: BallCommand):
        if (data.command == "send"):
            self.send_ball()
        else:
            print("error: unknown command, ball_move/command_received")
        
        
    def robot_scan_received(self, data):
        # wait until initialization is complete
        if not(self.initialized):
            return
        
   
    def set_start_ball(self, x, y, theta, v):
        # help from: https://www.programcreek.com/python/?code=marooncn%2Fnavbot%2Fnavbot-master%2Frl_nav%2Fscripts%2Fenv.py
        state = ModelState()
        state.model_name = 'soccer_ball'
        state.reference_frame = 'world'  # ''ground_plane'
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
        state.twist.linear.x = math.cos(theta)*v
        state.twist.linear.y = math.sin(theta)*v
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

    def set_random_ball_state(self):
        ball_y = self.south_goal_line+(self.north_goal_line-self.south_goal_line)*uniform(0,1)
        ball_x = self.mid_field_x
        #print(f"Initializing random ball at x,y {ball_x} {ball_y} ")
        self.last_ball_x = float('inf')
        dy = ball_y - self.mid_goal_y
        dx = ball_x - self.mid_goal_x
        ball_angle = math.pi+math.atan(dy/dx) + uniform(-1,1)*math.pi/20
        self.set_start_ball(ball_x, ball_y, ball_angle, self.ball_velocity)
        ball_state = BallInitState()
        ball_state.x = ball_x
        ball_state.y = ball_y
        ball_state.angle = ball_angle
        self.ball_state_pub.publish(ball_state)
        
    def compute_reward(self):
        state = self.get_state('soccer_ball','world')
        robot_state = self.get_state('turtlebot3_waffle_pi','world')
        #print(f'robot state in compute reward, x,y: {robot_state.pose.position.x} {robot_state.pose.position.y} ')

        IN_GOAL_REWARD = -100
        ROBOT_HIT_REWARD = 100
        MISSED_GOAL_REWARD = 50
        STILL_MOVING_REWARD = 0
        curr_ball_x = state.pose.position.x
        curr_ball_y = state.pose.position.y
        #print(f"before returning reward ball at x,y {curr_ball_x} {curr_ball_y}")
        if (curr_ball_x < C.GOAL_RIGHT and curr_ball_x > C.GOAL_LEFT
            and curr_ball_y < C.GOAL_TOP and curr_ball_y > C.GOAL_BOTTOM):
            print('Returning IN_GOAL_REWARD')
            self.last_ball_x = curr_ball_x
            return IN_GOAL_REWARD
        # check if ball went wide of the goal, 
        if (curr_ball_x < C.GOAL_RIGHT and 
            (curr_ball_y > C.GOAL_TOP or curr_ball_y < C.GOAL_BOTTOM)):
            print('Returning MISSED_GOAL_REWARD')
            self.last_ball_x = curr_ball_x
            return MISSED_GOAL_REWARD
        # Check if the ball has been hit by the robot and is therefore moving in opposite direction as 
        # when the last reward was computed, or if it is just still heading towards the robot.
        tolerance = 0.1
        if curr_ball_x > self.last_ball_x or np.isclose(curr_ball_x, self.last_ball_x, tolerance):
            # The robot hit the ball and reversed its direction
            print('Returning ROBOT_HIT_REWARD')
            self.last_ball_x = curr_ball_x
            return ROBOT_HIT_REWARD
        if (self.TIMES_UP == 1):
            print('Returning ROBOT_HIT_REWARD - times up')
            return MISSED_GOAL_REWARD
        else:
            print('Returning STILL_MOVING_REWARD')
            self.last_ball_x = curr_ball_x
            return STILL_MOVING_REWARD
        
    def send_ball(self):
        self.set_random_ball_state()
        self.TIMES_UP = 0 # timer to check for missed goal
        SLEEP_TIME = 5
        for _ in range(C.NUM_POS_SENDS - 1):
            rospy.sleep(SLEEP_TIME / C.NUM_POS_SENDS)
            reward = self.compute_reward()
            self.ball_res_pub.publish(reward)
        self.TIMES_UP = 1 # note that time is up, so either it's in the goal or not
        rospy.sleep(SLEEP_TIME / C.NUM_POS_SENDS)
        reward = self.compute_reward()
        self.ball_res_pub.publish(reward)
    
    def reset_goalie(self):
        state = ModelState()
        state.model_name = 'turtlebot3_waffle_pi'
        state.reference_frame = 'world'  # ''ground_plane'
        """
        Original version o below:
        # pose x -5.8 y 3.4 z 0 yaw 1.570796
        state.pose.position.x = -5.8
        state.pose.position.y = 3.4
        state.pose.position.z = 0
        quaternion = quaternion_from_euler(0, 0, 1.570796)
        """
        state.pose.position.x = C.GOALIE_X
        # Goalie needs to be in front of goal, not in middle
        state.pose.position.y = (C.GOAL_TOP + C.GOAL_BOTTOM) / 2
        state.pose.position.z = 0
        quaternion = quaternion_from_euler(0, 0, math.pi / 2)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")



    def run(self):
        rate = rospy.Rate(1)
        connections = self.ball_state_pub.get_num_connections()
        while connections < 1:
            rate.sleep()
            connections = self.ball_state_pub.get_num_connections()

        rate = rospy.Rate(1)
        NUM_SENDS = 1000
        for _ in range(NUM_SENDS):
            self.send_ball()
            print("Resetting goalie position for next iteration")
            self.reset_goalie()
        print("DONE")


        
if __name__=="__main__":

    bm = BallMove()
    bm.run()








