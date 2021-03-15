#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist, Vector3
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
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class Play_commands:

    def __init__(self):
        # once everything is setup initialized will be set to true
        self.initialized = False        
        rospy.init_node('play_commands')
        
        # Setup publishers and subscribers
        # subscribe to the lidar scan from the robot
        self.ball_command_pub = rospy.Publisher("robotics_final/BallCommand", BallCommand, queue_size=10)
        self.twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/robotics_final/ball_state", BallInitState, self.ball_state_received)
        rospy.Subscriber("/robotics_final/ball_result", BallResult, self.ball_result_received)
        rospy.Subscriber("scan", LaserScan, self.robot_scan_received);
        
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Create a default twist msg (all values 0)
        lin = Vector3()
        ang = Vector3()
        self.twist = Twist(linear=lin, angular=ang)
        self.moving = False
        self.initialized = True

    def robot_scan_received(self, data):
        dat_range = data.ranges[180:]
        range_data = np.array(dat_range) #convert to numpy array
        minval = min(range_data) # closest point in space
        mindir = np.argmin(range_data) # find direction to ball if found
        if minval < 10:
            print("ball is at %2f , %2f" % (minval,-45 + mindir))
    
        
    def move_goalie(self):
        state = self.get_state('soccer_ball','world')
        ball_y = state.pose.position.y  
        state = self.get_state('turtlebot3_waffle_pi','world')
        robot_y = state.pose.position.y
        kp_lin = .25
        dist = ball_y - robot_y
        self.twist.linear.x = kp_lin * dist
        self.twist_pub.publish(self.twist)

    def reset_goalie(self):
        state = ModelState()
        state.model_name = 'turtlebot3_waffle_pi'
        state.reference_frame = 'world'  # ''ground_plane'
        # pose x -5.8 y 3.4 z 0 yaw 1.570796
        state.pose.position.x = -5.8
        state.pose.position.y = 3.4
        state.pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 1.570796)
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

        # self.move_goalie()
        
        for i in range(10):
            self.reward = 0
            self.ball_command_pub.publish("send")
            while (self.reward == 0):
                self.move_goalie()
                rospy.sleep(1)
            self.twist_pub.publish(Twist()) # stop moving
            self.reset_goalie() # reset goalie position
        
        rospy.spin()

        
if __name__=="__main__":
    node = Play_commands()
    node.run()








