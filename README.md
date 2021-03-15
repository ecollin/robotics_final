# robotics_final
Introduction to robotics final project. Teaching the robot to play "pong" with reinforcement learning.

roscore

roslaunch robotics_final turtlebot3_pong.launch 

rosrun robotics_final ball_move.py

rosrun robotics_final play_commands.py


ball_move subscribes to the BallCommand thread and awaits a "send" command

ball_move publishes to the ball_state and ball_result threads and publishes the launch position and angle of the ball

Once the ball either scores a goal, misses the goal, or is deflected, ball_mobe publishes the reward, 10 for a save, and -10 for a goal

goals are detected by checking the ball position 5 seconds after launch


ball position is set using te set_model_state rospy service proxy

ball position is "get" using get_model_state rospy service proxy for detecting goal, or miss

on receiving a "send" command on the ballcommand thread, the ball_move node randomly places the ball on the midfeild line and aims toward the goal with a random angle near center of the goal.

play_commands.py setups up the publishers and subscribers for issuing ball send commands and reciving the ball position and state and the rewards. It then issues a ball send command and awaits a reward. this loops 10 times for now.




                  
