![simple goalie gif](https://github.com/ecollin/robotics_final/blob/main/goalierobot2.gif)

# robotics_final
Introduction to robotics final project. Teaching the robot to play "pong" with reinforcement learning.

# Robot Soccer/Pong Writeup
Katie Hughes, Elizabeth Singer, Enrique Collin

## How to run:
Terminal 1: roscore
Terminal 2: roslaunch robotics_final turtlebot3_pong.launch
Terminal 3: rosrun robotics_final ball_move.py
Terminal 4: rosrun robotics_final learning.py

## Project Description
Our project was inspired by both the idea of a robot goalie and the game of pong, where the turtlebot must stop a launched ball from entering a goal that is being defended. We wanted to make a robot that could move to the location of an incoming ball and stop it from entering the goal/hitting the area behind the robot. We aimed to do this using some form of a reinforcement learning algorithm. What makes this project interesting and different from what we've done in the past is that it involves a large, continuous state space of the ball pose (position, orientation, and velocity) on the field, as well as the pose of the robot. In the last QLearning project, there was a small, finite number of states which could be exhaustively searched. But in our goalie project, the soccer ball can come from any location along the middle line of the field we created, and at any angle, which is at the very least a computationally intractable number of states for performing traditional QLearning on a standard computer. We thus had to find an approach that was still using reinforcement learning but that reduced the overall complexity required.

Our project has 3 main components. 
#### Gazebo world
The first component is the Gazebo world we made. None of us had any experience creating a world in Gazebo, and we needed one on which we could arrange to automatically create and launch balls in a direction that the robot could attempt to block. The other components of the project necessarily build on this, because without a world, it's difficult to get into the specifics of how to implement the rest of the project.

#### Reinforcement learning
After much research into various reinforcement learning and adaptive control approaches, we decided that we would use a form of the simple QLearning algorithm with one fundamental change to make it consistent with the continuous nature of the world: we would incorporate a function that would map the continuous state of the world (robot and ball position and velocity) into a discrete number of states.  The methods that we used for this mapping through discretization are discussed in the next section). 

#### Discretization and Mapping
Mapping the continuous state space into a finite number of possible, learnable, states is a fundamentally important part of this project. We needed to be able to classify the states of our world using some combination of the robot position and the ball position. There are a number of different ways that we can discretize the state of the system. Our solution was to create a mapping from the continuous-valued x,y position of the ball to an integer-valued state by discretizing the area of the field into individual, equal sized squares, or pixels. Given the continuous-valued x,y position of the ball, this position can be mapped to a quantized, integer-valued x and y position. Similarly, the position of the robot can be mapped to a quantized, integer-valued position. The state of the overall learning algorithm can then use these two discretized, integer quantities, to index the current state.  Parameters of this discretization include the x and y resolution used in the mapping, which govern the number of states: a finer resolution leads to a more accurate model for the position of the robot and ball, but leads to more states in the Q matrix. More states in the matrix requires a longer time (more interactions) for the learning algorithm to converge. Not only does the learning algorithm have to visit more of the states, but the rewards need to propagate through these states in the learning updates. So there is a tradeoff between accuracy and learning time for the algorithm.

## System Architecture

#### Building Gazebo World

Our Gazebo world uses a set of models that are in the Gazebo model list to construct the playing field. We used a built-in soccer field model, a soccer-ball model, and two goal models to create the world that is used in the launch file. The robot always begins sitting in front of the same goal, which is controlled by the launch file, setting the turtlebot position and orientation. This same pose is used in the learning and training algorithm in a goalie reset function, which, after a ball has been successfully defended or a goal has been scored and reward issued, moves the goalie back to its original starting point.  The ball is launched at the goal with an initial velocity of 1.75 m/s from various positions across the midline. This velocity is a parameter of the ball_move.py function. Similarly, the ball can be launched from a random y position along the midline (x) of the field at an angle that is toward the goal, with a random dither in the angle. For training purposes, we also can (and did) restrict the number of launch positions by discretizing the y positions from which the ball can be launched. After a certain amount of time has passed, and the ball has either reached the goal, missed the goal, or stopped, the world resets with the robot in the same starting position and the ball launching from another random position across the midline. This behavior is defined in ball_move.py. We set the ball and robot initial states using rospy service proxy calls to get_model_state and set_model_state, which can be used to automatically locate and automatically move models in Gazebo. We set a random trajectory for the ball in set_random_ball_state(). This choses a random starting position somewhere directly across from the goal and passes it onto set_start_ball(), which physically moves the model in Gazebo. 

Rewards are published using custom messages at two instances during each kick of the ball. The first publication comes when the ball is about halfway between the goal and its starting point. This reward is simply 0, as the ball is still traveling towards the goal. The second publication comes when the ball has passed the robot. At this point, either the robot has blocked the ball (reward +100), the ball has missed the goal (reward +50), or the ball has entered the goal (reward -100). The position of the ball at each time can be accessed using the get_model_state service proxy. This logic is contained in the function compute_reward() in the ball_move.py script. To check whether or not the ball is in the goal,  we compare the position of the ball to the coordinates that the goal covers. That is, if the position of the ball after the kick is finished is within the four corners of the goal interior, then a goal has been scored. The net keeps the ball in this region if indeed a goal has been scored. If the ball is not in the goal, it is out of the goal, and no goal is scored. If the position of the ball is seen to have moved left to right, then it has been deflected by the goalie, and the system can be reset. If the ball is still moving to the left, but not in the goal, then the timer will detect when the kick is reset. This is also how we detect if the ball went wide of the goal. If enough time has passed so that the ball should be in one of these locations (indicated by TIMES_UP=True), we determine that it was a robot hit. In all other cases, we return the reward of 0, which indicates that the ball is still on its way to the goal. When running ball_move.py, the ball gets launched 1000 times, which is enough for convergence for the situations we tested with a constrained number of launch positions and a coarsely quantized field.  A larger number of launch positions and a finer discretization would require more training for convergence.

#### Messages
The messages that are used in this project include Gazebo messages for ModelState, GetModelState, and SetModelState. These are used to get and set the locations of the ball and the robot to both reset the world after a shot on goal, and to set the initial launch position and velocity and angle of the ball for each kick, as well as to set the position of the robot when taking the appropriate actions during the learning phase of the algorithm.  We also created messages BallCommand, BallReslult, and BallInitState so that the learning algorithm could request ball launches from the ball_move.py node, receive reward results from the ball_move node, and reset the goalie after finishing. To construct these custom messages, we created the appropriate .msg files, modified the CMakeLists file and the .xml file and performed a catkin_make. This enabled communication between the nodes not only for the final code, but also for testing during development.

#### initialization
The initialization in the ball_move.py node setup the node, created publishers for the ball_state and ball_result, and subscriber for the BallCommand threads. It also defined constants used in the node, including the locations of the four corners of the soccer field, the four corners of the soccer goal, the ball velocity, and the mid-field line.  Service proxy objects were also created for get_model_state and set_model_state to access the models in Gazebo to facilitate the learning process. Once initialized, the self.initialized variable was set to True.

#### Command_received
Command_received was used to respond to BallCommand message threads. If the data string in the message was “send” then a call to send_ball() as issued, otherwise, an error returned

#### set_start_ball(self, x, y, theta, v)
The function set_start_ball() was used to put a ball model on the soccer field at a specific location and moving at an initial velocity v, at angle theta. Using the message ModelState(), a state object is created, its model name is set to ‘soccer_ball’, in the reference frame ‘world’ and its pose is set to the x,y position and its twist is set to cos(theta)*v and sin(theta)*v to launch the ball at the requested velocity in the requested direction. 

#### set_random_ball_state()
The function set_random_ball_state placed the x and y coordinates of the ball to a random position on the midfield line. This was initially created using a uniform random variable, scaled by the field width, to put the ball at any position. To facilitate the q_learning, we changed this uniform random variable to a discrete random variable using randint, which let us select a smaller number of discrete positions and orientations, For example, using randint(3,4)/6 placed the ball at one of 2 positions: midfield, and 1/6th of the way north of midfield. Using randint(0,5)/6 would place the ball at one of 6 discrete positions across midline.  The ball is then set in motion by calling set_start_ball, and the ball initial state is published on the ball_state thread.

#### compute_reward()
The compute reward function has constants for -100 for a ball in goal, 100 for a robot deflecting the ball and 50 for the ball going wide of the goal, and 0 for the ball still in motion toward the goal. To compute the reward, the current position of the ball is obtained from a call to the get_state method in the soccer ball state, using  self.get_state(‘soccer_ball’,’world’). The robot state is similarly computed using the get_state method on the robot_state object. Checking to see if the ball position is within the goal, if so, then an IN_GOAL_REWARD of 100 is returned. The last_ball position is set to the current position before returning the reward. If the ball is outside the play area, then a MISSED_GOAL_REWARD is returned, and if the ball has moved to the right since the last recorded ball position, then a ROBOT_HIT_REWARD is issued. If none of these occur, and the TIMES_UP flag is set by a system timer, since the ball launched, then the ROBOT_HIT_REWARD is issued since it must have been deflected, otherwise it would be in goal, or out of bounds. If time is not up, then the ball is still moving and a STILL_MOVING_REWARD of zero is returned and the ball position is recorded in the last_ball variable.

#### send_ball()
Send_ball is a method within the ball_move node that manages sending the soccer ball. 
The function sets the random state in motion with set_random_ball_state() and uses a constant SLEEP_TIME (set to 4 seconds for example), that governs how long to wait before declaring the kick over. Using the parameter NUM_POS_SENDS (which governs how many times to check for reward during a kick), the function computers the reward with compute_reward(), publishes the reward, and if the reward is nonzero, keeps waiting for the reward to be non-zero or the timer to finish. Once the reward is nonzero, the routine is complete. If the timer finishes without a reward in the loop, then one last call to compute_reward is made, and the reward is published. 

#### reset_goalie()
The reset_goalie function creates a ModelState object ‘state’ for the ‘turtlebot3_waffle_pi’, sets its reference frame to the ‘world’ and sets its position and orientation to the original values from the launched world. This model state is then set using set_state(state).

#### ball_move.py()
The run method in the ball_move node initializes the number of balls to send for learning, issues a send_ball() command for each kick and a reset_goalie() after the kick is finished.

#### Reinforcement learning
We use Q-learning as our reinforcement learning algorithm. The bulk of the algorithm is similar to what we previously used for the Q-learning project. The main difference is how we defined the states of the system, and how we mapped the robot and ball positions into these states, which is described in the next section. The algorithm is defined in the function algorithm() in learning.py. We choose alpha = 1 and gamma = 0.5. For a certain number of iterations, we get the initial state number and calculate a random action that is either move left, move right, or stay put. Then, we apply this action using a function that automatically moves the gazebo model (described  more below). The function then sleeps until a reward is seen. If the reward is 0, then the ball has not yet reached the goal, and we calculate the maximum reward in this next state of the robot to include in the update to the q matrix. If the reward is nonzero, there is no next state as the ball has reached a final location, so this maximum is 0. The update to the Q-matrix is defined as Q[state][action] + alpha*(reward+gamma*maximum_next_state - Q[state][action]). If this is different from the previous value in Q[state][action] we update. 

Determining the convergence of the Q matrix was also tricky given the large number of states. For 5 possible shots on the goal, we waited for full convergence, which we defined as 50 iterations without an update to the Q-matrix. This took a very long time but eventually happened at around 800 total iterations. For the same setup, we also tried stopping the evaluation of the Q-matrix at 300 total iterations. In this scenario the Q-matrix had not converged, but it still was able to calculate the optimal actions with the partially filled matrix. This also took significantly less time. The current implementation stops the updates of the Q matrix either after 300 total iterations have passed or after there are 50 iterations without a Q-update. This is just to ensure that running the code takes a reasonable amount of time. While it is possible that convergence may have happened after 300 iterations, it is unlikely. If you want to wait for full convergence, you can set reward_num_threshold in algorithm() to around 1000 as convergence should happen at that many iterations. 

For the purpose of reinforcement learning, we also wrote a function to automatically teleport the robot to the desired location. This is similar to the phantom robot node for the q learning project, and significantly speeds up and simplifies the convergence of the Q matrix. The function that teleports the robot is the set_robot() function in the learning.py script, which takes in the desired x and y coordinates and moves the robot to that location. This function is called in apply_action(), which, given one of the actions (move left, move right, or stay put), either moves the robot 0.5 m to the left or right (corresponding to up or down on the y axis) or does nothing at all. 

Finally, we execute the optimal actions in execute_best_actions(). We take the current state of the robot and calculate the optimal action by looking up that row in the Q-matrix and looking at the maximum argument. Then we apply this action by teleporting the robot, and sleep until a reward is seen. This repeats continuously so to stop the script you need to CTRL-C. We attempted to drive the robot to the optimal location, but we encountered some major problems with this that are described in the Challenges section.

#### Discretization
As we said before, our solution was to discretize the field by dividing it up into equal sized squares and then when a continuous measurement was given, to find the square in which the ball or robot was positioned, which is easily done by division by the square resolution and rounding. The square size we started with, or our “resolution”, was 1mx1m. This is the size of the standard grid in gazebo for reference, in the image here.  To calculate a state we first determine which squares the robot and the ball are in. Then we take the difference in the number of squares that separate the robot and the ball. We have a difference in x and a difference in y. From this, we can convert to a unique state. Essentially a more accurate model could be achieved by making the squares smaller, however this means that the number of possible states increases and convergence will take longer. There was a balance between these two conflicting goals of an accurate state model and a system that would converge quickly.  When picking the resolution of the quantization, we wanted to maintain a reasonable number of states so that convergence would happen within a reasonable number of iterations, but still maintain a large enough number of states so that different field locations can be sufficiently distinguished for the goalie to be able to accurately defend the goal.

The portion of our code that handles discretization is the get_state_num() function in the learning.py script. First, we calculate the robot and ball’s exact positions using a rospy Service Proxy call to get_model_state. Then, we calculate which “box” on our grid the robot and ball lie in. We define the bottom left point of the field as (0,0) and calculate the box number as the ceiling of the x or y coordinate divided by the resolution. Then, we calculate dy and dx, which is the difference in the number of boxes in both the x and y direction. This information is then translated into a unique state. The number of states is dependent on the resolution choice. The smaller the resolution, the more x and y boxes you can have. The number of possible differences in the number of boxes is 2 times the number of boxes in that dimension minus 1. (For example, if there are 3 possible boxes in the x direction, and an object can either be in box 1, 2, or 3, the difference in number of x-boxes can range from -2 to +2). This means that the total number of states is (2*x_boxes-1)*(2*y_boxes-1), where x_boxes is the ceiling of the field width divided by resolution, and y_boxes is the ceiling of the field height divided by resolution. This method of discretization also uses the relative positions of the robot and the ball to each other (dx and dy) rather than the absolute positions (for example, x_box and y_box). We believe that making this choice reduces the number of possible states. Finally, since the ball is kicked from a certain set of locations, and the robot always stays in front of the goal, many of these states are never seen. For example, with a resolution of 0.4 m, there are 875 possible states, but the Q matrix is still able to converge in a reasonable amount of time since only a small subset of these states are ever encountered. 

## Challenges, Future Work, and Takeaways
### Challenges
#### Initial problems with Gazebo
The initial goal for our project was to make the robot play a variation of the classic game, pong. To achieve this we wanted to create a rectangular room for the turtlebot with a puck that is frictionless and able to bounce off of the robot and the wall. The attempt for the model is shown below, with the blue walls and the red puck. Unfortunately even after adjusting a ton of physical parameters of the puck and looking at a bunch of online resources we were unable to get the puck to bounce off the walls, when it hit the wall head on it would just stop. This was really frustrating . Eventually we realized that Gazebo has a bunch of built in models one of which includes a soccer field, as well as models for soccer goals and a soccer ball. It was easier to customize this world to suit our needs, so we decided to move forward creating a ground plane as the soccer field, adding two goals, a turtlebot, and a soccer ball. It is also shown below. Controlling the robot to kick the ball in this world is more difficult to do repeatedly, but if it moves in front of the ball, then the physics engine of Gazebo handles the interaction well and it can stop or deflect the ball in motion. This motivated the switch to having the robot defend the goal by moving to the right location and use the gazebo service proxy ro set the model state of the ball directly for the ball launches.
[original_world](original_world.png)
[final_world](final_world.png)
### Question of how to discretize
We spent a significant amount of time pondering how to discretize the world to make QLearning work efficiently. The method we decided on is described above, but actually doing the math and figuring out how many states there would be and how they would be defined was quite tricky. Drawing out the world, measuring the region covered by the ball in action and the goal helped us move past this and select reasonable balance between fidelity and convergence time.

#### Reinforcement Learning Convergence
When we first coded up the QLearning algorithm, it refused to converge, and updates to the QMatrix kept occurring frequently even after hundreds of iterations. After debugging the code (which did fix some errors), we turned towards problems with the discretization, thinking perhaps fine tuning the parameters there might improve the convergence. For instance, we tried adjusting the resolution (meters per each square) and the distance moved in each robot action. Intuitively, you can be more accurate and better approach the continuous nature of the world with smaller resolution and actions, but the convergence time will increase dramatically We considered the possibility that, due to the approximate nature of our discretization, and the uncertainty of the ball interaction with the robot and the goal posts, that we wouldn't be able to get the matrix to converge perfectly, since there is some degree of non-determinism in the physics interaction. At least there is a sensitive dependence on precise angles and interaction, so the amount of randomness we included, for example, in the ball launch angle could cause there to be an average best action, but no single one best action. For instance, perhaps being on one side of a rounded square versus another requires a different action to block an incoming ball, or a slight variation in the ball goalie interaction makes a close kick be a goal, versus a miss. As such, we tried adjusting the learning algorithm to make it converge more quickly and stop after a certain number of iterations, such as removing negative rewards and changing the learning rate. After a fair amount of time was spent on this problem, we managed to get the algorithm to converge by (1) making the number of ball send angles very limited and (2) waiting a long time for convergence. These results allowed us to validate that our code does indeed work even when we don't do (1) and send the ball from anywhere along the midfield line, but that it would just take a very long time to converge.

#### Ball Launching
Another challenge was figuring out how to launch the ball from a specific location, direction, and velocity, which we overcame using the rospy Service proxy. We found out about this by looking at the ROS Manuals online for information and examples and included a link to where we got inspiration for this idea. We gradually learned to do more: first, using small python nodes to create a ball, to create a ball and make it roll, to create a ball and make it launch at a particular angle and velocity, to do so randomly, etc. We were able to eventually have a ball launching node which is the physics simulation engine for our project, and of course which uses the Gazebo physics engine.
### Future Work
#### Other Reinforcement Learning Algorithms
If we had more time, we would like to try delving into other ML algorithms, in particular perhaps use one with neural networks. As mentioned above, the time it takes for the Q matrix to converge is perhaps our biggest problem, and we believe this is just intrinsic to the discretized QLearning approach we took. Other pong game QLearning implementations we found online learn for tens of thousands of iterations and several hours of GPU time. Some of these methods use all of the pixels of the pong game image as the input to a deep neural network to then output a state vector for reinforcement learning, either a Q-learning type approach or other reinforcement methods. As such, trying another algorithm that would either converge more quickly or provide a more powerful way to capture the state of the game might yield good results. The use of another algorithm might also enable us to shoot the ball from an arbitrary location and angle and still be able to learn the best policy, rather than needing to restrict the set of served angles to one of a fixed number of set locations. This is what we initially wanted our algorithm to accomplish but it was simply not possible using Q-learning given the time/computing power it would require and the computing machines available through the nomachine access. 

One idea that we considered in the spirit of what could we do if we had as much time as we wanted, was that  we’d like to try is NeuroEvolution of Augmenting Topologies, where we would iterate on a neural net using a genetic algorithm. We could make the world end if the robot doesn’t perform well enough for us to stop evolving, and then evaluate best neural networks by how long the robot lasts in the world, or how much reward it receives over time, evolving the neural net using the best of the current generation. We also considered and would like to try using a deep deterministic policy gradient, which works on continuous state/action space.  We knew both of these ideas may take a long time to train after we write code and that it may be difficult to implement without more theoretical knowledge. We also considered using the pytorch deep neural network package to train a DNN to decide on goalie actions by supervised training with the same rewards we used for q-learning. Instead of filling out a q-matrix with a state and action sequence, input variables are fed into the input layer of the DNN, and the output is a policy decision. This could be up/down like we did, or it could include more robot controls. The main difference between this approach and q-learning is that we would feed inputs into and need to train the DNN, which would require more computation. 
If we had more time, we would also improve our project by making the robot move with twist messages instead of using the rospy proxy servers to directly move the model state.  We would also try changing some of the dimensions of the problem to make the physics a little more deterministic, for example, I think maybe making the ball a little bigger might make things easier  The ball was moving too fast in this setup to reliably use scan messages and the lidar for sensing/perceiving the ball location using the robots sensors.  It would be interesting to also slow-down the ball sufficiently (and use a larger ball, so that it appears on the scan), and sense its location directly.  The current ball size is too small to be reliably picked up by the laser scanner even when it isn’t moving. At the velocities that the ball traveling in this project for learning, the laser scanner would not even register the ball at all.


#### Robot movement
As of now, the robot performs actions/moves just using the model states event. Given more time, we would make the turtlebot actually turn and move, rather than teleporting. We recently attempted to physically drive the robot to the optimally calculated location, but it was very difficult to get the timing of the movements right. It was also hard to stop the turtlebot from drifting, as it had to drive at a high speed to catch the ball in time, but doing so made it more susceptible to turns from the random noise. Specifically, trying to use odometry to tell when the robot has driven 0.5 m has a tendency to overshoot the correct location, whereas trying to use timing to determine when the robot has driven 0.5 m tends to severely undershoot the location (as the robot takes some time to accelerate to full speed). Our attempt at using odometry to drive the robot (which does not work correctly!) is in learning_odom.py. If you want to try running this script, simply run learning_odom.py instead of learning.py. We believe that if the ball was traveling slower it would be more feasible to drive the robot to the exact location. Play_commands_new.py implemented a simple proportional control goalie that moved the goalie using twist messages based on the difference between the position of the ball’s y coordinate and the robot’s y coordinate, which were obtained not using odometry, but rather using the get_model_state functionality.  Since this knows the precise location of the ball and the robot, this proportionally controlled goalie works well at stopping the ball but does not use learning, which was the goal of this project.  

#### Dimensionality
Given more time, another change we would like to try is changing some of the dimensions of the problem to make the physics a little more deterministic (that is, make the effects of quantization less significant). For example, perhaps making the ball a little bigger might make things be well-modeled with a sufficiently coarse quantization for rapid convergence. 

### Takeaways
* Training time can be very important for ML algorithms. In the last QLearning project, it took maybe a couple hundred iterations for the QMatrix to converge, and it was clear throughout the process that it was converging. With this project, it took significantly more time, and we had to greatly limit possible ball launches to even realize it was converging. We will make sure to account for this when choosing ML models in the future, as a model that takes so long to converge is increasingly more difficult to debug and work with.
* Don’t research for too long before implementing. In deciding which RL algorithms to use, we did copious reading and research, which allowed us to learn about a lot of great work in reinforcement learning. However, much of this work is for algorithms that run on powerful GPU computers and turned out to be too sophisticated for our project.  After selecting a few candidate approaches, the training time was so long, that the evaluation and iteration cycles made it difficult to make rapid changes to the model. If we had done substantially less research, we would’ve had more time to perfect our final code and algorithm. We might also have realized that QLearning was going to take too long to converge and tried a different approach, or figured out how to launch the nodes on a much more powerful computing server.. 
* Ask for help! Our group’s meeting with Sarah was very helpful for formally deciding on a learning algorithm and figuring out how to describe our states. While we might have been able to figure these things out by ourselves, but it was really helpful to get an outside perspective on what was the most feasible way to continue given the time crunch of the end of the quarter and end of the class. There is so much robotics to learn about and only so much time available to try things out. There is a lot of trial and error in this process, which is great fun, but it also makes development cycles longer than a pure coding project.




                  
