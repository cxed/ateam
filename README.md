A Team - System Integration Project 
===================================

![A Team](imgs/ateamlogo.png "There is no Plan B.")

Udacity Term 3 Final Capstone Project

## The A Team

### Members
  - 0 Chris X Edwards _@cxed_ UTC-7
  - 1 Vince Chan _@vincec_ UTC-7
  - 2 William Yuill _@williambeverly_ UTC+2
  - 3 Andreas Jankl _@jankl.andreas_ UTC+2
  - 4 Markus Meyerhofer _@markus.meyerhofer_ UTC+2

### Identification Note
Email addresses were requested but since this is a public forum
privacy concerns were discussed on Slack where **@stephen** (Stephen
Welch) said, "Ok, I've discussed this here, and I think we can do
without emails, so long as names are included." (2017-09-15)

Group member Slack usernames are also included.

### Sub Tasks
* Chris: easy stuff.
  - set up team resources (Slack, GitHub, Skype, etc),
  - documentation, coordination
  - testing, programming support, removing tabs
  - Vince did a _lot_ of testing and optimzing of everything
  - Vince also recorded videos
* Vince & William: waypoint updater a.k.a. getting the car moving at all.
  - making the car drive
  - making the car drive smoothly (direction _and_ speed)
  - making the car drive smoothly as long as we want
  - stopping the car when told that's a good idea
  - resuming the car when told that's a good idea
* Andreas & Markus: tl_detector, the classifiers.
  - detecting the state of any visible traffic lights
  - deciding when a stop should be done based on traffic light state
  - William also helped figure out how to get the classifier working
  - Chris built training image sets to prevent garbage in/garbage out

### Skype Calls
* **Success!** 2017-09-10 1500UTC=0800PDT=1700U+2
* **Success!** 2017-09-17 1500UTC=0900PDT=1800U+2
* **Success!** 2017-09-24 1500UTC=0900PDT=1800U+2
* **Success!** 2017-09-29 1430UTC=0830PDT=1730U+2

## Requirements

###  Smoothly follow waypoints in the simulator.
Our code properly guides the car through all of the waypoints. As
discussed by many people in `#p-system-integration` there were points
on the course where it was difficult to maintain control because of
mysterious simulator performance issues. Our different team members
had varying levels of success with this ranging from almost no
problems to very severe impassable points on the course _using the
same code_. Because so much effort was spent to overcome these high
interference zones, we ended up with an extremely stable and accurate
control system that guided the car nearly perfectly (special mention
of Vince's contribution is deserved here).

###  Stop at traffic lights when needed.
Our system takes the published video feed and crops out what we
believe is a sensible sub region. We followed the suggestion implied
in the project materials to locate that region dynamically using dead
reckoning but we found this was not especially helpful. With a
plausible traffic light published when near a known traffic light
location, the next task was to send this image to a classifier that
could determine if it was red, yellow, or green. We tried both a
Tensorflow system and a Keras system, eventually settling on the Keras
as easier to work with for this particular project. The classifier was
trained on a large set of images which, for the simulator, was
extracted from manual driving. For the bag file video, the frames
were extracted and a classifier trained on a large set of data derived
from them (13000 unique images of each of red, yellow, green).

###  Stop and restart PID controllers depending on the state of `/vehicle/dbw_enabled`.
We believe the `dbw_enabled` feature works as required. The simulator
can be put into and returned from "Manual" mode as expected.

### Confirm that traffic light detection works on real life images.
We have set the system up so that when launched with the site launcher
the software behaves with that in mind and can successfully detect
the state of the traffic light.

## Project Video
Here is a video showing perfect driving for an entire lap of the
simulator course.

Video: [A Team - Complete Simulation Success](https://youtu.be/XoXnJ4nqzmE)

[![A Team](https://img.youtube.com/vi/XoXnJ4nqzmE/0.jpg)](https://www.youtube.com/watch?v=XoXnJ4nqzmE)

Here is a video showing the team's project code running on http://ros.org[ROS]
http://wiki.ros.org/rosbag[bag] data of a real world site. When the
vehicle is within range, the classifier is activated and accurately
discerns the light's state.

Video: [A Team - Complete Site Success](https://youtu.be/gthHnvHFEnY)

[![A Team](https://img.youtube.com/vi/gthHnvHFEnY/0.jpg)](https://www.youtube.com/watch?v=gthHnvHFEnY)

## Project Components

### Nodes
![System Diagram](imgs/final-project-ros-graph-v2.png "System Diagram")

#### Perception
* Traffic light detection
* Obstacle detection

#### Planning
* Waypoint updater - Sets target velocity for each waypoint based on traffic light and obstacles.

#### Control
* Drive by wire ROS node -
  - input: target trajectory
  - output: control commands to vehicle
* Here's a handy flowchart for PID tuning.

### Car Info
* [ROS Interface to Lincoln MKZ DBW System](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/)
* Carla is a https://en.wikipedia.org/wiki/Lincoln_MKZ
* Curb weight = 3,713-3,911 lb (1,684-1,774 kg)

## Run
```
catkin_make && source devel/setup.sh && roslaunch launch/styx.launch
rosbag play -l just_traffic_light.bag
rqt_image_view /image_color
rqt_console
```

## References and Links
* [CarND-Capstone Repo](https://github.com/udacity/CarND-Capstone)
* [VM image](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/7e3627d7-14f7-4a33-9dbf-75c98a6e411b/concepts/8c742938-8436-4d3d-9939-31e40284e7a6?contentVersion=1.0.0&contentLocale=en-us)
* [Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.1)
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
* [Traffic Light Detection Test Video - a ROS bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing)
* [Starter Repo](https://github.com/udacity/CarND-System-Integration)
* [ROS Twist](http://docs.ros.org/jade/api/geometry_msgs/html/msg/Twist.html)
* [Team Sign Up Spreadsheet](https://docs.google.com/spreadsheets/d/17I_0q8tylk9Q_Y3GTSq738KkBIoS6SUt1quR5lPPAdg/edit#gid=0)

### Notable Slack Channels
* #ateam - **A Team** Slack channel.
* _#p-system-integration_ - Seems to be where this project is being discussed.
* _#sdc-ros_ - ROS topics.
* [Discussion Forum - System Integration](https://discussions.udacity.com/c/nd013-system-integration)

### Schedule

Project Due:
**2017-10-02**

Term Ends: 2017-10-16

```
Su Mo Tu We Th Fr Sa  
10 11 12 13 14 15 16  
17 18 19 20 21 22 23
24 25 26 27 28 29 30  September 2017       
 1  2  3  4  5  6  7  October 2017      
 8  9 10 11 12 13 14  
15 16 17 18 19 20 21  
```

