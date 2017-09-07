A Team - System Integration Project 
===================================

Udacity Term 3 
Due: 2017-10-06 <- Is this correct? It says "Due in 1 month" on 2017-09-06.

## The A Team
* Slack channel - "ateam"
* Members -
  0. Chris @cxed UTC-7
  1. Vince @vincec UTC-7
  2. William @williambeverly UTC+2
  3. Andreas @jankl.andreas UTC+2
  4. Markus @markus.meyerhofer UTC+2

[Team Sign Up Spreadsheet](https://docs.google.com/spreadsheets/d/17I_0q8tylk9Q_Y3GTSq738KkBIoS6SUt1quR5lPPAdg/edit#gid=0)
  
### Possible Division Of Labor
Team leader 0: set up team resources (slack channel, repository, etc),
              documentation, coordination, and programming support if needed.
Team member 1: waypoint updater
Team member 2: traffic light detector
Team member 3: drive-by-wire controller
Team member 4: testing

## Project Components

### Nodes
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


### Explicit Requirements
* Code via GitHub
* README.md

## TODO
1. The Waypoint Updater Node seems to be a prerequisite to many other components
  so it is recommended to work on it first.
2. DBW (Drive By Wire) Node
3. Traffic Light Detection
4. Waypoint Updater Node - complete.


## References and Links
* [CarND-Capstone Repo](https://github.com/udacity/CarND-Capstone)
* [VM image](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/7e3627d7-14f7-4a33-9dbf-75c98a6e411b/concepts/8c742938-8436-4d3d-9939-31e40284e7a6?contentVersion=1.0.0&contentLocale=en-us)
* [Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.1)
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
* [Traffic Light Detection Test Video - a ROS bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing)
* [Starter Repo](https://github.com/udacity/CarND-System-Integration)
* [ROS Twist](http://docs.ros.org/jade/api/geometry_msgs/html/msg/Twist.html)


![PID Tuning](http://support.motioneng.com/Downloads-Notes/Tuning/images/overshoot_flowchart.gif "PID Tuning")



