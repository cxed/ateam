#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, it's enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # define some velocity variables
        self.current_linear_velocity = None
        self.current_angular_velocity = None
        self.target_linear_velocity = None
        self.target_angular_velocity = None
        self.dbw_enabled = None
        self.min_speed = 0.0
        self.refresh_rate = 50

        # This is the twist controller object
        self.controller = Controller(
        	vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35),
        	fuel_capacity = rospy.get_param('~fuel_capacity', 13.5),
        	brake_deadband = rospy.get_param('~brake_deadband', .1),
        	decel_limit = rospy.get_param('~decel_limit', -5),
        	accel_limit = rospy.get_param('~accel_limit', 1.),
        	wheel_radius = rospy.get_param('~wheel_radius', 0.2413),
        	wheel_base = rospy.get_param('~wheel_base', 2.8498),
        	steer_ratio = rospy.get_param('~steer_ratio', 14.8),
        	max_lat_accel = rospy.get_param('~max_lat_accel', 3.),
        	max_steer_angle = rospy.get_param('~max_steer_angle', 8.),
        	min_speed = self.min_speed,
            refresh_rate = self.refresh_rate)


        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Define the subscribers
        # Subscribe to /current_velocity to obtain linear and angular velocities
        rospy.Subscriber('/current_velocity', TwistStamped, self.extract_current_velocities)

        # Subscribe to /twist_cmd to otain the target velocities from waypoint follower
        rospy.Subscriber('/twist_cmd', TwistStamped, self.extract_target_velocities)

        # Subscribe to /vehicle/dbw_enabled to obtain the dbw status
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.extract_dbw_status)

        self.loop()

    def loop(self):
        # You should only publish the control commands if dbw is enabled
        # if <dbw is enabled>:
        #   self.publish(throttle, brake, steer)
        # throttle = 0.5 # Range 0 to 1.
        # Official docs say: "...units of torque (N*m). The correct values
        # for brake can be computed using the desired acceleration, weight
        # of the vehicle, and wheel radius."
        # /dbw_node/vehicle_mass: 1080.0
        # Carla = https://en.wikipedia.org/wiki/Lincoln_MKZ
        # Curb weight = 3,713-3,911 lb (1,684-1,774 kg)
        # Decel_Force(newtons) = Mass_car(kg) * Max_decel(meter/s^2) 
        # MaxBrakeTorque(newton*meter) = Decel_Force(newtons) * wheel_radius(meters) / 4 wheels
        # MaxBrakeTorque(newton*meter) = Mass_car(kg) * Max_decel(meter/s^2) * wheel_radius(meters) / 4 wheels
        # 726 g/L density of gas. 13.5gal=51.1Liters, max fuel mass=37.1kg
        # 4 passengers = 280 kg
        # Let's just say 2000kg for a deployed car.
        # Note that rospy.get_param('~wheel_radius', 0.2413) but...
        # /dbw_node/wheel_radius: 0.335
        # (Chris independently calculated the wheel radius to be .340m, so let's go with .335)
        # MaxBrakeTorque(newton*meter) = 2000(kg) * 5(meter/s^2) * .335(meters) / 4 wheels
        # MaxBrakeTorque= 837.5Nm

        rate = rospy.Rate(self.refresh_rate)
        while not rospy.is_shutdown():
        	# preliminary attempt to get car moving
        	# block multiple calls if the velocity has already been set
        	# TODO add dbw_enabled
        	if self.current_linear_velocity is None or self.target_linear_velocity is None:
        		continue

    		throttle,brake,steer = self.controller.control(self.current_linear_velocity,self.current_angular_velocity,self.target_linear_velocity,self.target_angular_velocity)
    		self.publish(throttle, brake, steer)
        	
        	rate.sleep()

    
    def extract_current_velocities(self, msg):
    	# extract the current linear and angular velocities from TwistStamped type
    	self.current_linear_velocity = msg.twist.linear.x
    	self.current_angular_velocity = msg.twist.angular.z

        # rospy.logerr('Current linear velocity: ' + str(self.current_linear_velocity))
        # rospy.logerr('Current angular velocity: ' + str(self.current_angular_velocity))

    
    def extract_target_velocities(self, msg):
    	# extract the target linear and angular velocities from TwistStamped type
    	self.target_linear_velocity = msg.twist.linear.x
    	self.target_angular_velocity = msg.twist.angular.z
        # confirmed htat these are the only values populated in the msg

    
    def extract_dbw_status(self, msg):
    	# extract the boolean status of the dbw from the msg
    	self.dbw_enabled = msg.data


    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
