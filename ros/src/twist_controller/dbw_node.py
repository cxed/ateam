#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane
import math
import tf
import numpy as np
import warnings
from twist_controller import Controller
from pid import PID

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

MAX_STEERING = 0.43 # 25 degrees in radians

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
        self.refresh_rate = 10

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

        # Subscribe to /current_pose to get current position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # Subscribe to /final_waypoints to get final waypoints
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)

        # car's current pose
        self.current_pose = None

        # waypoints from final_waypoints topic
        self.final_waypoints = None

        self.timestamp = rospy.get_time()
        self.steering_controller = PID(0.16, 0.01, 0.8, mn=-MAX_STEERING, mx=MAX_STEERING)

        self.loop()

    def loop(self):
        '''Main publishing of car controls if car should be controlled.'''
        rate = rospy.Rate(self.refresh_rate)
        while not rospy.is_shutdown():

            # calculate delta time since last loop
            delta_time = rospy.get_time() - self.timestamp
            self.timestamp = rospy.get_time()

            # preliminary attempt to get car moving
            # block multiple calls if the velocity has already been set
            #if self.current_linear_velocity is None or self.target_linear_velocity is None:
            if (not self.dbw_enabled
                or self.current_linear_velocity is None
                or self.target_linear_velocity is None):
                continue

            throttle,brake,p_steer = self.controller.control(self.current_linear_velocity,
                                                           self.current_angular_velocity,
                                                           self.target_linear_velocity,
                                                           self.target_angular_velocity)

            # calculate cte
            cte = self.get_cte()

            # feed cte into pid to get steering
            c_steer = self.steering_controller.step(cte, delta_time)
            steer = c_steer

            # TODO: look into using both steering from cte and yaw_controller

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
        # reset the controller if dbw is switched off
        if self.dbw_enabled == True and msg.data == False:
            self.controller.reset()
            self.steering_controller.reset()

        # extract the boolean status of the dbw from the msg
        self.dbw_enabled = msg.data


    def publish(self, throttle, brake, steer):
        if throttle != 0:
            tcmd = ThrottleCmd()
            tcmd.enable = True
            tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
            tcmd.pedal_cmd = throttle
            self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        if brake != 0:
            bcmd = BrakeCmd()
            bcmd.enable = True
            bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
            bcmd.pedal_cmd = brake
            self.brake_pub.publish(bcmd)

    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg.waypoints

    def polyeval(self, coeffs, x):
        """ evaluate a polynomial
        """
        result = 0.0
        for i in range(len(coeffs)):
            result += coeffs[i] * pow(x, i)
        return result

    def get_cte(self):
        if self.current_pose != None and self.final_waypoints != None:
            # get yaw
            orientation = self.current_pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

            # car's current x and y
            px = self.current_pose.position.x
            py = self.current_pose.position.y

            # convert final way points to car's reference point of view
            ptsx_car = []
            ptsy_car = []
            for wp in self.final_waypoints:
                shift_x = wp.pose.pose.position.x - px
                shift_y = wp.pose.pose.position.y - py
                ptsx_car.append(shift_x * math.cos(-yaw) - shift_y * math.sin(-yaw))
                ptsy_car.append(shift_x * math.sin(-yaw) + shift_y * math.cos(-yaw))

            # cxed- Need to trap this because it likes to spew warnings. See:
            # https://stackoverflow.com/questions/21252541/how-to-handle-an-np-rankwarning-in-numpy
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    # np.polyfit returns coefficients, highest power first, so we need to reverse it
                    coeffs = np.polyfit(ptsx_car, ptsy_car, 3)
                except np.RankWarning:
                    return 0 # NOTE: This may be wrong!!!!
            coeffs = list(reversed(coeffs))

            cte = self.polyeval(coeffs, 0)
            #epsi = -math.atan(coeffs[1])
            return cte

        return 0



if __name__ == '__main__':
    DBWNode()
