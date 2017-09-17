#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import tf

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number
MPH_TO_MPS = 0.44704 # converions for miles per hour to meters per second
MAX_SPEED = 10 * MPH_TO_MPS # max speed for CARLA is 10 miles per hour - convert this to MPS
NARROW_SEARCH_RANGE = 10  # Number of waypoints to search current position back and forth


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.current_waypoints = None
        self.next_waypoint_index = None
        self.yaw = None

        rospy.spin()

    def pose_cb(self, msg):

        self.current_pose = msg

        # Calculate the yaw by utilising a transform from quaternion to euler
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.yaw = tf.transformations.euler_from_quaternion(quaternion)

        if (not rospy.is_shutdown() and (self.current_waypoints is not None)):
            next_waypoint_index = self.next_infront_waypoint()

            # set the target speed
            target_speed = MAX_SPEED

            # set the number of waypoints received from /base_waypoints
            number_waypoints = len(self.current_waypoints)

            lane = Lane()
            lane.header.frame_id = self.current_pose.header.frame_id
            lane.header.stamp = rospy.Time(0)


            # now create the waypoints ahead
            for i in range(LOOKAHEAD_WPS):
                # create a new waypoint, rather than ammending existing waypoints
                wp_new = Waypoint()
                # extract the desired waypoint, starting at the next_waypoint_index
                wp_extract = self.current_waypoints[next_waypoint_index]
                # copy the position contents of the base_waypoint to the new waypoint
                wp_new.pose = wp_extract.pose
                # set the target velocity of the new waypoint
                wp_new.twist.twist.linear.x = target_speed
                # add to the Lane waypoints list
                lane.waypoints.append(wp_new)
                # then increment the next_waypoint_index, considering circlic nature of list
                next_waypoint_index = (next_waypoint_index + 1) % number_waypoints

            self.final_waypoints_pub.publish(lane)


    def calculate_distance(self, waypoint):
        # calculates the euclidian distance between the vehicle (global) and the waypoint (global)
        waypoint_x = waypoint.pose.pose.position.x
        waypoint_y = waypoint.pose.pose.position.y

        delta_x = self.current_pose.pose.position.x - waypoint_x
        delta_y = self.current_pose.pose.position.y - waypoint_y
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    def closest_waypoint(self):
        # Method determines the closest waypoint, but does not take into account whether it is in-front or behind the vehicle
        # the determination of whether the waypoint is in front or behind is handled in next_infront_waypoint
        smallest_distance = 100000 # set to a large value
        closest_waypoint_idx = 0 # create an index for the closest waypoint

        start = 0 
        end = len(self.current_waypoints)

        if (self.next_waypoint_index):
            start = max(self.next_waypoint_index - NARROW_SEARCH_RANGE, 0) #limit it to min of zero
            end = min(self.next_waypoint_index + NARROW_SEARCH_RANGE, len(self.current_waypoints))
        
        for i in range(start, end):
            distance = self.calculate_distance(self.current_waypoints[i])
            if(distance < smallest_distance):
                closest_waypoint_idx = i
                smallest_distance = distance

        
        return closest_waypoint_idx   

    
    def next_infront_waypoint(self):
        # Method returns the next in-front waypoint, by first obtaining the closest waypoint
        # then checking whether this waypoint is in fact in-front of the vehicle
        # if the closest waypoint is found to be behind the vehicle, the closest_waypoint index is incremented
        
        # find the closest waypoint
        closest_waypoint_idx = self.closest_waypoint()

        # get the length of the waypoints list
        number_waypoints = len(self.current_waypoints)
        #rospy.logerr(number_waypoints)

        waypoint_x = self.current_waypoints[closest_waypoint_idx].pose.pose.position.x
        waypoint_y = self.current_waypoints[closest_waypoint_idx].pose.pose.position.y

        delta_x = waypoint_x - self.current_pose.pose.position.x
        delta_y = waypoint_y - self.current_pose.pose.position.y

        # transform x, so that it points directly in-front of the vehicle
        transformed_x = delta_x * math.cos(-self.yaw) - delta_y * math.sin(-self.yaw)
        
        # Next items are commented because they are not necessary
        # transformed_x = delta_x * math.cos(self.yaw) + delta_y * math.sin(self.yaw)
        # transformed_y = delta_x * math.sin(-self.yaw)  + delta_y * math.cos(-self.yaw)
        # relative_angle = math.atan2(transformed_y, transformed_x)

        # because x points in front of the vehicle, if it is negative, it means the object is behind
        # the vehicle - if that is the case, the next waypoint is the closest, so increment
        if transformed_x < 0.0:
            closest_waypoint_idx = (closest_waypoint_idx + 1) % number_waypoints

        self.next_waypoint_index = closest_waypoint_idx

        return closest_waypoint_idx
    
    # --------------------------------------------------------------------------------------------

    def waypoints_cb(self, waypoints):
        self.current_waypoints = waypoints.waypoints
        self.next_waypoint_index = None
        # we only need the message once, unsubscribe as soon as we got the message
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
