#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from std_msgs.msg import Int32
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

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MPH_TO_MPS = 0.44704 # converions for miles per hour to meters per second
NARROW_SEARCH_RANGE = 10  # Number of waypoints to search current position back and forth
MAX_DECEL = 1.0
REFRESH_RATE = 10

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoint_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.current_waypoints = None
        self.next_waypoint_index = None
        self.yaw = None
        self.last_waypoint_index = None
        # Waypoint index of a upcoming redlight, -1 if it does not exist
        self.traffic_stop_waypoint_index = -1
        self.current_velocity = None

        # get max speed from config
        self.max_speed = rospy.get_param('~max_speed_mph', 10) * MPH_TO_MPS
        
        # added self.target_speed
        self.target_speed = 0

        self.loop()
        
    def loop(self):
        rate = rospy.Rate(REFRESH_RATE)
        
        while not rospy.is_shutdown():
            rate.sleep()

            if self.current_waypoints is None or self.current_pose is None:
                continue

            next_waypoint_index = self.next_infront_waypoint()

            # set the target speed
            target_speed = self.max_speed

            # set the number of waypoints received from /base_waypoints
            number_waypoints = len(self.current_waypoints)

            lane = Lane()
            lane.header.frame_id = self.current_pose.header.frame_id
            lane.header.stamp = rospy.Time(0)

            if (self.traffic_stop_waypoint_index != -1 
                and self.is_in_braking_distance()):
                
                # we have decided that the waypoint published by the /traffic_waypoint 
                # is where we need to stop the car at
                stop_waypoint_index = self.traffic_stop_waypoint_index

                # TODO: handle wrapping
                #       Do we always generate LOOKAHEAD_WPS number of points? what to do
                #       if we will have points after the red traffic light? Just continue but
                #       set the velocity to 0 for those points? 
                decelerate_points_count = stop_waypoint_index - self.next_waypoint_index
                if decelerate_points_count > 0:
                    for i in range(decelerate_points_count):
                        wp_new = Waypoint()
                        wp_extract = self.current_waypoints[next_waypoint_index]
                        wp_new.pose = wp_extract.pose
                        lane.waypoints.append(wp_new)
                        wp_new.twist.twist.linear.x = target_speed
                        next_waypoint_index = (next_waypoint_index + 1) % number_waypoints
                    lane.waypoints = self.decelerate(lane.waypoints)
                
                #generate up to the LOOKAHEAD_WPS number of waypoints
                #fill it up with waypoints with zero velocity
                if decelerate_points_count < LOOKAHEAD_WPS:
                    for i in range(LOOKAHEAD_WPS - decelerate_points_count):
                        wp_new = Waypoint()
                        wp_extract = self.current_waypoints[next_waypoint_index]
                        wp_new.pose = wp_extract.pose
                        lane.waypoints.append(wp_new)
                        wp_new.twist.twist.linear.x = 0

            else:
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
                lane.waypoints = self.accelerate(lane.waypoints)

            self.final_waypoints_pub.publish(lane)
        

    def pose_cb(self, msg):

        self.current_pose = msg

        # Calculate the yaw by utilising a transform from quaternion to euler
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.yaw = tf.transformations.euler_from_quaternion(quaternion)

    def is_in_braking_distance(self):
        if self.current_velocity:
            p1 = self.current_waypoints[self.traffic_stop_waypoint_index].pose.pose.position
            p2 = self.current_waypoints[self.next_waypoint_index].pose.pose.position
            distance = self.calc_position_distance(p1, p2) - 2
            time = distance / self.current_velocity
            if distance < 1 or self.current_velocity / time >= 4:
                return True
        return False

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

        # check the distance
        self.check_next_waypoint_distance()

        self.log_waypoint_progress()

        # save next waypoint index to last waypoint index
        self.last_waypoint_index = self.next_waypoint_index

        return closest_waypoint_idx
    
    # --------------------------------------------------------------------------------------------

    def check_next_waypoint_distance(self):
        """ Check the distance between the current pose and next way point.
            Log a warning if the distance is larger than threshold
        """
        distance_limit = 3
        next_distance = self.calculate_distance(self.current_waypoints[self.next_waypoint_index])
        if next_distance > distance_limit:
            rospy.logwarn("large distance %s last idx %s next idx %s", 
                next_distance,
                self.last_waypoint_index,
                self.next_waypoint_index)

    def log_waypoint_progress(self):
        if self.next_waypoint_index != self.last_waypoint_index and not self.next_waypoint_index%10:
            rospy.loginfo('moving to waypoint %s', self.next_waypoint_index)

    def waypoints_cb(self, waypoints):
        self.current_waypoints = waypoints.waypoints
        self.next_waypoint_index = None

    def calc_position_distance(self, p1, p2):
        """ Calculate distance between two positions
        """
        return math.sqrt((p1.x - p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.calc_position_distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def accelerate(self, waypoints):
        # initialize starting velocity as current velocity
        if self.current_velocity != None:
            vel = self.current_velocity
        else:
            vel = 0
        #TODO: replace simple step up velocity accelecation
        step = 0.3
        for wp in waypoints:
            vel += step
            wp.twist.twist.linear.x = min(vel, self.max_speed)
        return waypoints

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def traffic_waypoint_cb(self, msg):
        if self.traffic_stop_waypoint_index != msg.data:
            rospy.loginfo("traffic light status change to %s", msg.data)
        self.traffic_stop_waypoint_index = msg.data

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
