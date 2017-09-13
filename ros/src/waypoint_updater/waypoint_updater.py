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


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None # Create a storage container for the base_waypoints msgs
        self.pose = None # Create a storage container for the pose data
        self.frame_id = None # Create a storage container for the frame_id

        rospy.spin()


    def pose_cb(self, msg):
        # pose callback - utilised to store the msg received from topic /current_pose and the frame_id
        # format of the received message is geometry_msgs/PoseStamped
        # - Header header
        # - Pose pose
        # -- Point point
        # -- Quaternion orientation

        # store the message
        self.pose = msg.pose
        self.frame_id = msg.header.frame_id


        # Then add the function here
        # the method finds the closest waypoint in front of the vehicle
        # creates an updated_waypoint list of length LOOKAHEAD_WPS
        # updated_waypoint velocities are updated with a target velocity
        # once completed, a message of Type Lane is constructed for publishing to final_waypoints_pub
        #
        # format of the create Lane message is styx_msgs/Lane
        # - Header header
        # - Waypoints[] waypoints
        # -- PoseStamped pose
        # -- TwistStamped twist

        if self.pose is not None and self.base_waypoints is not None and self.frame_id is not None:

            # find the index of the next infront waypoint
            idx = self.next_infront_waypoint(self.pose, self.base_waypoints)

            # simplistic max speed driving
            target_speed = MAX_SPEED

            
            # set up the next waypoints
            # TODO make it circlic
            infront_waypoints = self.base_waypoints[idx: idx + LOOKAHEAD_WPS]

            # set the speed
            for infront_waypoint in infront_waypoints:
                infront_waypoint.twist.twist.linear.x = target_speed

            # make a lane message

            lane = Lane()
            lane.header.frame_id = self.frame_id
            lane.waypoints = infront_waypoints
            lane.header.stamp = rospy.Time.now()

            # now publish the lane msg
            self.final_waypoints_pub.publish(lane)
            
            # Very bad use of rospy.logerr
            #rospy.logerr('Closest waypoint is: ' + str(idx))

        
    def waypoints_cb(self, waypoints):
        # waypoint callback - utilised to store the base waypoints, from topic /base_waypoints
        # these waypoints consist of the  complete list of all waypoints in-front and behind the vehicle
        # format of the message is styx_msgs/Lane
        # - Header header
        # - Waypoints[] waypoints
        # -- PoseStamped pose
        # -- TwistStamped twist
        self.base_waypoints = waypoints.waypoints

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
        # This is not used
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def calculate_distance(self, p1, p2):
        # calculates the euclidian distance between two points considering 3D (x,y,z)
        delta_x = p1.x - p2.x
        delta_y = p1.y - p2.y
        delta_z = p1.z - p2.z
        return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

    def closest_waypoint(self, pose, waypoints):
        # Method determines the closest waypoint, but does not take into account whether it is in-front or behind the vehicle
        # the determination of whether the waypoint is in front or behind is handled in next_infront_waypoint
        smallest_distance = self.calculate_distance(pose.position, waypoints[0].pose.pose.position) # calculate the distance on the first waypoint
        closest_waypoint_idx = 0 # create an index for the closest waypoint

        # Then loop through all of the rest of the waypoints to find the smallest distance
        for i, point in enumerate(waypoints):
            distance = self.calculate_distance(pose.position, point.pose.pose.position)
            if(distance < smallest_distance):
                closest_waypoint_idx = i
                smallest_distance = distance

        return closest_waypoint_idx   

    def next_infront_waypoint(self, pose, waypoints):
        # Method returns the next in-front waypoint, by first obtaining the closest waypoint
        # then checking whether this waypoint is in fact in-front of the vehicle
        # if the closest waypoint is found to be behind the vehicle, the closest_waypoint index is incremented
        
        # find the closest waypoint
        closest_waypoint_idx = self.closest_waypoint(pose, waypoints)

        # now check if the waypoint is indeed in front
        heading = math.atan2(waypoints[closest_waypoint_idx].pose.pose.position.y - pose.position.y, waypoints[closest_waypoint_idx].pose.pose.position.x - pose.position.x)

        # obtain the yaw
        _,_,yaw = tf.transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])

        angle = abs(yaw - heading)
        if(angle > math.pi / 4):
            closest_waypoint_idx += 1

        return closest_waypoint_idx


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
