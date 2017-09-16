#!/usr/bin/env python
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.best_waypoint = 0
        self.last_car_position = 0
        self.last_light_pos_wp = []
        self.IGNORE_FAR_LIGHT = 100.0
        self.simulator_debug_mode = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # we only need the message once, unsubscribe as soon as we got the message
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):

	if(self.simulator_debug_mode==1):
        	self.lights = msg.lights
        	light_wp, state = self.process_traffic_lights_simulation()

        	'''
        	Publish upcoming red lights at camera frequency.
        	Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        	of times till we start using it. Otherwise the previous stable 	state is
        	used.
        	'''
		if self.state != state:
		    self.state_count = 0
		    self.state = state
		elif self.state_count >= STATE_COUNT_THRESHOLD:
		    self.last_state = self.state
		    light_wp = light_wp if state == TrafficLight.RED else -1
		    self.last_wp = light_wp
		    self.upcoming_red_light_pub.publish(Int32(light_wp))
		else:
		    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
		self.state_count += 1

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
	if(self.simulator_debug_mode==0):
        	self.has_image = True
        	self.camera_image = msg
        	light_wp, state = self.process_traffic_lights()

        	'''
        	Publish upcoming red lights at camera frequency.
        	Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        	of times till we start using it. Otherwise the previous stable 	state is
        	used.
        	'''
		if self.state != state:
		    self.state_count = 0
		    self.state = state
		elif self.state_count >= STATE_COUNT_THRESHOLD:
		    self.last_state = self.state
		    light_wp = light_wp if state == TrafficLight.RED else -1
		    self.last_wp = light_wp
		    self.upcoming_red_light_pub.publish(Int32(light_wp))
		else:
		    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
		self.state_count += 1
        
    #Newly introduced function calculating a normal beeline distance
    def distance(self, p1, p2):
        if hasattr(p1,'x'):
            p1x= p1.x
        else:
            p1x= p1[0]
        if hasattr(p1,'y'):
            p1y= p1.y
        else:
            p1y= p1[1]
        if hasattr(p2,'x'):
            p2x= p2.x
        else:
            p2x= p2[0]
        if hasattr(p2,'y'):
            p2y= p2.y
        else:
            p2y= p2[1]
        delta_x = p1x - p2x
        delta_y = p1y - p2y
        return math.sqrt(delta_x*delta_x + delta_y*delta_y)	

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO - DONE - return the nearest waypoint for given position
        best_waypoint = self.best_waypoint
        if self.waypoints is not None:
            waypoints = self.waypoints.waypoints
            min_dist = self.distance(pose.position, waypoints[0].pose.pose.position)
            for i, point in enumerate(waypoints):
                dist = self.distance(pose.position, point.pose.pose.position)
                if dist < min_dist:
                    best_waypoint = i
                    min_dist = dist
        self.best_waypoint = best_waypoint
        return best_waypoint
        
    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO - DONE - Use tranform and rotation to calculate 2D position of light in image

        # From quaternion to Euler angles:
        x = self.pose.pose.orientation.x
        y = self.pose.pose.orientation.y
        z = self.pose.pose.orientation.z
        w = self.pose.pose.orientation.w
        
        # Determine car heading:
        t3 = +2.0 * (w * z + x*y)
        t4 = +1.0 - 2.0 * (y*y + z*z)
        theta = math.degrees(math.atan2(t3, t4))

        Xcar = (cord_y-self.pose.pose.position.y)*math.sin(math.radians(theta))-(self.pose.pose.position.x-cord_x)*math.cos(math.radians(theta))
        Ycar = (cord_y-self.pose.pose.position.y)*math.cos(math.radians(theta))-(cord_x-self.pose.pose.position.x)*math.sin(math.radians(theta))

        self.distance_to_light = Xcar
        self.deviation_of_light = Ycar

        objectPoints = np.array([[float(Xcar), float(Ycar), 0.0]], dtype=np.float32)

        rvec = (0,0,0)
        tvec = (0,0,0)

        cameraMatrix = np.array([[fx,  0, image_width/2],
                                [ 0, fy, image_height/2],
                                [ 0,  0,  1]])
        distCoeffs = None

        ret, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

        x = int(ret[0,0,0])
        y = int(ret[0,0,1])

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO - DONE - use light location to zoom in on traffic light in image
        
        if ((x is None) or (y is None) or (x < 0) or (y<0) or
            (x>self.config['camera_info']['image_width']) or (y>self.config['camera_info']['image_height'])):
            return TrafficLight.UNKNOWN
        else:
            # Cropped for the classifier from Markus which would need to ingest bgr8 images that are of size 300x200 (Can be changed if needed)
            cropped_image = cv2.resize(cv_image,(300, 200), interpolation = cv2.INTER_CUBIC)

        #Get classification
        return self.light_classifier.get_classification(cropped_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        #TODO - DONE - find the closest visible traffic light (if one exists)
        
        #Find where the vehicle is and safe it in car position
        light_positions = self.config['light_positions']
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
            if car_position is not None:
                self.last_car_position = car_position
        
        # Attribute the light positions to waypoints
        light_pos_wp = []
        if self.waypoints is not None:
            wp = self.waypoints
            for i in range(len(light_positions)):
                # FIX: See note in get_closest_waypoint_light. l_pos can't be a list!
                # cxe: I just changed distance function to handle both. Maybe a bad idea.
		# AJankl: Put it back to the state from my implmentation lpos is not a list cause it says light_positions[i] so its only one element of the array
                l_pos = self.get_closest_waypoint_light(wp, light_positions[i])
                light_pos_wp.append(l_pos)
            self.last_light_pos_wp = light_pos_wp
        else:
            light_pos_wp = self.last_light_pos_wp
            
        # Get the id of the next light
        if self.last_car_position > max(light_pos_wp):
             light_num_wp = min(light_pos_wp)
        else:
            light_delta = light_pos_wp[:]
            light_delta[:] = [x - self.last_car_position for x in light_delta]
            light_num_wp = min(i for i in light_delta if i > 0) + self.last_car_position

        light_idx = light_pos_wp.index(light_num_wp)
        light = light_positions[light_idx]
        
        # FIX: distance_light does not seem to be defined.
        #light_distance = self.distance_light(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        light_distance = self.distance(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        
	#Fix changed handling of simulator. Not being done in this function anymore
        if light:
            if light_distance >= self.IGNORE_FAR_LIGHT:
                return -1, TrafficLight.UNKNOWN
            else:
                state = self.get_light_state(light)
                return light_num_wp, state
            
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def process_traffic_lights_simulation(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        #TODO - DONE - find the closest visible traffic light (if one exists)
        
	#Get the traffic light positions not from the config but from the vehicle/traffic_lights topic
        light_positions = []
	for i in range(len(self.lights)):
		light_positions.append(self.lights[i].pose.pose.position)

        #Find where the vehicle is and safe it in car position
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
            if car_position is not None:
                self.last_car_position = car_position
        
        # Attribute the light positions to waypoints
        light_pos_wp = []
        if self.waypoints is not None:
            wp = self.waypoints
            for i in range(len(light_positions)):
                # FIX: See note in get_closest_waypoint_light. l_pos can't be a list!
                # cxe: I just changed distance function to handle both. Maybe a bad idea.
		# AJankl: Put it back to the state from my implmentation lpos is not a list cause it says light_positions[i] so its only one element of the array
                l_pos = self.get_closest_waypoint_light(wp, light_positions[i])
                light_pos_wp.append(l_pos)
            self.last_light_pos_wp = light_pos_wp
        else:
            light_pos_wp = self.last_light_pos_wp
            
        # Get the id of the next light
        if self.last_car_position > max(light_pos_wp):
             light_num_wp = min(light_pos_wp)
        else:
            light_delta = light_pos_wp[:]
            light_delta[:] = [x - self.last_car_position for x in light_delta]
            light_num_wp = min(i for i in light_delta if i > 0) + self.last_car_position

        light_idx = light_pos_wp.index(light_num_wp)
        light = light_positions[light_idx]
        
        # FIX: distance_light does not seem to be defined.
        #light_distance = self.distance_light(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        light_distance = self.distance(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        
	#Fix changed handling of simulator. Not being done in this function anymore
        if light:
            if light_distance >= self.IGNORE_FAR_LIGHT:
                return -1, TrafficLight.UNKNOWN
            else:
                state = self.lights[light_idx].state
                return light_num_wp, state
            
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN
        
    def get_closest_waypoint_light(self, wp, l_pos):
        best_waypoint = None
        waypoints = wp.waypoints
        # l_pos should not be a list. 
        # The second arg, waypoints[0].pose.pose.position is type: geometry_msgs.msg._Point.Point
        min_dist = self.distance(l_pos, waypoints[0].pose.pose.position)
        for i, point in enumerate(waypoints):
            dist = self.distance(l_pos, point.pose.pose.position)
            if dist < min_dist:
                best_waypoint = i
                min_dist = dist
        return best_waypoint

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
