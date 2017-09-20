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
import numpy as np
import time

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

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=0)
        # A publisher to show us the cropped images. Maybe disable when no longer needed.
        # Refer to styx/conf.py for more hints.
        self.cropped_pub = rospy.Publisher("/crop_image",Image, queue_size=2) 

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
        self.IGNORE_FAR_LIGHT = 25.0
        self.IGNORE_FAR_LIGHT_SIMULATOR = 50.0
        self.simulator_debug_mode = 1
        self.save_images_simulator = 1

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # we only need the message once, unsubscribe as soon as we got the message
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
	self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint """
        self.has_image = True
        self.camera_image = msg
	#rospy.loginfo('[TLNode] Start of TL Node')
	if(self.simulator_debug_mode==1):
		light_wp, state = self.process_traffic_lights_simulation()
	elif(self.simulator_debug_mode==0):
		light_wp, state = self.process_traffic_lights()
	#rospy.loginfo('[TLNode] End of process traffic lights with result' + str(state) + str(light_wp))

	'''
	Publish upcoming red lights at camera frequency.
	Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
	of times till we start using it. Otherwise the previous stable 	state is
	used.
	'''
	if self.state != state:
	    self.state_count = 0
	    self.state = state
	    #rospy.loginfo('[TLNode_Simu] End of TL Node. Detected change in traffic light but will need more confirmations before publishing in case its a redlight')
	if self.state_count >= STATE_COUNT_THRESHOLD:
	    self.last_state = self.state
	    light_wp = light_wp if state == TrafficLight.RED else -1
	    self.last_wp = light_wp
	    #rospy.loginfo('[TLNode_Simu] End of TL Node publishing upcoming redlight at waypoint ' + str(light_wp) + str(state))
	    self.upcoming_red_light_pub.publish(Int32(light_wp))

	if self.state_count < STATE_COUNT_THRESHOLD:	
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

	#AJankl copied this function from: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    def Quaternion_toEulerianAngle(self, x, y, z, w):
	ysqr = y*y
	
	t0 = +2.0 * (w * x + y*z)
	t1 = +1.0 - 2.0 * (x*x + ysqr)
	X = math.degrees(math.atan2(t0, t1))

	t2 = +2.0 * (w*y - z*x)
	t2 =  1 if t2 > 1 else t2
	t2 = -1 if t2 < -1 else t2
	Y = math.degrees(math.asin(t2))

	t3 = +2.0 * (w * z + x*y)
	t4 = +1.0 - 2.0 * (ysqr + z*z)
	Z = math.degrees(math.atan2(t3, t4))

	return Z 

    def project_to_image_plane_2(self, point_in_world):
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

        #TODO Use tranform and rotation to calculate 2D position of light in image
        # Ouch... Then there's this:
        # https://discussions.udacity.com/t/focal-length-wrong/358568/12

        # Variable Naming Note: Abc --> A=object (Car or stopLight), b=coord system (world,zeroed,car), c=geom variable
        #======== DONE: Need to get this information out of the input argument.
        Lwx,Lwy,Lwz = point_in_world[0], point_in_world[1], 0
        #======== DONE: Need to get this information out of its ROS package?
        Cwx,Cwy,Cwz = self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z 
	#AJankl calculated this via converting a quaternion to euler angles
	x_=self.pose.pose.orientation.x
	y_=self.pose.pose.orientation.y
	z_=self.pose.pose.orientation.z
	w_=self.pose.pose.orientation.w
	Cwtheta = math.radians(self.Quaternion_toEulerianAngle(x_,y_,z_,w_)) # Cwtheta in radians! Used quaternion conversion for this
        #L0x,L0y,L0z = Lwx-Cwx,Lwy-Cwy,Lwz-Cwz # Stoplight position for coords moved so car is at 0,0,0.
        L0x,L0y,L0z = Lwx-Cwx,Lwy-Cwy,0 # Stoplight position for coords moved so car is at 0,0,0.
        # Maybe check for L0y=0 conditions, already lined up (ahead or behind).
        Lctheta = math.atan2(L0y,L0x) # Direction (radians) from car to stoplight when car is at 0,0,0.
        LcR = math.sqrt(L0x*L0x + L0y*L0y) # Distance from stoplight to car.
        Lcphi = Lctheta-Cwtheta # Angle between car bearing and angle to stoplight.
        Lcx= LcR * math.cos(Lcphi) # How far stoplight is ahead of car.
        Lcy= LcR * math.sin(Lcphi) # How far stoplight is laterally over from car's current path.
        hw,hh= image_width/2, image_height/2 # Screen half width and half height.

        # Simple screen projection given aligned system.
        # https://en.wikipedia.org/wiki/3D_projection
        y = hh + int(fy*L0z/Lcx) # Screen vertical = focal_len_vert * stoplight_dist_high / stoplight_dist_ahead
        x = hw + int(fx*Lcy/Lcx) # Screen horiz = focal_len_horiz * stoplight_dist_over / stoplight_dist_ahead
        if Lcx<0 or x<-hw or x>hw or y<-hh or y>hh:
            return False #???? Or (-1,-1), basically, it's not on the screen.
        else:
            return (x,y)

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

        # Get quaternion values from self orientation
        x_ = self.pose.pose.orientation.x
        y_ = self.pose.pose.orientation.y
        z_ = self.pose.pose.orientation.z
        w_ = self.pose.pose.orientation.w
        
        # Determine car heading via quaternion to Euler conversion
	car_heading=self.Quaternion_toEulerianAngle(x_,y_,z_,w_)

	Lwx=point_in_world[0]
	Lwy=point_in_world[1]

	# How far stoplight is ahead of car.
        Lcx = (Lwy-self.pose.pose.position.y)*math.sin(math.radians(car_heading))-(self.pose.pose.position.x-Lwx)*math.cos(math.radians(car_heading))
	# How far stoplight is laterally over from car's current path.
        Lcy = (Lwy-self.pose.pose.position.y)*math.cos(math.radians(car_heading))-(Lwx-self.pose.pose.position.x)*math.sin(math.radians(car_heading))

	#Object point is already in car coordinate system now
        objectPoints = np.array([[float(Lcx), float(Lcy), 5.0]], dtype=np.float32)

	#set transfromations zero as everything is in car cosy
        rvec = (0,0,0)
        tvec = (0,0,0)

	#create camera matrix
        cameraMatrix = np.array([[fx,  0, image_width/2],
                                [ 0, fy, image_height/2],
                                [ 0,  0,  1]])
        distCoeffs = None

	# Same as simple screen projection given aligned system.
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

        x, y = self.project_to_image_plane(light)

	rospy.loginfo('[TLNode_Real] Projected points '+str(x)+' '+str(y))

        #TODO - DONE - use light location to zoom in on traffic light in image
        
        if ((x is None) or (y is None) or (x < 0) or (y<0) or
            (x>self.config['camera_info']['image_width']) or (y>self.config['camera_info']['image_height'])):
            return TrafficLight.UNKNOWN
        else:
            # Cropped for the classifier from Markus which would need to ingest bgr8 images that are of size 300x200 (Can be changed if needed)
            cv_cropped_image = cv_image[(y-150):(y+150),(x-100):(x+100)]
            # A publisher to show the cropped images. Enable definition in __init__ also.
	    cv_marked_image = cv_image.copy()
 	    cv_marked_image = cv2.drawMarker(cv_marked_image,(x,y),(0,0,255),markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)

	    if(self.save_images_simulator==1):
            	self.time=time.clock()
		path = '/home/student/Pictures/simulated/'
            	cv2.imwrite(path+str(int(self.time*1000))+'.jpg',cv_cropped_image)
	    	rospy.loginfo('[TLNode_Real] Saved Image from simulator ')
	    self.cropped_pub.publish(self.bridge.cv2_to_imgmsg(cv_cropped_image, "bgr8"))
	    #self.cropped_pub.publish(cropped_image)
            

        #Get classification
        return self.light_classifier.get_classification(cv_cropped_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        #TODO - DONE - find the closest visible traffic light (if one exists)
        
        #Find where the vehicle is and safe it in car position
        #light_positions = self.config['light_positions']
        #if self.pose:
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
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
	if len(light_pos_wp) is not 0:
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

	#rospy.loginfo('[TLNode_Simu] Last light position as given by topic' + str(light_positions[len(light_positions)-1]))
	
        #Find where the vehicle is and safe it in car position
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
            if car_position is not None:
                self.last_car_position = car_position

	#rospy.loginfo('[TLNode_Simu] Current car ' + str(car_position))

        # Attribute the light positions to waypoints
        light_pos_wp = []
        if self.waypoints is not None:
            wp = self.waypoints
            for i in range(len(light_positions)):
                # FIX: See note in get_closest_waypoint_light. l_pos can't be a list!
                # cxe: I just changed distance function to handle both. Maybe a bad idea.
		# AJankl: Put it back to the state from my implmentation lpos is not a list cause it says light_positions[i] so its only one element of the array
                l_pos = self.get_closest_waypoint_light(wp, light_positions[i])
		
		#if(i==(len(light_positions)-1)):
			#rospy.loginfo('[TLNode_Simu] Light: '+ str(i) + ' at position ' + str(light_positions[i]))
			#rospy.loginfo('[TLNode_Simu] Belongs to waypoint: '+ str(l_pos) + ' at position ' + str(wp.waypoints[l_pos].pose.pose.position))

                light_pos_wp.append(l_pos)
            self.last_light_pos_wp = light_pos_wp
        else:
            light_pos_wp = self.last_light_pos_wp

        # Get the id of the next light
	if len(light_pos_wp) is not 0:
		# This branch gets taken in case the vehicle is almost through the loop. After the last light. Then the next light can only be the one that comes first in the loop.
        	if self.last_car_position > max(light_pos_wp):
			light_num_wp = min(light_pos_wp)
        	else:
            		light_delta = light_pos_wp[:]
            		light_delta[:] = [x - self.last_car_position for x in light_delta]
            		light_num_wp = min(i for i in light_delta if i > 0) + self.last_car_position

        	light_idx = light_pos_wp.index(light_num_wp)
        	light = light_positions[light_idx]

		#rospy.loginfo('[TLNode_Simu] Light identified to be nearest to car: '+ str(light_idx) + ' at position ' + str(light))
		#rospy.loginfo('[TLNode_Simu] Belongs to waypoint: '+ str(light_num_wp) + ' at position ' + str(self.waypoints.waypoints[light_num_wp].pose.pose.position))
		#rospy.loginfo('[TLNode_Simu] Car is at waypoint: '+ str(car_position) + ' at position ' + str(wp.waypoints[car_position].pose.pose.position))
        
        	# FIX: distance_light does not seem to be defined.
        	#light_distance = self.distance_light(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        	light_distance = self.distance(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)

		#rospy.loginfo('[TLNode_Simu] Distance to light: '+ str(light_distance))
        
	#Fix changed handling of simulator. Not being done in this function anymore
        if light:
            if light_distance >= self.IGNORE_FAR_LIGHT_SIMULATOR:
                return -1, TrafficLight.UNKNOWN
            else:
                state = self.lights[light_idx].state
		#rospy.loginfo('[TLNode_Simu] Light is in state: '+ str(state))
		#rospy.loginfo('[TLNode_Simu] Return values therefore: '+ str(light_num_wp)+ ' , '+str(state))
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
