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
        self.cropped_pub = rospy.Publisher("/crop_image",Image, queue_size=100) 

        self.bridge = CvBridge()
	#TODO Markus Meyerhofe. Please uncomment in case classifier is up and running
        #self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.best_waypoint = 0
        self.last_car_position = 0
        self.last_light_pos_wp = []
        self.IGNORE_FAR_LIGHT_REAL = 21.5
        self.IGNORE_FAR_LIGHT_SIMULATOR_DATA_COLLECTION = 25.0
        self.IGNORE_FAR_LIGHT_SIMULATOR = 50.0
        self.IGNORE_LOW_DISTANCE_LIGHT_SIMULATOR = 1.0
        self.simulator_debug_mode = 1
        self.simulator_classifier_mode = 0
        self.realimages_classifier_mode = 0
        self.save_images_simulator = 0
        self.save_images_real = 0

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

        self.has_image = True
        self.camera_image = msg
	#rospy.loginfo('[TLNode] Start of TL Node')
	if(self.simulator_debug_mode==1):
		light_wp, state = self.process_traffic_lights_simulation()
	elif((self.realimages_classifier_mode==1 or self.simulator_classifier_mode==1) and (self.simulator_debug_mode==0)):
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
        objectPoints = np.array([[float(Lcx), float(Lcy), 0.0]], dtype=np.float32)

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

	if x>(image_width) or y>(image_height):
            	return (False, False) # basically, it's not on the screen.
        else:
        	return (x,y)


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

	if (self.simulator_classifier_mode==1):
        	x, y = self.project_to_image_plane(light)
	if (self.realimages_classifier_mode==1):
        	x, y =  684, 548

	rospy.loginfo('[TLNode_Real] Projected points '+str(x)+' '+str(y))

        #TODO - DONE - use light location to zoom in on traffic light in image
        if ((x is False) or (y is False)):
            return TrafficLight.UNKNOWN
        else:
            # Cropped for the classifier from Markus which would need to ingest bgr8 images that are of size 300x200 (Can be changed if needed)
	    #height, width, channels = cv_image.shape
	    #rospy.loginfo('[TLNode_Real] Image size ' + str(height)+','+str(width)+','+str(channels),)
	    cv_cropped_image = cv_image.copy()
	    if (self.simulator_classifier_mode==1):
            	#cv_cropped_image = cv_image[(y-150):(y+150),(x-100):(x+100)]
        	cv_cropped_image = cv_image[(y-225):(y+225),(x-150):(x+150)]
	    	cv_cropped_image = cv2.resize(cv_cropped_image,(200,300),interpolation = cv2.INTER_CUBIC)
	    if (self.realimages_classifier_mode==1):
        	cv_cropped_image = cv_image[(y-375):(y+375),(x-250):(x+250)]
	    	cv_cropped_image = cv2.resize(cv_cropped_image,(200,300),interpolation = cv2.INTER_CUBIC)


	    if(self.save_images_simulator==1):
            	self.time=time.clock()
		path = '/home/student/Pictures/simulated/'
            	cv2.imwrite(path+str(int(self.time*1000))+'.jpg',cv_cropped_image)
	    	rospy.loginfo('[TLNode_Real] Saved Image from simulator ')

	    if(self.save_images_real==1):
            	self.time=time.clock()
		path = '/home/student/Pictures/bagfiles/'
            	cv2.imwrite(path+str(int(self.time*1000))+'.jpg',cv_cropped_image)
	    	rospy.loginfo('[TLNode_Real] Saved Image from bagfile ')
	    
	    # A publisher to show the cropped images.
	    self.cropped_pub.publish(self.bridge.cv2_to_imgmsg(cv_cropped_image, "bgr8"))
            
        #Get classification
	#TODO Markus Meyerhofer. Please change in case classifier is up and running
        #return self.light_classifier.get_classification(cv_cropped_image)
	return TrafficLight.UNKNOWN

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
        light_positions = self.config['stop_line_positions']
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
	if len(light_pos_wp) is not 0:
		if self.last_car_position > max(light_pos_wp):
		     light_num_wp = min(light_pos_wp)
		else:
		    light_delta = light_pos_wp[:]
		    light_delta[:] = [x - self.last_car_position for x in light_delta]
		    light_num_wp = min(i for i in light_delta if i >= 0) + self.last_car_position

		light_idx = light_pos_wp.index(light_num_wp)
		light = light_positions[light_idx]

		#rospy.loginfo('[TLNode_Real] Light identified to be nearest to car: '+ str(light_idx) + ' at position ' + str(light))
		#rospy.loginfo('[TLNode_Real] Belongs to waypoint: '+ str	(light_num_wp) + ' at position ' + str(self.waypoints.waypoints[light_num_wp].pose.pose.position))
		rospy.loginfo('[TLNode_Real] Car is at waypoint: '+ str(car_position) + ' at waypoint position ' + str(wp.waypoints[car_position].pose.pose.position) + ' at car position ' + str(self.pose.pose.position))
		
		# FIX: distance_light does not seem to be defined.
		#light_distance = self.distance_light(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
		light_distance = self.distance(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
		
		target_light_too_far = 1
		if(self.simulator_classifier_mode==1):
			if light_distance <=self.IGNORE_FAR_LIGHT_SIMULATOR_DATA_COLLECTION:
				target_light_too_far = 0
		if(self.realimages_classifier_mode==1):
			rospy.loginfo('[TLNode_Real] Waypoint difference: '+ str(light_num_wp - car_position))
			rospy.loginfo('[TLNode_Real] Waypoint logic state: '+ str(light_distance <=self.IGNORE_FAR_LIGHT_REAL and (light_num_wp - car_position)>=-15 and (light_num_wp - car_position)<=-6))
			if light_distance <=self.IGNORE_FAR_LIGHT_REAL and (((light_num_wp - car_position)>=-15 and (light_num_wp - car_position)<=-9) or ((light_num_wp - car_position)>30)):
				 target_light_too_far = 0
				 rospy.loginfo('[TLNode_Real] Classification to be invoked: ')


		rospy.loginfo('[TLNode_Real] Distance to light: '+ str(light_distance))
        
	#Fix changed handling of simulator. Not being done in this function anymore
        if light:
            if  target_light_too_far == 1:
                return -1, TrafficLight.UNKNOWN
            else:
		rospy.loginfo('[TLNode_Real] Invoke classification')
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

	rospy.loginfo('[TLNode_Simu] Current car ' + str(self.last_car_position))

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
            		light_num_wp = min(i for i in light_delta if i >= 0) + self.last_car_position

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
            if light_distance >= self.IGNORE_FAR_LIGHT_SIMULATOR or light_distance <=self.IGNORE_LOW_DISTANCE_LIGHT_SIMULATOR:
                return -1, TrafficLight.UNKNOWN
            else:
                state = self.lights[light_idx].state
		#rospy.loginfo('[TLNode_Simu] Light is in state: '+ str(state))
		#rospy.loginfo('[TLNode_Simu] Return values therefore: '+ str(light_num_wp)+ ' , '+str(state))
		if (self.last_car_position<=light_num_wp-25):
                	return light_num_wp-25, state
		else:
                	return -1, TrafficLight.UNKNOWN
            
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
