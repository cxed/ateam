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
import os

STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

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
        self.cropped_pub = rospy.Publisher("/crop_image",Image, queue_size=100) 

        self.bridge = CvBridge()
        #TODO Markus Meyerhofer. Please uncomment in case classifier is up and running
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        #Self introduced variables being used below in the code
        #Waypoint variables making sure we always know where we are even if a message once gets missed        
        self.best_waypoint = 0
        self.last_car_position = 0
        self.last_light_pos_waypoints = []
        self.last_stop_pos_waypoints = []

        #Variables that make sure the light is not being considered if its way out
        self.IGNORE_FAR_LIGHT_REAL = 20.5
        self.IGNORE_FAR_LIGHT_SIMULATOR_DATA_COLLECTION = 25.0
        self.IGNORE_FAR_LIGHT_SIMULATOR = 25.0
        self.IGNORE_LOW_DISTANCE_LIGHT_SIMULATOR = 1.0

        #Variables for configuring this script:
        #simulator_debug_mode = 1 is for using the script with the results from /vehicle/traffic_lights
        #simulator_classifier_mode = is for using the script with the simulator with a classifier working
        #simulator_classifier_mode = is for using the script with the real images from the bagfiles with a classifier working
        self.simulator_debug_mode = False
        self.simulator_classifier_mode = 1
        self.realimages_classifier_mode = 0
        self.save_images_simulator = 0
        self.save_images_real = False
        self.c = 0 # Counter for distinct naming of images.


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        self.has_image = True
        self.camera_image = msg
        #Start of the node since and image came in. Take different branch depending on the configuration
        if self.simulator_debug_mode:
            light_wp, state = self.process_traffic_lights_simulation()
        elif ((self.realimages_classifier_mode==1 or self.simulator_classifier_mode==1) and not self.simulator_debug_mode):
            light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable         state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        if self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
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
            for i in range(len(waypoints)):
                dist = self.distance(pose.position, waypoints[i].pose.pose.position)
                if dist < min_dist:
                    best_waypoint = i
                    min_dist = dist
        self.best_waypoint = best_waypoint
        return best_waypoint

    # AJankl copied this function from:
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
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

        # Get the points in the world
        Lwx=point_in_world[0]
        Lwy=point_in_world[1]

        # How far is the stoplight ahead of the car. Do calculate it by simply using normal algebra
        Lcx = (Lwy-self.pose.pose.position.y)*math.sin(math.radians(car_heading))-(self.pose.pose.position.x-Lwx)*math.cos(math.radians(car_heading))
        # How far is the stoplight laterally over from car's current path.Do calculate it by simply using normal algebra
        Lcy = (Lwy-self.pose.pose.position.y)*math.cos(math.radians(car_heading))-(Lwx-self.pose.pose.position.x)*math.sin(math.radians(car_heading))

        #Object point is already in car coordinate system now
        objectPoints = np.array([[float(Lcx), float(Lcy), 5.0]], dtype=np.float32)

        #set transfromations zero as everything is in car cosy
        rvec = (0,0,0)
        tvec = (0,0,0)

        #create camera matrix
        cameraMatrix = np.array([[fx,  0, image_width/2 ],
                                 [ 0, fy, image_height/2],
                                 [ 0,  0, 1             ]])
        distCoeffs = None

        # Same as simple screen projection given aligned system. !! we do not do any real transformation as rvec and tvec are zeroe
        ret, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

        #Get the points in world on pixel coordinates
        x = int(ret[0,0,0])
        y = int(ret[0,0,1])

        # check if x,y basically is not on the screen. and if so just return false as something is wrong
        if x>(image_width) or y>(image_height):
            return (False, False) 
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
        # ===== Chris added this to extract images interactively on demand by doing `touch /dev/shmX`.
        #triggerfile= '/dev/shm/X'
        #if os.path.isfile(triggerfile):
            #self.c += 1
            #cv2.imwrite('/tmp/simcapture/C%05d.png'%self.c, cv_image)
            #rospy.loginfo('[TLNode_Simu] Saved Image from bagfile - %05d.png'%self.c)
            #os.remove(triggerfile)
            
        # Invoke projection only for the simulator where this works reasonably well for the
        # bagfiles just take the middle of the image and crop around that
        if (self.simulator_classifier_mode==1):
            x, y = self.project_to_image_plane(light)
        if (self.realimages_classifier_mode==1):
            x, y =  684, 548

        #TODO - DONE - use light location to zoom in on traffic light in image
        if ((x is False) or (y is False)): # if not (x and y)?
            return TrafficLight.UNKNOWN
        else:
            # Cropped around the traffic light for the classifier from Markus which would need
            # to ingest bgr8 images that are of size 300x200 (Can be changed if needed)
            # Crops are different for bagfiles and simulator as the image size is different
            cv_cropped_image = cv_image.copy()
            if (self.simulator_classifier_mode==1):
                cv_cropped_image = cv_image[(y-225):(y+225),(x-150):(x+150)]
                cv_cropped_image = cv2.resize(cv_cropped_image,(200,300),interpolation = cv2.INTER_CUBIC)
            if (self.realimages_classifier_mode==1):
                cv_cropped_image = cv_image[(y-375):(y+375),(x-250):(x+250)]
                cv_cropped_image = cv2.resize(cv_cropped_image,(200,300),interpolation = cv2.INTER_CUBIC)

            #This gets only taken if one wants to save images for the classifier training. Not being used in end system
            if(self.save_images_simulator==1):
                    self.time=time.clock()
                    path = '/home/student/Pictures/simulated/'
                    cv2.imwrite(path+str(int(self.time*1000))+'.png',cv_cropped_image)
                    rospy.loginfo('[TLNode_Simu] Saved Image from simulator ')

            if self.save_images_real:
                    self.time=time.clock()
                    path = '/tmp/simcapture/'
                    cv2.imwrite(path+str(int(self.time*1000))+'.png',cv_cropped_image)
                    rospy.loginfo('[TLNode_Real] Saved Image from bagfile ')
            
            # A publisher to show the cropped images.
            self.cropped_pub.publish(self.bridge.cv2_to_imgmsg(cv_cropped_image, "bgr8"))
            
        #Get classification
        #TODO Markus Meyerhofer. Please change in case classifier is up and running
        return self.light_classifier.get_classification(cv_cropped_image)
        #return TrafficLight.UNKNOWN

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
        
        # Attribute the light positions to waypoints to use this in later steps
        light_pos_waypoints = []
        if self.waypoints is not None:
            temp_waypoints = self.waypoints
            for i in range(len(light_positions)):
                l_pos = self.get_closest_waypoint_light(temp_waypoints, light_positions[i])
                light_pos_waypoints.append(l_pos)
            self.last_light_pos_waypoints = light_pos_waypoints
        else:
            light_pos_waypoints = self.last_light_pos_waypoints
            
        # Get the id of the next light
        if len(light_pos_waypoints) is not 0:
                # This branch gets taken in case the vehicle is almost through the loop. After the last light.
                # Then the next light can only be the one that comes first in the loop.
                if self.last_car_position > max(light_pos_waypoints):
                     closest_light_wp = min(light_pos_waypoints)
                #This branch gets taken to determine the closest car when its unclear which it is. It calculates the difference between the waypoint of the light and the car and takes the one that is closest to the car but it can't be behind the car then it gets an arbitrary high value in in waypoint difference so it won't get taken
                else:
                    waypoint_difference = []
                    for i in range(len(light_pos_waypoints)):
                        difference = light_pos_waypoints[i]-car_position
                        if(difference >=0):
                            waypoint_difference.append(difference)
                        else:
                            waypoint_difference.append(10000)
                    closest_light_wp=min(waypoint_difference) + car_position

                #With the basis of aboves determined light wp now find the actual light
                light_idx = light_pos_waypoints.index(closest_light_wp)
                light = light_positions[light_idx]
                
                #This wholes block purpose is to determine if we should even bother invoking the classifier. If its far out then I won't do so. For the bagfiles I needed to alter the logic a bit cause unluckily the loop is done so it the car switches from waypoint 60 to zero just in front of the light so the distance function is a bit more complicated.
                light_distance = self.distance(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
                
                target_light_too_far = 1
                if(self.simulator_classifier_mode==1):
                    if light_distance <=self.IGNORE_FAR_LIGHT_SIMULATOR_DATA_COLLECTION:
                        target_light_too_far = 0
                if(self.realimages_classifier_mode==1):
                    if light_distance <=self.IGNORE_FAR_LIGHT_REAL and (((closest_light_wp - car_position)>=-15 and (closest_light_wp - car_position)<=-9) or ((closest_light_wp - car_position)>30)):
                        target_light_too_far = 0
        
        #Either call the classifier or don't bother as the closest light is still far out
        if light:
            if  target_light_too_far == 1:
                return -1, TrafficLight.UNKNOWN
            else:
                rospy.loginfo('[TLNode_Real] Invoke classification')
                state = self.get_light_state(light)
                rospy.loginfo('[TLNode_Real] Classification returned: ' + str(state))
                return closest_light_wp, state
            
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
        
        #Find where the vehicle is and safe it in car position
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
            if car_position is not None:
                self.last_car_position = car_position

        #Get the stopline positions
        stop_positions = self.config['stop_line_positions']
        
        #Attribute the stopline positions to waypoints to use this in later steps
        stop_pos_waypoints = []
        if self.waypoints is not None:
            temp_waypoints = self.waypoints
            for i in range(len(stop_positions)):
                l_pos = self.get_closest_waypoint_light(temp_waypoints, stop_positions[i])
                stop_pos_waypoints.append(l_pos)
            self.last_stop_pos_waypoints = stop_pos_waypoints
        else:
            stop_pos_waypoints = self.last_stop_pos_waypoints

        #Get the traffic light positions not from the config but from the vehicle/traffic_lights topic
        light_positions = []
        for i in range(len(self.lights)):
            light_positions.append(self.lights[i].pose.pose.position)

        # Attribute the light positions to waypoints to use this in later steps
        light_pos_waypoints = []
        if self.waypoints is not None:
            temp_waypoints = self.waypoints
            for i in range(len(light_positions)):

                l_pos = self.get_closest_waypoint_light(temp_waypoints, light_positions[i])
                
                light_pos_waypoints.append(l_pos)
            self.light_pos_waypoints = light_pos_waypoints
        else:
            light_pos_waypoints = self.last_light_pos_waypoints

        # Get the id of the next light
        if len(light_pos_waypoints) is not 0:
            # This branch gets taken in case the vehicle is almost through the loop. After the last light.
            # Then the next light can only be the one that comes first in the loop.
            if self.last_car_position > max(light_pos_waypoints):
                closest_light_wp = min(light_pos_waypoints)
            # This branch gets taken to determine the closest car when it's
            # unclear which it is. It calculates the difference between the waypoint of the
            # light and the car and takes the one that is closest to the car but it can't be
            # behind the car then it gets an arbitrary high value in in waypoint difference
            # so it won't get taken
            else:
                waypoint_difference = []
                for i in range(len(light_pos_waypoints)):
                    difference = light_pos_waypoints[i]-car_position
                    if(difference >=0):
                        waypoint_difference.append(difference)
                    else:
                        waypoint_difference.append(10000)
                closest_light_wp=min(waypoint_difference) + car_position

            light_idx = light_pos_waypoints.index(closest_light_wp)
            light = light_positions[light_idx]
            #also need stopline not just line
            stoplight = stop_pos_waypoints[light_idx]

            #Calculate distance to stopline not light
            light_distance = self.distance(self.waypoints.waypoints[stoplight].pose.pose.position, self.waypoints.waypoints[self.last_car_position].pose.pose.position)

        
        # Either evaluate the light status from vehicle/traffic_lights or don't bother as the closest light is still far out.
        if light:
            if light_distance >= self.IGNORE_FAR_LIGHT_SIMULATOR:
                return -1, TrafficLight.UNKNOWN
            else:
                # cxed- Enable if needed. Trying to clean up log output.
                #rospy.loginfo('[TLNode_Real] Invoke evaluating traffic lights topic')
                state = self.lights[light_idx].state
                #rospy.loginfo('[TLNode_Simu] traffic lights topic returned: ' + str(state))
                if self.state is not self.last_state: # A new state has been detected. Worth logging.
                    if self.state == 0:
                        rospy.loginfo('[TLNode_Simu] Light change: RED')
                    elif self.state == 1:
                        rospy.loginfo('[TLNode_Simu] Light change: YELLOW')
                    elif self.state == 2:
                        rospy.loginfo('[TLNode_Simu] Light change: GREEN')
                    # 4 could be "NONE" ? I.e. already beyond any light of concern.
                    else:
                        rospy.loginfo('[TLNode_Simu] Light change: UNKNOWN '+str(self.state))

                # Since the traffic lights topic publishes the light instead of the stoplight I needed to have this logic introduced
                if (self.last_car_position<=stoplight):
                    return stoplight, state
                else:
                    return -1, TrafficLight.UNKNOWN
            
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN
        
    #Same as get closest waypoint with the difference that it does not safe the last waypoint to a permanent variable as this is only for the car pos not for the lights
    def get_closest_waypoint_light(self, given_waypoints, light_position):
        best_waypoint = None
        waypoints = given_waypoints.waypoints
        min_dist = self.distance(light_position, waypoints[0].pose.pose.position)
        for i in range(len(waypoints)):
            dist = self.distance(light_position, waypoints[i].pose.pose.position)
            if dist < min_dist:
                best_waypoint = i
                min_dist = dist
        return best_waypoint

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
