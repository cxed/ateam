from pid import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858 # this is in kg / gallon it seems
MPH_TO_MPS = 0.44704
MAX_STEERING = 0.43 # 25 degrees in radians


class Controller(object):
    def __init__(self, **kwargs):
        self.vehicle_mass = kwargs["vehicle_mass"]
        self.fuel_capacity = kwargs["fuel_capacity"]
        self.brake_deadband = kwargs["brake_deadband"]
        self.decel_limit = kwargs["decel_limit"]
        self.accel_limit = kwargs["accel_limit"]
        self.wheel_radius = kwargs["wheel_radius"]
        self.wheel_base = kwargs["wheel_base"]
        self.steer_ratio = kwargs["steer_ratio"]
        self.max_lat_accel = kwargs["max_lat_accel"]
        self.max_steer_angle = kwargs["max_steer_angle"]
        self.min_speed = kwargs["min_speed"]
        self.refresh_rate = kwargs["refresh_rate"]

        # get max speed from config
        self.max_speed = rospy.get_param('/waypoint_updater/max_speed_mph', 10) * MPH_TO_MPS

        # create a refresh rate of 50 Hz
        #self.refresh_rate = 0.02

        # initialise PID controller
        # self.linear_velocity_PID = PID(1.0, 0.1, 0.5, mn=0, mx=self.max_speed) # replaced with below
        self.throttle_control = PID(1.0, 0.1, 0.0, mn=0.0, mx=1.0)
        

        # for steering, clamp the output to +- 25 degrees (in radians)
        self.angular_velocity_PID = PID(5.0, 0.1,0.5,mn=-MAX_STEERING, mx=MAX_STEERING)
        
        # create a yaw controller
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, self.min_speed, self.max_lat_accel, self.max_steer_angle)

        self.timestamp = None

    def reset(self):
        self.linear_velocity_PID.reset()
        self.angular_velocity_PID.reset()
        

    def control(self, current_linear_velocity, current_angular_velocity, target_linear_velocity,  target_angular_velocity):

        if self.timestamp == None:
            self.timestamp = rospy.get_time()
            return 0.0, 0.0, 0.0

        delta_time = rospy.get_time() - self.timestamp
        self.timestamp = rospy.get_time()

        # presummably the mass is dependent on remaining fuel
        # so in the control step, update the mass based on the available fuel
        # although this doesn't seem to change for the project - good practice
        vehicle_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY

        # calculate a velocity error
        # velocity_error = target_linear_velocity - current_linear_velocity # removed and replaced with acceleration
        
        # calculate the acceleration required
        acceleration = max(min((target_linear_velocity - current_linear_velocity), self.accel_limit), self.decel_limit)
        
        # calculate the throttle, with limitations set between 0 and 1
        throttle = self.throttle_control.step(acceleration, delta_time)
        
        # removed and replaced with the above
        # pass the error to the PID controller, with a delta sample time
        # throttle_cmd = self.linear_velocity_PID.step(velocity_error, delta_time)
        # then limit the acceleration
        # acceleration = throttle_cmd - current_linear_velocity
        # throttle = min(acceleration, self.accel_limit)
        
        # Obtain the two components for the steering
        #corrective_steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        #predictive_steer = self.angular_velocity_PID.step(target_angular_velocity, 1.0 / self.refresh_rate)

        # add the two components to produce final steer value
        #steer = corrective_steer + predictive_steer

        steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)

        # TODO refine braking
        brake = 0
        # simple braking just so we can get the car to stop at the light
        if acceleration < 0:
            brake = 0
            throttle = 0
            if acceleration < -self.brake_deadband:
                throttle = 0
                brake = max(acceleration + self.brake_deadband, self.decel_limit) * vehicle_mass * self.wheel_radius * -1

        return throttle, brake, steer
