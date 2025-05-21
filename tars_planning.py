# 02_tars_planning.py

import numpy as np
from tars_config import (
    MAX_STEER, MAX_SPEED, MIN_SPEED, STRAIGHT_SPEED, TURN_THRESHOLD,
    WHEELBASE, LOOKAHEAD_DISTANCE
)

class LanePlanner:
    def __init__(self):
        # self.MAX_STEER = 0.5
        # self.MAX_SPEED = 0.5
        # self.MIN_SPEED = 0.5
        # self.STRAIGHT_SPEED = 0.33
        # self.TURN_THRESHOLD = 0.15
        # self.DEFAULT_LANE_WIDTH = 700  # This is no longer needed
        self.MAX_STEER = MAX_STEER
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.STRAIGHT_SPEED = STRAIGHT_SPEED
        self.TURN_THRESHOLD = TURN_THRESHOLD
        self.WHEELBASE = WHEELBASE
        self.LOOKAHEAD_DISTANCE = LOOKAHEAD_DISTANCE

    # Modified plan method to accept the single lane center x coordinate and image center x coordinate
    # It will now calculate the deviation based on these two points.
    def plan(self, lane_center_x: float | None, image_center_x: int):
        if lane_center_x is None:
            return 0.0, 0.0, 0.0

        # Calculate deviation from the image center
        deviation = (lane_center_x - image_center_x) / image_center_x
        deviation = np.clip(deviation, -1.0, 1.0)

        # Pure Pursuit calculation
        # Convert deviation to a target point in the look-ahead distance
        target_x = self.LOOKAHEAD_DISTANCE * deviation
        
        # Calculate the steering angle using Pure Pursuit formula
        # α = arctan(2L * sin(α) / ld)
        # where L is wheelbase, ld is look-ahead distance
        steering_angle = np.arctan2(2 * self.WHEELBASE * target_x, 
                                  self.LOOKAHEAD_DISTANCE**2)
        
        # Convert steering angle to wheel speed difference
        # For 4-wheel independent drive, we'll use the steering angle to determine
        # the speed difference between left and right wheels
        steering = np.clip(steering_angle, -self.MAX_STEER, self.MAX_STEER)

        # Calculate base speed
        if abs(steering) > 0.08:
            linear_speed = self.MIN_SPEED
        else:
            linear_speed = self.STRAIGHT_SPEED

        # Clamp linear speed
        linear_speed = np.clip(linear_speed, -self.MAX_SPEED, self.MAX_SPEED)

        return linear_speed, steering, deviation
