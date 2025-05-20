# 02_tars_planning.py

import numpy as np

class LanePlanner:
    def __init__(self):
        self.MAX_STEER = 0.5
        self.MAX_SPEED = 0.5
        self.MIN_SPEED = 0.5
        self.STRAIGHT_SPEED = 0.33
        self.TURN_THRESHOLD = 0.15
        # self.DEFAULT_LANE_WIDTH = 700  # This is no longer needed

    # Modified plan method to accept the single lane center x coordinate and image center x coordinate
    # It will now calculate the deviation based on these two points.
    def plan(self, lane_center_x: float | None, image_center_x: int):
        deviation = 0.0
        steering = 0.0
        linear_speed = 0.0

        # If no lane center is detected, stop the robot.
        if lane_center_x is None:
            return 0.0, 0.0, 0.0

        # Calculate deviation from the image center
        deviation = (lane_center_x - image_center_x) / image_center_x

        # Clamp deviation to a reasonable range
        deviation = np.clip(deviation, -1.0, 1.0)

        # Calculate steering proportional to deviation
        steering = self.MAX_STEER * deviation
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)

        # Determine linear speed based on steering action
        # If steering is non-zero (robot is turning), use MIN_SPEED (0.5)
        # Otherwise (robot is going straight), use STRAIGHT_SPEED (0.3)
        if abs(steering) > 0.08: # Use a small tolerance for floating point comparison
             linear_speed = self.MIN_SPEED # 0.5
        else:
             linear_speed = self.STRAIGHT_SPEED # 0.3

        # Clamp linear speed to the defined range
        linear_speed = np.clip(linear_speed, -self.MAX_SPEED, self.MAX_SPEED)

        # Return the calculated control values and the deviation for potential logging/debugging
        return linear_speed, steering, deviation
