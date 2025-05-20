# 03_tars_control.py

class RobotController:
    def __init__(self, base):
        self.base = base

    def send_control(self, linear_speed, steering):
        self.base.base_velocity_ctrl(linear_speed, steering)
