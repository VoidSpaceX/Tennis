import numpy as np
import time

class CalculationSpeed:
    def __init__(self, scale):
        self.scale = scale
        self.prev_center = None
        self.prev_time = None
        self.current_speed = 0.0
        
    def add_position(self, x, y):
        current_time = time.time()
        
        if self.prev_center is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                pixel_dist = np.hypot(x - self.prev_center[0], y - self.prev_center[1])
                meter_dist = pixel_dist * self.scale
                self.current_speed = meter_dist / dt
            else:
                self.current_speed = 0.0
                
        self.prev_center = (x, y)
        self.prev_time = current_time
            
    def get_speed(self):
        return self.current_speed