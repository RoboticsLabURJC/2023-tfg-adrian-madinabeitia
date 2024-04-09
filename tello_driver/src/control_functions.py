import numpy as np
import cv2

def save_timestamps(file_path, timestamps):
    np.save(file_path, timestamps)


def save_profiling(file_path, profiling_data):
    with open(file_path, 'w') as file:
        for entry in profiling_data:
                file.write(entry + '\n')

class PID:
    def __init__(self, min, max):
        self.min = min
        self.max = max

        self.prev_error = 0
        self.int_error = 0
        
        # Angular values as default
        self.KP = 1.0
        self.KD = 0.0
        self.KI = 0.0
    
    def set_pid(self, kp, kd, ki):
        self.KP = kp
        self.KD = kd
        self.KI = ki
    
    def get_pid(self, vel):        
        
        if (vel <= self.min):
            vel = self.min
        if (vel >= self.max):
            vel = self.max
        
        self.int_error += vel
        dev_error = vel - self.prev_error
        
        # Controls the integral value
        if (self.int_error > self.max):
            self.int_error = self.max
        if self.int_error < self.min:
            self.int_error = self.min

        self.prev_error = vel

        out = self.KP * vel + self.KI * self.int_error + self.KD * dev_error
        
        if (out > self.max):
            out = self.max
        if out < self.min:
            out = self.min
            
        return out
