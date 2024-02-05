import numpy as np
import cv2

def save_timestamps(file_path, timestamps):
    np.save(file_path, timestamps)


def save_profiling(file_path, profiling_data):
    with open(file_path, 'w') as file:
        for entry in profiling_data:
                file.write(entry + '\n')


def search_top_line(image):
    img_height = image.shape[0]
    first_nonzero_row = 0

    for row in range(img_height):
        if np.any(image[row] != 0):
            first_nonzero_row = row
            break
    
    return first_nonzero_row


def search_bottom_line(image):
    img_height = image.shape[0]
    last_nonzero_row = img_height - 1  # Start from the bottom

    for row in range(img_height - 1, -1, -1):
        if np.any(image[row] != 0):
            last_nonzero_row = row
            break
    
    return last_nonzero_row


def band_midpoint(image, topw, bottomW):
    img_roi = image[topw:bottomW, :]
    non_zero_pixels = cv2.findNonZero(img_roi)

    if non_zero_pixels is None:
        return (0, 0)

    centroid = np.mean(non_zero_pixels, axis=0)
    centroid = centroid.flatten().astype(int)

    return centroid.tolist()

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
