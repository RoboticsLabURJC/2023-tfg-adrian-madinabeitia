import numpy as np

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
    img_width = image.shape[1]
    img_height = image.shape[0]

    x = 0
    y = 0
    count = 0

    # Checks the image limits
    init = max(topw, 0)
    end = min(bottomW, img_height-1)

    for row in range(init, end):
        for col in range(img_width):

            comparison = image[row][col] != 0
            if comparison.all():
                y += row
                x += col 
                count += 1

    if count == 0:
        return (0, 0)

    return [int(x / count), int(y / count)]

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
